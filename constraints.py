"""Hard shape constraints and the projection machinery that lets a first-order minimizer run ON the
constraint manifold.

This replaces the eqSoftBody PENALTY springs with equality CONSTRAINTS. The springs hold the shape
with a stiff quadratic (kEdge ~ 1/l0^2, kArea ~ 1/A0^2); those stiff modes are the fastest-oscillating
directions in the Hessian, they set FIRE's timestep ceiling (dtMax = 0.03, above which it goes NaN)
and they dominate the condition number that sets FIRE's linear convergence rate. Rigidity is the
stiff-spring limit, so constraining a term is the SAME model with the penalty taken to infinity --
and taking it there removes those modes from the dynamics entirely, leaving only the physical contact
stiffness. What is left is a smaller, far better conditioned problem.

Each shape term the springs offer can be made rigid independently, and the rows are written in the
same RELATIVE / dimensionless scaling the springs use (so a residual is a fractional error and one
tolerance means the same thing for every family):

  ``edge``       C_k = |r_{k+1} - r_k|^2 / l0_p^2 - 1 = 0     one row per edge (n rows)
  ``perimeter``  C_P = (sum_k |r_{k+1} - r_k|) / P0_p - 1 = 0  one row
  ``area``       C_A = A_p / A0_p - 1 = 0                      one row, shoelace

``edge`` + ``area`` rigidifies triangles and quads completely (up to rigid motion) and leaves an n-gon
the same few internal flex modes the per-edge springs allowed. ``perimeter`` + ``area`` is the loose
alternative: shape free to mold against its neighbors, only size and shape index pinned.

Every PER-OBJECT constraint is intra-polygon, so its Jacobian is block diagonal and each block is
handled independently -- a batched (P, m, .) decomposition rather than a global one. The moment
constraints below are the exception: they couple the whole packing by construction.

Two operations are exposed:

  projectVector     removes the constraint-normal component of a force or velocity, leaving the
                    tangent-space part.
  projectPositions  SHAKE: Newton-iterates the minimum-norm correction until the constraints are
                    satisfied to ~1e-15, i.e. retracts a drifted configuration back onto the manifold.

Both use RAW coordinates with no periodic wrap, matching softBody: the constraints are intrinsic to a
single polygon, and ``wrapPolygonsIntoCell`` only ever translates a WHOLE polygon by a cell vector,
which leaves every edge length, the perimeter and the area exactly invariant. So wrapping and
constraining commute.

Both are built from a rank-revealing SVD of J itself, never of the Gram matrix J J^T. That matters
twice over. First, some combinations are rank deficient BY CONSTRUCTION -- the perimeter is a function
of the edge lengths, and a triangle's area is already determined by its three edge lengths -- so
truncating the SVD drops the redundant direction cleanly where a ridge would only damp it (see
``redundancyReason``). Second, forming J J^T squares the condition number, and the ridge needed to
keep it invertible leaks ~1e-9 of the constraint-NORMAL direction back into the supposedly tangential
force, which would put a floor under the residual at exactly the tolerance we are trying to beat.
Working from J keeps the projector exact to roundoff.

Vertex counts are NOT assumed uniform: the blocks are padded to the largest count and carried with a
validity mask, so a 4-gon wall around 32-gons or a mix of shapes all work. Padded slots contribute
all-zero rows, which the truncated SVD drops for free.

``targetArea`` is per POLYGON but ``targetEdgeLength`` is per VERTEX (the edge leaving it), so the
edge targets are gathered through the same ragged index table as the coordinates -- see
``edgeTargets``.

Two flavors of constraint live here, and they compose:

  ShapeConstraints         per-OBJECT equalities -- every polygon meets its own targets. Block
                           diagonal, so the projection is a batched per-polygon decomposition.
  DistributionConstraints  global MOMENT equalities -- only ``sum_i t_i^k`` is held, leaving the
                           individual shapes free to trade with each other. A handful of dense rows
                           coupling the whole packing.
  CompositeConstraints     both at once, exactly, without ever forming a global dense Jacobian.
"""

import warnings

import numpy as np



# np.linalg.qr ONLY TAKES STACKED (3-D) INPUT FROM NUMPY 1.22. The stacked svd and solve used elsewhere
# in this file have accepted stacks since 1.8, so this one call is the whole portability surface, and on
# an older numpy it raises "Array must be two-dimensional" from deep inside the constraint projection --
# which reads as a malformed Jacobian rather than a version gap. Probed ONCE at import, because
# _qrFactor is on the per-step path and the loop fallback is only for the old case.
try:
    np.linalg.qr(np.zeros((1, 2, 2)))
    _STACKED_QR = True
except (np.linalg.LinAlgError, ValueError):
    _STACKED_QR = False

_RANK_RCOND = 1e-12
_MIN_EDGE_LENGTH = 1e-15
# Floor on |A| in the shape index P / sqrt(A), whose gradient carries A^(-3/2). A polygon that has
# collapsed or folded to zero signed area would otherwise divide by zero; the constraint is meaningless
# there anyway, and self-repulsion is what actually prevents it.
_MIN_AREA = 1e-30
# Floor on the DIMENSIONLESS distortion d_i when a negative moment exponent is in play. Unlike the
# deviation families, whose k = -1 barrier diverges before its quantity can reach zero, a polygon
# constrained in DIRECT distortion can sit exactly at d = 0: it is a regular polygon, an entirely
# ordinary state, and the isoperimetric theorem makes it the floor rather than a singularity to be
# avoided. The row is meaningless there, so the base is clamped instead of dividing by zero.
_MIN_DISTORTION = 1e-12
# Halvings allowed when backtracking a moment retraction step (2^-25 ~ 3e-8 of the Newton step).
_BACKTRACK_STEPS = 25
# How far over the isoperimetric bound a target area may sit before the set is called infeasible (see
# ``ShapeConstraints.infeasibleReason``). A REGULAR polygon sits exactly ON the bound, and
# ``generateEquilateralPolygons`` builds regular polygons, so the default target set lands a fraction of
# an ulp over it -- measured +2.2e-16 for n = 4, 6, 8. The slack has to clear that comfortably while
# staying far below anything that matters: real infeasibility from independently drawn area and
# perimeter targets came in at 2.2e-2, eleven orders above this.
_FEASIBILITY_SLACK = 1e-9
# Conditioning below which moment rows are numerically parallel and their residual stops meaning
# anything. Measured on 8 polygons ramping the shape deficit down 1000x: 2 rows held 8.6e-02,
# 4 rows 2.7e-02, 6 rows collapsed to 2.0e-09 while still reporting a 1.6e-12 residual.
_MOMENT_CONDITIONING_FLOOR = 1e-3

# Smallest deviation a NEGATIVE moment exponent will divide by. A barrier row carries delta^(k-1) with
# k < 1, so a term sitting exactly at its ideal is a division by zero -- and a single NaN launders
# through the rank test into an all-NaN Jacobian that reads as full rank. Floored rather than guarded
# downstream, where the damage is already done.
_MIN_DEVIATION = 1e-12

# Relative gain in the moment residual that counts as PROGRESS, and how many consecutive passes may
# fail to make it before the alternation gives up. A converging retraction beats this comfortably --
# the four-row set the budget was sized for gained about 39% a pass -- so these only ever fire on a
# target the geometry cannot deliver, where the remaining budget buys nothing and is charged once per
# minimizer step. See CompositeConstraints.projectPositions.
_MOMENT_STALL_GAIN = 0.01
_MOMENT_STALL_PATIENCE = 8


# UNVERIFIED(Cam)
class _RaggedBlocks:
    """The padded per-polygon block structure shared by every constraint flavor here.

    RAGGED BY DESIGN. Vertex counts differ in any real packing -- a 4-gon wall around 32-gons, or
    mixed shapes -- so the blocks are PADDED to the largest count and carried with a validity mask
    rather than reshaped to a dense (P, n, 2). Padded slots contribute all-zero constraint rows, which
    the rank-revealing SVD drops for free, so padding costs accuracy nothing.

    A CONTAINER is excluded: it is a pinned wall, so every column of its block would be zeroed and its
    rows carry no information. Excluding it here is what keeps a wall out of every constraint and
    every moment sum -- a signed wall area of -1 mixed into ``sum_i A_i`` alongside 0.16-sized
    polygons poisons the whole distribution.

    Caches only the structure; targets and coordinates are read from the packing on every call, so
    retargeting (e.g. ``setBiPerimeter``) is picked up with no rebuild.
    """

    def __init__(self, packing):
        starts = np.asarray(packing.startIndices, dtype = int)
        container = getattr(packing, "containerIndex", None)
        numBlocks = packing.numPolygons if container is None else int(container)
        counts = np.diff(starts)[:numBlocks]
        if counts.size == 0:
            raise ValueError("constraints need at least one non-container polygon.")

        self.numConstrained = int(numBlocks)
        self.numCoveredVertices = int(starts[numBlocks])
        self.numVertices = int(packing.numVertices)
        self.counts = counts
        self.n = int(counts.max())
        self.ragged = bool(np.any(counts != counts[0]))
        self.numPolygons = self.numConstrained

        local = np.arange(self.n)[None, :]
        valid = local < counts[:, None]
        base = starts[:numBlocks, None]
        wrapped = np.where(valid, local, 0)
        safeCounts = np.maximum(counts, 1)[:, None]
        self.valid = valid
        self.index = base + wrapped
        self.nextIndex = base + np.where(valid, (wrapped + 1) % safeCounts, 0)
        self.prevIndex = base + np.where(valid, (wrapped - 1) % safeCounts, 0)
        self.prevLocal = np.where(valid, (wrapped - 1) % safeCounts, 0)
        # WHICH vertices a flatness family covers. A cascade flattens every OTHER vertex so the flat
        # ones can then be removed exactly; constraining all of them would drive the corners flat too
        # and collapse the polygon. The mask lives on the PACKING so it survives the constraint rebuilds
        # setConstraints performs, and so the per-object and moment forms -- and halveNumEdges -- all
        # act on the same vertices by construction rather than by luck.
        mask = getattr(packing, "diagonalMask", None)
        self.diagonalSelected = valid if mask is None else (
            valid & np.asarray(mask, dtype = bool)[self.index])

    def diagonalTargets(self, packing):
        """Per-VERTEX skip-one diagonal targets as a padded (P, maxN) block.

        Indexed by the vertex the diagonal is centred on, i.e. entry k is the target for
        ``|r_{k+1} - r_{k-1}|``. Gathered through the ragged index table exactly as ``edgeTargets`` is,
        and for the same reason."""
        return packing.targetDiagonal[self.index]

    def edgeTargets(self, packing):
        """Per-EDGE rest lengths as a padded (P, maxN) block.

        ``packing.targetEdgeLength`` is indexed by VERTEX (the edge leaving it), not by polygon, so it
        has to be gathered through the ragged index table like the coordinates are. Slicing it
        ``[:numPolygons]`` instead silently hands every polygon the first few vertices' targets, which
        is correct only when the packing happens to be uniform and monodisperse."""
        return packing.targetEdgeLength[self.index]

    def _frame(self, packing):
        """(r, rNext, rPrev, e) as padded (P, maxN, 2) blocks, gathered through the ragged index
        tables. Edges of padded slots are zeroed so they cannot contribute."""
        pos = packing.positions.reshape(-1, 2)
        r = pos[self.index]
        rNext = pos[self.nextIndex]
        rPrev = pos[self.prevIndex]
        e = np.where(self.valid[:, :, None], rNext - r, 0.0)
        return r, rNext, rPrev, e

    def edgeLengths(self, packing):
        """Padded (P, maxN) actual edge lengths; padded slots are zero."""
        _, _, _, e = self._frame(packing)
        return np.sqrt(np.einsum("pkc,pkc->pk", e, e))

    def diagonalSpans(self, packing):
        """Padded (P, maxN, 2) skip-one vectors ``r_{k+1} - r_{k-1}``; padded slots are zero.

        The vertex itself does not appear, which is what makes this a measure of the TURNING ANGLE at
        k rather than of its edges: with edges of length l meeting at turning angle theta, the span is
        ``2 l cos(theta/2)``."""
        _, rNext, rPrev, _ = self._frame(packing)
        return np.where(self.valid[:, :, None], rNext - rPrev, 0.0)

    def diagonalLengths(self, packing):
        """Padded (P, maxN) skip-one distances; padded slots are zero."""
        span = self.diagonalSpans(packing)
        return np.sqrt(np.einsum("pkc,pkc->pk", span, span))

    # UNVERIFIED(Cam)
    def regularShapeIndex(self):
        """(P,) floor of ``P / sqrt(A)`` per polygon, ``sqrt(4 n tan(pi/n))``: 4 for a square, ~3.72
        for a hexagon. Depends on the vertex count, which is why the raw shape index of a mixed-n
        packing cannot be summed -- ``quantity('shape')`` divides it out."""
        counts = self.counts.astype(float)
        return np.sqrt(4.0 * counts * np.tan(np.pi / counts))

    def idealEdgeLengths(self, packing, kappa = None):
        """(P,) edge length an EQUILATERAL polygon of shape index ``kappa`` would have at each
        polygon's own ACTUAL area: ``kappa sqrt(A) / n``.

        ``kappa = None`` means each polygon's own regular floor ``sqrt(4 n tan(pi/n))``, which reduces
        this to ``sqrt(4 A tan(pi/n) / n)`` -- the regular n-gon's edge, which is what the deviation
        families measure against.

        The point of the argument is that the ideal is derived from the polygon's OWN area rather than
        from a stored target, so it follows a size that is free to move. That is what lets a shape be
        pinned while its size stays a degree of freedom."""
        counts = self.counts.astype(float)
        areas = np.maximum(np.abs(self.areas(packing)), _MIN_AREA)
        index = self.regularShapeIndex() if kappa is None else float(kappa)
        return index * np.sqrt(areas) / counts

    def flatTargets(self, packing):
        """Padded (P, maxN) target flatness per vertex, from ``packing.flatTarget``; 1.0 if unset.

        PER VERTEX and RAMPABLE, which a collective row cannot be. A single row holding the SUM of
        d/(a+b) is either degenerate or ineffective and cannot be both: aimed at exactly ``count`` it
        does force every term to 1 (each is at most 1, so the sum can only reach count if all do), but
        that is a maximum of the quantity and the row's gradient vanishes there -- measured, the
        conditioning fell to 2.02e-10 and LAPACK failed. Backed off to 0.99 count it is well
        conditioned but holds only the MEAN, and 80 vertices can average 0.99 while the worst sits at
        0.80 -- measured over eight ramp rounds, which is why a packing run flattened far too slowly.

        One row per vertex has neither problem: each is pulled toward its own target, and the target
        can be walked from where that vertex started to just short of flat, keeping every row off the
        boundary where its gradient dies."""
        target = getattr(packing, "flatTarget", None)
        if target is None:
            # CAPTURE, do not command. An unset target used to read 1.0, which is exactly flat -- so
            # merely switching the family on drove every selected vertex onto the triangle-inequality
            # boundary in one projection and collapsed the polygons (measured: kappa went infinite).
            # Enabling a constraint must be inert; moving it is what setFlatTargets is for, and this
            # mirrors how the moment families capture their reference at construction.
            return self.flatness(packing)
        return np.asarray(target, dtype = float)[self.index]

    def flatness(self, packing):
        """Padded (P, maxN) skip-one distance divided by its OWN two adjacent edges, ``d / (a + b)``.

        Dimensionless, and 1 exactly when the vertex is flat -- the triangle inequality makes it the
        upper limit, reached only when the three points are collinear. Scale-free is what makes it
        usable while the polygon SIZES are a free degree of freedom: an absolute diagonal target would
        read a polygon that merely shrank as less flat, and so would fight the size DOF instead of
        flattening anything.

        Preferred over the turning angle for a reason that was measured: a live-angle constraint's
        gradient VANISHES at a straight vertex (4e-10 against 4.4 for a length-based form at 180
        degrees), so it cannot pull the last of the way to flat."""
        lengths = self.edgeLengths(packing)
        entering = np.take_along_axis(lengths, self.prevLocal, axis = 1)
        total = entering + lengths
        safe = np.where(total > _MIN_EDGE_LENGTH, total, 1.0)
        return np.where(self.valid, self.diagonalLengths(packing) / safe, 0.0)

    def areas(self, packing):
        """(P,) actual signed shoelace areas of the covered polygons."""
        r, rNext, _, _ = self._frame(packing)
        cross = np.where(self.valid,
                         r[:, :, 0] * rNext[:, :, 1] - rNext[:, :, 0] * r[:, :, 1], 0.0)
        return 0.5 * cross.sum(axis = 1)


# UNVERIFIED(Cam)
class ShapeConstraints(_RaggedBlocks):
    """Hard PER-OBJECT shape constraints: every polygon meets its own targets.

    ``area`` / ``perimeter`` / ``edge`` select which shape terms are rigid, matching the term names of
    ``Model.setSpringConstants``; at least one must be on. Rows are ordered edges, then perimeter, then
    area.
    """

    def __init__(self, packing, area = False, perimeter = False, edge = False, diagonal = False,
                 equilateral = None, flatten = False):
        if not (area or perimeter or edge or diagonal or equilateral is not None or flatten):
            raise ValueError("ShapeConstraints needs at least one of area / perimeter / edge / "
                             "diagonal / equilateral / flatten.")
        # EQUILATERAL AT A FIXED SHAPE INDEX, WITH THE SIZE FREE. ``edge = True`` cannot express this:
        # it pins each edge to a stored number, which fixes the size too. Here the target is the
        # polygon's OWN live area -- every edge must equal ``kappa sqrt(A) / n`` -- so the shape is
        # pinned (equal edges at that shape index) while the size is left as a degree of freedom for
        # the packing to trade globally.
        #
        # n rows per polygon on 2n - 3 shape+size DOF, leaving n - 3: that is n - 4 of shape plus the
        # one size. The n - 4 is the compliance, and it reaches zero at n = 4, where an equilateral
        # quadrilateral with kappa = 4 is a square and nothing else.
        self.equilateral = None if equilateral is None else float(equilateral)
        # ``edge`` means "held at a STORED target"; ``equilateral`` holds it at a LIVE one. Either way
        # the edge spring must be dropped, or it fights the constraint it duplicates -- so the energy
        # asks this rather than reading ``edge`` directly.
        self.edgeHeld = bool(edge) or self.equilateral is not None
        # PER-OBJECT FLATNESS, the endgame of a flattening ramp. The moment form holds d/(a+b) as a
        # DISTRIBUTION, which is what lets polygons share the work -- but its two rows go numerically
        # parallel as the width it holds goes to zero, so it cannot finish. Measured at n = 8: the
        # worst selected vertex plateaued at 0.9997 whether the ramp took 12 rounds or 200, with the
        # conditioning at 2.7e-07 against a floor of 1e-03. One row per SELECTED vertex, target exactly
        # 1, has no such limit. Anneal close on moments, then switch to this.
        self.flatten = bool(flatten)
        super().__init__(packing)

        self.area = bool(area)
        self.perimeter = bool(perimeter)
        self.edge = bool(edge)
        # THE SKIP-ONE DIAGONAL |r_{k+1} - r_{k-1}| PINS THE TURNING ANGLE. Edge lengths alone leave a
        # polygon free to flex -- an equilateral quadrilateral is any rhombus -- so a shape TEMPLATE
        # needs the angles too. Holding them as distances rather than angles keeps this family
        # identical in kind to ``edge``: same ragged gather, same quadratic residual, no wrapping and
        # no arctangent. Together edge + diagonal fix the shape up to rigid motion and reflection.
        self.diagonal = bool(diagonal)
        self.numPolygons = self.numConstrained
        # Rows: one per padded edge slot, then per diagonal slot, then perimeter, then area.
        self.numConstraints = (self.n * self.edge + self.n * self.diagonal
                               + self.n * (self.equilateral is not None) + self.n * self.flatten
                               + self.perimeter + self.area)
        needed = ([("targetEdgeLength", self.edge), ("targetDiagonal", self.diagonal),
                   ("targetPerimeter", self.perimeter),
                   ("targetArea", self.area)])
        missing = [name for name, on in needed if on and getattr(packing, name, None) is None]
        if missing:
            raise ValueError(f"constraints need {', '.join(missing)} set on the packing.")

    def redundancyReason(self):
        """Why this constraint set is rank deficient, or None when it has full rank.

        Rank deficiency is HANDLED (the truncated SVD drops the dependent direction), but it means a
        row is doing no work, so it is worth saying out loud once rather than silently."""
        if self.edge and self.perimeter:
            return ("perimeter is the sum of the edge lengths, so constraining both makes the "
                    "perimeter row dependent on the edge rows")
        if self.edge and self.area and self.n == 3:
            return ("a triangle's area is already fixed by its three edge lengths, so the area row "
                    "is dependent on the edge rows")
        return None

    def families(self):
        """Active families as a tuple of names, in row order."""
        return tuple(name for name, _ in self.rowFamilies())

    # UNVERIFIED(Cam)
    def rowFamilies(self):
        """``[(name, slice)]`` over the rows of ``residual`` and ``jacobian``, in the order they are
        actually assembled.

        The single source of truth for that order, because there were three copies of it and they had
        DRIFTED: ``families`` still listed ``edge, perimeter, area`` from before ``diagonal``,
        ``flatten`` and ``equilateral`` existed, so it both omitted the new families and named them in
        the wrong order. Nothing had noticed, because ``families`` was only ever printed. It stops
        being cosmetic the moment a diagnostic uses it to say WHICH family owns a broken row."""
        widths = (("edge", self.n * self.edge), ("diagonal", self.n * self.diagonal),
                  ("flatten", self.n * self.flatten),
                  ("equilateral", self.n * (self.equilateral is not None)),
                  ("perimeter", int(self.perimeter)), ("area", int(self.area)))
        out, start = [], 0
        for name, width in widths:
            if width:
                out.append((name, slice(start, start + width)))
                start += width
        return out

    def infeasibleReason(self, packing):
        """Why this target set admits NO configuration at all, or None when it is satisfiable.

        Constraining a polygon's area AND its edge lengths (or its perimeter) over-determines it, and
        the two can flatly contradict each other. For any n-gon of perimeter P the enclosed area obeys
        the isoperimetric bound

            A <= P^2 / (4 n tan(pi / n)),

        with equality only for the REGULAR polygon -- so a quad with four edges of length l can never
        enclose more than l^2. Ask for more and SHAKE cannot converge, because there is nowhere to
        converge to: the residual sticks at O(1) and a minimizer grinds forever against it.

        This is easy to trip by accident. ``generateEquilateralPolygons`` lands exactly ON the bound
        (a regular n-gon), so perturbing the area and the perimeter INDEPENDENTLY -- two calls to
        ``setLogNormalTargetArea`` / ``setLogNormalTargetPerimeter`` -- puts about half the polygons a
        hair over it. Reported in terms of the shape index p = P / sqrt(A), whose floor is
        sqrt(4 n tan(pi/n)) (4 for a square, and ~3.72 for a hexagon)."""
        if self.equilateral is not None:
            floor = self.regularShapeIndex()
            below = np.nonzero(self.equilateral < floor - 1e-12)[0]
            if below.size:
                worst = int(below[np.argmax(floor[below])])
                return (f"equilateral = {self.equilateral} is below the regular floor "
                        f"{floor[worst]:.6f} of polygon {worst}'s {int(self.counts[worst])}-gon. No "
                        f"polygon of that vertex count can have a shape index under its regular "
                        f"value, so the rows describe a shape that does not exist. Raise the shape "
                        f"index, or reduce the vertex count -- the floor RISES as n falls "
                        f"(3.5506 at 32, 4.0000 at 4).")
        if not (self.area and (self.edge or self.perimeter)):
            return None
        A0 = np.asarray(packing.targetArea, dtype = float)[:self.numConstrained]
        if self.edge:
            P0 = np.where(self.valid, self.edgeTargets(packing), 0.0).sum(axis = 1)
        else:
            P0 = np.asarray(packing.targetPerimeter, dtype = float)[:self.numConstrained]
        counts = self.counts.astype(float)
        maximum = P0 ** 2 / (4.0 * counts * np.tan(np.pi / counts))
        excess = A0 / np.maximum(maximum, 1e-300) - 1.0
        bad = np.nonzero(excess > _FEASIBILITY_SLACK)[0]
        if bad.size == 0:
            return None
        worst = int(bad[np.argmax(excess[bad])])
        floor = np.sqrt(4.0 * counts * np.tan(np.pi / counts))
        index = P0 / np.sqrt(np.maximum(A0, 1e-300))
        return (f"{bad.size} of {self.numConstrained} polygons ask for more area than their edge "
                f"targets can enclose, so the constraint set is infeasible. Worst is polygon {worst}: "
                f"target area is {1.0 + excess[worst]:.4f}x the maximum, i.e. shape index "
                f"{index[worst]:.6f} against a floor of {floor[worst]:.6f} for a "
                f"{int(counts[worst])}-gon. Constrain area alone, or set the area and perimeter "
                f"targets together instead of independently, or hold the edges by their DISTRIBUTION "
                f"(Model.setConstraints(area = True, edge = [1, 2])) so no per-edge target can "
                f"contradict the area.")

    def residual(self, packing):
        """Constraint residual C, shape (P, m); every entry is a FRACTIONAL error, so one tolerance
        applies uniformly across families. Padded edge slots yield exactly zero."""
        r, rNext, _, e = self._frame(packing)
        stop = self.numConstrained
        lengthSquared = np.einsum("pkc,pkc->pk", e, e)
        blocks = []
        if self.edge:
            l0 = self.edgeTargets(packing)
            blocks.append(np.where(self.valid, lengthSquared / l0 ** 2 - 1.0, 0.0))
        if self.diagonal:
            _, rNextD, rPrevD, _ = self._frame(packing)
            span = np.where(self.valid[:, :, None], rNextD - rPrevD, 0.0)
            d0 = self.diagonalTargets(packing)
            spanSquared = np.einsum("pkc,pkc->pk", span, span)
            blocks.append(np.where(self.valid, spanSquared / d0 ** 2 - 1.0, 0.0))
        if self.flatten:
            # d / (a + b) - 1 on the SELECTED vertices, zero elsewhere so the block stays rectangular
            # (an all-zero row is discarded by the truncated SVD).
            blocks.append(np.where(self.diagonalSelected,
                                   self.flatness(packing) - self.flatTargets(packing), 0.0))
        if self.equilateral is not None:
            # l_k^2 / l0^2 - 1 with l0 = kappa sqrt(A) / n, the same fractional form the edge rows use
            # -- but l0 is a live function of the polygon's own area rather than a stored target.
            ideal = self.idealEdgeLengths(packing, self.equilateral)
            blocks.append(np.where(self.valid, lengthSquared / (ideal ** 2)[:, None] - 1.0, 0.0))
        if self.perimeter:
            P0 = packing.targetPerimeter[:stop]
            blocks.append((np.sqrt(lengthSquared).sum(axis = 1) / P0 - 1.0)[:, None])
        if self.area:
            A0 = packing.targetArea[:stop]
            cross = np.where(self.valid,
                             r[:, :, 0] * rNext[:, :, 1] - rNext[:, :, 0] * r[:, :, 1], 0.0)
            blocks.append((0.5 * cross.sum(axis = 1) / A0 - 1.0)[:, None])
        return np.concatenate(blocks, axis = 1)

    def jacobian(self, packing):
        """Constraint Jacobian J = dC/dr, shape (P, m, 2 maxN), block diagonal by polygon. Padded
        slots leave all-zero rows and columns, which the truncated SVD discards."""
        stop = self.numConstrained
        P, n, m = self.numPolygons, self.n, self.numConstraints
        _, rNext, rPrev, e = self._frame(packing)
        J = np.zeros((P, m, n, 2))
        row = 0

        if self.edge:
            k = np.arange(n)
            weight = (2.0 / self.edgeTargets(packing) ** 2)[:, :, None]
            J[:, k, k, :] = -weight * e
            nxt = np.where(self.valid, (k[None, :] + 1) % np.maximum(self.counts, 1)[:, None], 0)
            pIdx = np.arange(P)[:, None]
            J[pIdx, k[None, :], nxt, :] = weight * e
            J[:, :n, :, :] *= self.valid[:, :, None, None]     # padded edge rows are all-zero
            row += n
        if self.diagonal:
            # d(|r_{k+1} - r_{k-1}|^2)/dr is +2 span at the FOLLOWING vertex and -2 span at the
            # PRECEDING one; the centre vertex k does not appear, which is what makes this a pure
            # angle constraint rather than a second edge constraint.
            k = np.arange(n)
            span = np.where(self.valid[:, :, None], rNext - rPrev, 0.0)
            weight = (2.0 / self.diagonalTargets(packing) ** 2)[:, :, None]
            pIdx = np.arange(P)[:, None]
            nxt = np.where(self.valid, (k[None, :] + 1) % np.maximum(self.counts, 1)[:, None], 0)
            J[pIdx, row + k[None, :], nxt, :] = weight * span
            J[pIdx, row + k[None, :], self.prevLocal, :] = -weight * span
            J[:, row:row + n, :, :] *= self.valid[:, :, None, None]
            row += n
        if self.flatten:
            # The same three-term gradient as the diagonal moment rows, without the moment weighting:
            #   at k+1 :  span/(d S)  -  (d/S^2) bHat
            #   at k-1 : -span/(d S)  +  (d/S^2) aHat
            #   at k   :              -  (d/S^2) (aHat - bHat)
            k = np.arange(n)
            span = np.where(self.valid[:, :, None], rNext - rPrev, 0.0)
            length = np.sqrt(np.einsum("pkc,pkc->pk", span, span))
            safeSpan = np.where(length > _MIN_EDGE_LENGTH, length, 1.0)
            spanUnit = np.where(self.valid[:, :, None], span / safeSpan[:, :, None], 0.0)
            edgeLength = np.sqrt(np.einsum("pkc,pkc->pk", e, e))
            safeEdge = np.where(edgeLength > _MIN_EDGE_LENGTH, edgeLength, 1.0)
            unit = np.where(self.valid[:, :, None], e / safeEdge[:, :, None], 0.0)
            entering = np.take_along_axis(unit, self.prevLocal[:, :, None].repeat(2, axis = 2),
                                          axis = 1)
            total = np.take_along_axis(edgeLength, self.prevLocal, axis = 1) + edgeLength
            safeTotal = np.where(total > _MIN_EDGE_LENGTH, total, 1.0)
            t = np.where(self.valid, length / safeTotal, 0.0)
            pick = self.diagonalSelected
            nxt = np.where(self.valid, (k[None, :] + 1) % np.maximum(self.counts, 1)[:, None], 0)
            pIdx = np.arange(P)[:, None]
            inv = np.where(pick, 1.0 / safeTotal, 0.0)
            J[pIdx, row + k[None, :], nxt, :] += (inv[:, :, None] * spanUnit
                                                  - (inv * t)[:, :, None] * unit)
            J[pIdx, row + k[None, :], self.prevLocal, :] += (-inv[:, :, None] * spanUnit
                                                             + (inv * t)[:, :, None] * entering)
            J[:, row + k, k, :] += -(inv * t)[:, :, None] * (entering - unit)
            J[:, row:row + n, :, :] *= pick[:, :, None, None]
            row += n
        if self.equilateral is not None:
            # C_k = l_k^2 n^2 / (kappa^2 A) - 1, so the row carries TWO gradients:
            #   d(l_k^2)/dr  -- the usual two-vertex edge term, and
            #   dA/dr        -- the shoelace gradient, which touches EVERY vertex of the polygon.
            # The second is what makes the size free: shrinking a polygon lowers l and l0 together and
            # the row does not notice, so only the SHAPE is held.
            k = np.arange(n)
            ideal = self.idealEdgeLengths(packing, self.equilateral)
            area = np.where(np.abs(self.areas(packing)) > _MIN_AREA, self.areas(packing), _MIN_AREA)
            lengthSquared = np.einsum("pkc,pkc->pk", e, e)
            scale = (1.0 / ideal ** 2)[:, None]
            # dA/dr_i = 1/2 perp(r_{i+1} - r_{i-1}), perp(x, y) = (y, -x)
            span = np.where(self.valid[:, :, None], rNext - rPrev, 0.0)
            areaGradient = 0.5 * np.stack([span[:, :, 1], -span[:, :, 0]], axis = 2)
            pIdx = np.arange(P)[:, None]
            nxt = np.where(self.valid, (k[None, :] + 1) % np.maximum(self.counts, 1)[:, None], 0)
            for slot in range(n):
                J[:, row + slot, :, :] -= (lengthSquared[:, slot] * scale[:, 0]
                                           / area)[:, None, None] * areaGradient
            J[:, row + k, k, :] += -2.0 * scale[:, :, None] * e
            J[pIdx, row + k[None, :], nxt, :] += 2.0 * scale[:, :, None] * e
            J[:, row:row + n, :, :] *= self.valid[:, :, None, None]
            row += n
        if self.perimeter:
            length = np.sqrt(np.einsum("pkc,pkc->pk", e, e))
            safeLength = np.where(length > _MIN_EDGE_LENGTH, length, 1.0)
            uhat = np.where(self.valid[:, :, None], e / safeLength[:, :, None], 0.0)
            rolled = np.take_along_axis(uhat, self.prevLocal[:, :, None].repeat(2, axis = 2), axis = 1)
            J[:, row, :, :] = np.where(self.valid[:, :, None], rolled - uhat, 0.0) \
                              / packing.targetPerimeter[:stop, None, None]
            row += 1
        if self.area:
            gradArea = 0.5 * np.stack(
                [rNext[:, :, 1] - rPrev[:, :, 1], rPrev[:, :, 0] - rNext[:, :, 0]], axis = -1
            )
            J[:, row, :, :] = np.where(self.valid[:, :, None], gradArea, 0.0) \
                              / packing.targetArea[:stop, None, None]

        pinned = getattr(packing, "pinned", None)
        if pinned is not None:
            free = ~pinned[self.index] & self.valid
            J *= free[:, None, :, None]
        return J.reshape(P, m, 2 * n)

    @staticmethod
    def _qrFactor(J):
        """``(Q, R)`` with ``J^T = Q R`` per polygon, or ``None`` when any block is rank deficient.

        THE FAST PATH, and it replaces the SVD for everything the SVD was doing here. Both quantities
        this class needs come out of it, because the columns of ``Q`` span the row space of ``J`` just
        as the kept rows of ``Vh`` do:

            normalBasis = Q^T
            J^+ C       = Q R^-T C        (solve R^T y = C, then Q y)

        the second because ``J J^T = R^T Q^T Q R = R^T R`` for full row rank, so
        ``J^+ = J^T (J J^T)^-1 = Q R (R^T R)^-1 = Q R^-T``. Note this never forms ``J J^T`` -- ``R``
        carries the same condition number as ``J``, not its square, so the objection that rules out the
        normal equations does not apply.

        Measured against the thin SVD it replaces: 5.9x faster at n = 16, 4.3x at n = 32 with 128
        polygons, with the projectors agreeing to 5.2e-14. The SVD was 93% of SHAKE's cost and SHAKE
        was ~50% of a FIRE step.

        RETURNS None RATHER THAN GUESSING when the factorization is not trustworthy. Plain QR is not
        rank revealing, and rank deficiency here is real: a triangle's area is determined by its three
        edges (measured 24 of 32 rows kept, condition 2.4e16), ``perimeter`` is a function of the edge
        rows, and a RAGGED packing pads its blocks with all-zero rows. In each case the caller falls
        back to the truncated SVD, which drops the redundant directions properly.

        A NON-FINITE BLOCK ALSO RETURNS None, and that test cannot be folded into the rank test below:
        QR propagates a NaN silently, and every comparison against a NaN is False, so
        ``diagonal <= tol`` REPORTS FULL RANK on a block that is entirely NaN. The poisoned factor then
        leaves here as a normal basis, and the first symptom is a MOMENT row going non-finite several
        projections downstream -- which points at the moment constraint, the one innocent party.
        Falling through to ``_decompose`` instead raises there, naming the cause."""
        if not np.all(np.isfinite(J)):
            return None
        stacked = np.swapaxes(J, 1, 2)
        if _STACKED_QR:
            Q, R = np.linalg.qr(stacked)
        else:
            factored = [np.linalg.qr(block) for block in stacked]
            Q = np.stack([pair[0] for pair in factored])
            R = np.stack([pair[1] for pair in factored])
        diagonal = np.abs(np.diagonal(R, axis1 = 1, axis2 = 2))
        largest = diagonal.max(axis = 1, keepdims = True)
        if np.any(diagonal <= _RANK_RCOND * np.maximum(largest, _MIN_EDGE_LENGTH)):
            return None
        return Q, R

    # UNVERIFIED(Cam)
    def poisonedInputReport(self, packing, J = None):
        """Which unfloored input went bad: the POSITIONS or a stored TARGET -- and which rows it hit.

        The distinction is the whole diagnosis and it cannot be read off the Jacobian, because both
        faults present identically there (every polygon non-finite at once). They have opposite causes
        and opposite fixes: non-finite positions mean the MINIMIZER diverged and the constraint layer
        is an innocent bystander, while a bad target means a SCHEDULE or transient step walked one of
        the stored arrays to zero, negative or NaN and the geometry is still fine."""
        lines = []
        positions = np.asarray(packing.positions, dtype = float).reshape(-1, 2)
        bad = ~np.isfinite(positions).all(axis = 1)
        lines.append(f"    POSITIONS: {int(bad.sum())} of {positions.shape[0]} vertices non-finite"
                     + (f", max|finite| {np.abs(positions[~bad]).max():.3e}"
                        if (~bad).any() else ""))
        for name in ("targetArea", "targetPerimeter", "targetEdgeLength", "targetDiagonal"):
            values = getattr(packing, name, None)
            if values is None:
                continue
            values = np.asarray(values, dtype = float)
            finite = np.isfinite(values)
            trouble = []
            if not finite.all():
                trouble.append(f"{int((~finite).sum())} non-finite")
            if (finite & (values <= 0.0)).any():
                trouble.append(f"{int((finite & (values <= 0.0)).sum())} <= 0 "
                               f"(min {values[finite].min():.3e})")
            lines.append(f"    {name}: {', '.join(trouble) if trouble else 'clean'}")
        if J is not None:
            J = np.asarray(J, dtype = float)
            broken = ~np.isfinite(J).all(axis = 2)
            hit = [name for name, rows in self.rowFamilies() if broken[:, rows].any()]
            lines.append(f"    families with a non-finite row: {hit}")
        return "\n" + "\n".join(lines)

    @staticmethod
    def _decompose(J, context = None):
        """Rank-revealing thin SVD ``J = U diag(S) Vh`` per polygon, plus the kept-rank mask.

        The rows of ``Vh`` where the mask holds are an ORTHONORMAL basis of the constraint-normal
        space; the rest span the redundant directions and are dropped. Everything else in this class
        is one contraction away from here."""
        if not np.all(np.isfinite(J)):
            # LAPACK answers a non-finite matrix with a bare "SVD did not converge", which says
            # nothing about the cause. The cause is upstream: the rows divide by the targets, so a
            # target that has gone to zero, negative or NaN lands here rather than where it broke.
            broken = np.nonzero(~np.isfinite(J).all(axis = (1, 2)))[0]
            raise FloatingPointError(
                f"constraint Jacobian is not finite for polygon(s) {broken.tolist()}.\n"
                f"    NOT DEGENERATE GEOMETRY. Every geometric divisor in this file is floored, and "
                f"a sweep to exactly zero area confirms it: a polygon squashed flat, or shrunk to a "
                f"point, gives a Jacobian as large as 4e+57 but still FINITE "
                f"(tests/degenerateJacobianCheck.py). So a collapsed polygon is not what this is, and "
                f"looking for one wastes the run.\n"
                f"    Two things are NOT floored, and it is one of them: the POSITIONS "
                f"(np.isfinite(packing.positions).all() -- a minimizer that took a NaN step), or a "
                f"stored TARGET, which the area and perimeter rows divide by raw "
                f"(getTargetAreas(), getTargetPerimeters())."
                f"{'' if context is None else context()}")
        U, S, Vh = np.linalg.svd(J, full_matrices = False)
        keep = S > _RANK_RCOND * S.max(axis = 1, keepdims = True)
        return U, S, Vh, keep

    def normalBasis(self, packing):
        """Orthonormal basis of the constraint-NORMAL space, shape (P, m, 2n), with redundant
        directions zeroed out.

        Exposed separately because it depends only on the POSITIONS: a minimizer step projects both
        the force and the velocity at the same configuration, so building this once and passing it to
        both ``projectVector`` calls halves the cost of the dominant term.

        Via QR when the blocks are full rank, via the truncated SVD when they are not. The two give
        different bases of the SAME subspace -- signs and rotations differ -- which is invariant here
        because every consumer uses the projector ``V^T V``, never the individual vectors."""
        J = self.jacobian(packing)
        factored = self._qrFactor(J)
        if factored is not None:
            return np.swapaxes(factored[0], 1, 2)
        _, _, Vh, keep = self._decompose(
            J, context = lambda: self.poisonedInputReport(packing, J))
        return np.where(keep[:, :, None], Vh, 0.0)

    def projectVector(self, packing, vector, basis = None):
        """Tangent-space part of a flat (2N,) force or velocity: ``w - sum_i (v_i . w) v_i`` over the
        orthonormal constraint-normal basis ``v_i``.

        Gathers the constrained vertices through the ragged index tables, projects, and scatters back;
        anything not covered (a container, padded slots) passes through untouched."""
        V = self.normalBasis(packing) if basis is None else basis
        full = np.array(vector, dtype = float).reshape(-1, 2)
        w = full[self.index].reshape(self.numPolygons, -1)
        tangent = (w - np.einsum("pmd,pm->pd", V, np.einsum("pmd,pd->pm", V, w)))
        tangent = tangent.reshape(self.numPolygons, self.n, 2)
        rows, cols = np.nonzero(self.valid)
        full[self.index[rows, cols]] = tangent[rows, cols]
        return full.reshape(-1)

    def projectPositions(self, packing, tol = 1e-14, maxIter = 20, stallFactor = 0.5,
                         stallCeiling = 1e-10):
        """SHAKE: pull ``packing.positions`` back onto the constraint manifold, in place.

        Newton-iterates the minimum-norm correction ``dr = -J^+ C`` (pseudoinverse, from the same
        truncated SVD), which converges quadratically from any nearby configuration. Returns
        ``(iterations, maxAbsResidual)``.

        Stops on EITHER ``tol`` or stagnation. The stagnation test is the one that matters: the
        residual has a roundoff floor that GROWS with system size -- ~8e-15 at N=64 but ~4e-14 at
        N=128 -- so a fixed absolute ``tol`` is unreachable for a large enough packing and the loop
        would burn every iteration re-achieving the same answer. That cost is not hypothetical: it
        made a FIRE step at N=128 spend 124 ms in SHAKE against a 23 ms force evaluation.

        Stagnation requires BOTH a failure to shrink by ``stallFactor`` AND a residual already below
        ``stallCeiling``. The ceiling is essential: Newton's convergence is only quadratic ONCE it is
        close, and the first call after a spring-relaxed build starts far enough out that an early
        iteration can legitimately gain less than a factor of two. Testing the ratio alone aborts
        that solve and silently leaves the shapes off the manifold (measured: max|C| = 5e-01 at
        N=128 -- a broken packing, not a slow one). Below the ceiling the residual is at its floor
        and a stalled ratio is unambiguous; above it, keep iterating.

        The FIRST call after a spring-relaxed build takes a finite jump: stiff springs only hold the
        shape approximately, so the starting configuration sits slightly off the manifold. That is
        expected -- afterwards each step's drift is O(dt^2) and one or two iterations clear it."""
        r = packing.positions.reshape(-1, 2)
        previous = np.inf
        for iteration in range(1, maxIter + 1):
            C = self.residual(packing)
            worst = float(np.abs(C).max())
            if worst < tol or (worst < stallCeiling and worst > stallFactor * previous):
                return iteration - 1, worst
            previous = worst
            J = self.jacobian(packing)
            factored = self._qrFactor(J)
            if factored is not None:
                # dr = J^+ C = Q R^-T C: forward-substitute through the lower triangle, then apply Q.
                Q, R = factored
                y = _forwardSubstitute(R, C)
                step = np.einsum("pdj,pj->pd", Q, y).reshape(self.numPolygons, self.n, 2)
            else:
                U, S, Vh, keep = self._decompose(
                    J, context = lambda: self.poisonedInputReport(packing, J))
                y = np.where(keep, np.einsum("pij,pi->pj", U, C) / np.where(keep, S, 1.0), 0.0)
                step = np.einsum("pjd,pj->pd", Vh, y).reshape(self.numPolygons, self.n, 2)
            rows, cols = np.nonzero(self.valid)
            r[self.index[rows, cols]] -= step[rows, cols]
        return maxIter, float(np.abs(self.residual(packing)).max())

    def maxResidual(self, packing):
        """Largest fractional constraint violation max|C| -- the drift diagnostic for a constrained
        run. It should sit at the SHAKE tolerance (~1e-14) from the first step to the last."""
        return float(np.abs(self.residual(packing)).max())


# UNVERIFIED(Cam)
# UNVERIFIED(Cam)
def _forwardSubstitute(R, C):
    """Solve ``R^T y = C`` per polygon for upper-triangular ``R``, batched over the leading axis.

    ``np.linalg.solve`` would do this, but it LU-factorizes with pivoting a matrix already known to be
    triangular -- measured 0.985 ms against the QR's own 1.955 ms, so a third of the fast path was
    spent rediscovering structure the factorization just produced. Substitution is one loop over the m
    rows with the batch vectorized underneath, which is the right way round: m is small (33 at n = 32)
    while the batch is the large axis."""
    R = np.asarray(R, dtype = float)
    C = np.asarray(C, dtype = float)
    y = np.empty_like(C)
    for i in range(C.shape[1]):
        total = C[:, i] if i == 0 else C[:, i] - np.einsum("pk,pk->p", R[:, :i, i], y[:, :i])
        y[:, i] = total / R[:, i, i]
    return y


def _orthonormalRows(rows, context = None):
    """Orthonormal basis for the row space of ``rows``, dropping numerically dependent rows.

    A truncated SVD rather than Gram-Schmidt: the moment rows are deliberately near-dependent (see
    ``DistributionConstraints``), and the SVD reports the rank instead of normalizing roundoff into a
    spurious basis vector.

    ``context`` is a zero-argument callable returning extra text for the non-finite warning, evaluated
    ONLY when that warning fires. Naming the broken polygon is what makes the warning actionable, and
    the caller is the only one holding the packing -- but building the report on every call would put
    a geometry sweep inside the projection hot loop for a case that almost never happens."""
    rows = np.atleast_2d(np.asarray(rows, dtype = float))
    if rows.size == 0:
        return np.zeros((0, rows.shape[-1]))
    # NON-FINITE ROWS FIRST. LAPACK reports "SVD did not converge" on them, which reads as an
    # ill-conditioning problem and is not one: a single finite row cannot fail to decompose. A NaN or
    # inf in a constraint gradient is a real fault upstream -- a collapsed polygon, a zero-length edge
    # -- so it is named rather than absorbed, and the row is dropped so the run survives to report it.
    finite = np.isfinite(rows).all(axis = 1)
    if not finite.all():
        detail = "" if context is None else context()
        warnings.warn(
            f"\n*** {int((~finite).sum())} of {rows.shape[0]} constraint rows are NOT FINITE ***\n"
            f"    A NaN or inf in a constraint gradient means the geometry underneath it is broken --"
            f" a collapsed polygon, a zero-length edge, an area that has reached zero -- not that the"
            f" constraint is badly conditioned.\n"
            f"    Those rows are dropped so the run continues, but the state is already wrong: check"
            f" getAreas() and getEdgeLengths() for zeros before trusting anything after this point."
            f"{detail}",
            stacklevel = 3)
        rows = rows[finite]
        if rows.shape[0] == 0:
            return np.zeros((0, rows.shape[-1]))
    try:
        _, S, Vh = np.linalg.svd(rows, full_matrices = False)
    except np.linalg.LinAlgError:
        # MODIFIED GRAM-SCHMIDT, which needs no decomposition to converge. LAPACK can still give up on
        # rows that have gone numerically parallel -- a schedule walking a constraint toward a BOUNDARY
        # does exactly that, since the gradient of a bounded quantity vanishes at its bound. Losing an
        # hour-long run to that is the wrong failure mode. Rows that carry no direction fall out here
        # by construction rather than being discarded by a rank test afterwards.
        basis = []
        for row in rows:
            v = np.array(row, dtype = float)
            for previous in basis:
                v -= float(previous @ v) * previous
            norm = float(np.sqrt(v @ v))
            if norm > _RANK_RCOND * max(float(np.abs(row).max()), 1e-300):
                basis.append(v / norm)
        return np.array(basis) if basis else np.zeros((0, rows.shape[-1]))
    if S[0] <= 0.0:
        return np.zeros((0, rows.shape[-1]))
    return Vh[S > _RANK_RCOND * S[0]]


# UNVERIFIED(Cam)
def _brokenGeometryReport(blocks, packing, rows = None, blockBasis = None):
    """Locate a non-finite constraint row: WHICH polygon, and did the fault enter the geometry, the
    raw moment gradient, or the block basis the row was projected through?

    That last distinction is the one worth paying for. A composite moment row is
    ``block.projectVector(momentGradient)``, so three separate things can poison it, and they have
    completely different causes: broken COORDINATES (a collapsed polygon), a broken MOMENT GRADIENT
    (a reference divided down to zero) or a broken BLOCK BASIS (a per-object Jacobian that went
    singular under QR). The composed row looks identical in all three cases, and the previous warning
    could only say that it happened.

    Column indices are mapped back to polygons through the ragged index table, so the report names the
    polygon rather than a coordinate offset."""
    lines = []
    positions = np.asarray(packing.positions, dtype = float).reshape(-1, 2)
    owner = np.full(positions.shape[0], -1, dtype = int)
    owner[blocks.index[blocks.valid]] = np.nonzero(blocks.valid)[0]

    bad = ~np.isfinite(positions).all(axis = 1)
    if bad.any():
        lines.append(f"    COORDINATES are not finite: {int(bad.sum())} vertices, "
                     f"polygons {sorted(set(owner[bad].tolist()))}")
    else:
        areas = np.abs(blocks.areas(packing))
        lengths = np.where(blocks.valid, blocks.edgeLengths(packing), np.inf)
        worstArea = int(np.argmin(areas))
        worstEdge = np.unravel_index(int(np.argmin(lengths)), lengths.shape)
        lines.append(f"    smallest |area| {areas[worstArea]:.3e} at polygon {worstArea}; "
                     f"smallest edge {lengths[worstEdge]:.3e} at polygon {worstEdge[0]}")

    if rows is not None:
        rows = np.atleast_2d(np.asarray(rows, dtype = float))
        columns = np.nonzero(~np.isfinite(rows).all(axis = 0))[0]
        if columns.size:
            touched = sorted(set(owner[columns // 2].tolist()))
            lines.append(f"    non-finite COLUMNS: {columns.size} of {rows.shape[1]}, "
                         f"polygons {touched}")

    if blockBasis is not None:
        basis = np.asarray(blockBasis, dtype = float)
        broken = ~np.isfinite(basis).all(axis = tuple(range(1, basis.ndim)))
        lines.append(f"    BLOCK BASIS non-finite for polygons {np.nonzero(broken)[0].tolist()}"
                     if broken.any() else
                     "    block basis is finite, so the fault is in the moment gradient itself")
    return "\n" + "\n".join(lines) if lines else ""


# UNVERIFIED(Cam)
class DistributionConstraints(_RaggedBlocks):
    """Global MOMENT equalities on the ACTUAL geometry: hold ``sum_i t_i^k`` and let the individual
    shapes trade with each other freely.

        Phi_k = sum_i t_i(r)^k = const,    k in the requested list

    The contrast with ``ShapeConstraints`` is the whole point. There, every polygon is pinned to its
    own target -- Nn + N equalities, and the shapes cannot change at all. Here only a handful of
    numbers describing the DISTRIBUTION are held, so a polygon is free to grow or stretch as long as
    another compensates. That is the soft-to-hard annealing handle: relax with the distribution wide
    and the shapes pliable, then drive the width to zero and recover rigidity.

    Note these are moments of the geometry ITSELF, not of separate target variables. There are no new
    degrees of freedom and no springs, so nothing chases anything: the constraint is imposed directly
    on the configuration, which sidesteps the double-optimization machinery in ``transient.py`` (and
    its blind spot, the overlap normalizer's target dependence) entirely.

    TWO THINGS TO KNOW.

    First, the k = 1 and k = 2 rows become PARALLEL as the distribution narrows, so monodispersity
    cannot be imposed by driving the second moment to its floor -- anneal to a small width and hand off
    to ``ShapeConstraints`` for the last step. The degeneracy is gradual, not a cliff: measured on
    hexagons, ``sigma2 / sigma1 = 0.40 CV`` across seven decades of width. That means there is no rank
    drop to wait for -- both rows stay formally independent at every width the geometry can even
    represent -- while the projector's noise amplification grows as 1 / CV. The handoff has to be
    decided on the WIDTH, not detected from a rank; ``rank`` is exposed for diagnosis, but it will read
    full until the ratio reaches the truncation threshold, long after the constraint has stopped being
    usefully transverse.

    Second, edge-length moments do not control SHAPE. Holding ``sum l`` and ``sum l^2`` leaves a quad
    free to be a rhombus or a dart, so pair them with a hard per-object AREA constraint
    (``CompositeConstraints``) rather than using them alone: fixing the area both stops a polygon
    collapsing and, together with the second moment, bounds elongation, since one edge growing long
    forces the rest to give up length quadratically.

    That is not a guarantee against FOLDING, though. The shoelace area is SIGNED, so a figure-eight
    polygon whose two lobes differ by A0 satisfies the area constraint exactly -- self-repulsion is what
    actually keeps the loop simple, and it stays load-bearing here. Worth monitoring the shape index
    ``P / sqrt(A)`` per polygon (4 for a square) alongside the energy: it is the cheap scalar that says
    how far the intermediate states have wandered from the shape being searched for.
    """

    def __init__(self, packing, moments, area = False, edge = False, shape = False,
                 deviation = False, fullShapeMoments = False, diagonal = False):
        if not (area or edge or shape or diagonal):
            raise ValueError("DistributionConstraints needs at least one of area / edge / diagonal / "
                             "shape.")
        # Whether the shape family may carry the full exponent list in DIRECT mode. Off by default for
        # the reason in ``familyMoments``; ``Model.setConstraints(distortion = [...])`` turns it on,
        # which is the spelling for taking moments of the DIMENSIONLESS distortion.
        self.fullShapeMoments = bool(fullShapeMoments)
        super().__init__(packing)
        # UNVERIFIED(Cam)
        # DEVIATION MODE: take moments of the distance from the ideal rather than of the quantity
        # itself. This exists to make a NEGATIVE exponent usable as a barrier, which is what fixes the
        # endgame. Holding sum_i delta_i^-1 keeps every deviation strictly positive with a gradient
        # carrying -delta^-2 -- it BLOWS UP as delta -> 0, where the direct shape budget's gradient
        # VANISHES (measured row norm 1.55e-01 -> 2.24e-02 as the budget fell 0.030 -> 0.0009, after
        # which the retraction had to be backtracked to stop it hurling the packing away). Best
        # conditioned exactly where the direct form dies.
        self.deviation = bool(deviation)
        self.moments = [int(k) for k in np.atleast_1d(moments)]
        if not self.moments:
            raise ValueError("DistributionConstraints needs at least one exponent; k = 1 alone holds "
                             "the mean.")
        if 0 in self.moments:
            raise ValueError("k = 0 is not a usable moment: sum_i t_i^0 is just the count, which is "
                             "constant for any geometry, so it constrains nothing.")
        self.area = bool(area)
        self.edge = bool(edge)
        # The skip-one distances held as a DISTRIBUTION rather than per object. Per-object diagonals
        # pin every polygon to one template, which is n rows per polygon and over-determines the shape
        # (n edges + n diagonals + 1 area = 2n+1 rows on 2n-3 shape DOF -- measured rank 13 of 17 at
        # n = 8). Moments cost len(exponents) rows for the WHOLE packing and leave the polygons free to
        # trade cornered-ness between themselves while the distribution is held.
        self.diagonal = bool(diagonal)
        self.shape = bool(shape)
        # No per-object perimeter here: it is a sum of edge lengths, so its distribution is what edge
        # moments already describe. Present so the energy can read the same flags off any constraint
        # object and drop the matching springs.
        self.perimeter = False
        self.edgeHeld = self.edge
        self.reference = {name: self.momentValues(self.quantity(packing, name), self.familyMoments(name))
                          for name in self.families()}

    def families(self):
        """Active families as a tuple of names, in row order."""
        return tuple(name for name, on in
                     (("area", self.area), ("edge", self.edge), ("diagonal", self.diagonal),
                      ("shape", self.shape)) if on)

    # UNVERIFIED(Cam)
    def familyMoments(self, name):
        """Exponents used for one family.

        ``shape`` always gets ``[1]`` and nothing else, whatever was requested -- UNLESS this is a
        deviation set. Its quantity is the NONNEGATIVE distortion ``d_i >= 0``, so the first moment
        alone is already a hard squeeze: a sum of nonnegative terms can only reach zero with every term
        at zero. Higher moments of the same quantity add rows that vanish faster than the first and only
        degrade the conditioning.

        In DEVIATION mode every family gets the full exponent list, because that is the whole point --
        the ``k = -1`` row is what keeps each deviation positive while the ``k = +1`` row squeezes the
        mean, and pairing them is what stops the distribution narrowing by collapsing one polygon onto
        the floor while another stays bent."""
        if self.deviation:
            return self.moments
        if name == "shape" and not self.fullShapeMoments:
            return [1]
        return self.moments

    @property
    def numRows(self):
        return sum(len(self.familyMoments(name)) for name in self.families())

    def quantity(self, packing, name):
        """The flat 1-D vector of geometric values a family takes moments of: one entry per polygon
        for ``area`` and ``shape``, one per (valid) EDGE for ``edge``.

        In DEVIATION mode these are distances from the ideal instead -- see ``deviations``."""
        if self.deviation:
            return self.deviations(packing, name)
        if name == "area":
            return self.areas(packing)
        if name == "edge":
            return self.edgeLengths(packing)[self.valid]
        if name == "diagonal":
            return self.flatness(packing)[self.diagonalSelected]
        if name == "shape":
            return self.distortions(packing)
        raise ValueError(f"unknown moment family {name!r}; use 'area', 'edge', 'diagonal' or "
                         f"'shape'.")

    # UNVERIFIED(Cam)
    # UNVERIFIED(Cam)
    def deviations(self, packing, name):
        """Distance from the ideal, NONNEGATIVE BY CONSTRUCTION, for one family.

            shape   delta_i   = P_i - g_i sqrt(A_i)       g_i = sqrt(4 n_i tan(pi/n_i))
            area    alpha_i   = A0_i - A_i
            edge    eps_ik    = |l_ik - l0_i|             l0_i from the polygon's own ACTUAL area

        ``shape`` is the ISOPERIMETRIC DEFICIT and its nonnegativity is a theorem, not a constraint to
        enforce: no polygon of area A can have perimeter below the regular one's, with equality only for
        the regular polygon. Measuring against the polygon's ACTUAL area rather than its target is what
        keeps shape and size on separate axes -- a polygon that shrank is not thereby called distorted.

        ``area`` is SHRINK-ONLY, so it is nonnegative exactly while the packing has not inflated. That
        is safe here because ``Model.getPackingFraction`` measures from the actual geometry, so
        shrinking LOWERS the reported density rather than inflating it; a polygon cannot pay for a
        better phi by getting smaller. A negative value means something grew past its target, which the
        ``k = -1`` guard in ``momentValues`` will refuse rather than silently square away.

        IT MUST BE SEEDED BEFORE IT CAN BE USED WITH A NEGATIVE EXPONENT. On a fresh packing alpha is
        exactly zero -- ``setPackingFraction`` scales the geometry and the targets together, so the two
        never separate (measured -2.8e-17, roundoff of either sign) -- and a barrier is singular there.
        Shrink the polygons about their centroids first, leaving the targets alone, the same way
        ``Model.spreadShapes`` seeds shape spread before the moments narrow it. The slack is an
        annealing freedom: how much a polygon may shrink to squeeze past an obstruction, opened
        deliberately and then closed.

        ``edge`` carries an absolute value, so it has a KINK where an edge crosses its ideal length.
        Under a ``k = -1`` barrier that kink is unreachable (the barrier diverges before it), which is
        consistent but has a consequence worth stating: the barrier then prevents any individual edge
        from being exactly ideal. For driving a packing to regular polygons prefer ``shape``, whose zero
        is the thing actually wanted; ``edge`` deviations are for holding a SPREAD away from zero."""
        if name == "diagonal":
            # THE FLATNESS DEFICIT, nonnegative by the triangle inequality: d <= a + b for any three
            # points, with equality exactly when they are collinear. So 1 - d/(a+b) >= 0 and vanishes
            # precisely at flat -- the same kind of theorem-backed one-sided deficit as the
            # isoperimetric one, and it makes a single-row BUDGET well posed where the moments are not.
            #
            # An earlier version refused this case, on the grounds that "a skip-one distance has no
            # one-sided ideal -- a turning angle can err in either direction". That is true of a
            # diagonal LENGTH against a stored target, which is what that family holds. It is not true
            # of flatness against 1, which is a bound rather than a target.
            return 1.0 - self.flatness(packing)[self.diagonalSelected]
        if name == "shape":
            areas = np.maximum(np.abs(self.areas(packing)), _MIN_AREA)
            perimeters = np.where(self.valid, self.edgeLengths(packing), 0.0).sum(axis = 1)
            return perimeters - self.regularShapeIndex() * np.sqrt(areas)
        if name == "area":
            targets = np.asarray(packing.targetArea, dtype = float)[:self.numPolygons]
            return targets - np.abs(self.areas(packing))
        if name == "edge":
            ideal = self.idealEdgeLengths(packing)
            return np.abs(self.edgeLengths(packing) - ideal[:, None])[self.valid]
        raise ValueError(f"unknown moment family {name!r}; use 'area', 'edge' or 'shape'.")

    # UNVERIFIED(Cam)
    def distortions(self, packing):
        """(P,) relative shape distortion ``d_i = P_i / (sqrt(A_i) g_i) - 1``, zero exactly at the
        regular n-gon and positive for everything else.

        This is the quantity an ANNEAL over shapes drives to zero, and it is a different axis from the
        area moments entirely: fixing every area says nothing about shape, since a fixed-area
        quadrilateral is still any quadrilateral. Normalizing by the n-dependent floor ``g_i`` makes
        one number meaningful for a packing of mixed vertex counts.

        The area is taken as ``|A|`` so a clockwise polygon is not reported as infinitely distorted;
        the shape index is orientation-blind by construction."""
        areas = np.maximum(np.abs(self.areas(packing)), _MIN_AREA)
        perimeters = np.where(self.valid, self.edgeLengths(packing), 0.0).sum(axis = 1)
        return perimeters / (np.sqrt(areas) * self.regularShapeIndex()) - 1.0

    def momentValues(self, t, moments = None):
        """``[sum_i t_i^k for k in moments]`` for a flat vector of geometric values."""
        t = np.asarray(t, dtype = float)
        moments = self.moments if moments is None else moments
        if min(moments) < 0 and t.min() <= 0.0:
            raise ValueError(
                f"a negative moment was requested but a value has reached {t.min():.3e}; sum t^k "
                "diverges there. Negative k exists to REPEL zero, so it cannot be applied to a "
                "quantity that has already arrived -- drop the negative exponent or start from a "
                "configuration with no collapsed polygon.")
        return np.array([np.sum(t ** k) for k in moments])

    # UNVERIFIED(Cam)
    def rescale(self, factor):
        """Carry the conserved moments through a uniform rescaling of the whole packing.

        A moment constraint holds ABSOLUTE sums -- ``sum_i l_i`` and ``sum_i l_i^2`` for
        ``edge = [1, 2]`` -- so scaling every polygon by ``factor`` moves them by ``factor^k`` even
        though the DISTRIBUTION is completely unchanged. Left uncorrected the constraint reads a pure
        size change as a violation and the retraction fights the density move: measured, a x1.10
        compression under ``edge = [1, 2]`` left a residual of 4.06e-01 (the retraction did not
        converge) and dragged the packing fraction back by 0.117%, against 3.55e-15 and no drift for
        the same move with the edges held per object.

        The exponent is the family's LENGTH DIMENSION: areas go as ``factor^2``, lengths as ``factor``,
        and the direct shape distortion is dimensionless so it does not move at all. In deviation mode
        every family measures a distance from its own ideal, which scales with the ideal -- the shape
        deviation ``P - g sqrt(A)`` is a length, not a ratio."""
        factor = float(factor)
        if factor == 1.0:
            return self
        for name in self.families():
            if self.deviation:
                power = 2.0 if name == "area" else 1.0
            else:
                power = {"area": 2.0, "edge": 1.0, "diagonal": 1.0, "shape": 0.0}[name]
            exponents = np.asarray(self.familyMoments(name), dtype = float)
            self.reference[name] = self.reference[name] * factor ** (power * exponents)
        return self

    # UNVERIFIED(Cam)
    def setReference(self, name, values):
        """Move one family's conserved values to a NEW target, so a schedule can walk them.

        The reference is otherwise captured once at construction, which is right for a quantity being
        CONSERVED but wrong for one being ANNEALED: the shape budget ``sum_i d_i`` has to be driven
        down over the run, and the constraint is what carries it there. Retarget, then SHAKE.

        Nothing is retracted here -- the caller decides when to project, since a schedule usually wants
        to retarget and relax in one step."""
        if name not in self.families():
            raise ValueError(f"{name!r} is not an active family; active: {self.families()}")
        values = np.atleast_1d(np.asarray(values, dtype = float))
        expected = len(self.familyMoments(name))
        if values.size != expected:
            raise ValueError(f"family {name!r} has {expected} row(s); got {values.size} value(s).")
        self.reference[name] = values
        return self

    def values(self, packing):
        """All conserved quantities, families in row order."""
        return np.concatenate([self.momentValues(self.quantity(packing, name),
                                                 self.familyMoments(name))
                               for name in self.families()])

    # UNVERIFIED(Cam)
    def _familyScale(self, name):
        """Row normalizer(s) for one family.

        Everything but ``shape`` is scaled by |reference|, making the residual FRACTIONAL so that one
        tolerance covers every exponent -- without it the rows span many orders of magnitude (``sum
        l^-1`` against ``sum l^4``) and the least-squares solve is dominated by whichever happens to be
        largest.

        The direct (non-deviation) ``shape`` budget cannot use that scaling, because its reference is
        DRIVEN TO ZERO: a fractional error against a vanishing target diverges, and near the end of an
        anneal it would dominate every other row for no reason. It is normalized by the polygon count
        instead, so its residual reads as an absolute error in the MEAN distortion -- a quantity that
        stays meaningful at zero.

        DEVIATION families go back to |reference| even for ``shape``. Their targets never reach zero --
        the ``k = -1`` row diverges first, which is the whole reason the deviation form exists -- so a
        fractional residual stays well defined all the way down, and it is the right thing: the two
        rows differ by many orders (``sum delta`` against ``sum 1/delta``) and only a relative measure
        puts them on one tolerance."""
        if name == "shape" and not self.deviation:
            # PER ROW, and the two signs of k need OPPOSITE normalizers -- a moment list is now
            # possible here (``distortion = [...]``) where only k = 1 used to be.
            #   k > 0: the reference goes to ZERO as the packing straightens, so a fractional residual
            #          diverges exactly where the anneal is heading. Normalize by the count, making it
            #          an absolute error in a mean.
            #   k < 0: the reference DIVERGES instead, and count-normalizing understates the row by
            #          orders of magnitude. Measured: with d_i down to 4.5e-03 the k = -1 reference is
            #          ~2.2e+02 against a count of 6, and scaling it by 6 let that row dominate the
            #          least-squares solve and fold a polygon on the first retraction step -- caught
            #          only because momentValues refuses a negative base.
            counts = float(max(self.numPolygons, 1))
            reference = np.maximum(np.abs(self.reference[name]), 1e-300)
            return np.array([counts if k > 0 else reference[j]
                             for j, k in enumerate(self.familyMoments(name))])
        return np.maximum(np.abs(self.reference[name]), 1e-300)

    def _scales(self):
        """Row normalizers, families in row order."""
        return np.concatenate([self._familyScale(name) for name in self.families()])

    def residual(self, packing):
        """Fractional moment residual, shape (numRows,)."""
        reference = np.concatenate([self.reference[name] for name in self.families()])
        return (self.values(packing) - reference) / self._scales()

    def jacobian(self, packing):
        """``dC/dr``, shape (numRows, 2 numVertices), dense -- a moment couples the WHOLE packing, so
        unlike the per-object rows this has no block structure. Container and pinned columns are zero.

        The rows are contractions of the per-object gradients already used by ``ShapeConstraints``:
        ``d(sum_i t_i^k)/dr = sum_i k t_i^(k-1) dt_i/dr``, i.e. the same shoelace and unit-vector
        gradients weighted by ``k t^(k-1)`` and summed instead of kept separate.

        The ``shape`` row combines both of those gradients rather than picking one, since the shape
        index is a ratio of perimeter to root area:

            d(P / (g sqrt(A))) / dr = (1 / (g sqrt(A))) dP/dr - (P / (2 g A^(3/2))) dA/dr

        with ``dP/dr`` the sum of edge unit vectors and ``dA/dr`` the shoelace gradient.

        NOTE this row goes DEGENERATE at the answer. The shape index is MINIMIZED at the regular
        polygon, so its gradient vanishes there -- exactly where the anneal is trying to arrive. The
        row's contribution to the smallest singular value therefore decays as the distortion does, and
        ``conditioning`` falls with it: the ramp has to hand off to per-object ``ShapeConstraints``
        while the constraint is still transverse, the same way the edge moments do."""
        _, rNext, rPrev, e = self._frame(packing)
        rows = []
        areaGradient = np.where(self.valid[:, :, None], 0.5 * np.stack(
            [rNext[:, :, 1] - rPrev[:, :, 1], rPrev[:, :, 0] - rNext[:, :, 0]], axis = -1), 0.0)
        length = self.edgeLengths(packing)
        safe = np.where(length > _MIN_EDGE_LENGTH, length, 1.0)
        uhat = np.where(self.valid[:, :, None], e / safe[:, :, None], 0.0)
        for name in self.families():
            t = self.quantity(packing, name)
            scale = self._familyScale(name)
            if self.deviation:
                rows.extend(self._deviationRows(packing, name, t, scale, areaGradient, uhat, safe))
            elif name == "area":
                for j, k in enumerate(self.moments):
                    weight = float(k) * t ** (k - 1)
                    block = weight[:, None, None] * areaGradient
                    row = np.zeros((self.numVertices, 2))
                    np.add.at(row, self.index[self.valid], block[self.valid])
                    rows.append(row.reshape(-1) / scale[j])
            elif name == "shape":
                signedArea = self.areas(packing)
                absArea = np.maximum(np.abs(signedArea), _MIN_AREA)
                perimeter = np.where(self.valid, length, 0.0).sum(axis = 1)
                g = self.regularShapeIndex()
                # d|A|/dr carries the sign of A: a clockwise polygon's shoelace gradient points the
                # other way, and the shape index depends on |A|.
                orientation = np.where(signedArea < 0.0, -1.0, 1.0)
                perimeterWeight = 1.0 / (g * np.sqrt(absArea))
                areaWeight = -orientation * perimeter / (2.0 * g * absArea ** 1.5)
                # ``d(sum_i d_i^k)/dr = sum_i k d_i^(k-1) dd_i/dr``, the inner derivative being the two
                # blocks below. With the default single k = 1 the factor is 1 and this is the row that
                # was here before; a moment LIST (``distortion = [...]``) is what makes the loop earn
                # its keep. A negative exponent needs d_i > 0 -- that is the barrier's whole nature --
                # so the base is floored rather than allowed to divide by zero on an exactly regular
                # polygon, which is reachable here in a way it is not in the deviation form.
                for j, k in enumerate(self.familyMoments(name)):
                    base = np.maximum(t, _MIN_DISTORTION) if k < 1 else t
                    factor = float(k) * base ** (k - 1)
                    row = np.zeros((self.numVertices, 2))
                    block = (factor * perimeterWeight)[:, None, None] * uhat
                    np.add.at(row, self.nextIndex[self.valid], block[self.valid])
                    np.add.at(row, self.index[self.valid], -block[self.valid])
                    block = (factor * areaWeight)[:, None, None] * areaGradient
                    np.add.at(row, self.index[self.valid], block[self.valid])
                    rows.append(row.reshape(-1) / scale[j])
            elif name == "diagonal":
                # t = d / (a + b). THREE gradients, not one: the span's, and the two adjacent edges'
                # through the denominator.
                #     at k+1 :  span/(d S)  -  (d/S^2) bHat
                #     at k-1 : -span/(d S)  +  (d/S^2) aHat
                #     at k   :              -  (d/S^2) (aHat - bHat)
                # with aHat the unit of the edge ENTERING k and bHat the one LEAVING it. The centre
                # vertex appears only through the denominator, which is what keeps this a measure of
                # the angle rather than of the edges.
                span = self.diagonalSpans(packing)
                length = self.diagonalLengths(packing)
                safeSpan = np.where(length > _MIN_EDGE_LENGTH, length, 1.0)
                spanUnit = np.where(self.valid[:, :, None], span / safeSpan[:, :, None], 0.0)
                entering = np.take_along_axis(uhat, self.prevLocal[:, :, None].repeat(2, axis = 2),
                                              axis = 1)
                edge = self.edgeLengths(packing)
                total = np.take_along_axis(edge, self.prevLocal, axis = 1) + edge
                safeTotal = np.where(total > _MIN_EDGE_LENGTH, total, 1.0)
                t = self.flatness(packing)
                pick = self.diagonalSelected
                # This branch is DIRECT mode only: the deviation form of "diagonal" is dispatched
                # above, to _deviationRows, before the loop ever reaches this elif -- the two share
                # the span/entering/edge geometry but differ in the weight (t vs 1 - t) and are kept
                # in their respective functions rather than interleaved here.
                for j, k in enumerate(self.moments):
                    weight = np.where(pick, float(k) * np.where(pick, t, 1.0) ** (k - 1), 0.0)
                    atNext = (weight / safeTotal)[:, :, None] * spanUnit \
                        - (weight * t / safeTotal)[:, :, None] * uhat
                    atPrev = -(weight / safeTotal)[:, :, None] * spanUnit \
                        + (weight * t / safeTotal)[:, :, None] * entering
                    atHere = -(weight * t / safeTotal)[:, :, None] * (entering - uhat)
                    row = np.zeros((self.numVertices, 2))
                    np.add.at(row, self.nextIndex[pick], atNext[pick])
                    np.add.at(row, self.prevIndex[pick], atPrev[pick])
                    np.add.at(row, self.index[pick], atHere[pick])
                    rows.append(row.reshape(-1) / scale[j])
            else:
                for j, k in enumerate(self.moments):
                    weight = np.where(self.valid, float(k) * safe ** (k - 1), 0.0)
                    block = weight[:, :, None] * uhat
                    row = np.zeros((self.numVertices, 2))
                    np.add.at(row, self.nextIndex[self.valid], block[self.valid])
                    np.add.at(row, self.index[self.valid], -block[self.valid])
                    rows.append(row.reshape(-1) / scale[j])
        J = np.array(rows)
        pinned = getattr(packing, "pinned", None)
        if pinned is not None:
            J = J.reshape(len(rows), self.numVertices, 2)
            J[:, pinned, :] = 0.0
            J = J.reshape(len(rows), -1)
        return J

    # UNVERIFIED(Cam)
    def _deviationRows(self, packing, name, t, scale, areaGradient, uhat, safe):
        """Moment rows for one DEVIATION family: ``d(sum delta^k)/dr = sum k delta^(k-1) d(delta)/dr``.

        Only ``d(delta)/dr`` differs from the direct families, and it is built from the same two
        gradients they already use -- the shoelace ``dA/dr`` and the edge unit vectors ``dl/dr``:

            shape   d/dr [P - g sqrt(A)]  =  dP/dr - (g / (2 sqrt(A))) d|A|/dr
            area    d/dr [A0 - |A|]       =  -d|A|/dr
            edge    d/dr |l - l0|         =  sign(l - l0) [dl/dr - (c / (2 sqrt(A))) d|A|/dr]

        with ``d|A|/dr`` carrying the polygon's orientation, since a clockwise loop's shoelace gradient
        points the other way.

        The ``k = -1`` row is the one that matters: its weight is ``-delta^-2``, so it GROWS without
        bound as the deviation shrinks. That is the point -- it is the barrier, and it is best
        conditioned exactly where the direct shape budget's gradient vanishes."""
        signedArea = self.areas(packing)
        absArea = np.maximum(np.abs(signedArea), _MIN_AREA)
        orientation = np.where(signedArea < 0.0, -1.0, 1.0)
        counts = self.counts.astype(float)
        if name == "diagonal":
            # Same three gradients as the DIRECT diagonal branch in `jacobian` -- the span's and the
            # two adjacent edges' through the denominator -- computed once here since they do not
            # depend on k. Only the weight differs, built from delta = 1 - t rather than t.
            span = self.diagonalSpans(packing)
            diagLength = self.diagonalLengths(packing)
            safeSpan = np.where(diagLength > _MIN_EDGE_LENGTH, diagLength, 1.0)
            spanUnit = np.where(self.valid[:, :, None], span / safeSpan[:, :, None], 0.0)
            entering = np.take_along_axis(uhat, self.prevLocal[:, :, None].repeat(2, axis = 2),
                                          axis = 1)
            edgeLength = self.edgeLengths(packing)
            total = np.take_along_axis(edgeLength, self.prevLocal, axis = 1) + edgeLength
            safeTotal = np.where(total > _MIN_EDGE_LENGTH, total, 1.0)
            tFull = self.flatness(packing)
            pick = self.diagonalSelected
        rows = []

        for j, k in enumerate(self.moments):
            row = np.zeros((self.numVertices, 2))
            if name == "diagonal":
                # d(delta^k)/dr = k delta^(k-1) d(delta)/dr = -k delta^(k-1) dt/dr, so this reuses the
                # direct branch's atNext/atPrev/atHere exactly, with the weight's base swapped from t
                # to delta = 1 - t and the sign flipped for the chain rule through (1 - t).
                #
                # FLOORED FOR k < 1: a vertex sitting exactly at flat makes delta = 0, and a barrier
                # exponent divides by it. One NaN here poisons the whole Jacobian -- the rank test
                # reads `<=` on a NaN as False and reports an all-NaN block as full rank -- so this is
                # floored at the source rather than guarded downstream.
                base = np.where(pick, 1.0 - tFull, 1.0)
                if k < 1:
                    base = np.maximum(base, _MIN_DEVIATION)
                weight = np.where(pick, -float(k) * base ** (k - 1), 0.0)
                atNext = (weight / safeTotal)[:, :, None] * spanUnit \
                    - (weight * tFull / safeTotal)[:, :, None] * uhat
                atPrev = -(weight / safeTotal)[:, :, None] * spanUnit \
                    + (weight * tFull / safeTotal)[:, :, None] * entering
                atHere = -(weight * tFull / safeTotal)[:, :, None] * (entering - uhat)
                np.add.at(row, self.nextIndex[pick], atNext[pick])
                np.add.at(row, self.prevIndex[pick], atPrev[pick])
                np.add.at(row, self.index[pick], atHere[pick])
            elif name == "area":
                weight = float(k) * t ** (k - 1)
                block = (-orientation * weight)[:, None, None] * areaGradient
                np.add.at(row, self.index[self.valid], block[self.valid])
            elif name == "shape":
                weight = float(k) * t ** (k - 1)
                block = weight[:, None, None] * uhat
                np.add.at(row, self.nextIndex[self.valid], block[self.valid])
                np.add.at(row, self.index[self.valid], -block[self.valid])
                areaWeight = -weight * orientation * self.regularShapeIndex() / (2.0 * np.sqrt(absArea))
                block = areaWeight[:, None, None] * areaGradient
                np.add.at(row, self.index[self.valid], block[self.valid])
            elif name == "edge":
                # t is flattened over the valid edges, so the weights have to be scattered back into
                # the padded (P, maxN) block before they can multiply per-edge gradients.
                padded = np.zeros(self.valid.shape)
                padded[self.valid] = float(k) * t ** (k - 1)
                ideal = self.idealEdgeLengths(packing)
                sign = np.where(self.valid, np.sign(safe - ideal[:, None]), 0.0)
                weight = padded * sign
                block = weight[:, :, None] * uhat
                np.add.at(row, self.nextIndex[self.valid], block[self.valid])
                np.add.at(row, self.index[self.valid], -block[self.valid])
                # Every edge of a polygon shares one l0, so their area terms sum onto that polygon.
                c = np.sqrt(4.0 * np.tan(np.pi / counts) / counts)
                areaWeight = -weight.sum(axis = 1) * orientation * c / (2.0 * np.sqrt(absArea))
                block = areaWeight[:, None, None] * areaGradient
                np.add.at(row, self.index[self.valid], block[self.valid])
            else:
                raise ValueError(f"unknown moment family {name!r}.")
            rows.append(row.reshape(-1) / scale[j])
        return rows

    def normalBasis(self, packing):
        """Orthonormal basis of the constraint-NORMAL space, shape (rank, 2 numVertices)."""
        rows = self.jacobian(packing)
        return _orthonormalRows(
            rows, context = lambda: _brokenGeometryReport(self, packing, rows = rows))

    def rank(self, packing):
        """Live rank of the moment rows -- below ``numRows`` once exponents go numerically dependent.

        A blunt diagnostic. Because the mean/variance degeneracy is LINEAR in the width, this reads
        full until the width is within roundoff, so it will not warn you in time -- use
        ``conditioning`` for that."""
        return int(self.normalBasis(packing).shape[0])

    # UNVERIFIED(Cam)
    def rowNorms(self, packing):
        """2-norm of each scaled constraint row, as a dict keyed by ``"family k"``.

        The transversality measure ``conditioning`` cannot provide for a SINGLE row: a ratio of
        singular values is identically 1 when there is only one, so a ``shape``-only constraint always
        reports perfect conditioning right up to the moment it stops working. What actually degrades is
        the row's own magnitude -- the shape index is minimized at the regular polygon, so its gradient
        vanishes as the budget is driven to zero, and a retraction dividing by that norm takes ever
        larger steps. This is the number an anneal should watch to decide when to hand off."""
        J = self.jacobian(packing)
        names = [f"{name} {k}" for name in self.families() for k in self.familyMoments(name)]
        return {name: float(np.linalg.norm(row)) for name, row in zip(names, J)}

    def conditioning(self, packing):
        """Smallest-to-largest singular value ratio of the moment rows.

        The usable measure of how transverse the constraint still is, and it decays in direct
        proportion to the distribution's width (measured ``0.40 CV`` for k = [1, 2] on hexagons). Its
        reciprocal is roughly the factor by which the projection amplifies force noise, so it is what
        decides when an anneal should hand off to per-object constraints."""
        singular = np.linalg.svd(self.jacobian(packing), compute_uv = False)
        if singular.size == 0 or singular[0] <= 0.0:
            return 0.0
        return float(singular[-1] / singular[0])

    def projectVector(self, packing, vector, basis = None):
        """Tangent-space part of a flat (2N,) force or velocity."""
        V = self.normalBasis(packing) if basis is None else basis
        w = np.array(vector, dtype = float).reshape(-1)
        if V.shape[0] == 0:
            return w
        return w - V.T @ (V @ w)

    def projectPositions(self, packing, tol = 1e-12, maxIter = 20, **ignored):
        """Newton-retract the positions until every moment is back on its reference.

        The minimum-norm correction ``dr = -J^+ C`` from a truncated SVD, exactly as SHAKE does for the
        per-object constraints. The tolerance is looser by default (1e-12 rather than 1e-14) because a
        moment sums N terms, so its roundoff floor is N times a single polygon's.

        The step is BACKTRACKED against the 2-norm of the residual rather than taken whole. For the
        area and edge moments the full step is accepted every time -- they are close to linear in the
        coordinates -- but the ``shape`` row is not: its gradient vanishes at the regular polygon it is
        aiming for, so Newton divides by an ever-smaller number and, undamped, throws the packing to
        infinity. Measured on 6 squares: an undamped halving of the budget left the hard areas wrong by
        a factor of 860. Halving the step until the merit function actually decreases costs nothing
        where Newton was already working and turns divergence into slow progress where it was not."""
        merit = lambda: float(np.linalg.norm(self.residual(packing)))
        for iteration in range(1, maxIter + 1):
            C = self.residual(packing)
            worst = float(np.abs(C).max()) if C.size else 0.0
            if worst < tol:
                return iteration - 1, worst
            J = self.jacobian(packing)
            U, S, Vh = np.linalg.svd(J, full_matrices = False)
            keep = S > _RANK_RCOND * S[0] if S.size else np.zeros(0, dtype = bool)
            if not np.any(keep):
                return iteration - 1, worst
            step = Vh[keep].T @ ((U[:, keep].T @ C) / S[keep])
            before = merit()
            saved = packing.positions.copy()
            scale = 1.0
            for _ in range(_BACKTRACK_STEPS):
                packing.positions[:] = saved - scale * step
                if merit() < before:
                    break
                scale *= 0.5
            else:
                # No downhill step at any scale: the row has gone degenerate at this configuration.
                # Leave the geometry where it was and report -- the caller's schedule is asking for a
                # budget the constraint can no longer reach, which is the signal to hand off.
                packing.positions[:] = saved
                return iteration, worst
        return maxIter, float(np.abs(self.residual(packing)).max())

    def maxResidual(self, packing):
        """Largest fractional moment violation."""
        C = self.residual(packing)
        return float(np.abs(C).max()) if C.size else 0.0

    def retarget(self, packing, polydispersity):
        """Re-aim the conserved moments at the SAME mean but the requested coefficient of variation,
        without touching the geometry -- the annealing handle.

        For a distribution of mean mu and CV c the moments follow ``E[t^k] = mu^k (1+c^2)^(k(k-1)/2)``
        (exact for a log-normal). With k in {1, 2} that is just "hold the mean, set the variance" and
        involves no distributional assumption at all, which is the case worth using; higher exponents
        commit to the log-normal family. Only the REFERENCE moves here, so follow this with a
        ``projectPositions`` to bring the geometry onto it.

        NARROWING ONLY. Asking for a width much wider than the geometry currently has does not work,
        and the reason is the same degeneracy that forces the endgame handoff: the variance is at a
        MINIMUM on a monodisperse configuration, so its gradient vanishes there and no first-order step
        increases it -- Newton is being asked to solve x^2 = eps starting from x = 0. It overshoots by
        ~1/conditioning and wanders instead of widening. Seed the width geometrically first (see
        ``Model.spreadShapes``, which stretches each polygon at CONSTANT area), then ramp down.

        The ``shape`` family is skipped: its budget is not a width around a mean but a distance from a
        FLOOR, so a CV says nothing about it. Drive that one with ``setReference('shape', ...)``."""
        c = float(polydispersity)
        for name in self.families():
            if name == "shape":
                continue
            realized = self.polydispersity(packing)[name]
            if c > max(10.0 * realized, 1e-3):
                warnings.warn(
                    f"\n*** widening a moment constraint does not work ***\n    asked for CV = {c:.3g} "
                    f"on '{name}' but the geometry is at {realized:.3g}. The variance is at a minimum "
                    f"there, so its gradient vanishes and the retraction has no first-order direction "
                    f"to widen along -- it will overshoot and wander rather than spread. Seed the "
                    f"width geometrically with Model.spreadShapes({c:.3g}) (constant area), then ramp "
                    f"the moments DOWN from there.", stacklevel = 3)
        for name in self.families():
            if name == "shape":
                continue
            t = self.quantity(packing, name)
            mu = float(t.mean())
            factor = 1.0 + c * c
            self.reference[name] = np.array(
                [t.size * mu ** k * factor ** (0.5 * k * (k - 1)) for k in self.moments])
        return self

    # UNVERIFIED(Cam)
    def shapeBudget(self, packing = None):
        """The held (or, given a packing, the REALIZED) total distortion ``sum_i d_i``.

        The one number a shape anneal walks down. Zero means every polygon is regular: the terms are
        nonnegative, so the sum cannot hide a distorted polygon behind a compensating one the way an
        edge-length moment can."""
        if not self.shape:
            raise ValueError("no shape family is active; construct with shape = True.")
        if packing is None:
            return float(self.reference["shape"][0])
        return float(np.sum(self.distortions(packing)))

    # UNVERIFIED(Cam)
    def setShapeBudget(self, budget):
        """Aim the shape row at a new total distortion. Follow with ``projectPositions``."""
        return self.setReference("shape", [float(budget)])

    # UNVERIFIED(Cam)
    def requestedPolydispersity(self, packing):
        """The width the stored reference moments ASK for, per family, as a dict; ``{}`` when it cannot
        be read off. The counterpart of ``polydispersity``, which measures what was achieved.

        Only defined for the ``[1, 2]`` exponent set, where mean and variance ARE the reference and the
        inversion is exact: ``mean = M1 / count``, ``var = M2 / count - mean^2``. Higher or negative
        exponents constrain combinations that no single width describes, so nothing is guessed.

        Exists because a residual is the wrong number to hand someone asking "does this matter?". A
        fractional moment residual of 1e-07 sounds alarming and is six orders below a ramp step; the
        degenerate case this warning was BUILT for reported 1.59e-12 while missing its target by a
        factor of 6.2. Requested against achieved says which of the two you have."""
        if sorted(self.moments) != [1, 2]:
            return {}
        result = {}
        for name in self.families():
            if name == "shape":
                continue
            reference = np.asarray(self.reference[name], dtype = float)
            order = list(self.familyMoments(name))
            if sorted(order) != [1, 2]:
                continue
            first = float(reference[order.index(1)])
            second = float(reference[order.index(2)])
            count = float(np.asarray(self.quantity(packing, name)).size)
            if count <= 0.0:
                continue
            mean = first / count
            variance = second / count - mean * mean
            if mean > 0.0 and variance >= 0.0:
                result[name] = float(np.sqrt(variance) / mean)
        return result

    def polydispersity(self, packing):
        """Realized std/mean of each family's geometric values, as a dict -- what the moments are
        actually holding, measured rather than requested. ``shape`` is reported as its MEAN, not a CV:
        it is a one-sided distance from a floor, so a spread around its mean is not the quantity of
        interest and would read zero for a packing of identical non-squares."""
        result = {}
        for name in self.families():
            t = self.quantity(packing, name)
            result[name] = float(np.mean(t)) if name == "shape" \
                else float(np.std(t) / np.mean(t))
        return result


# UNVERIFIED(Cam)
class CompositeConstraints:
    """Per-object constraints AND global moment constraints together, projected EXACTLY.

    The combination Cam's square search wants is hard per-object AREAS with the edge lengths held only
    by their distribution: the area is what sets the packing fraction, so letting it wander would let
    the polygons quietly shrink to fit and report a packing that is not one, while the edges are free
    to reshape a square into a same-area rectangle -- real exploration that cannot cheat on phi. It
    also removes the over-determination that makes per-object area + per-object edges generically
    infeasible (see ``ShapeConstraints.infeasibleReason``).

    Naively this means one Jacobian with both row sets, but that would throw away the block structure:
    the per-object rows are block diagonal and get a batched per-polygon SVD, while a single dense
    ((N + k) x 2Nn) factorization is hopeless at any real N. Instead the two are composed exactly.
    Writing ``B`` for the block-normal space and ``g_j`` for the moment gradients, the combined normal
    space is ``B + span{g_j}``, and an orthonormal basis for it is the block basis together with an
    orthonormalized ``{g_j - proj_B g_j}``. That second piece is just the block projector applied to
    each moment row -- a handful of extra calls, since there are only ever a few moments.

    The same decomposition makes the retraction cheap: because those basis vectors already lie in the
    block TANGENT space, correcting the moments along them disturbs the per-object constraints only at
    second order, which the next block SHAKE sweep absorbs.
    """

    def __init__(self, block, distribution, momentTol = 1e-12, momentMaxIter = None):
        if block is None and distribution is None:
            raise ValueError("CompositeConstraints needs at least one constraint set.")
        self.block = block
        self.distribution = distribution
        self.momentTol = float(momentTol)
        # Which alarms have already been raised by THIS constraint set. Both messages quote the live
        # residual and pass count, and Python keys its duplicate suppression on the message TEXT, so
        # numbers that drift by a digit defeat it completely: a single stalled sweep emitted the same
        # warning hundreds of times, differing only in "5.130e-01 after 179 passes". Each alarm is
        # worth exactly one report per constraint set -- the condition, not the arithmetic, is the news.
        self._raised = set()
        # Outer (alternation) passes, counted separately from the block SHAKE's own iteration budget so
        # a pathological configuration costs O(momentMaxIter * maxIter) rather than O(maxIter^2).
        #
        # The budget has to cover the ALTERNATION, not just one Newton descent: each moment step is
        # taken in the block tangent space and disturbs the per-object rows at second order, which the
        # next block SHAKE removes, so the pair converges geometrically rather than quadratically. The
        # old fixed default of 8 cut that off mid-descent -- measured on a perturbed walled packing it
        # left per-object 3.15e-08 and moments 4.70e-08, where the same retraction reaches 1.22e-15 and
        # 7.74e-13 given room.
        #
        # It SCALES WITH THE ROW COUNT because each extra moment row costs conditioning, and a
        # geometric rate set by the conditioning needs proportionally more passes. Measured on the
        # shape deviation family: 2 rows condition at 1.5e-01, 3 at 6.2e-02, 4 at 2.4e-02, and the
        # four-row set silently under-converged at 24 passes (relative error 3.5e-01 in the deficit,
        # hard areas off by 3.5e-06) while reaching 7.3e-13 and 1.3e-15 at 80. Exiting on tolerance
        # means a generous budget costs nothing when convergence is quick -- 80 and 200 were identical.
        if momentMaxIter is None:
            rows = 1 if distribution is None else max(int(distribution.numRows), 1)
            momentMaxIter = max(24, 20 * rows)
        self.momentMaxIter = int(momentMaxIter)

    def _constrains(self, term):
        """Whether a shape term is held here AT ALL, per-object or by its distribution.

        Read by the energy to decide which springs to drop. A moment-constrained term drops its spring
        just as a rigid one does: the per-object target is exactly what the moment formulation
        replaces, so leaving the spring on would pull every polygon back toward the individual target
        the constraint was chosen to abolish."""
        if self.block is not None and getattr(self.block, term, False):
            return True
        return self.distribution is not None and term in self.distribution.families()

    @property
    def area(self):
        return self._constrains("area")

    @property
    def edge(self):
        return self._constrains("edge")

    @property
    def edgeHeld(self):
        return (self.block is not None and getattr(self.block, "edgeHeld", self.block.edge)) \
            or self._constrains("edge")

    @property
    def perimeter(self):
        return self.block is not None and self.block.perimeter

    # UNVERIFIED(Cam)
    @property
    def shape(self):
        """Whether the shape distortion is held. Unlike the others this NEVER drops a spring: there is
        no shape-index spring in eqSoftBody, so the flag is read only by the anneal."""
        return self.distribution is not None and self.distribution.shape

    # UNVERIFIED(Cam)
    def shapeBudget(self, packing = None):
        """Delegate to the moment set, so a caller need not know how the constraints were composed."""
        if self.distribution is None:
            raise ValueError("no moment constraints are active.")
        return self.distribution.shapeBudget(packing)

    # UNVERIFIED(Cam)
    def setShapeBudget(self, budget):
        if self.distribution is None:
            raise ValueError("no moment constraints are active.")
        self.distribution.setShapeBudget(budget)
        return self

    def families(self):
        block = () if self.block is None else self.block.families()
        moment = () if self.distribution is None else tuple(
            name + " (moments)" for name in self.distribution.families())
        return block + moment

    def redundancyReason(self):
        return None if self.block is None else self.block.redundancyReason()

    def infeasibleReason(self, packing):
        return None if self.block is None else self.block.infeasibleReason(packing)

    def normalBasis(self, packing):
        """The composed basis: the block basis plus the block-tangent part of the moment rows."""
        blockBasis = None if self.block is None else self.block.normalBasis(packing)
        extra = np.zeros((0, packing.positions.size))
        if self.distribution is not None:
            raw = self.distribution.jacobian(packing)
            rows = raw
            if self.block is not None:
                rows = np.array([self.block.projectVector(packing, g, basis = blockBasis)
                                 for g in rows])
            extra = _orthonormalRows(
                rows, context = lambda: _brokenGeometryReport(
                    self.distribution, packing, rows = raw, blockBasis = blockBasis))
        return {"block": blockBasis, "extra": extra}

    def projectVector(self, packing, vector, basis = None):
        if basis is None:
            basis = self.normalBasis(packing)
        w = np.array(vector, dtype = float).reshape(-1)
        if self.block is not None:
            w = self.block.projectVector(packing, w, basis = basis["block"])
        extra = basis["extra"]
        if extra.shape[0]:
            w = w - extra.T @ (extra @ w)
        return w

    def projectPositions(self, packing, tol = 1e-14, maxIter = 20, **shakeKwargs):
        """Retract onto BOTH constraint sets, alternating block SHAKE with a moment correction taken
        in the block tangent space. Returns ``(iterations, maxAbsResidual)`` over both sets."""
        total = 0
        worst = np.inf
        # STOP EARLY WHEN THE RESIDUAL STOPS FALLING. The budget above assumes the alternation is
        # CONVERGING, geometrically, and pays for the slow tail of that. An UNREACHABLE target does not
        # converge at all, and there the budget is spent in full, every pass, for nothing -- and it is
        # spent per MINIMIZER STEP, since every step retracts. Measured on 5 rounded polygons under
        # load: a reachable target cost 42 passes per step with 23% of the wall clock in the retraction,
        # while a target past what the geometry could deliver cost 263 passes per step and 99%. That is
        # the difference between a relaxation and a hang, and the warning below already says the right
        # thing about it -- "no iteration count fixes that" -- so there is no reason to spend them.
        #
        # The test is on the BEST residual seen, not on monotone descent: a good moment step can
        # legitimately raise the residual for the next block SHAKE to absorb, and requiring descent
        # between passes stalled this retraction at 3.15e-08 where it otherwise reaches ~1e-14. A
        # genuinely converging alternation improves far faster than the threshold -- the four-row set
        # this budget was sized for gained about 39% per pass -- so it never trips.
        best = np.inf
        stalled = 0
        for _ in range(self.momentMaxIter):
            blockResidual = 0.0
            if self.block is not None:
                iterations, blockResidual = self.block.projectPositions(
                    packing, tol = tol, maxIter = maxIter, **shakeKwargs)
                total += iterations
            momentResidual = 0.0
            if self.distribution is not None:
                momentResidual = self.distribution.maxResidual(packing)
            worst = max(blockResidual, momentResidual)
            if self.distribution is None or momentResidual < self.momentTol:
                return total, worst
            if momentResidual < best * (1.0 - _MOMENT_STALL_GAIN):
                best = momentResidual
                stalled = 0
            else:
                stalled += 1
                if stalled >= _MOMENT_STALL_PATIENCE:
                    self._warnIfUnconverged(packing, total, stalled = True)
                    return total, worst
            basis = self.normalBasis(packing)
            extra = basis["extra"]
            if extra.shape[0] == 0:
                return total, worst
            C = self.distribution.residual(packing)
            M = self.distribution.jacobian(packing) @ extra.T
            y = np.linalg.lstsq(M, C, rcond = _RANK_RCOND)[0]
            # Backtracked for the same reason as the pure-moment retraction: a ``shape`` row's gradient
            # vanishes at the regular polygon it is aiming for, so an undamped step divides by a
            # near-zero singular value and hurls the packing away -- far enough that the next block
            # SHAKE cannot bring the areas back (measured: areas wrong by a factor of 860). The
            # area and edge rows accept the full step on the first try, so this costs them nothing.
            step = extra.T @ y
            # BACKTRACK ONLY WHEN A SHAPE ROW IS PRESENT. That row is the one whose gradient misbehaves
            # -- vanishing at the regular polygon in the direct form, diverging in the deviation form --
            # and it is what an undamped step turns into a catastrophe (measured: hard areas wrong by a
            # factor of 860). The area and edge moment rows are near-linear in the coordinates and take
            # the full step safely; damping them costs real accuracy, because this merit is measured
            # BETWEEN alternation passes, where a good step can legitimately raise the moment residual
            # for the next block SHAKE to absorb. Requiring monotone descent there stalled the composite
            # retraction at 3.15e-08 / 4.70e-08 where the undamped step reaches ~1e-14.
            if getattr(self.distribution, "shape", False):
                before = float(np.linalg.norm(C))
                saved = packing.positions.copy()
                scale = 1.0
                for _ in range(_BACKTRACK_STEPS):
                    packing.positions[:] = saved - scale * step
                    if float(np.linalg.norm(self.distribution.residual(packing))) < before:
                        break
                    scale *= 0.5
                else:
                    # No downhill step at any scale. This is a FAILURE exit, not a converged one, and
                    # it is the one a degenerate row set takes.
                    packing.positions[:] = saved
                    self._warnIfUnconverged(packing, total)
                    return total, worst
            else:
                packing.positions -= step
            total += 1
        self._warnIfUnconverged(packing, total)
        return total, worst

    # UNVERIFIED(Cam)
    def _warnIfUnconverged(self, packing, passes, stalled = False):
        """Say so when the retraction stops without meeting the moment tolerance.

        SILENCE IS THE DANGEROUS CASE. The per-object rows are still satisfied exactly -- the block
        SHAKE runs every pass -- so the packing looks healthy while the moment targets were simply never
        reached. Measured, a six-exponent set missed its deficit target by a FACTOR OF 6 with the hard
        areas exact to 8.9e-16 and nothing said.

        Too many rows on too few values is the usual cause, and it is self-inflicted: the more of the
        distribution the rows determine, the closer they drive it to monodisperse, and a monodisperse
        quantity makes EVERY moment row parallel. Measured on 8 polygons ramping the deficit down 1000x,
        2 rows conditioned at 8.6e-02, 4 at 2.7e-02, and 6 collapsed to 2.0e-09."""
        if self.distribution is None:
            return
        residual = self.distribution.maxResidual(packing)
        conditioning = self.distribution.conditioning(packing)
        # TWO SEPARATE ALARMS, because the residual alone detects neither failure reliably.
        #
        # A healthy retraction lands just above a 1e-12 fractional tolerance all the time (measured
        # 1.4e-12 for two rows, 6.4e-12 for four), so warning at the tolerance itself cries wolf on
        # every run. Hence the 100x margin.
        #
        # And a DEGENERATE set can report a small residual while missing badly: the six-exponent set
        # measured 1.59e-12 fractional residual with its total deficit off by a factor of 6.2. So low
        # conditioning is warned on in its own right, whatever the residual says.
        if residual > 100.0 * self.momentTol:
            if "retraction" in self._raised:
                return
            self._raised.add("retraction")
            # THE RESIDUAL ALONE CANNOT ANSWER "does this matter?", so the widths go in beside it.
            # A fractional 1e-07 reads alarming and is six orders under a ramp step; the degenerate
            # set this warning exists for reported 1.59e-12 while missing by a factor of 6.2. Asked
            # against achieved is the comparison that separates them, and it is free to compute.
            asked = self.distribution.requestedPolydispersity(packing)
            got = self.distribution.polydispersity(packing)
            widths = "  ".join(
                f"{name}: asked {asked[name]:.6f}, got {got[name]:.6f} "
                f"({abs(got[name] - asked[name]):.1e} off)"
                for name in sorted(asked) if name in got)
            comparison = (f"\n    {widths}\n    Judge it on THOSE, not on the residual: if the widths "
                          f"agree to far better than a step of your ramp, this is cosmetic."
                          if widths else "")
            warnings.warn(
                f"\n*** moment retraction did not converge ***\n"
                f"    residual {residual:.3e} after {passes} passes (tolerance {self.momentTol:.1e}); "
                f"conditioning {conditioning:.2e} across {self.distribution.numRows} rows.\n"
                f"    The per-object constraints ARE still satisfied, so the geometry is valid -- it is "
                f"the moment TARGETS that were not reached.{comparison}\n"
                + (f"\n    STOPPED EARLY, after {passes} passes: the residual went "
                   f"{_MOMENT_STALL_PATIENCE} consecutive passes without falling by "
                   f"{100 * _MOMENT_STALL_GAIN:.0f}%, which is what an unreachable target looks like "
                   f"from the inside. The remaining budget is not being spent, because it would be "
                   f"charged once per MINIMIZER STEP and buys nothing."
                   if stalled else
                   f"\n    A residual that barely moves across passes means the targets are "
                   f"UNREACHABLE, not that the budget is short -- no iteration count fixes that.")
                + f" Repeats from this constraint set are suppressed.",
                stacklevel = 4)
        elif conditioning < _MOMENT_CONDITIONING_FLOOR:
            if "conditioning" in self._raised:
                return
            self._raised.add("conditioning")
            warnings.warn(
                f"\n*** moment rows have gone numerically parallel ***\n"
                f"    conditioning {conditioning:.2e} across {self.distribution.numRows} rows, below "
                f"{_MOMENT_CONDITIONING_FLOOR:.0e}.\n"
                f"    The reported residual ({residual:.2e}) is NOT trustworthy at this conditioning -- "
                f"measured, a six-exponent set showed 1.6e-12 while its total was off by 6.2x. Use "
                f"fewer exponents; no iteration budget fixes this. Repeats are suppressed.",
                stacklevel = 4)

    def maxResidual(self, packing):
        worst = 0.0 if self.block is None else self.block.maxResidual(packing)
        if self.distribution is not None:
            worst = max(worst, self.distribution.maxResidual(packing))
        return worst
