"""Many-body assembly, relaxation and initialization for the polygon contact law of ``polyContact``.

The law itself is a pair interaction; this is everything around it -- which pairs interact, how the
gradients assemble, how a configuration is relaxed, and how a valid one is built in the first place.
Follows the build order in ``notes/polygonContact/HANDOFF.md`` steps 4, 6 and 7.

BODIES ARE STORED CSR, never as a uniform (N, M, 2) block: ``positions`` is (V, 2) and
``startIndices`` is (N+1,), so vertex counts may differ per body.

PERIODICITY IS APPLIED AT THE BODY-PAIR LEVEL, NEVER PER VERTEX (handoff trap T7). Wrapping individual
vertices gives a body straddling the boundary edges that span the whole box, after which crossings,
parity and nearest-feature all return nonsense on a perfectly well-formed body. Vertices are stored
unwrapped; only the minimum-image SHIFT between two bodies is ever applied, and it is applied to the
whole body at once.

THE VALIDITY RATIO IS CHECKED, NOT ASSUMED. ``d_B`` has a ridge at ~the inradius and crossing it
reverses the sign of the repulsion, so ``relax`` and ``compress`` both monitor ``dMax / rIn`` and
``compress`` rejects and halves any step that exceeds its threshold. Disjoint initialization is
necessary but NOT sufficient: one large compression step from a certified-disjoint lattice reproduced
a ratio of exactly 1.0000.
"""

# UNVERIFIED(Cam)

import warnings

import numpy as np

import polyContact as pc


class BodySet:
    """A CSR collection of simple polygons, optionally in a periodic square box."""

    # UNVERIFIED(Cam)
    def __init__(self, loops, boxSize = None):
        loops = [pc.makeCounterClockwise(np.asarray(loop, dtype = float)) for loop in loops]
        self.positions = np.concatenate(loops, axis = 0)
        self.startIndices = np.cumsum([0] + [len(loop) for loop in loops]).astype(int)
        self.boxSize = None if boxSize is None else float(boxSize)
        # Index of a body whose REGION IS ITS EXTERIOR (a confining wall, wound clockwise), or None.
        # Such a body must never be culled -- see ``circumradii``.
        self.exterior = None

    @property
    def count(self):
        return len(self.startIndices) - 1

    def loop(self, body):
        return self.positions[self.startIndices[body]:self.startIndices[body + 1]]

    def loops(self):
        return [self.loop(body) for body in range(self.count)]

    def copy(self):
        clone = BodySet.__new__(BodySet)
        clone.positions = self.positions.copy()
        clone.startIndices = self.startIndices.copy()
        clone.boxSize = self.boxSize
        clone.exterior = self.exterior
        return clone


def bodyCentroids(bodies):
    return np.stack([bodies.loop(b).mean(axis = 0) for b in range(bodies.count)])


def circumradii(bodies):
    """Per-body circumradius, used by BOTH the host and device broad phases to cull distant pairs.

    AN EXTERIOR BODY GETS A RADIUS THAT CANNOT CULL, and this is not an optimization detail -- it is a
    correctness requirement. The cull asks whether two bodies are further apart than the sum of their
    radii, which is right for two BOUNDED obstacles. A confining wall's region is its complement, which
    is unbounded: a polygon far outside the box is not far from the obstacle, it is deep inside it.
    Culling that pair sets its confinement energy and force to exactly zero, so a body that drifts out
    can never be pushed back -- a one-way door. Measured on one square walked out of a unit box, the
    force was -1.8e-03 at a centroid of 1.0 and EXACTLY 0.0 from 1.5 onward, and a cascade run duly
    ended with a wall penetration of 59.

    The radius is made large enough to reach every body rather than infinite, so the same arithmetic
    works unchanged in the CUDA driver, which uploads these values and culls device-side."""
    centroids = bodyCentroids(bodies)
    radii = np.array([np.linalg.norm(bodies.loop(b) - centroids[b], axis = 1).max()
                      for b in range(bodies.count)])
    exterior = getattr(bodies, "exterior", None)
    if exterior is not None and bodies.count:
        reach = np.linalg.norm(centroids - centroids[int(exterior)], axis = 1) + radii
        radii[int(exterior)] = float(reach.max()) + float(radii.max())
    return radii


# UNVERIFIED(Cam)
def candidatePairs(bodies):
    """``(first, second, shift)`` for every body pair whose circumradii overlap.

    Conservative: two bodies can only touch if their bounding circles do, so nothing that contributes
    is dropped. ``shift`` is the minimum-image displacement applied to the SECOND body as a whole --
    see the module docstring on why it is never applied per vertex.

    This is the body-level broad phase. The handoff's step 4 also calls for a uniform grid over all
    EDGES of all bodies, which would make the per-pair work O(M) rather than O(M^2); that is a further
    optimization and is not built yet."""
    centroids = bodyCentroids(bodies)
    radii = circumradii(bodies)
    # A SINGLE minimum image is only valid while a body is smaller than half the box. Past that the
    # shift flips discontinuously at the half-box and the energy jumps -- measured, four hexagons of
    # circumradius 0.5 compressed into a box of 1.4688 (a body spanning 68% of it) put the diagonal
    # pairs exactly on the flip and broke the gradient's finite difference at 50%, while net force
    # stayed at 3e-18 and reported nothing.
    if bodies.boxSize is not None and 2.0 * radii.max() > 0.5 * bodies.boxSize:
        warnings.warn(
            f"\n*** a body spans {2 * radii.max() / bodies.boxSize:.0%} of the box "
            f"(> 50%) ***\n    The single minimum-image shift is no longer well defined: it flips "
            f"discontinuously at the half-box, so the energy jumps and the gradient stops being its "
            f"derivative. Use a larger box or fewer bodies.", stacklevel = 2)
    pairs = []
    for first in range(bodies.count):
        for second in range(first + 1, bodies.count):
            offset = centroids[second] - centroids[first]
            shift = np.zeros(2)
            if bodies.boxSize is not None:
                shift = -bodies.boxSize * np.round(offset / bodies.boxSize)
            if np.linalg.norm(offset + shift) < radii[first] + radii[second]:
                pairs.append((first, second, shift))
    return pairs


# UNVERIFIED(Cam)
def systemEnergyGradient(bodies, stiffness = 1.0, useCuda = None, wallStiffness = 1.0):
    """``(energy, gradient)`` for the whole system; ``gradient`` has the shape of ``positions``.

    ``E = 1/2 sum over ORDERED pairs``, which for each unordered pair is exactly the symmetrized
    ``contactGradient``. Both halves are accumulated so no body is privileged.

    The GPU takes it when one is present -- 19-67x, agreeing to relE 2e-16 and max|dg| ~1e-17. Pass
    ``useCuda = False`` for the numpy path, which is what the contracts compare against.

    ``wallStiffness`` MULTIPLIES the stiffness for pairs involving ``bodies.exterior``, leaving every
    body-body pair alone. It is exact rather than an approximation, because both the energy and the
    gradient are linear in the stiffness -- there is no second law for walls, only a different k.

    It exists because the two terms are ALTERNATIVES rather than independent contributions. A confined
    packing under stress relieves itself through whichever is softer, and escaping through the boundary
    lowers the confinement for everyone while overlapping a neighbour relieves nothing globally. At
    equal stiffness the wall loses outright: measured, a packing sitting at its target contact energy
    carried 100.00% of it in wall penetration, 1.83e-19 between bodies, and a pair overlap of EXACTLY
    zero -- nothing touching anything, held together entirely by leaking out of its container."""
    exterior = getattr(bodies, "exterior", None)
    exterior = -1 if exterior is None else int(exterior)
    if useCuda is not False:
        try:
            import cudaOverlap
        except ImportError:
            cudaOverlap = None
        if cudaOverlap is not None and cudaOverlap.isAvailable() and bodies.count >= 2:
            maxCount = int(np.diff(bodies.startIndices).max())
            if maxCount <= 64:
                return cudaOverlap.polyContactCuda(bodies, stiffness, wallStiffness = wallStiffness)
    energy = 0.0
    gradient = np.zeros_like(bodies.positions)
    for first, second, shift in candidatePairs(bodies):
        loopA = bodies.loop(first)
        loopB = bodies.loop(second) + shift
        pairStiffness = stiffness
        if exterior >= 0 and (first == exterior or second == exterior):
            pairStiffness *= wallStiffness
        pairEnergy, gradientA, gradientB = pc.contactGradient(loopA, loopB, pairStiffness)
        if pairEnergy == 0.0 and not gradientA.any() and not gradientB.any():
            continue
        energy += pairEnergy
        gradient[bodies.startIndices[first]:bodies.startIndices[first + 1]] += gradientA
        gradient[bodies.startIndices[second]:bodies.startIndices[second + 1]] += gradientB
    return energy, gradient


# UNVERIFIED(Cam)
def systemValidity(bodies, samples = 200):
    """``(worstRatio, worstPair)`` of ``dMax / rIn`` over all interacting pairs.

    THE one hard constraint. Past the medial-axis ridge the repulsion reverses sign and the bodies are
    pulled through, so this is a correctness monitor rather than an accuracy one. ``rIn`` is measured
    per body and the SMALLER of the pair is used, because for a limbed shape it is the limb half-width
    that matters, not the particle size."""
    radii = [pc.inradius(bodies.loop(b)) for b in range(bodies.count)]
    worst, where = 0.0, None
    for first, second, shift in candidatePairs(bodies):
        depth = pc.maximumDepth(bodies.loop(first), bodies.loop(second) + shift, samples)
        if depth == 0.0:
            continue
        ratio = depth / max(min(radii[first], radii[second]), 1e-300)
        if ratio > worst:
            worst, where = ratio, (first, second)
    return worst, where


# UNVERIFIED(Cam)
def relax(bodies, maximumIterations = 500, gradientTolerance = 1e-10, stiffness = 1.0,
          frozen = None):
    """L-BFGS on the analytic gradient. Returns ``(energy, maxGradient, iterations)``, in place.

    L-BFGS RATHER THAN FIRE, and the handoff is emphatic. From a 1e-4 perturbation of a minimum with
    the identical gradient, L-BFGS reached 9.8e-11 in 41 iterations where FIRE stalled at 5.5e-6 after
    150. FIRE is fine for a far-from-minimum rough phase and nothing else. The energy is C2, so Newton
    would converge quadratically on a fixed contact set, but the Hessian is NOT guaranteed positive
    definite -- corner contacts contribute negative transverse terms -- so a Newton variant would need
    a trust region or a modified factorization.

    ``frozen`` is a boolean mask over vertices whose gradient is zeroed, for walls or pins.

    THIS RELAXES EVERY VERTEX INDEPENDENTLY, so it is only meaningful with a shape term supplied by the
    caller. On its own the energy is purely repulsive and its global minimum is every body shrunk to a
    POINT: measured, 16 hexagons at phi 0.9755 went to E = 0 in ONE iteration by shrinking to 68% of
    their area. Use ``relaxRigid`` for rigid bodies, or drive this from ``Model``, which supplies the
    shape springs and constraints."""
    from scipy.optimize import minimize

    shape = bodies.positions.shape
    mask = None if frozen is None else np.asarray(frozen, dtype = bool)

    def objective(flat):
        bodies.positions[:] = flat.reshape(shape)
        energy, gradient = systemEnergyGradient(bodies, stiffness)
        if mask is not None:
            gradient[mask] = 0.0
        return energy, gradient.ravel()

    result = minimize(objective, bodies.positions.ravel(), jac = True, method = "L-BFGS-B",
                      options = {"maxiter": maximumIterations, "gtol": gradientTolerance,
                                 "ftol": 0.0, "maxcor": 20})
    bodies.positions[:] = result.x.reshape(shape)
    energy, gradient = systemEnergyGradient(bodies, stiffness)
    if mask is not None:
        gradient[mask] = 0.0
    return energy, float(np.abs(gradient).max()), int(result.nit)


# UNVERIFIED(Cam)
def certifiedLattice(shape, count, boxSize = None, spacingFactor = 1.02):
    """Bodies on a square lattice with spacing > 2 x circumradius, which CERTIFIES ``E == 0``.

    Step 1 of the handoff's initialization protocol. The certificate is the point: it is checked, not
    assumed, by ``assert energy == 0`` at step 2. Returns a ``BodySet`` whose box is sized to hold the
    lattice when ``boxSize`` is None."""
    shape = pc.makeCounterClockwise(np.asarray(shape, dtype = float))
    radius = np.linalg.norm(shape - shape.mean(axis = 0), axis = 1).max()
    perSide = int(np.ceil(np.sqrt(count)))
    spacing = spacingFactor * 2.0 * radius
    if boxSize is None:
        boxSize = perSide * spacing
    loops = []
    for index in range(count):
        centre = np.array([(index % perSide + 0.5) * spacing, (index // perSide + 0.5) * spacing])
        loops.append(shape - shape.mean(axis = 0) + centre)
    return BodySet(loops, boxSize = boxSize)


# UNVERIFIED(Cam)
def compress(bodies, targetBoxSize, stiffness = 1.0, initialStep = 0.02, rejectAbove = 0.35,
             minimumStep = 1e-4, maximumIterations = 400, verbose = False):
    """Adaptively compress toward ``targetBoxSize``, rejecting any step that breaks validity.

    Steps 3 and 4 of the protocol. After each increment the system is relaxed and ``dMax / rIn`` is
    tested; a step exceeding ``rejectAbove`` is UNDONE and halved rather than accepted. The threshold
    is 0.35 rather than 1 because the failure is a discrete jump, not creep -- one large step from a
    certified-disjoint lattice (half-size 1.218 -> 0.90) reproduced a ratio of exactly 1.0000.

    Compression is affine on the body CENTROIDS only; the shapes are rigid here, so vertices move with
    their centroid and no body is distorted."""
    step = initialStep
    history = []
    while bodies.boxSize > targetBoxSize + 1e-12 and step > minimumStep:
        factor = max(targetBoxSize / bodies.boxSize, 1.0 - step)
        saved = bodies.positions.copy()
        savedBox = bodies.boxSize

        centroids = bodyCentroids(bodies)
        for body in range(bodies.count):
            block = slice(bodies.startIndices[body], bodies.startIndices[body + 1])
            bodies.positions[block] += (factor - 1.0) * centroids[body]
        bodies.boxSize = savedBox * factor

        energy, worstGradient, iterations = relaxRigid(bodies, maximumIterations,
                                                        stiffness = stiffness)
        ratio, where = systemValidity(bodies)
        if ratio > rejectAbove:
            bodies.positions[:] = saved
            bodies.boxSize = savedBox
            step *= 0.5
            if verbose:
                print(f"  reject  box {savedBox:.5f}  dMax/rIn {ratio:.3f} at {where}  step -> {step:.4f}")
            continue
        history.append({"boxSize": bodies.boxSize, "energy": energy, "maxGradient": worstGradient,
                        "validity": ratio, "iterations": iterations})
        if verbose:
            print(f"  accept  box {bodies.boxSize:.5f}  E {energy:.4e}  |g| {worstGradient:.2e}  "
                  f"dMax/rIn {ratio:.3f}  ({iterations} L-BFGS iters)")
    return history


# UNVERIFIED(Cam)
def packingEnergyForce(packing, stiffness = 1.0, wallStiffness = 1.0):
    """``(energy, force)`` for a pyPolygon ``packing`` under the depth-contact law; force is (V, 2).

    The adapter between this law and the rest of the project. Force is MINUS the gradient.

    THE CONTAINER IS THE EXTERIOR REGION, NOT AN OBSTACLE. Handed to the pair law as drawn, a wall
    polygon would be an obstacle containing every body whole, and the law would push them all OUT. The
    confining region is the wall's COMPLEMENT, and membership is read from the winding, so the wall is
    passed CLOCKWISE and the same integral becomes ``int over (dP outside the box) of d^3``. Verified
    against a dense-quadrature reference sharing no code with this module: exact at faces AND corners
    to 1e-10, force to 7.6e-10 (``tests/wallFrameCheck.py``).

    Winding is normalized here rather than assumed. ``addShape`` accepts a container drawn either way
    -- ``energies.containerOrientationSign`` exists precisely because the rest of the project does not
    care -- and a wall of the wrong handedness inverts the membership test into an attractive well,
    which is what collapsed five squares onto a point on 2026-08-01.

    THE WALL IS JUST ANOTHER BODY once its winding is inverted, so it rides the SAME batched pair loop
    -- and the same CUDA kernel -- as everything else. It is emphatically not a separate per-body Python
    loop: that spelling cost 76.9 ms of a 92.3 ms force evaluation at N = 11, against 12.7 ms for the
    whole body-body term. ``confinementEnergyGradient`` keeps that slow spelling as the reference the
    batched path is checked against; the two agree to 1e-19.

    ``BodySet.__init__`` would undo the inversion -- it normalizes every loop to counter-clockwise -- so
    the set is built through ``__new__``. That normalization is a real guard, not an inconvenience:
    handing this function a counter-clockwise wall turns confinement into a 250x-too-large attractive
    well pulling every body into the boundary."""
    container = getattr(packing, "containerIndex", None)
    vertices = packing.positions.reshape(-1, 2)
    starts = np.asarray(packing.startIndices, dtype = int)

    positions = vertices
    wallSlice = None
    if container is not None:
        wall = slice(starts[container], starts[container + 1])
        if pc.signedArea(vertices[wall]) > 0.0:
            positions = vertices.copy()
            positions[wall] = vertices[wall][::-1]
            wallSlice = wall

    bodies = BodySet.__new__(BodySet)
    bodies.positions = positions
    bodies.startIndices = starts
    bodies.boxSize = None if (packing.box is None or container is not None) else 1.0
    # The wall's region is its exterior, so it must reach every body however far one has drifted.
    bodies.exterior = None if container is None else int(container)

    energy, gradient = systemEnergyGradient(bodies, stiffness, wallStiffness = wallStiffness)
    if wallSlice is not None:
        gradient[wallSlice] = gradient[wallSlice][::-1]
    return float(energy), -gradient


# UNVERIFIED(Cam)
def confinementEnergyGradient(packing, stiffness = 1.0):
    """``(energy, gradient)`` for every body against the container's EXTERIOR; gradient is (V, 2).

    The container is one more body in the master form, its region being the complement of the drawn
    wall, so the ordered-pair sum is symmetrized exactly as ``contactEnergy`` symmetrizes body-body
    contact: half the body's boundary lying outside the box weighted by distance to the box, plus half
    the box's boundary lying inside the body weighted by distance to the body. Treating only the first
    half would be a different law for walls than for bodies.

    Unlike body-body contact this term has NO validity limit. The exterior of a convex region has no
    medial axis -- beyond an edge the nearest feature is that edge, beyond a corner it is that corner,
    and the seam between them is a normal line where two INCIDENT features tie, a C^1 seam rather than
    a jump in grad d. So ``dMax/rIn`` does not cap how hard a wall may be pressed. A NONCONVEX container
    does have a medial axis and this no longer holds."""
    container = int(packing.containerIndex)
    vertices = packing.positions.reshape(-1, 2)
    starts = np.asarray(packing.startIndices, dtype = int)
    wallSlice = slice(starts[container], starts[container + 1])
    wall = vertices[wallSlice]

    reversed_ = pc.signedArea(wall) > 0.0
    exterior = wall[::-1].copy() if reversed_ else wall.copy()

    energy = 0.0
    gradient = np.zeros_like(vertices)
    for body in range(container):
        block = slice(starts[body], starts[body + 1])
        loop = vertices[block]
        pairwise, towardBody, towardWall = pc.contactGradient(loop.copy(), exterior, stiffness)
        energy += pairwise
        gradient[block] += towardBody
        gradient[wallSlice] += towardWall[::-1] if reversed_ else towardWall
    return float(energy), gradient


# UNVERIFIED(Cam)
def rigidGradient(bodies, stiffness = 1.0):
    """``(energy, perBodyGradient)`` in RIGID-BODY coordinates: ``(dE/dx, dE/dy, dE/dtheta)`` each.

    The chain rule from the vertex gradient. With ``x_i = c + R(theta)(v_i - c) + t``,

        dE/dt      = sum_i g_i
        dE/dtheta  = sum_i g_i . J (x_i - c),    J = [[0, -1], [1, 0]]

    ``J (x_i - c)`` is the velocity of vertex i under rotation about the body centroid, so the angular
    term is just the torque about that centroid."""
    energy, gradient = systemEnergyGradient(bodies, stiffness)
    centroids = bodyCentroids(bodies)
    perBody = np.zeros((bodies.count, 3))
    for body in range(bodies.count):
        block = slice(bodies.startIndices[body], bodies.startIndices[body + 1])
        rows = gradient[block]
        offset = bodies.positions[block] - centroids[body]
        perBody[body, 0:2] = rows.sum(axis = 0)
        perBody[body, 2] = float(np.sum(rows[:, 1] * offset[:, 0] - rows[:, 0] * offset[:, 1]))
    return energy, perBody


# UNVERIFIED(Cam)
def relaxRigid(bodies, maximumIterations = 2000, gradientTolerance = 1e-14, stiffness = 1.0,
               frozen = None):
    """L-BFGS over RIGID-BODY degrees of freedom -- three per body, not two per vertex.

    This is what the initialization protocol wants: the shapes are given and the question is whether
    they FIT, so letting vertices move independently answers a different question and answers it
    trivially (see ``relax``). Returns ``(energy, maxRigidGradient, iterations)``.

    ``maximumIterations`` is deliberately generous. The handoff's open item 1 records that its two
    contacting configurations reached only 8.6e-9 "with the L-BFGS iteration cap binding, not the
    tolerance", and says to raise it."""
    from scipy.optimize import minimize

    reference = bodies.positions.copy()
    centroids = bodyCentroids(bodies)
    offsets = [reference[bodies.startIndices[b]:bodies.startIndices[b + 1]] - centroids[b]
               for b in range(bodies.count)]
    movable = np.ones(bodies.count, dtype = bool)
    if frozen is not None:
        for body in range(bodies.count):
            block = slice(bodies.startIndices[body], bodies.startIndices[body + 1])
            movable[body] = not bool(np.asarray(frozen)[block].all())

    def place(state):
        state = state.reshape(bodies.count, 3)
        for body in range(bodies.count):
            angle = state[body, 2]
            cosine, sine = np.cos(angle), np.sin(angle)
            turned = offsets[body] @ np.array([[cosine, sine], [-sine, cosine]])
            block = slice(bodies.startIndices[body], bodies.startIndices[body + 1])
            bodies.positions[block] = centroids[body] + turned + state[body, 0:2]

    def objective(flat):
        place(flat)
        energy, perBody = rigidGradient(bodies, stiffness)
        perBody[~movable] = 0.0
        return energy, perBody.ravel()

    result = minimize(objective, np.zeros(3 * bodies.count), jac = True, method = "L-BFGS-B",
                      options = {"maxiter": maximumIterations, "gtol": gradientTolerance,
                                 "ftol": 0.0, "maxcor": 20})
    place(result.x)
    energy, perBody = rigidGradient(bodies, stiffness)
    perBody[~movable] = 0.0
    return energy, float(np.abs(perBody).max()), int(result.nit)
