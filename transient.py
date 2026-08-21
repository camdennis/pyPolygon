"""Transient (tunable) target degrees of freedom, following Arzash, Tah, Liu & Manning,
"Rigidity of Epithelial Tissues as a Double Optimization Problem", Phys. Rev. Research 7, 013157
(2025), arXiv:2312.11683.

Normally each polygon's targets are fixed inputs and the shape terms pull the polygon toward them.
Here the TARGETS themselves become degrees of freedom minimized jointly with the vertex positions --
a double optimization. Left free they would simply chase the actual shapes and collapse the elastic
energy to zero, so the distribution is pinned by holding a set of MOMENTS fixed:

    phi_k = sum_i t_i^k = const,     k in the requested list (their eq: k in {-1,-2,-3,1,2,3})

Fixing k = 1 holds the mean, k = 2 the spread, and a NEGATIVE k penalises any target approaching
zero, which is what keeps the distribution from degenerating. The paper's headline result is that
tuning stiffnesses leaves the rigidity transition alone, but making preferred shapes or areas tunable
induces spatial correlations in the targets that SHIFT the transition -- so the chasing is the
physics, not an artifact.

Following the paper, the constraint is imposed by projecting the target force into the constraint
tangent space,

    f^c = f - J0 J0^T f,

with the rows of J0 (the moment gradients, dphi_k/dt_j = k t_j^(k-1)) ORTHONORMALIZED by modified
Gram-Schmidt. No separate mass or step size is introduced for the target DOF: the projection plus the
energy's own scale set the relative motion, exactly as in the reference.
"""

import warnings

import numpy as np


# UNVERIFIED(Cam)
class MomentConstraints:
    """Holds ``sum_i t_i^k`` fixed for each requested exponent ``k``, by projection.

    Built per call from the current target vector, since the Jacobian depends on it."""

    def __init__(self, moments):
        moments = [int(k) for k in np.atleast_1d(moments)]
        if not moments:
            raise ValueError("setMoments needs at least one exponent; k = 1 alone holds the mean.")
        if 0 in moments:
            raise ValueError("k = 0 is not a usable moment: sum_i t_i^0 = N is constant for any "
                             "targets, so it constrains nothing.")
        self.moments = moments

    def values(self, targets):
        """The conserved quantities ``[sum t^k for k in moments]`` at the given targets."""
        return np.array([np.sum(targets.astype(float) ** k) for k in self.moments])

    def basis(self, targets):
        """Orthonormal basis of the constraint-normal space, shape (numMoments, N).

        Rows are the moment gradients ``dphi_k/dt_j = k t_j^(k-1)``, orthonormalized by modified
        Gram-Schmidt (as in the reference). A row that collapses under orthogonalization -- a
        redundant or degenerate moment -- is dropped rather than normalized into noise."""
        t = np.asarray(targets, dtype = float)
        rows = np.array([k * t ** (k - 1) for k in self.moments])
        kept = []
        for row in rows:
            for done in kept:
                row = row - np.dot(row, done) * done
            norm = np.linalg.norm(row)
            if norm > 1e-10 * max(1.0, np.linalg.norm(t)):
                kept.append(row / norm)
        return np.array(kept) if kept else np.zeros((0, t.size))

    def project(self, targets, vector):
        """Tangent-space part of a target force: ``f - sum_i (v_i . f) v_i``."""
        V = self.basis(targets)
        f = np.asarray(vector, dtype = float)
        if V.shape[0] == 0:
            return f
        return f - V.T @ (V @ f)

    def restore(self, targets, reference, tol = 1e-12, maxIter = 60):
        """Pull ``targets`` back onto the constraint surface (the SHAKE analogue for the moments).

        Damped Newton on the minimum-norm correction. Three things make this robust that the obvious
        version is not, each of which was observed failing:

        ROW SCALING. The rows span enormous ranges -- with k in {1, 2, -1, 4} the Jacobian entries go as
        t^-2 through t^3, so on targets of order 0.1 the rows differ by ~10^6 and whichever happens to
        be largest dominates the solve. Dividing each row by its own reference makes the system
        dimensionless, so every moment is enforced to the same RELATIVE accuracy.

        SVD, NOT GRAM. The old version formed ``J J^T`` with a 1e-14 ridge. That squares an already
        terrible condition number, and it overflowed outright during an anneal (``RuntimeWarning:
        overflow encountered in matmul``, then a non-finite Jacobian downstream). ``constraints.py``
        documents the same lesson for the position constraints: work from J itself.

        POSITIVITY AND DESCENT. A Newton step from far away can overshoot a target through zero, and a
        negative target makes both the shape terms and any negative moment meaningless -- ``sum t^-1``
        then diverges. Each step is halved until every target stays positive AND the scaled residual
        actually decreases, so the iteration cannot walk off the surface it is trying to reach."""
        t = np.array(targets, dtype = float)
        reference = np.asarray(reference, dtype = float)
        scale = np.maximum(np.abs(reference), 1e-300)
        floor = max(1e-9 * float(np.mean(np.abs(t))), 1e-300)
        for _ in range(maxIter):
            residual = (self.values(t) - reference) / scale
            worst = float(np.abs(residual).max())
            if worst <= tol:
                return t
            J = np.array([k * t ** (k - 1) for k in self.moments]) / scale[:, None]
            if not np.all(np.isfinite(J)):
                raise FloatingPointError(
                    "moment Jacobian is not finite while restoring the target distribution; a target "
                    "has reached zero or gone negative. Drop the negative exponents from setMoments, "
                    "or ask for fewer moments -- with M exponents on N targets only N - M degrees of "
                    "freedom remain, and the surface can become unreachable.")
            # Merit is the 2-NORM, not the max-norm. Gauss-Newton only guarantees descent in the
            # 2-norm; the max-norm can rise on a perfectly good step when the binding moment swaps,
            # which stalls the line search and aborts a run that was converging fine.
            merit = float(np.linalg.norm(residual))
            newton = np.linalg.lstsq(J, residual, rcond = 1e-12)[0]
            # Steepest descent on the same merit, as a fallback direction: for a small enough step it
            # ALWAYS decreases, so it rescues the cases where the Gauss-Newton direction is poor.
            descent = J.T @ residual
            trial = None
            for direction in (newton, descent):
                norm = float(np.linalg.norm(direction))
                if norm <= 0.0 or not np.isfinite(norm):
                    continue
                damping = 1.0
                for _ in range(60):
                    candidate = t - damping * direction
                    if np.all(candidate > floor):
                        if float(np.linalg.norm((self.values(candidate) - reference) / scale)) < merit:
                            trial = candidate
                            break
                    damping *= 0.5
                if trial is not None:
                    break
            if trial is None:
                # Stalled. This is recoverable -- the caller is usually mid-anneal and the next step
                # re-solves from a slightly different state -- so return the best targets found rather
                # than aborting the run. The moments are reported so a real divergence is still visible.
                warnings.warn(
                    f"\n*** moment restore stalled ***\n"
                    f"    no positive step reduces the residual (worst {worst:.3e}); the conserved "
                    f"moments are left drifted by that much. Returning the best targets found. "
                    f"Persistent stalls mean {len(self.moments)} moments on {t.size} targets is too "
                    f"tight, or a negative exponent is fighting a target driven toward zero.",
                    stacklevel = 3)
                return t
            t = trial
        return t


# UNVERIFIED(Cam)
class TransientTargets:
    """The tunable-target state: which target arrays are free, and the moments pinning them.

    ``targetArea`` and ``targetEdgeLength`` are treated as INDEPENDENT families, each with its own
    moment constraints, matching ``setLogNormalTargetArea`` / ``setLogNormalTargetPerimeter``. Every
    free target is constrained by the same requested exponents.
    """

    def __init__(self, packing, moments, area = True, perimeter = True):
        if not (area or perimeter):
            raise ValueError("transient DOF need at least one of area / perimeter to be free.")
        self.area = bool(area)
        self.perimeter = bool(perimeter)
        self.constraints = MomentConstraints(moments)
        # A CONTAINER wall must stay out of every moment sum. Its area is SIGNED and negative (the wall
        # is wound the other way so the packing sits outside it), so including it makes sum_i A0_i the
        # difference of a -1 wall and the polygons -- for 5 squares at phi = 0.8 that reference is
        # -0.2, and the restore then drives the polygon targets NEGATIVE chasing it (measured: -227,
        # after which the constraint Jacobian divides by it and the run dies). The wall's targets are
        # not degrees of freedom in any case: it is pinned.
        self.span = {"targetArea": self.numFree(packing, "targetArea"),
                     "targetEdgeLength": self.numFree(packing, "targetEdgeLength")}
        self.reference = {}
        for name in self.families():
            self.reference[name] = self.constraints.values(self.free(packing, name))

    @staticmethod
    def numFree(packing, name):
        """How many leading entries of a target array are free DOF, i.e. exclude the container."""
        container = getattr(packing, "containerIndex", None)
        if container is None:
            return len(getattr(packing, name))
        if name == "targetArea":
            return int(container)
        return int(packing.startIndices[int(container)])

    def free(self, packing, name):
        """The free (non-container) slice of a target family."""
        return np.asarray(getattr(packing, name), dtype = float)[:self.span[name]]

    def setFree(self, packing, name, values):
        """Write back a free slice, leaving the container's entry untouched."""
        getattr(packing, name)[:self.span[name]] = values

    def families(self):
        return tuple(n for n, on in (("targetArea", self.area),
                                     ("targetEdgeLength", self.perimeter)) if on)

    def momentDrift(self, packing):
        """Largest RELATIVE departure of any conserved moment from its initial value."""
        worst = 0.0
        for name in self.families():
            reference = self.reference[name]
            current = self.constraints.values(self.free(packing, name))
            worst = max(worst, float(np.abs((current - reference)
                                            / np.maximum(np.abs(reference), 1e-300)).max()))
        return worst

    def retarget(self, packing, polydispersity):
        """Re-aim the conserved moments at a distribution with the SAME mean but the requested
        coefficient of variation, without moving the targets themselves.

        This is the handle an anneal needs: ``setMoments`` pins the moments at whatever they happen to
        be, which freezes the distribution's width forever. Driving the width down instead -- from
        polydisperse (easy to pack, shallow landscape, many near-degenerate solutions) toward
        monodisperse (the hard problem) -- follows a solution branch rather than dropping into a
        random basin.

        For a log-normal of mean mu and CV c the moments are closed-form:
        ``sum t = N mu``, ``sum t^2 = N mu^2 (1 + c^2)``, ``sum t^-1 = N (1 + c^2) / mu``, and in
        general ``E[t^k] = mu^k (1 + c^2)^(k(k-1)/2)``. Only the REFERENCE changes here; the next
        ``restore`` pulls the targets onto it."""
        for name in self.families():
            t = self.free(packing, name)
            n = t.size
            mu = float(t.mean())
            factor = 1.0 + polydispersity ** 2
            self.reference[name] = np.array(
                [n * mu ** k * factor ** (0.5 * k * (k - 1)) for k in self.constraints.moments])
        return self

    def restore(self, packing):
        """Re-impose every family's moments in place (after a step has drifted them)."""
        for name in self.families():
            restored = self.constraints.restore(self.free(packing, name), self.reference[name])
            self.setFree(packing, name, restored)
            if name == "targetEdgeLength":
                packing.syncTargetPerimeter()


# UNVERIFIED(Cam)
def targetForces(packing, kEdge, kArea):
    """``-dE/d(targets)`` from the RELATIVE eqSoftBody shape terms, analytically.

    With the relative form the library uses,

        E_edge = (kEdge/2) sum_k (l_k - l0)^2 / l0^2,   E_area = (kArea/2) sum_p (A_p - A0)^2 / A0^2

    the target derivatives are

        dE/dl0 = -kEdge sum_{k in p} [ (l_k - l0)/l0^2 + (l_k - l0)^2/l0^3 ]
        dE/dA0 = -kArea [ (A_p - A0)/A0^2 + (A_p - A0)^2/A0^3 ]

    Both vanish when the polygon already meets its target, so a free target chases the realised shape
    -- which is precisely the coupling that makes this a double optimization.

    SCOPE: only the shape terms are differentiated. The overlap normalizer also depends on
    targetArea (norm = A0[A] + A0[B]), and that contribution is NOT included here -- it would need the
    per-pair overlap areas, which the CUDA driver does not return. Targets therefore feel geometric
    incompatibility but not the overlap normalization, which is faithful to the vertex-model
    reference (where targets appear only in the elastic energy) but IS an approximation for this
    model. Returns ``{"targetArea": fA, "targetEdgeLength": fL}`` (forces, i.e. -dE/dt)."""
    from softBody import backboneArea, backboneEdgeLengths
    l = backboneEdgeLengths(packing)
    A = backboneArea(packing)
    shapeId = packing.shapeId
    counts = np.diff(packing.startIndices)

    # Every EDGE target is its own degree of freedom, so this stays per-edge (no bincount).
    l0 = packing.targetEdgeLength
    stretch = l - l0
    forceEdge = kEdge * (stretch / l0 ** 2 + stretch ** 2 / l0 ** 3)

    A0 = packing.targetArea
    diff = A - A0
    dEdA0 = -kArea * (diff / A0 ** 2 + diff ** 2 / A0 ** 3)
    return {"targetArea": -dEdA0, "targetEdgeLength": forceEdge}
