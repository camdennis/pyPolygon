# UNVERIFIED(Cam)
"""SEQUENTIAL LINEAR PROGRAMMING: squeeze a valid packing until its contacts actually engage.

THIS IS THE STEP THE RELAXATION PROTOCOL CANNOT DO. A protocol that ends by bisecting down to zero
overlap stops at the largest size that happens to be valid with the arrangement FROZEN. The depth law
has zero energy and zero force at zero overlap, so nothing rearranges, and the packing is left with
slack distributed unevenly across its contacts -- measured on a 26-square packing: eleven wall gaps
between 1.00e-03 and 1.15e-03, pair gaps around 2e-03, and no single rescale that closes them, because
scaling changes every gap by a different amount.

THE SLACK IS THE WHOLE SHORTFALL. That same packing sat 0.214% above the record, and 1.5e-03 of a side
accumulated along a chain of six squares spanning the box is exactly 0.01 -- the entire deficit. So the
arrangement was right and only the fit was loose.

THE METHOD IS TORQUATO AND JIAO'S ADAPTIVE SHRINKING CELL, specialized to squares. Hold the squares
unit and MINIMIZE THE BOX SIDE ``s``, subject to non-overlap, over the centers, the angles and ``s``
at once. Non-overlap between convex polygons is a disjunction -- SOME separating axis must work -- and
therefore not convex; but fixing one axis per pair and requiring separation along it is a convex
RESTRICTION that is tight at contact, and a linear one once the rotations are linearized. So each
iteration is a linear program in ``3N + 1`` variables, solved inside a trust region and re-linearized.

The contact graph is then the ACTIVE SET at convergence, which is the point: no tolerance has to be
guessed, because the solver decides which constraints are tight.
"""
import numpy as np
from scipy.optimize import linprog

import contacts as ct

_TURN = np.array([[0.0, -1.0], [1.0, 0.0]])
_CORNERS = 0.5 * np.array([[1.0, 1.0], [-1.0, 1.0], [-1.0, -1.0], [1.0, -1.0]])


# UNVERIFIED(Cam)
def corners(centers, angles):
    """The four corners of every unit square, shape ``(N, 4, 2)``."""
    c, s = np.cos(angles), np.sin(angles)
    rotation = np.stack([np.stack([c, -s], -1), np.stack([s, c], -1)], -2)
    return np.einsum("iab,kb->ika", rotation, _CORNERS) + centers[:, None, :]


# UNVERIFIED(Cam)
def separatingAxis(vertices, i, j):
    """``(gap, owner, other, edge, normal)`` for the BEST separating axis of two squares.

    Positive gap means disjoint. The axis is searched over all eight edge normals, which is exhaustive
    for convex polygons: if no edge normal separates them, they overlap."""
    best = (-np.inf, i, j, 0, np.array([1.0, 0.0]))
    for owner, other in ((i, j), (j, i)):
        for edge in range(4):
            e = vertices[owner, (edge + 1) % 4] - vertices[owner, edge]
            normal = np.array([e[1], -e[0]])
            normal /= np.linalg.norm(normal)
            gap = (vertices[other] @ normal).min() - (vertices[owner] @ normal).max()
            if gap > best[0]:
                best = (gap, owner, other, edge, normal)
    return best


# UNVERIFIED(Cam)
def neighborPairs(centers, cutoff = 2.6):
    """Pairs close enough to matter. The cutoff is in SQUARE SIDES; two unit squares cannot touch
    with centers more than sqrt(2) apart, so 2.6 leaves room for them to approach during a step."""
    out = []
    for i in range(len(centers)):
        for j in range(i + 1, len(centers)):
            if np.linalg.norm(centers[i] - centers[j]) <= cutoff:
                out.append((i, j))
    return out


# UNVERIFIED(Cam)
def worstViolation(centers, angles, side, cutoff = 2.6):
    """The deepest overlap in the configuration, as a positive number, or 0.0 if it is valid."""
    vertices = corners(centers, angles)
    worst = 0.0
    for i, j in neighborPairs(centers, cutoff):
        worst = max(worst, -separatingAxis(vertices, i, j)[0])
    worst = max(worst, -vertices.min(), vertices.max() - side)
    return float(worst)


# UNVERIFIED(Cam)
def _program(centers, angles, side, radius, margin, cutoff, relief = False):
    """The linear program for one trust-region step: minimize ``ds`` subject to linearized non-overlap.

    Unknowns are ``dx, dy, dtheta`` per square then ``ds``. A vertex moves as
    ``dv = (dx, dy) + dtheta * T (v - c)`` with ``T`` the quarter turn, which is the linearization of
    the rotation and the only approximation in the whole program.

    THE MARGIN IS WHAT MAKES IT CONVERGE. The linearization errs by ``O(dtheta^2)``, so an LP solution
    sitting exactly on a constraint lands just outside the true feasible set, the step is rejected and
    the trust region collapses -- measured, it stalled at 5.62627 with contacts still 1e-05 apart.
    Demanding a margin that shrinks quadratically with the trust region keeps every accepted step
    strictly valid while still going to zero."""
    count = len(centers)
    sIndex = 3 * count
    columns = sIndex + 2 if relief else sIndex + 1
    vertices = corners(centers, angles)
    rows, limits = [], []

    for i in range(count):
        for k in range(4):
            lever = _TURN @ (vertices[i, k] - centers[i])
            for axis in (0, 1):
                row = np.zeros(columns)
                row[3 * i + axis], row[3 * i + 2] = -1.0, -lever[axis]
                rows.append(row)
                limits.append(vertices[i, k, axis] - margin)
                row = np.zeros(columns)
                row[3 * i + axis], row[3 * i + 2], row[sIndex] = 1.0, lever[axis], -1.0
                rows.append(row)
                limits.append(side - vertices[i, k, axis] - margin)

    for i, j in neighborPairs(centers, cutoff):
        _, owner, other, _, normal = separatingAxis(vertices, i, j)
        turned = _TURN @ normal
        for k in range(4):
            arm = vertices[other, k] - centers[owner]
            row = np.zeros(columns)
            row[3 * other:3 * other + 2] -= normal
            row[3 * other + 2] -= normal @ (_TURN @ (vertices[other, k] - centers[other]))
            row[3 * owner:3 * owner + 2] += normal
            row[3 * owner + 2] -= turned @ arm
            rows.append(row)
            limits.append(normal @ arm - 0.5 - margin)

    matrix = np.array(rows)
    bounds = [(-radius, radius)] * (sIndex + 1)
    if relief:
        # ONE SHARED SLACK ON EVERY ROW, minimized: this is a Chebyshev feasibility program, so it
        # drives the DEEPEST violation down rather than the sum, which is what "make it valid" means.
        matrix[:, -1] = -1.0
        bounds.append((0.0, None))
    cost = np.zeros(columns)
    cost[-1] = 1.0
    return linprog(cost, A_ub = matrix, b_ub = np.array(limits), bounds = bounds, method = "highs")


# UNVERIFIED(Cam)
def relieve(centers, angles, side, radius = 0.05, minRadius = 1e-13, maxSteps = 600,
            cutoff = 2.6, verbose = True):
    """Push an INVALID configuration apart until it is valid, holding the box side as free as it needs.

    A RELAXED PACKING IS NOT AUTOMATICALLY VALID and one that is not will silently defeat the squeeze:
    every trial step measures as violating, the trust region collapses, and the input comes back
    unchanged looking converged. Measured on ``tests/test26``: squares 16 and 17 overlapped by 1.4e-02
    of a side with vertices 7e-04 outside the box, which reported ``s = 5.5969`` -- BETTER than the
    best known 5.6213, because it was not a packing at all. A side that beats the record is the tell.

    Same linear program with one shared slack added to every row and that slack minimized, which drives
    the DEEPEST violation to zero rather than the total."""
    centers, angles, side = np.array(centers, float), np.array(angles, float), float(side)
    count = len(centers)
    violation = worstViolation(centers, angles, side, cutoff)
    if verbose and violation > 0.0:
        print(f"  invalid on entry: worst violation {violation:.3e} -- relieving")
    for step in range(maxSteps):
        if violation <= 0.0:
            break
        result = _program(centers, angles, side, radius, 0.0, cutoff, relief = True)
        if result.success:
            move = result.x[:3 * count].reshape(-1, 3)
            trial = (centers + move[:, :2], angles + move[:, 2], side + result.x[3 * count])
            got = worstViolation(*trial, cutoff = cutoff)
            if got < violation:
                centers, angles, side = trial
                violation = got
                continue
        radius *= 0.5
        if radius < minRadius:
            break
    if verbose:
        state = "valid" if violation <= 0.0 else f"STILL INVALID by {violation:.3e}"
        print(f"  relieved in {step} steps: s = {side:.12f}, {state}")
    return centers, angles, side


# UNVERIFIED(Cam)
def squeeze(centers, angles, side, radius = 0.02, minRadius = 1e-13, maxSteps = 4000,
            cutoff = 2.6, restorations = 8, verbose = True):
    """Minimize the box side over positions, angles and ``s`` at once. Returns ``(centers, angles, s)``.

    Every accepted iterate is a VALID packing, so the returned side is an honest upper bound on the
    optimum for this arrangement at every point along the way, not only at the end.

    THE MARGIN IS FOUND BY RESTORATION, NOT SET FROM THE TRUST RADIUS. Tying it to the radius looks
    natural -- the linearization errs by ``O(dtheta^2)`` and the radius bounds ``dtheta`` -- but the
    radius is an upper bound the step rarely attains, so the margin becomes a FLOOR ON THE CONTACT
    GAPS: measured, a run with margin ``r^2 / 2`` and ``r`` recovering to 0.02 left gaps stuck at
    1.6e-04, 2.4e-04, 3.2e-04, which is that floor exactly. Instead the program is solved with no
    margin, the trial is measured against the TRUE geometry, and if it overlaps the margin is set to
    the overlap it actually produced and the program re-solved. Gaps then close as far as the step
    really allows."""
    centers, angles, side = np.array(centers, float), np.array(angles, float), float(side)
    if worstViolation(centers, angles, side, cutoff) > 0.0:
        centers, angles, side = relieve(centers, angles, side, cutoff = cutoff, verbose = verbose)
    for step in range(maxSteps):
        margin, trial = 0.0, None
        for _ in range(restorations):
            result = _program(centers, angles, side, radius, margin, cutoff)
            if not result.success:
                break
            move = result.x[:-1].reshape(-1, 3)
            candidate = (centers + move[:, :2], angles + move[:, 2], side + result.x[-1])
            violation = worstViolation(*candidate, cutoff = cutoff)
            if violation <= 1e-15:
                trial = candidate
                break
            margin += 1.1 * violation
        if trial is None:
            radius *= 0.5
            if radius < minRadius:
                break
            continue
        gained = side - trial[2]
        centers, angles, side = trial
        radius = min(radius * 1.4, 0.02)
        if verbose and step % 100 == 0:
            print(f"  step {step:5d}  radius {radius:.2e}  s = {side:.12f}")
        if gained < 1e-15:
            radius *= 0.5
            if radius < minRadius:
                break
    if verbose:
        print(f"  stopped after {step + 1} steps: s = {side:.12f}")
    return centers, angles, side


# UNVERIFIED(Cam)
def fromPacking(packing):
    """``(centers, angles, side)`` in UNIT-SQUARE units, which is what ``squeeze`` works in."""
    state, _ = ct.unitState(packing)
    count = (len(state) - 1) // 4
    centers = np.stack([state[0:4 * count:4], state[1:4 * count:4]], axis = -1)
    angles = np.arctan2(state[3:4 * count:4], state[2:4 * count:4])
    return centers, angles, float(state[-1])


# UNVERIFIED(Cam)
def asPacking(centers, angles, side):
    """A packing object carrying the three attributes ``contacts`` and ``refine`` read.

    Unit squares in a box ``[0, s]^2``, so ``contacts.unitState`` round-trips it unchanged."""
    loops = list(corners(centers, angles))
    loops.append(np.array([[0.0, 0.0], [0.0, side], [side, side], [side, 0.0]]))

    class Squeezed:
        pass

    packing = Squeezed()
    packing.positions = np.concatenate(loops)
    packing.startIndices = np.arange(len(loops) + 1) * 4
    packing.containerIndex = len(loops) - 1
    return packing


# UNVERIFIED(Cam)
def closePacking(packing, **options):
    """``(packing, side)`` with the slack squeezed out. The one call that goes in front of ``refine``.

    Mirrors ``contacts.unitState``: a packing in, a side out, so it drops straight into the pipeline."""
    centers, angles, side = squeeze(*fromPacking(packing), **options)
    return asPacking(centers, angles, side), side
