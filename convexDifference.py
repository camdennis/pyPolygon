"""Convex differences tree for a simple polygon, and the R-function depth built on it.

Implements the nonconvex remedy of ``notes/softDepth.tex`` sec:nonconvex, in the CSG form the note
allows -- "arbitrary nesting of unions, intersections and complements" -- rather than as a union of a
partition. THE DISTINCTION IS NOT COSMETIC. Measured on an L-shape whose true interior depth is 0.500
along a line crossing an artificial diagonal:

    partition into convex pieces + softmax     0.0139   (36x too small)
    convex differences tree (this module)      0.4861   (off by exactly eps log 2)

A partition puts every artificial diagonal into the zero set: ``max(f_1, f_2)`` vanishes wherever two
pieces' boundaries meet, and an internal seam is precisely such a place, so the composed depth creases
down to ~0 along it. With ``phi ~ h^(5/2)`` that is a factor of 316 of missing contact energy. The
remedy is to build the shape by DIFFERENCE from a containing convex body, so that no boundary appears
where the shape has no boundary.

THE CONSTRUCTION (Woodwark's convex differences tree). A simple polygon is its convex hull minus the
"pockets" -- the regions between the hull and the polygon -- and each pocket is itself a simple polygon,
so the same statement recurses:

    Omega = hull(Omega) \\ union_i P_i,        P_i = hull(P_i) \\ union_j P_ij,   ...

Every node is convex, so ``softDepth`` applies to it unchanged, and the composition is

    h(node) = softmin( h_hull(node), -h(child_1), -h(child_2), ... )

intersection being softmin and complement being negation.

WHY THE VERTEX GRADIENT IS EASY HERE. Every node's polygon is built from ORIGINAL vertices of the input
loop: hull vertices are original, and a pocket is an original chain closed by a chord whose two
endpoints are also original. No node introduces a new point, so a node's vertex derivative scatters
straight back to the input indices with no interpolation and no reindexing.

WHAT COMPOSITION COSTS. The eikonal identity |grad h|^2 - eps lap(h) = 1 holds only for a single
softmin of affine functions and is NOT preserved by the composition, so the cheapest verification test
in the note is unavailable here and finite differences replace it (sec:nonconvex says this explicitly).
The Hessian also stops being sign-definite: complement flips its sign.
"""

# UNVERIFIED(Cam)

import warnings

import numpy as np

from softDepth import softDepth, depthGradientHessian


# UNVERIFIED(Cam)
def signedArea(loop):
    """Twice-signed area / 2 by the shoelace formula; positive for a CCW loop."""
    loop = np.asarray(loop, dtype = float)
    following = np.roll(loop, -1, axis = 0)
    return 0.5 * float(np.sum(loop[:, 0] * following[:, 1] - following[:, 0] * loop[:, 1]))


# UNVERIFIED(Cam)
def convexHullIndices(loop):
    """Indices of the CCW convex hull of ``loop``, as INDICES INTO ``loop`` rather than points.

    Indices rather than coordinates because the whole tree has to stay addressable back to the input
    vertices for the gradient. Monotone chain, so collinear points are dropped -- a hull vertex with a
    straight angle would give two identical half-planes and a duplicated softmin term (sec:numerics
    warns about exactly that)."""
    loop = np.asarray(loop, dtype = float)
    order = sorted(range(len(loop)), key = lambda i: (loop[i, 0], loop[i, 1]))

    def half(sequence):
        chain = []
        for index in sequence:
            while len(chain) >= 2:
                a, b = loop[chain[-2]], loop[chain[-1]]
                if (b[0] - a[0]) * (loop[index][1] - a[1]) - (b[1] - a[1]) * (loop[index][0] - a[0]) > 0:
                    break
                chain.pop()
            chain.append(index)
        return chain

    lower = half(order)
    upper = half(order[::-1])
    return lower[:-1] + upper[:-1]


class DifferenceNode:
    """One node: a CONVEX loop, minus the regions of its children.

    ``indices`` are positions in the ORIGINAL input loop, so ``vertices[node.indices]`` recovers the
    node's polygon at any configuration and the gradient scatters back by the same map."""

    # UNVERIFIED(Cam)
    def __init__(self, indices, children = None):
        self.indices = list(indices)
        self.children = list(children) if children else []

    def nodeCount(self):
        return 1 + sum(child.nodeCount() for child in self.children)

    def depthOfTree(self):
        return 1 + max((child.depthOfTree() for child in self.children), default = 0)


# UNVERIFIED(Cam)
def buildDifferenceTree(loop, indices = None, maximumDepth = 32):
    """The convex differences tree of a simple CCW polygon.

    ``indices`` addresses a sub-chain of the original loop (used by the recursion); leave it None at the
    top. Returns a ``DifferenceNode`` whose region is ``hull(indices) \\ union(children)``.

    A pocket is the region between a hull edge and the original chain it skips over. Its boundary is
    that chain closed by the hull chord, and it is traversed CLOCKWISE when the parent is CCW, so it is
    reversed to keep every node's normals outward -- ``softDepth`` reads orientation from the winding."""
    loop = np.asarray(loop, dtype = float)
    if indices is None:
        indices = list(range(len(loop)))
        if signedArea(loop) < 0.0:
            indices = indices[::-1]

    points = loop[indices]
    hullLocal = convexHullIndices(points)
    if len(hullLocal) < 3 or maximumDepth <= 0:
        return DifferenceNode(indices)

    # Rotate the hull so its order follows the chain's own order, then read off the skipped runs.
    onHull = sorted(hullLocal, key = lambda local: local)
    hullSet = set(onHull)
    children = []
    for position, local in enumerate(onHull):
        following = onHull[(position + 1) % len(onHull)]
        span = (following - local) % len(indices)
        if span <= 1:
            continue
        chain = [indices[(local + step) % len(indices)] for step in range(span + 1)]
        pocketPoints = loop[chain]
        if abs(signedArea(pocketPoints)) < 1e-15:
            continue
        if signedArea(pocketPoints) < 0.0:
            chain = chain[::-1]
        children.append(buildDifferenceTree(loop, chain, maximumDepth - 1))

    return DifferenceNode([indices[local] for local in onHull], children)


# Warn when a pocket corner is sharp enough that the exterior bias eats half its contact energy.
# sin(theta/2)^(5/2) < 0.5 is theta < 98.6 degrees.
_SHARP_POCKET_ENERGY_FACTOR = 0.5
_warnedPockets = set()


# UNVERIFIED(Cam)
def worstPocketFactor(loop, node):
    """``(energyFactor, angle)`` for the SHARPEST complemented corner in the tree, or ``(1.0, pi)``.

    *** KNOWN, DELIBERATE, UNFIXED BIAS -- see TODO.md. ***

    Outside a convex piece the half-plane softmin reads the largest single half-plane VIOLATION, not the
    Euclidean distance. At a corner of interior angle ``theta`` that is short by ``sin(theta/2)``
    (notes eq:extfactor), so the depth is under-read and, since ``phi ~ h^(5/2)``, the contact energy is
    short by ``sin(theta/2)^(5/2)``. Only COMPLEMENTED nodes -- the pockets -- are ever evaluated from
    outside, so only their corners carry it, and those corners are exactly the reflex vertices of the
    original shape.

    Measured on builder packings: median energy factor 0.81-0.96 at kappa=4, which is benign. The TAIL
    is not: a 2.2-degree pocket at n=32 reads 1e-4 of its true contact energy, i.e. that notch is
    effectively invisible. At kappa=20 even the MEDIAN is 0.354. The error is single-signed, so it
    biases rather than averaging out.

    Accepted deliberately (Cam's call) to keep ``ell_i`` affine along an edge, which the exact envelope
    walk, the single-contact-interval guarantee and the closed-form stretch integral of sec:chord all
    depend on. The targeted fix, if the tail ever matters, is exterior-exact distances on POCKET NODES
    ONLY -- ``min_i dist(x, segment_i)``, which coincides with ``min_i ell_i`` inside -- leaving the
    convex fast path untouched."""
    worst, worstAngle = 1.0, np.pi
    loop = np.asarray(loop, dtype = float)
    for child in node.children:
        corners = loop[child.indices]
        previous = np.roll(corners, 1, axis = 0)
        following = np.roll(corners, -1, axis = 0)
        back, forward = previous - corners, following - corners
        cosine = np.einsum("ij,ij->i", back, forward) / (
            np.hypot(*back.T) * np.hypot(*forward.T) + 1e-300)
        angles = np.arccos(np.clip(cosine, -1.0, 1.0))
        factors = np.sin(0.5 * angles) ** 2.5
        position = int(np.argmin(factors))
        if factors[position] < worst:
            worst, worstAngle = float(factors[position]), float(angles[position])
        childWorst, childAngle = worstPocketFactor(loop, child)
        if childWorst < worst:
            worst, worstAngle = childWorst, childAngle
    return worst, worstAngle


def warnOnSharpPockets(loop, node):
    """Emit a warning when the tree contains a pocket corner sharp enough to matter.

    Deduplicated on a rounded message so a minimizer rebuilding trees every step does not spam, but
    NOT silenced: today's lesson is that a silent factor-of-10000 is worse than a noisy one."""
    factor, angle = worstPocketFactor(loop, node)
    if factor >= _SHARP_POCKET_ENERGY_FACTOR:
        return factor
    # Bucketed by 5 degrees so a packing of many similar shapes warns a handful of times rather than
    # once per polygon per force evaluation. Deliberately NOT once-per-session: Cam asked to be
    # reminded, and a bias this large should stay visible.
    bucket = int(np.degrees(angle) // 5)
    if bucket not in _warnedPockets:
        _warnedPockets.add(bucket)
        warnings.warn(
            f"\n*** sharp pocket: corner angle ~{np.degrees(angle):.0f} deg carries only "
            f"{factor:.2e} of its true contact energy ***\n"
            f"    KNOWN, DELIBERATE, UNFIXED exterior bias of the complemented node. Depth is short by "
            f"sin(theta/2), energy by sin(theta/2)^(5/2).\n"
            f"    See convexDifference.worstPocketFactor and TODO.md. The fix, if this matters, is "
            f"exterior-exact distances on POCKET NODES ONLY.", stacklevel = 3)
    return factor


# UNVERIFIED(Cam)
def treeDepth(points, loop, node, epsilon):
    """Composed soft depth ``h`` at each point, by nested softmin and complement.

    ``h(node) = softmin( h_hull(node), -h(child_1), ... )`` in the shifted form, which matters as much
    here as it does for the single softmin: the exponents are otherwise unbounded.

    Returns ``(h, weights)`` where ``weights`` is the list of softmin weights at THIS node, one per
    term, in the order (hull, child_1, child_2, ...). The recursion returns enough for the caller to
    chain-rule, and ``treeDepthGradient`` does exactly that."""
    points = np.atleast_2d(np.asarray(points, dtype = float))
    loop = np.asarray(loop, dtype = float)
    terms = [softDepth(points, loop[node.indices], epsilon)[0]]
    for child in node.children:
        terms.append(-treeDepth(points, loop, child, epsilon)[0])
    stack = np.stack(terms, axis = 1)

    lowest = stack.min(axis = 1, keepdims = True)
    shifted = np.exp(-(stack - lowest) / epsilon)
    total = shifted.sum(axis = 1, keepdims = True)
    h = (lowest - epsilon * np.log(total))[:, 0]
    return h, shifted / total


# UNVERIFIED(Cam)
def treeDepthGradient(points, loop, node, epsilon):
    """``(h, gradient)`` of the composed depth with respect to the POINT.

    The chain rule through a softmin is the same weighted mean as (10): with
    ``h = softmin_k(g_k)``, ``grad h = sum_k w_k grad g_k``. A complement contributes ``-grad`` of its
    child, so the signs alternate with nesting depth automatically."""
    points = np.atleast_2d(np.asarray(points, dtype = float))
    loop = np.asarray(loop, dtype = float)
    h, weights = treeDepth(points, loop, node, epsilon)

    _, hullWeights, hullNormals, _ = softDepth(points, loop[node.indices], epsilon)
    hullGradient, _ = depthGradientHessian(hullWeights, hullNormals, epsilon)
    gradient = weights[:, 0:1] * hullGradient
    for position, child in enumerate(node.children):
        _, childGradient = treeDepthGradient(points, loop, child, epsilon)
        gradient = gradient - weights[:, position + 1:position + 2] * childGradient
    return h, gradient


# UNVERIFIED(Cam)
def subtractIntervals(keep, remove):
    """``keep \\ remove`` for two lists of disjoint, ascending [lo, hi] intervals."""
    result = []
    for lo, hi in keep:
        pieces = [(lo, hi)]
        for cutLo, cutHi in remove:
            nextPieces = []
            for pieceLo, pieceHi in pieces:
                if cutHi <= pieceLo or cutLo >= pieceHi:
                    nextPieces.append((pieceLo, pieceHi))
                    continue
                if cutLo > pieceLo:
                    nextPieces.append((pieceLo, cutLo))
                if cutHi < pieceHi:
                    nextPieces.append((cutHi, pieceHi))
            pieces = nextPieces
        result.extend(piece for piece in pieces if piece[1] - piece[0] > 0.0)
    return result


# UNVERIFIED(Cam)
def treeContactIntervals(edgeStart, edgeEnd, loop, node, epsilon):
    """Sub-intervals of [0, 1] along one edge on which the COMPOSED depth is non-negative.

    A union of intervals rather than one, because composition destroys concavity. But the set algebra
    is exact and needs no new root finding: ``h(node) = softmin(h_hull, -h(child), ...)`` is at most
    each of its terms, so

        {h(node) >= 0}  is contained in  {h_hull >= 0} \\ union_children {h(child) >= 0}

    and each node's own hull IS convex, so its own set is the single interval that
    ``softDepth.contactIntervals`` already returns. Recursion plus interval subtraction does the rest.

    Containment rather than equality means the domain is very slightly OVER-estimated -- by the O(eps)
    softmin gap. That is the safe direction: ``phi`` vanishes wherever ``h < 0``, so an over-wide domain
    contributes exactly zero there, whereas an under-wide one would silently drop contact."""
    from softDepth import contactIntervals
    hull = np.asarray(loop, dtype = float)[node.indices]
    lower, upper = contactIntervals(np.asarray([edgeStart, edgeEnd], dtype = float),
                                    hull, epsilon)
    own = [(float(lower[0]), float(upper[0]))] if upper[0] > lower[0] else []
    for child in node.children:
        own = subtractIntervals(own, treeContactIntervals(edgeStart, edgeEnd, loop, child, epsilon))
        if not own:
            break
    return own


# UNVERIFIED(Cam)
def accumulateVertexGradient(points, loop, node, epsilon, coefficient, out):
    """``out[j] += sum_q coefficient[q] * d h(x_q) / d v_j``, scattered onto ORIGINAL vertex indices.

    The chain rule through the tree: ``dh/dv = sum_k w_k dg_k/dv`` with ``w_k`` the softmin weights, the
    hull term contributing the lever rule (43) on its own edges and each child contributing MINUS its
    own gradient. Because every node addresses original vertices, the scatter needs no interpolation."""
    from softDepth import loopFrame
    points = np.atleast_2d(np.asarray(points, dtype = float))
    loop = np.asarray(loop, dtype = float)
    _, weights = treeDepth(points, loop, node, epsilon)

    indices = np.asarray(node.indices, dtype = int)
    hull = loop[indices]
    _, tangents, normals, lengths, _ = loopFrame(hull)
    _, softWeights, _, feet = softDepth(points, hull, epsilon)
    scaled = coefficient * weights[:, 0]
    toOwnEnd = np.einsum("p,pi,ij->ij", scaled, softWeights * (1.0 - feet), normals)
    toNextEnd = np.einsum("p,pi,ij->ij", scaled, softWeights * feet, normals)
    perVertex = toOwnEnd + np.roll(toNextEnd, 1, axis = 0)
    np.add.at(out, indices, perVertex)

    for position, child in enumerate(node.children):
        accumulateVertexGradient(points, loop, child, epsilon,
                                 -coefficient * weights[:, position + 1], out)


# UNVERIFIED(Cam)
def collectNodeIntervals(loopA, loopB, node, epsilon, out):
    """Per-node contact interval for EVERY edge of ``loopA`` at once, keyed by node.

    Vectorized over edges deliberately. ``contactIntervals`` already solves all of a loop's edges in one
    branch-free sweep, so calling it once per NODE costs ~50 numpy calls; calling it once per
    (edge, node) costs ~50 per edge and is what made the first version unusable -- 4291 ``softDepth``
    calls for a single N=4, n=8 force evaluation, essentially all of it interpreter overhead on
    two-element arrays."""
    from softDepth import contactIntervals
    hull = np.asarray(loopB, dtype = float)[node.indices]
    out[id(node)] = contactIntervals(loopA, hull, epsilon)
    for child in node.children:
        collectNodeIntervals(loopA, loopB, child, epsilon, out)
    return out


def _edgeIntervals(node, edge, cache):
    """The composed contact set on one edge, from the cached per-node intervals."""
    lower, upper = cache[id(node)]
    own = [(float(lower[edge]), float(upper[edge]))] if upper[edge] > lower[edge] else []
    for child in node.children:
        if not own:
            break
        own = subtractIntervals(own, _edgeIntervals(child, edge, cache))
    return own


# UNVERIFIED(Cam)
def treeEdgeLoopEnergyForce(loopA, loopB, node, epsilon, stiffness, order = 8, panelsPerEpsilon = 4.0):
    """``int_{dA} phi(h_B) dl`` with ``h_B`` the COMPOSED depth of a nonconvex loop B.

    Returns ``(energy, forcesA, forcesB)``.

    COMPOSITE GAUSS, panels capped at ``panelsPerEpsilon * epsilon``, rather than the envelope-split
    rule used for convex loops. The envelope walk relies on the integrand's features being the softmin's
    half-plane switches, exactly locatable because each ``ell_i`` is affine along the edge. Under
    composition the dominant TERM switches too, and those crossings are not affine, so the clean
    construction no longer applies. Capping the panel at a few multiples of ``epsilon`` resolves a
    feature of width ``epsilon`` without needing to know where it is -- measured 2.7e-09 at 4 eps with
    order 8 against the exact envelope rule on the convex case.

    EVERYTHING IS BATCHED ACROSS EDGES. The per-node root solves go through one vectorized sweep, and
    every quadrature point of every edge is handed to the tree in a single array, so the number of numpy
    calls scales with the number of TREE NODES rather than with nodes times edges times panels. The
    convex path is untouched and still takes the faster envelope route."""
    from softDepth import contactLaw
    loopA = np.asarray(loopA, dtype = float)
    loopB = np.asarray(loopB, dtype = float)
    count = len(loopA)
    nextIndex = (np.arange(count) + 1) % count
    forcesA = np.zeros_like(loopA)
    forcesB = np.zeros_like(loopB)

    starts = loopA
    ends = loopA[nextIndex]
    directions = ends - starts
    lengths = np.hypot(directions[:, 0], directions[:, 1])

    # ONLY THE ROOT HULL'S INTERVAL IS SOLVED FOR. The composed depth is at most each of its terms, so
    # {h >= 0} is contained in the root hull's own interval, and that hull is convex so the interval is
    # the single one contactIntervals already returns. Subtracting the children would tighten the
    # domain but costs one root solve per NODE, and the root solve is the dominant term -- 0.51 s of a
    # 0.9 s evaluation. Over-estimating the domain is free: phi vanishes wherever h < 0, so the extra
    # nodes contribute exactly zero rather than contributing wrongly.
    from softDepth import contactIntervals
    rootLower, rootUpper = contactIntervals(loopA, loopB[node.indices], epsilon)
    from softDepth import _gaussRule
    gaussNodes, gaussWeights = _gaussRule(int(order))

    localParts, weightParts, edgeParts = [], [], []
    for edge in range(count):
        for lower, upper in ([(float(rootLower[edge]), float(rootUpper[edge]))]
                             if rootUpper[edge] > rootLower[edge] else []):
            panels = max(1, int(np.ceil((upper - lower) * lengths[edge]
                                        / (panelsPerEpsilon * epsilon))))
            bounds = np.linspace(lower, upper, panels + 1)
            lo, hi = bounds[:-1, None], bounds[1:, None]
            local = (0.5 * (hi + lo) + 0.5 * (hi - lo) * gaussNodes[None, :]).ravel()
            localParts.append(local)
            weightParts.append((lengths[edge] * 0.5 * (hi - lo) * gaussWeights[None, :]).ravel())
            edgeParts.append(np.full(local.shape, edge, dtype = int))
    if not localParts:
        return 0.0, forcesA, forcesB

    local = np.concatenate(localParts)
    weight = np.concatenate(weightParts)
    edgeIndex = np.concatenate(edgeParts)
    points = starts[edgeIndex] + local[:, None] * directions[edgeIndex]

    depth, gradient = treeDepthGradient(points, loopB, node, epsilon)
    density, first, _ = contactLaw(depth, stiffness)
    energy = float(np.dot(weight, density))

    # Node force -dE/dx = -phi'(h) grad h, split barycentrically onto the edge's two endpoints.
    nodeForces = -(weight * first)[:, None] * gradient
    np.add.at(forcesA, edgeIndex, (1.0 - local)[:, None] * nodeForces)
    np.add.at(forcesA, nextIndex[edgeIndex], local[:, None] * nodeForces)

    # The measure moves: dl = |e| dt gives a tangential pair, equal and opposite, torque-free.
    perEdge = np.zeros(count)
    np.add.at(perEdge, edgeIndex, weight * density)
    live = lengths > 0.0
    tangential = np.zeros_like(loopA)
    tangential[live] = (perEdge[live] / lengths[live] ** 2)[:, None] * directions[live]
    forcesA += tangential
    np.subtract.at(forcesA, nextIndex, tangential)

    accumulateVertexGradient(points, loopB, node, epsilon, -(weight * first), forcesB)
    return energy, forcesA, forcesB
