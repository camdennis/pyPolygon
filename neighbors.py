"""Geometry-attached neighbor balls with a Verlet skin -- the candidate pairs the intersection scan
actually has to test.

``energies.updateIntersections`` was written as a full double loop over polygon pairs, and says so in
its own docstring. That is not a cold path: ``sharpContainerEnergyForce`` calls it on every sharp force
evaluation and ``Model.getOverlapArea`` calls it on every packing verdict, so the quadratic scan is the
hot loop of a sweep even when the overlap energy itself runs on the GPU. Its cost is O(N^2) polygon
pairs times O(n^2) edge pairs, and the n^2 factor is the one that bites for 16- and 32-gons.

TWO LEVELS OF BALL, both attached to the geometry rather than to the domain:

  polygon   center = centroid, radius = covering radius (the true max vertex distance)
  edge      center = midpoint,  radius = half the edge length

A uniform grid was considered and rejected for two reasons. The configurations of interest are large
loops with big VOID areas, and a grid allocates cells for empty space. And a grid's cell size is set by
the LARGEST edge -- a container wall spans the whole box, so ``cellSize = 1.0`` collapses the grid to a
single cell and buys nothing. Balls size themselves per object and are indifferent to both.

WHY THE CRITERION IS EXACT. Two segments that cross at a point ``p`` have their midpoints within half
their own lengths of ``p``, so

    |mid_i - mid_j|  <=  |e_i|/2 + |e_j|/2

always holds for a genuine crossing. Testing with ``<`` plus a nonnegative skin therefore cannot miss
one. The same argument at the polygon level is the covering-radius test: two polygons whose centroids
are farther apart than the sum of their covering radii cannot touch anywhere.

THE SKIN IS WHAT MAKES REUSE SAFE. A pair excluded at build time is separated by more than ``skin``
beyond contact. If every vertex has moved less than ``skin/2`` since the build, two edges can have
closed that gap by at most ``skin``, so the excluded pair still cannot be crossing. Hence the rule:

    rebuild when   max_v |r_v - r_v^built|  >  skin/2

which is checked on every use, never assumed. A silent miss here does not crash -- it under-reports
overlap, makes an invalid packing look valid, and corrupts the density a sweep reports. That is why
``tests/neighborCheck.py`` asserts exact set equality against the all-to-all reference rather than
agreement to a tolerance.

The container needs no special case: a wall edge simply gets a big ball, and the polygon level sees a
covering radius of the box half-diagonal, which is permissive rather than exclusionary. This matters
because the sharp container term is written on the assumption that the scan is all-to-all -- see
``sharpContainerEnergyForce``, which notes a neighbor-list version "would have to guarantee container
pairs are never culled". Nothing here culls them, and ``tests/neighborCheck.py`` check 6 pins that down
with a polygon deliberately pushed through the wall rather than leaving it to argument.
"""

# UNVERIFIED(Cam)

import numpy as np

from packing import minImageShift


# Default skin as a fraction of the mean COVERING RADIUS -- the polygon's size, not its edge length.
#
# Edge length is the wrong scale and badly so at large n. A 32-gon at the same area as a square has
# edges eight times shorter, but its vertices move on the scale of the POLYGON, so an edge-derived skin
# invalidates the list almost every step: measured on 32 32-gons, an edge-based skin of 5.7e-03 gave a
# reuse of 2.25 force evaluations per rebuild, against 542 for squares. The covering radius is a
# geometric size that does not collapse as n grows, so one fraction works across shapes.
_DEFAULT_SKIN_FRACTION = 0.25
# How many force evaluations one rebuild should buy. The adaptive skin solves for this.
#
# IT BARELY MATTERS, which is worth knowing before anyone tunes it. Scanned 5 / 10 / 20 / 40 on 32
# 32-gons, the candidate count moved 1451 -> 13019 (9x) while the runtime stayed 28.3-29.2s -- inside
# run-to-run scatter. On 12 squares the skin clamps at its ceiling for every target and the runtime does
# not move at all. The list stopped being the bottleneck once it existed; SHAKE is ~50% of a step.
# Left at 40 because nothing argues for another value. ``statistics()['reuse']`` reports the realized
# figure if a future configuration does make this matter.
_DEFAULT_TARGET_REUSE = 40.0
# Bounds on the adaptive skin, as fractions of the mean polygon covering radius. They exist to stop the
# feedback running away in either direction, NOT for correctness -- any skin is safe, since the
# staleness test is what guarantees no crossing is missed.
_MIN_SKIN_FRACTION = 0.02
_MAX_SKIN_FRACTION = 1.0


# UNVERIFIED(Cam)
def edgeBalls(packing):
    """``(midpoints, halfLengths)`` per EDGE, indexed by the vertex the edge leaves from.

    The tight bounding ball of a segment, and edge-edge is the granularity the crossing test works in.
    A per-VERTEX ball would have to be conservative by a whole edge length to cover the same segments."""
    r = packing.positions.reshape(-1, 2)
    nextIndex = np.asarray(packing.next, dtype = int)
    edge = r[nextIndex] - r
    return 0.5 * (r + r[nextIndex]), 0.5 * np.hypot(edge[:, 0], edge[:, 1])


# UNVERIFIED(Cam)
def polygonBalls(packing):
    """``(centroids, coveringRadii)`` per polygon -- the coarse level.

    Delegates to ``energies.polygonCentroidsRadii`` rather than recomputing: that one measures the
    covering radius from the ACTUAL vertices, which is both tighter than the old ``targetPerimeter/4``
    (by ~1.55x on a decagon) and rigorous, since a target is something the springs let the real
    perimeter exceed."""
    from energies import polygonCentroidsRadii
    return polygonCentroidsRadii(packing)


# UNVERIFIED(Cam)
def meanPolygonRadius(packing):
    """Mean covering radius over the ORDINARY polygons, container excluded.

    The container has to be left out for the same reason every other distribution here leaves it out:
    it is a pinned wall whose covering radius is the box half-diagonal, so it is neither representative
    of the geometry that moves nor comparable in size to it. Measured on 11 squares in a unit box, the
    polygons sit at 0.1913 and the wall at 0.7071, and including it drags the mean to 0.2343 -- a 22%
    inflation of a number the skin is derived from."""
    radii = polygonBalls(packing)[1]
    container = getattr(packing, "containerIndex", None)
    if container is not None:
        radii = radii[:int(container)]
    return float(np.mean(radii)) if radii.size else 0.0


# UNVERIFIED(Cam)
class NeighborList:
    """Candidate edge pairs whose balls overlap, cached across steps and rebuilt on a skin trigger.

    Built and used through ``candidates(packing)``, which rebuilds only when the geometry has moved far
    enough to invalidate the cache. Everything else on this object is diagnostics."""

    def __init__(self, packing, skin = None, targetReuse = None, useCuda = True):
        # A FIXED skin is the wrong shape of knob, and the geometric default only gets half of it
        # right. The skin has two effects with different natural units. Its COST -- how many extra
        # candidate pairs it admits -- scales with skin/r, r the polygon size, so a geometric scale is
        # correct there and the covering radius is the one that does not collapse as n grows
        # (R/edge = 1/(2 sin(pi/n)) runs 0.71 at n=4 to 5.1 at n=32). Its BENEFIT -- how many steps the
        # list survives -- is skin / (2 x per-step displacement), which is DYNAMICS and has nothing to
        # do with geometry: measured reuse ran 542 on one sweep and 8.0 on another at the same formula.
        #
        # So the skin is derived from a target REUSE instead, using the displacement the list already
        # has to measure for its staleness test. The knob becomes "how many force evaluations should a
        # rebuild buy", which has an interpretable optimum -- the ratio of rebuild cost to per-step
        # pair-test cost -- and it self-corrects as the dynamics change, which they do constantly
        # inside a sweep as sigma ramps, phi drops and the tier switches.
        self.adaptive = skin is None
        self.useCuda = bool(useCuda)
        self.targetReuse = float(_DEFAULT_TARGET_REUSE if targetReuse is None else targetReuse)
        if self.targetReuse < 1.0:
            raise ValueError(f"targetReuse must be at least 1, got {self.targetReuse}")
        # The geometric value is still the STARTING point: there is no displacement history yet, and
        # it is the right order of magnitude.
        self.skin = float(skin) if skin is not None \
            else _DEFAULT_SKIN_FRACTION * meanPolygonRadius(packing)
        if self.skin < 0.0:
            raise ValueError(f"skin must be non-negative, got {self.skin}")
        self.floor = _MIN_SKIN_FRACTION * meanPolygonRadius(packing)
        self.ceiling = _MAX_SKIN_FRACTION * meanPolygonRadius(packing)
        self._built = None                  # positions at the last build
        self._pairs = None                  # (edgeI, edgeJ) global edge indices
        self._sinceBuild = 0                # uses since the last rebuild, for the reuse estimate
        self.builds = 0
        self.uses = 0

    def stale(self, packing):
        """Whether any vertex has moved more than ``skin/2`` since the list was built.

        Half the skin, not the whole of it, because BOTH edges of a pair may be moving toward each
        other -- the gap closes by up to twice the per-vertex displacement."""
        if self._built is None:
            return True
        # THE VERTEX-COUNT CHECK MUST COME FIRST. It was written after the subtraction, where it can
        # never be reached: numpy raises on the mismatched broadcast before the guard is consulted.
        # Any operation that changes the vertex count -- halveNumEdges, doubleNumEdges -- then crashes
        # here instead of triggering the rebuild the guard was put there to trigger.
        if self._built.shape[0] != packing.numVertices:
            return True
        moved = packing.positions.reshape(-1, 2) - self._built
        return float(np.hypot(moved[:, 0], moved[:, 1]).max()) > 0.5 * self.skin

    def candidates(self, packing):
        """``(edgeI, edgeJ)`` candidate edge-index arrays, rebuilding first if the cache has gone
        stale."""
        self.uses += 1
        self._sinceBuild += 1
        if self.stale(packing):
            self.rebuild(packing)
        return self._pairs

    def rebuild(self, packing):
        """Recompute the candidate list from scratch and mark the positions it was built at.

        Retunes the skin first, from what the run that just ended actually did. Runs on the GPU when
        one is available -- the device pass applies the SAME two levels, verified to produce identical
        candidate sets across 15 configurations including walls and skins out to 0.2."""
        if self.adaptive:
            self._retune(packing)
        self._pairs = self._broadPhase(packing)
        self._built = packing.positions.reshape(-1, 2).copy()
        self._sinceBuild = 0
        self.builds += 1
        return self._pairs

    def _broadPhase(self, packing):
        """Candidate pairs from the GPU when one is present, from numpy otherwise.

        The fallback is silent, unlike the energy tiers' loud one, because there is nothing to warn
        about: both paths compute the same set, and the CPU pass is not the bottleneck it was before
        the list existed. If the device call fails it raises rather than degrading, since a partial
        candidate list is the one failure the whole design refuses to tolerate."""
        if self.useCuda:
            try:
                import cudaOverlap
            except ImportError:
                cudaOverlap = None
            if cudaOverlap is not None and cudaOverlap.isAvailable():
                return cudaOverlap.neighborPairsCuda(packing, skin = self.skin)
        return candidateEdgePairs(packing, skin = self.skin)

    def _retune(self, packing):
        """Set the skin from the displacement rate the last interval measured.

        The list survives until some vertex moves ``skin/2``, so with a per-step displacement ``d`` it
        buys ``skin / (2 d)`` uses. Inverting for a target,

            skin = 2 * targetReuse * d

        and ``d`` is read off the interval that just ended -- the largest displacement since the last
        build, divided by the number of uses it lasted. No scale is assumed.

        BOUNDED both ways, because this feeds back on itself. A run that momentarily barely moves would
        otherwise drive the skin toward zero and rebuild every step forever; one that lurches would
        inflate it until every pair is a candidate. The bounds are geometric -- fractions of the polygon
        size -- which is the one place a geometric scale genuinely belongs, since it is the pair COST
        that the ceiling is protecting against. Correctness never depends on the value: any skin is
        safe, because the staleness test is what guarantees no crossing is missed."""
        if self._built is None or self._sinceBuild < 1:
            return
        # Count first, exactly as in ``stale``: this guard was also written after the subtraction that
        # needs it, so a changed vertex count raised on the broadcast instead of skipping the retune.
        # There is nothing to measure across a topology change anyway -- the displacement of a vertex
        # that no longer exists is not a rate.
        if self._built.shape[0] != packing.numVertices:
            return
        moved = packing.positions.reshape(-1, 2) - self._built
        travelled = float(np.hypot(moved[:, 0], moved[:, 1]).max())
        if travelled <= 0.0:
            return
        perUse = travelled / float(self._sinceBuild)
        self.skin = float(np.clip(2.0 * self.targetReuse * perUse, self.floor, self.ceiling))

    def invalidate(self):
        """Force a rebuild on the next use. Call after any move the staleness test cannot see -- a
        packing rebuilt from different geometry, or a vertex count change."""
        self._built = None
        self._pairs = None
        return self

    def statistics(self):
        """``{'skin', 'builds', 'uses', 'reuse', 'pairs'}`` -- ``reuse`` is the mean number of uses per
        rebuild, which is the number that says whether the skin is sized well."""
        pairs = 0 if self._pairs is None else int(self._pairs[0].size)
        return dict(skin = self.skin, builds = self.builds, uses = self.uses,
                    reuse = (self.uses / self.builds) if self.builds else 0.0, pairs = pairs,
                    adaptive = self.adaptive, targetReuse = self.targetReuse,
                    useCuda = self.useCuda)


# UNVERIFIED(Cam)
def candidateEdgePairs(packing, skin = 0.0):
    """``(edgeI, edgeJ)`` global edge indices on DIFFERENT polygons whose bounding balls overlap.

    Two levels, coarse first. The polygon test removes whole pairs at O(N^2) with a tiny vectorized
    constant; the edge test then runs only inside the survivors, which is where the O(n^2) factor that
    dominates for 16- and 32-gons actually gets cut.

    Periodicity uses ONE shift per polygon pair, taken from the polygons' first vertices, matching what
    ``updateIntersections`` and the CUDA kernel both do -- the single-image assumption the model is
    built on. Applying it at the polygon level keeps the edge test in a single consistent frame."""
    r = packing.positions.reshape(-1, 2)
    starts = np.asarray(packing.startIndices, dtype = int)
    box = packing.box
    numPoly = int(packing.numPolygons)

    centroids, radii = polygonBalls(packing)
    mid, half = edgeBalls(packing)

    edgeI = []
    edgeJ = []
    for A in range(numPoly):
        a0, a1 = int(starts[A]), int(starts[A + 1])
        for B in range(A + 1, numPoly):
            b0, b1 = int(starts[B]), int(starts[B + 1])
            shift = minImageShift(r[b0] - r[a0], box)
            delta = centroids[B] + shift - centroids[A]
            if float(np.hypot(*delta)) >= radii[A] + radii[B] + skin:
                continue
            # Surviving polygon pair: test its edge balls, vectorized over the (nA, nB) block.
            mA, hA = mid[a0:a1], half[a0:a1]
            mB, hB = mid[b0:b1] + shift, half[b0:b1]
            separation = np.hypot(*(mA[:, None, :] - mB[None, :, :]).T).T
            close = separation < (hA[:, None] + hB[None, :] + skin)
            if not close.any():
                continue
            localA, localB = np.nonzero(close)
            edgeI.append(localA + a0)
            edgeJ.append(localB + b0)

    if not edgeI:
        return np.zeros(0, dtype = int), np.zeros(0, dtype = int)
    return np.concatenate(edgeI), np.concatenate(edgeJ)
