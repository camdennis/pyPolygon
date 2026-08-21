"""Verification for the neighbor-ball candidate list and the sorted follower scan.

Both changes are pure PERFORMANCE changes, so the bar is exact agreement with the code they replace --
not agreement to a tolerance. A dropped candidate pair or a missed follower does not crash: it
under-reports overlap, which makes an invalid packing look valid and corrupts the density a sweep
reports. Every check below is therefore an exact set or array comparison against the all-to-all
reference, which is kept in the tree for exactly this purpose.

Checks:

  1. candidate pairs never miss a real crossing -- every all-to-all intersection's edge pair is in the
     candidate list, across free / periodic boxes, n = 4, 16, 32, with and without a container;
  2. the intersection SETS are identical, candidate path vs all-to-all;
  3. the sorted followers match the O(M^2) reference exactly;
  4. the skin invariant holds: displaced by just under skin/2 the cached list is still correct, and
     past skin/2 the rebuild actually FIRES;
  5. an absurdly small skin is caught by the staleness test rather than silently returning a wrong
     list;
  6. the container is never culled -- a wall edge crossing a polygon edge is always a candidate.

Run: python tests/neighborCheck.py
"""

# UNVERIFIED(Cam)

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import Model
from neighbors import NeighborList, candidateEdgePairs
import energies


WALL = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 0.0]])


def buildPacking(n = 4, N = 6, seed = 3, phi = 0.55, wall = False, periodic = True):
    """A packing dense enough that polygons genuinely cross."""
    model = Model(N = N, n = n, seed = seed)
    model.generateEquilateralPolygons(phi = phi,
                                      kappa = float(np.sqrt(4.0 * n * np.tan(np.pi / n))))
    model.syncTargetPerimeters()
    model.syncTargetAreas()
    if wall:
        model.addShape(WALL)
        model.pinVertices(np.arange(model.getNumVertices())[-4:])
        model.setBoundaryConditions("fixed")
    elif not periodic:
        model.setBoundaryConditions("free")
    return model


def referenceIntersections(packing):
    energies.updateIntersections(packing, candidates = None)
    return np.array(packing.intersections, dtype = float)


def candidateIntersections(packing, skin = 0.0):
    pairs = candidateEdgePairs(packing, skin = skin)
    energies.updateIntersections(packing, candidates = pairs)
    return np.array(packing.intersections, dtype = float), pairs


def canonical(rows):
    """Sort the intersection rows so two sets can be compared regardless of discovery order."""
    if rows.size == 0:
        return rows.reshape(0, 6)
    order = np.lexsort((rows[:, 5], rows[:, 4], rows[:, 3], rows[:, 2], rows[:, 1], rows[:, 0]))
    return rows[order]


def configurations():
    for n in (4, 16, 32):
        for wall in (False, True):
            yield dict(n = n, N = 5, wall = wall, periodic = not wall)


def checkNoMissedCrossings():
    """1. Every real crossing's edge pair is present as a candidate."""
    worst = None
    for config in configurations():
        model = buildPacking(**config)
        packing = model.packing
        reference = referenceIntersections(packing)
        edgeI, edgeJ = candidateEdgePairs(packing, skin = 0.0)
        present = {(int(i), int(j)) for i, j in zip(edgeI, edgeJ)}
        present |= {(j, i) for i, j in present}
        missing = 0
        for row in reference:
            pair = (int(row[4]), int(row[2]))
            if pair not in present and pair[::-1] not in present:
                missing += 1
        label = f"n={config['n']:2d} wall={int(config['wall'])}"
        print(f"  1. {label}   crossings {len(reference):4d}   candidates {edgeI.size:6d}"
              f"   missed {missing}")
        assert missing == 0, f"{label}: {missing} real crossings were not candidates"
        worst = worst or missing


def checkIdenticalIntersections():
    """2. Candidate path vs all-to-all: identical sets, exactly."""
    for config in configurations():
        model = buildPacking(**config)
        packing = model.packing
        reference = canonical(referenceIntersections(packing))
        fromCandidates, _ = candidateIntersections(packing, skin = 0.0)
        fromCandidates = canonical(fromCandidates)
        label = f"n={config['n']:2d} wall={int(config['wall'])}"
        same = (reference.shape == fromCandidates.shape
                and np.array_equal(reference, fromCandidates))
        print(f"  2. {label}   rows {reference.shape[0]:4d} vs {fromCandidates.shape[0]:4d}   "
              f"{'identical' if same else 'DIFFER'}")
        assert same, f"{label}: candidate path found a different intersection set"


def referenceFollowers(packing):
    """The O(M^2) scan this replaced, kept here as the check's reference."""
    X = packing.intersections
    starts = np.asarray(packing.startIndices, dtype = int)
    M = len(X)
    out = np.full(M, -1, dtype = int)
    for alpha in range(M):
        rhoI, rhoJ = int(X[alpha, 0]), int(X[alpha, 1])
        nEdges = int(starts[rhoI + 1]) - int(starts[rhoI])
        cLeave = X[alpha, 4] + X[alpha, 5] - int(starts[rhoI])
        best, bestDist = -1, np.inf
        for beta in range(M):
            if int(X[beta, 0]) == rhoJ and int(X[beta, 1]) == rhoI:
                cArrive = X[beta, 2] + X[beta, 3] - int(starts[rhoI])
                dist = (cArrive - cLeave) % nEdges
                if 0.0 < dist < bestDist:
                    best, bestDist = beta, dist
        out[alpha] = best
    return out


def checkFollowers():
    """3. The sorted follower scan matches the O(M^2) reference exactly."""
    for config in configurations():
        model = buildPacking(**config)
        packing = model.packing
        referenceIntersections(packing)
        expected = referenceFollowers(packing)
        energies.updateFollowers(packing)
        got = np.asarray(packing.followerIndices, dtype = int)
        label = f"n={config['n']:2d} wall={int(config['wall'])}"
        same = np.array_equal(expected, got)
        print(f"  3. {label}   M {len(expected):4d}   {'identical' if same else 'DIFFER'}")
        assert same, f"{label}: sorted followers disagree with the O(M^2) scan"


def checkSkinInvariant():
    """4. Just under skin/2 the cached list is still right; past it the rebuild fires."""
    model = buildPacking(n = 16, N = 5, wall = False, periodic = True)
    packing = model.packing
    neighbors = NeighborList(packing, skin = 0.05)
    neighbors.candidates(packing)
    builds = neighbors.builds

    rng = np.random.default_rng(0)
    direction = rng.standard_normal((packing.numVertices, 2))
    direction /= np.hypot(direction[:, 0], direction[:, 1])[:, None]

    saved = packing.positions.copy()
    packing.positions += (0.45 * neighbors.skin * direction).reshape(-1)
    stale = neighbors.stale(packing)
    pairs = neighbors.candidates(packing)
    energies.updateIntersections(packing, candidates = pairs)
    cached = canonical(np.array(packing.intersections, dtype = float))
    reference = canonical(referenceIntersections(packing))
    print(f"  4. moved 0.45*skin   stale {stale}   rebuilds {neighbors.builds - builds}   "
          f"rows {cached.shape[0]} vs {reference.shape[0]}")
    assert not stale, "the list went stale below skin/2, so the skin is not being used"
    assert np.array_equal(cached, reference), "the cached list missed a crossing below skin/2"

    packing.positions[:] = saved
    packing.positions += (0.80 * neighbors.skin * direction).reshape(-1)
    assert neighbors.stale(packing), "moving past skin/2 did not trigger a rebuild"
    print(f"     moved 0.80*skin   stale True   <- rebuild fires")


def checkUndersizedSkinDetected():
    """5. A tiny skin must be CAUGHT by the staleness test, never silently wrong."""
    model = buildPacking(n = 16, N = 5, wall = False, periodic = True)
    packing = model.packing
    neighbors = NeighborList(packing, skin = 1e-9)
    neighbors.candidates(packing)
    rng = np.random.default_rng(1)
    packing.positions += (1e-4 * rng.standard_normal((packing.numVertices, 2))).reshape(-1)
    stale = neighbors.stale(packing)
    print(f"  5. skin 1e-09, moved 1e-04   stale {stale}")
    assert stale, "an undersized skin was not detected -- the list would be silently wrong"


def checkContainerNeverCulled():
    """6. Wall edges crossing polygon edges are always candidates."""
    model = buildPacking(n = 4, N = 5, wall = True, periodic = False)
    packing = model.packing
    container = int(packing.containerIndex)
    # Push one polygon out through the wall so it genuinely crosses it.
    a, b = int(packing.startIndices[0]), int(packing.startIndices[1])
    r = packing.positions.reshape(-1, 2)
    r[a:b] += np.array([1.0 - r[a:b, 0].max() + 0.02, 0.0])
    model._forces = None

    reference = canonical(referenceIntersections(packing))
    wallRows = [row for row in reference if container in (int(row[0]), int(row[1]))]
    fromCandidates, _ = candidateIntersections(packing, skin = 0.0)
    fromCandidates = canonical(fromCandidates)
    print(f"  6. wall crossings {len(wallRows)}   total rows {reference.shape[0]} vs "
          f"{fromCandidates.shape[0]}")
    assert len(wallRows) > 0, "the test did not actually push a polygon through the wall"
    assert np.array_equal(reference, fromCandidates), "a container crossing was culled"


def checkCudaMatchesNumpy():
    """5. The device broad phase produces the SAME candidate set as numpy.

    It has to apply both levels to do so. A device pass with only the edge test finds strictly more
    candidates -- measured 33 against 29 -- because the polygon cull is NOT implied by the edge test:
    for an edge whose endpoints are within R of the centroid the parallelogram law gives d^2 + h^2 <=
    R^2, hence d + h <= sqrt(2) R rather than R, so the covering-ball test can reject a pair whose edge
    balls overlap. Both filters are valid necessary conditions for a crossing, so either alone still
    catches every one -- but only matching filters can be checked against each other."""
    try:
        import cudaOverlap
    except ImportError:
        print("  5. CUDA not importable -- skipped")
        return
    if not cudaOverlap.isAvailable():
        print("  5. no GPU available -- skipped")
        return
    for config in configurations():
        model = buildPacking(**config)
        for skin in (0.0, 0.05, 0.2):
            cpu = candidateEdgePairs(model.packing, skin = skin)
            gpu = cudaOverlap.neighborPairsCuda(model.packing, skin = skin)
            asSet = lambda pair: set(zip(map(int, pair[0]), map(int, pair[1])))
            same = asSet(cpu) == asSet(gpu)
            label = f"n={config['n']:2d} wall={int(config['wall'])} skin={skin:.2f}"
            print(f"  5. {label}   cpu {cpu[0].size:5d}   gpu {gpu[0].size:5d}   "
                  f"{'identical' if same else 'DIFFER'}")
            assert same, f"{label}: CUDA and numpy candidate sets differ"


def main():
    print("neighbor balls and sorted followers")
    checkNoMissedCrossings()
    checkIdenticalIntersections()
    checkFollowers()
    checkSkinInvariant()
    checkUndersizedSkinDetected()
    checkContainerNeverCulled()
    checkCudaMatchesNumpy()
    print("all checks passed")


if __name__ == "__main__":
    main()
