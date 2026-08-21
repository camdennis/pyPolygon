"""Checks on drawing the EDGE targets from a distribution instead of making them all equal.

The point of the build is to turn the edge-length spread into a degree of freedom an anneal can
actually withdraw. Equilateral polygons have no within-polygon spread at all, so the pooled edge CV is
inherited entirely from the size distribution, sits exactly on the floor that fixed areas impose, and a
ramp aimed at zero is asking for a state the build already occupies.

  0  the draw holds the SHAPE INDEX -- every polygon comes out at kappa, not merely on average
  1  the draw holds the SIZE -- perimeters and areas are untouched, so only the shape changed
  2  the spread is really WITHIN polygons, and the variance decomposition is exact
  3  the realized geometry follows the targets, not just the targets themselves
  4  an infeasible spread is REFUSED, at a threshold that agrees with an independent construction
  5  the spread is above the reachable floor, i.e. there is something for a ramp to take back
  6  ``cyclicArea`` agrees with closed forms it cannot have got from the same algebra
  7  ragged vertex counts are handled -- never assume uniform n

Check 6 is the one that keeps 4 honest: the feasibility bound is only worth having if the maximum area
it computes is right, so it is measured against the regular n-gon formula and against Heron, neither of
which shares any code with it. Check 4 then closes the loop by asking the SPRINGS where the limit is
and comparing that to where the bound says it should be.

    python tests/edgePolydispersityCheck.py
"""

# UNVERIFIED(Cam)

import os
import sys
import warnings

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(_ROOT))
sys.path.insert(0, _ROOT)

import pyPolygon as pp
import anneal
import build
from softBody import backboneEdgeLengths, backboneArea

_KAPPA = 4.0


def make(cv, N = 11, n = 32, seed = 42, phi = 0.3):
    model = pp.Model(N = N, n = n, seed = seed)
    model.generatePolygons(phi = phi, kappa = _KAPPA, edgePolydispersity = cv)
    return model


def checkShapeIndex():
    """CHECK 0: every polygon holds kappa exactly, at every spread.

    The whole proposition is "same shape ratio, different edges". A mean that lands on 4.0 would not be
    enough -- the constraint the notebook then imposes is per polygon, so the spread across polygons is
    what matters, and it has to be at the build tolerance rather than merely small."""
    ok = True
    for cv in (0.0, 0.05, 0.1, 0.2, 0.3):
        model = make(cv)
        kappa = model.getShapeIndices()[:model.N]
        worst = float(np.abs(kappa - _KAPPA).max())
        # From the TARGETS as well: the geometry inherits the build's finite tolerance, the targets
        # should be exact by construction since the draw never touches area or perimeter.
        target = float(np.abs(model.packing.targetPerimeter[:model.N]
                              / np.sqrt(model.packing.targetArea[:model.N]) - _KAPPA).max())
        good = worst < 1e-5 and target < 1e-12
        ok = ok and good
        print(f"  cv {cv:.2f}   geometry |kappa - 4| {worst:.2e}   targets {target:.2e}"
              f"   {'ok' if good else 'FAIL'}")
    print(f"  CHECK 0 shape index held {'PASS' if ok else 'FAIL'}")
    return ok


def checkSizeUntouched():
    """CHECK 1: the draw changes shape ONLY -- same perimeters, same areas, same packing fraction.

    Distinct from check 0: a build could hold every kappa and still have resized the polygons, since
    kappa is scale-free. Compared against the cv = 0 build from the same seed, which draws the same
    areas because the edge draw is taken after them."""
    base = make(0.0)
    ok = True
    for cv in (0.1, 0.3):
        model = make(cv)
        perimeter = float(np.abs(model.packing.targetPerimeter / base.packing.targetPerimeter
                                 - 1.0).max())
        area = float(np.abs(model.packing.targetArea / base.packing.targetArea - 1.0).max())
        phi = abs(model.getPackingFraction() / base.getPackingFraction() - 1.0)
        good = perimeter < 1e-14 and area < 1e-14 and phi < 1e-6
        ok = ok and good
        print(f"  cv {cv:.2f}   dPerimeter {perimeter:.2e}   dArea {area:.2e}   dPhi {phi:.2e}"
              f"   {'ok' if good else 'FAIL'}")
    print(f"  CHECK 1 size untouched {'PASS' if ok else 'FAIL'}")
    return ok


def checkDecomposition():
    """CHECK 2: the spread lands WITHIN polygons, and pooled^2 = within^2 + between^2 exactly.

    The identity is what makes the diagnostic worth reading -- it says the pooled CV an anneal
    constrains is exactly the two parts in quadrature, so a within of zero means the pooled number is
    pinned at the size spread and cannot move. Checked here against a spread built by the OTHER handle
    (``setLogNormalScale``, which resizes whole polygons), which must land entirely in ``between``."""
    ok = True
    # 1e-6, not 0: these are the realized lengths, so they carry the build's finite relax tolerance.
    model = make(0.0)
    split = model.getEdgePolydispersity()
    ok = ok and split["within"] < 1e-6 and split["between"] < 1e-6
    print(f"  equilateral, monodisperse: within {split['within']:.2e}  between "
          f"{split['between']:.2e}")

    sized = make(0.0)
    sized.setLogNormalScale(polydispersity = 0.25)
    split = sized.getEdgePolydispersity()
    good = split["within"] < 1e-6 and split["between"] > 0.05
    ok = ok and good
    print(f"  sizes spread only:         within {split['within']:.2e}  between "
          f"{split['between']:.4f}   {'ok' if good else 'FAIL'}")

    both = make(0.2)
    both.setLogNormalScale(polydispersity = 0.25)
    split = both.getEdgePolydispersity()
    quadrature = float(np.hypot(split["within"], split["between"]))
    identity = abs(quadrature - split["pooled"])
    good = split["within"] > 0.15 and split["between"] > 0.05 and identity < 1e-12
    ok = ok and good
    print(f"  both:                      within {split['within']:.4f}  between "
          f"{split['between']:.4f}  pooled {split['pooled']:.4f}   "
          f"|quadrature - pooled| {identity:.2e}   {'ok' if good else 'FAIL'}")
    print(f"  CHECK 2 variance decomposition {'PASS' if ok else 'FAIL'}")
    return ok


def checkRealized():
    """CHECK 3: the GEOMETRY is non-equilateral, not merely the targets.

    A target set nothing reaches would be a spread on paper only. Measured on the realized edge lengths
    straight out of the build, against the realized target CV."""
    ok = True
    for cv in (0.1, 0.3):
        model = make(cv)
        starts = np.asarray(model.packing.startIndices, dtype = int)
        lengths = backboneEdgeLengths(model.packing)[:starts[model.N]]
        targets = np.asarray(model.packing.targetEdgeLength, dtype = float)[:starts[model.N]]
        realized = float(np.std(lengths) / np.mean(lengths))
        wanted = float(np.std(targets) / np.mean(targets))
        good = abs(realized / wanted - 1.0) < 0.01
        ok = ok and good
        print(f"  cv {cv:.2f}   realized edge CV {realized:.4f}   target CV {wanted:.4f}"
              f"   {'ok' if good else 'FAIL'}")
    print(f"  CHECK 3 geometry follows the targets {'PASS' if ok else 'FAIL'}")
    return ok


def checkRefusal():
    """CHECK 4: too wide a draw is refused, and the bound agrees with what the SPRINGS can do.

    Unequal edges enclose less area than equal ones of the same total, so they raise the floor on the
    shape index; past the point where that floor reaches kappa the targets describe a polygon that does
    not exist. The risk in a bound like this is that it is arithmetically self-consistent and simply
    wrong, so it is compared against an INDEPENDENT witness: relax a single polygon onto edge targets
    of a given spread and read the shape index it can actually reach. Below the bound the springs must
    make kappa; above it they must fall short."""
    n = 32
    edge = np.ones(n)
    print(f"  floor at n = {n}: equal edges {build.minShapeIndex(edge):.4f} "
          f"(regular {build.regularShapeIndex(n):.4f})")
    ok = abs(build.minShapeIndex(edge) - build.regularShapeIndex(n)) < 1e-12

    # Bisect the bound in the SPREAD, using a fixed draw so the two sides compare like with like.
    draw = build._logNormalDraw(n, 1.0, 7)
    draw = np.log(draw / draw.mean())
    widthOf = lambda t: np.exp(t * draw) / np.exp(t * draw).mean()

    def feasible(t):
        """Wide enough and the edges cannot close at all, which is infeasible a fortiori."""
        try:
            return build.minShapeIndex(widthOf(t)) < _KAPPA
        except ValueError:
            return False

    lo, hi = 0.0, 1.0
    while feasible(hi):
        hi *= 2.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if feasible(mid):
            lo = mid
        else:
            hi = mid
    critical = 0.5 * (lo + hi)
    print(f"  this draw's edges support kappa = 4 up to a CV of "
          f"{float(np.std(widthOf(critical))):.4f}")

    for scale, expect in ((0.85, True), (1.15, False)):
        edges = widthOf(scale * critical)
        packing = build.buildEquilateralPacking(1, n, _KAPPA, areaKind = "mono", phi = 0.05, rng = 3)
        packing.targetEdgeLength = edges * float(packing.targetPerimeter[0]) / n
        packing.syncTargetPerimeter()
        build.shapeBackbones(packing, maxSteps = 400000, fThreshold = 1e-12)
        reached = float(build.shapeIndices(packing)[0])
        made = abs(reached - _KAPPA) < 1e-3
        ok = ok and made == expect
        print(f"  spread {scale:.2f}x the bound: springs reach kappa {reached:.4f}   "
              f"{'made it' if made else 'fell short'}   {'ok' if made == expect else 'FAIL'}")

    # And the builder refuses rather than approximating. The bound is PERMISSIVE -- a 32-gon at
    # kappa = 4 carries an edge CV near 2 before its edges stop enclosing enough area -- so the
    # refusal is a backstop and not a limit any sane spread will meet.
    def builds(cv):
        try:
            make(cv, N = 4, n = 8)
            return True
        except ValueError as error:
            return "polydispersity" not in str(error)

    lo, hi = 0.0, 1.0
    while builds(hi):
        hi *= 2.0
    for _ in range(20):
        mid = 0.5 * (lo + hi)
        if builds(mid):
            lo = mid
        else:
            hi = mid
    ok = ok and lo > 0.5 and not builds(hi)
    print(f"  builder accepts up to a requested cv of {lo:.3f} at n = 8 and refuses above it")
    print(f"  CHECK 4 infeasible spread refused {'PASS' if ok else 'FAIL'}")
    return ok


def checkAboveFloor():
    """CHECK 5: the spread leaves the anneal something to withdraw.

    The reason for the whole build. With the areas held rigid the edge CV cannot fall below the CV of
    ``sqrt(A0)`` -- that is ``anneal._reachableWidth``, and an equilateral build sits ON it, so
    ``setTargetPolydispersity`` clamps every request and the ramp is a no-op. The drawn build has to
    start strictly above the floor, and the distance above it is the usable range."""
    ok = True
    for cv in (0.0, 0.2):
        model = make(cv)
        model.setLogNormalScale(polydispersity = 0.25)
        model.setConstraints(area = True, perimeter = True, edge = [1, 2])
        pooled = model.getPolydispersity().get("edge", 0.0)
        floor = anneal._reachableWidth(model)
        headroom = pooled / floor - 1.0
        good = (headroom < 1e-3) if cv == 0.0 else (headroom > 0.5)
        ok = ok and good
        print(f"  cv {cv:.2f}   pooled {pooled:.4f}   floor {floor:.4f}   "
              f"headroom {100 * headroom:+7.2f}%   {'ok' if good else 'FAIL'}")

    # And the ramp must actually move it: aim at the floor and confirm the width follows.
    model = make(0.2)
    model.setLogNormalScale(polydispersity = 0.25)
    model.setConstraints(area = True, perimeter = True, edge = [1, 2])
    before = model.getEdgePolydispersity()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.setTargetPolydispersity(anneal._reachableWidth(model))
    after = model.getEdgePolydispersity()
    good = after["within"] < 0.5 * before["within"]
    ok = ok and good
    print(f"  ramp to the floor: within {before['within']:.4f} -> {after['within']:.4f}   "
          f"{'ok' if good else 'FAIL'}")
    print(f"  CHECK 5 room above the floor {'PASS' if ok else 'FAIL'}")
    return ok


def checkCyclicArea():
    """CHECK 6: the maximum-area bound, against closed forms that share none of its code.

    ``cyclicArea`` is the whole basis of the feasibility refusal, so it is measured rather than
    reasoned about. Two independent witnesses: the regular n-gon's area (equal edges) and Heron's
    formula (every triangle is cyclic, so the bound is exact there, including the obtuse case where the
    circumcenter falls OUTSIDE and the longest edge subtends the reflex arc -- a separate branch)."""
    ok = True
    worst = 0.0
    for n in (3, 4, 5, 8, 16, 32, 64):
        a = np.full(n, 0.37)
        exact = n / 4.0 * 0.37 ** 2 / np.tan(np.pi / n)
        worst = max(worst, abs(build.cyclicArea(a) / exact - 1.0))
    ok = ok and worst < 1e-14
    print(f"  regular n-gons, n = 3..64: worst relative error {worst:.2e}")

    a = np.array([10.0, 6.0, 6.0])
    s = a.sum() / 2.0
    heron = float(np.sqrt(s * np.prod(s - a)))
    obtuse = abs(build.cyclicArea(a) / heron - 1.0)
    ok = ok and obtuse < 1e-12
    print(f"  obtuse triangle (circumcenter outside): relative error {obtuse:.2e}")

    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(2000):
        a = rng.random(3) + 0.1
        if a.max() >= a.sum() - a.max():
            continue
        s = a.sum() / 2.0
        worst = max(worst, abs(build.cyclicArea(a) / np.sqrt(s * np.prod(s - a)) - 1.0))
    ok = ok and worst < 1e-10
    print(f"  2000 random triangles vs Heron: worst relative error {worst:.2e}")

    # A closed polygon that cannot exist must say so rather than returning a number.
    try:
        build.cyclicArea(np.array([1.0, 0.2, 0.2]))
        refused = False
    except ValueError:
        refused = True
    ok = ok and refused
    print(f"  impossible edge set refused: {refused}")
    print(f"  CHECK 6 cyclic area {'PASS' if ok else 'FAIL'}")
    return ok


def checkRagged():
    """CHECK 7: mixed vertex counts, since nothing may assume uniform n.

    ``getEdgePolydispersity`` weights the between-polygon term by vertex count for exactly this case.
    Built by hand rather than through the builder, which is uniform by construction, and verified
    against a direct per-polygon computation."""
    rng = np.random.default_rng(5)
    counts = [4, 7, 13, 5]
    blocks, starts = [], [0]
    for k, count in enumerate(counts):
        radius = 0.05 * (1.0 + k)
        angle = np.arange(count) * 2.0 * np.pi / count
        blocks.append(np.stack([radius * np.cos(angle), radius * np.sin(angle)], axis = 1)
                      + rng.random(2))
        starts.append(starts[-1] + count)
    from packing import Packing
    from enums import EnergyType
    positions = np.vstack(blocks).reshape(-1)
    edges = rng.lognormal(0.0, 0.3, size = starts[-1]) * 0.1
    packing = Packing(positions, starts, box = None, energyType = EnergyType.eqSoftBody,
                      targetEdgeLength = edges,
                      targetArea = np.full(len(counts), 0.01))
    packing.syncTargetPerimeter()

    model = pp.Model.__new__(pp.Model)
    model.packing = packing
    split = model.getEdgePolydispersity()

    # Against the GEOMETRY, which is what getEdgePolydispersity reads.
    edges = backboneEdgeLengths(packing)
    mean = edges.mean()
    perPolygon = [edges[a : b] for a, b in zip(starts[:-1], starts[1:])]
    within = np.sqrt(sum(len(e) * np.var(e) for e in perPolygon) / len(edges)) / mean
    between = np.sqrt(sum(len(e) * (e.mean() - mean) ** 2 for e in perPolygon) / len(edges)) / mean
    ok = (abs(split["within"] - within) < 1e-14 and abs(split["between"] - between) < 1e-14
          and abs(np.hypot(within, between) - split["pooled"]) < 1e-14)
    print(f"  counts {counts}: within {split['within']:.6f} (direct {within:.6f})   "
          f"between {split['between']:.6f} (direct {between:.6f})")
    print(f"  CHECK 7 ragged vertex counts {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    print("edge polydispersity", flush = True)
    results = []
    for name, check in (("shape index held", checkShapeIndex),
                        ("size untouched", checkSizeUntouched),
                        ("variance decomposition", checkDecomposition),
                        ("geometry follows targets", checkRealized),
                        ("infeasible spread refused", checkRefusal),
                        ("room above the floor", checkAboveFloor),
                        ("cyclic area", checkCyclicArea),
                        ("ragged vertex counts", checkRagged)):
        print(f"\n{name}:", flush = True)
        results.append(check())
    print(f"\n{sum(results)}/{len(results)} checks passed")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
