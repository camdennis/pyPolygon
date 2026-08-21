"""Checks on the excess-energy density controller: is the control parameter what it claims to be?

``holdExcessEnergy`` replaces "start at this density" with "start this far above jamming". Three things
have to be true for that to be a real improvement rather than a renaming:

  0  the density move is AFFINE and the container never moves
  1  the number is DIMENSIONLESS -- the same value means the same geometry at any size and stiffness
  2  the energy it reads is the CONTACT term, not the total (which carries live shape springs)
  3  the controller is TWO-SIDED -- it converges onto the target from either direction
  4  it works without a container, where the polygons resize instead of the box
  5  end to end, a sweep started BELOW jamming now finds the packing instead of reporting its input
  6  the load at a held state sits between BODIES rather than in the wall

Check 5 is the only one measured against something outside this code: six unit squares, whose optimal
container side is exactly 3.0. Check 6 is the one that would have caught the original failure in a
single line -- the excess was being satisfied entirely by the packing extruding through its container
while nothing inside touched at all.

    python tests/excessEnergyCheck.py
"""

# UNVERIFIED(Cam)

import os
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(_ROOT))
sys.path.insert(0, _ROOT)

import pyPolygon as pp
import anneal
from softBody import eqSoftBodyEnergyForce


def build(N = 6, n = 8, seed = 42, side = 3.75, container = True, wallStiffness = 100.0):
    """A relaxed packing on the depth tier, deliberately LOOSE so the controller has to compress.

    ``wallStiffness`` defaults to 100 rather than the library's 1.0 so these checks run in the regime a
    confined packing is actually usable in. At 1.0 the wall is softer than a body contact and the
    packing relieves stress by extruding through it -- measured, 100% of the contact energy in wall
    penetration with a pair overlap of exactly zero -- which would make every check below a statement
    about a leak rather than about a packing."""
    model = pp.Model(N = N, n = n, seed = seed)
    model.generateEquilateralPolygons(phi = N / side ** 2, kappa = 4.0)
    model.syncTargetPerimeters()
    model.syncTargetAreas()
    if container:
        model.addShape(np.array([[0, 0], [0, 1], [1, 1], [1, 0]]))
        model.pinVertices(np.arange(model.getNumVertices())[-4:])
        model.setBoundaryConditions("fixed")
    model.setConstraints(area = True, edge = True)
    model.minimizeFIRE(maxUnbalancedForce = 1e-3, maxSteps = 4000)
    model.setDepthContact(stiffness = 1.0, wallStiffness = wallStiffness)
    model.minimizeLBFGS(maxUnbalancedForce = 1e-6, maxSteps = 2000)
    return model


def rescaleEverything(model, factor):
    """Scale the WHOLE system -- geometry, box and every length target -- about the origin.

    A pure change of units, so nothing physical moves. Every dimensionless quantity must come back
    unchanged, which is what makes this a test of the dimensional analysis in ``anneal.energyScale``
    rather than of the packing."""
    packing = model.packing
    packing.positions *= factor
    packing.targetArea *= factor ** 2
    packing.targetEdgeLength *= factor
    if packing.targetDiagonal is not None:
        packing.targetDiagonal *= factor
    packing.syncTargetPerimeter()
    model._forces = None
    model._energy = None
    return model


def checkAffine():
    """CHECK 0 -- the density move is AFFINE and leaves the container alone.

    Two properties, both of which the first version of ``_setDensity`` failed by scaling the box:
    the container's area is untouched, and every polygon's centroid stays where it was in the box
    (fractional coordinates fixed) while the polygons themselves scale. That combination IS affine
    compression written in the box's frame -- shrink the box by 1/f about its centre, carry the
    centroids with it, then rescale by f to restore the box.

    The failure it guards against is silent and only bites on the way OUT: walking the wall away from
    a cluster that never expands drops the contact energy to the noise floor while the interior has
    not moved at all, so the controller reads "unjammed" off a packing it never decompressed."""
    model = build()
    stop = int(model.packing.containerIndex)
    starts = np.asarray(model.packing.startIndices, dtype = int)
    r = model.packing.positions.reshape(-1, 2)
    before = np.array([r[starts[i]:starts[i + 1]].mean(axis = 0) for i in range(stop)])
    boxBefore = model.getBoxArea()
    areaBefore = model.getAreas()[:stop].copy()

    factor = 1.21
    anneal._setDensity(model, model.getPackingFraction() * factor)

    r = model.packing.positions.reshape(-1, 2)
    after = np.array([r[starts[i]:starts[i + 1]].mean(axis = 0) for i in range(stop)])
    centroidDrift = float(np.abs(after - before).max())
    boxDrift = abs(model.getBoxArea() / boxBefore - 1.0)
    areaRatio = float(np.abs(model.getAreas()[:stop] / areaBefore - factor).max())
    print(f"  box area drift {boxDrift:.2e}   centroid drift {centroidDrift:.2e}   "
          f"polygon area ratio off by {areaRatio:.2e} (expected exactly {factor})")
    ok = boxDrift < 1e-15 and centroidDrift < 1e-12 and areaRatio < 1e-12
    print(f"  CHECK 0 affine, box fixed    {'PASS' if ok else 'FAIL'}")
    return ok


def checkDimensionless(excess = 1e-6):
    """CHECK 1 -- a change of units leaves the excess alone, and the raw energy does not.

    Measured from a genuinely LOADED state rather than a nearly-touching one. At the noise floor the
    contact energy is a sum of near-cancelling terms at depths around 1e-04 of an edge, and cubing that
    leaves only a few digits: the same test run from a 2.3e-11 state reproduced the expected factor of
    81 to 3.7e-08 rather than to roundoff, which says nothing about the scaling and everything about
    where it was sampled."""
    model = build()
    model.holdExcessEnergy(excess, maxUnbalancedForce = 1e-6)
    before, rawBefore = model.getExcessEnergy(), model.getContactEnergy()
    # A POWER OF TWO, so the change of units is EXACT and the test can demand exact invariance.
    # Multiplying by 4 only moves a binary exponent, so every intermediate reproduces bit for bit and
    # the excess must come back unchanged to the last bit -- not merely close.
    factor = 4.0
    rescaleEverything(model, factor)
    after, rawAfter = model.getExcessEnergy(), model.getContactEnergy()
    # The raw energy carries k L^4 on this tier, so it must move by exactly factor^4. If it does not,
    # the scale in energyScale is the wrong power and the excess is only accidentally invariant.
    rawRatio = rawAfter / rawBefore if rawBefore else float("nan")
    drift = abs(after / before - 1.0) if before else float("nan")
    print(f"  excess {before:.6e} -> {after:.6e}   relative drift {drift:.2e}")
    print(f"  raw    {rawBefore:.6e} -> {rawAfter:.6e}   ratio off by "
          f"{abs(rawRatio / factor ** 4 - 1.0):.2e}  (expected exactly {factor ** 4:.0f})")
    # EXACT, and it took a measurement to learn that it could be. This check first ran with factor 3.0
    # and drifted anywhere from 5e-13 to 1e-07 between runs, which I put down to absolute tolerances in
    # the geometric predicates and gated loosely at 1e-6. Wrong diagnosis. Sweeping the factor against
    # the energy level:
    #
    #   factor 2.00, 4.00   drift EXACTLY 0.00e+00 at every energy level
    #   factor 3.00, 1.10   drift 1e-14 .. 8e-12, implied |dE| only 1e-20 .. 1e-22
    #
    # The discriminator is not the energy at all -- it is whether the factor is a POWER OF TWO. 2 and 4
    # move a binary exponent and nothing else, so the computation reproduces bit for bit; 3 and 1.1
    # perturb every coordinate at the ULP level and the cubic law carries it through. The scatter was
    # my own rescaling arithmetic, not the contact law, and the invariance is exact.
    ok = drift == 0.0 and rawRatio == factor ** 4
    print(f"  CHECK 1 dimensionless        {'PASS' if ok else 'FAIL'}")
    return ok


def checkContactOnly(excess = 1e-6):
    """CHECK 2 -- the contact energy really excludes the springs, and they really are live.

    The second half needs the edge spring turned ON explicitly: ``Model.__init__`` leaves ``kEdge`` at
    0.0 (only ``setSpringConstants`` raises it to 1), so without that call the spring energy is exactly
    zero and the test would be comparing nothing against nothing.

    Run at ``wallStiffness = 1`` on purpose. This is a bookkeeping check -- does the total minus the
    springs equal the contact term -- and it holds identically at any wall stiffness, so paying the
    stiff wall's slow confinement path here buys nothing. It costs plenty: at 100 this single check ran
    for about fifteen minutes against seconds at 1."""
    model = build(wallStiffness = 1.0)
    model.holdExcessEnergy(excess, maxUnbalancedForce = 1e-6)
    # Springs OFF: both shape terms constrained, so the total energy IS the contact energy.
    model.setConstraints(area = True, edge = True)
    model.calcForceEnergy()
    total, contact = model.getEnergy(), model.getContactEnergy()
    rigidGap = abs(total - contact) / max(abs(total), 1e-300)
    # Springs ON: area constrained, edges left to the spring, so the total must EXCEED the contact by
    # exactly the spring energy -- computed here from softBody directly rather than by subtraction.
    #
    # The spring is STRETCHED DELIBERATELY, by moving the edge targets off the geometry. Relying on the
    # packing to be loaded is what broke the earlier version of this check: at wallStiffness = 1 the
    # packing is unjammed, every shape sits exactly on its target, and the spring energy came out at
    # 1.1e-29 -- nothing to compare against, and the check would have passed vacuously without its
    # "is the spring live" guard. The identity being tested is bookkeeping and holds at ANY
    # configuration, so no relaxation is wanted here at all; only a state where the term is nonzero.
    model.setSpringConstants(area = 1.0, edge = 1.0)
    model.setConstraints(area = True, perimeter = True, edge = False)
    upTo = int(model.packing.startIndices[int(model.packing.containerIndex)])
    model.packing.targetEdgeLength[:upTo] *= 1.05
    model.calcForceEnergy()
    total, contact = model.getEnergy(), model.getContactEnergy()
    spring, _ = eqSoftBodyEnergyForce(model.packing, model.kEdge, 0.0, relative = True)
    softGap = abs((total - contact) - spring) / max(abs(spring), 1e-300)
    print(f"  constrained: |total - contact| / total = {rigidGap:.2e}")
    print(f"  sprung:      spring energy {spring:.6e}, total - contact {total - contact:.6e}, "
          f"mismatch {softGap:.2e}")
    # The second half is only meaningful if the spring is actually carrying something; a zero spring
    # would make the test pass for the wrong reason.
    ok = rigidGap < 1e-12 and softGap < 1e-10 and spring > 1e-12 * abs(contact)
    print(f"  CHECK 2 contact term only    {'PASS' if ok else 'FAIL'}")
    return ok


def checkTwoSided(excess = 1e-6, tolerance = 0.05):
    """CHECK 3 -- the controller converges onto the target from BOTH directions.

    What is asserted is convergence, not agreement. The two runs are NOT expected to land at the same
    density and it would be wrong to demand it: the excess is a function of the CONFIGURATION, and
    compressing a loose packing arrives at a different arrangement from loading one and letting it back
    out. Measured here, both reach the requested energy while sitting 137% apart in phi (0.4067 against
    0.9626), the DECOMPRESSING run the denser.

    Do not read a rule into that sign. An earlier reading of this same check gave 4.3% the other way,
    and the difference was the setup, not the landscape -- it predated the excess counting body contact
    only and predated the wall being stiffenable, so the "compression is denser" figure quoted from it
    does not stand. The spread is the useful output and it is REPORTED, never tested: a threshold on it
    would assert a property of the landscape rather than of the controller, and the two readings above
    are exactly why that would be a mistake."""
    fromBelow = build()
    gotBelow, phiBelow = fromBelow.holdExcessEnergy(excess, tolerance = tolerance, maxUnbalancedForce = 1e-6)

    fromAbove = build()
    # Load it three decades past the target first, so this genuinely arrives from the other side. The
    # controller sets its own starting point: naming a DENSITY instead is how the first version of this
    # test broke itself -- setPackingFraction(x3.3) grows every polygon 1.8x linearly about its own
    # centroid, which carries the system past the contact law's dMax/rIn << 1 limit, where the
    # repulsion reverses sign and the energy reads SMALL at phi = 1.478. Naming an ENERGY cannot leave
    # the regime the law is valid in, because the energy is what the law reports.
    fromAbove.holdExcessEnergy(excess * 1000.0, tolerance = tolerance,
                              maxUnbalancedForce = 1e-6)
    startedAt = fromAbove.getExcessEnergy()
    gotAbove, phiAbove = fromAbove.holdExcessEnergy(excess, tolerance = tolerance, maxUnbalancedForce = 1e-6)

    band = float(np.exp(tolerance))
    inBand = (excess / band <= gotBelow <= excess * band
              and excess / band <= gotAbove <= excess * band)
    spread = abs(phiAbove / phiBelow - 1.0)
    print(f"  from below: excess {gotBelow:.4e}  phi {phiBelow:.6f}")
    print(f"  from above: excess {gotAbove:.4e}  phi {phiAbove:.6f}  (started at {startedAt:.3e})")
    print(f"  requested {excess:.1e} +/- {100 * (band - 1):.0f}%;  both converged: {inBand}")
    print(f"  measured path dependence: {100 * spread:.3f}% in phi at equal energy "
          f"({'compression denser' if phiBelow > phiAbove else 'decompression denser'})")
    # The "from above" leg is only a real second direction if it started far above; otherwise it is
    # the first leg again under another name.
    ok = inBand and startedAt > 10.0 * excess
    print(f"  CHECK 3 two-sided            {'PASS' if ok else 'FAIL'}")
    return ok


def checkPeriodic(excess = 1e-6, tolerance = 0.05):
    """CHECK 4 -- with no container the polygons resize instead of the box, and it still converges."""
    model = build(container = False)
    got, phi = model.holdExcessEnergy(excess, tolerance = tolerance, maxUnbalancedForce = 1e-6)
    band = float(np.exp(tolerance))
    ok = excess / band <= got <= excess * band
    print(f"  periodic: excess {got:.4e}  phi {phi:.6f}")
    print(f"  CHECK 4 no container         {'PASS' if ok else 'FAIL'}")
    return ok


def checkSweepFromBelow(excess = 1e-6):
    """CHECK 5 -- end to end: energySweep started BELOW jamming still finds the packing.

    This is the failure the whole controller exists to remove. Handed a density below jamming, the old
    sweep relaxed once, found nothing overlapping, warned that the packing was "already valid" and
    returned that density -- a lower bound it had been given rather than a result. With
    ``excessEnergy`` the sweep compresses onto the target first, so the anneal has contacts to
    rearrange.

    Six SQUARES, whose optimum container side is exactly 3.0, so the answer is checkable against
    something that is not this code. The rigid pipeline previously measured 3.0035."""
    N = 6
    model = pp.Model(N = N, n = 4, seed = 42)
    # Deliberately loose: side 3.6 against an optimum of 3.0, so nothing is touching at the start.
    model.generateEquilateralPolygons(phi = N / 3.6 ** 2, kappa = 4.0)
    model.syncTargetPerimeters()
    model.syncTargetAreas()
    model.addShape(np.array([[0, 0], [0, 1], [1, 1], [1, 0]]))
    model.pinVertices(np.arange(model.getNumVertices())[-4:])
    model.setBoundaryConditions("fixed")
    model.setConstraints(area = True, edge = True)
    model.minimizeFIRE(maxUnbalancedForce = 1e-3, maxSteps = 4000)
    model.setModelType("depth")
    model.minimizeLBFGS(maxUnbalancedForce = 1e-6, maxSteps = 2000)
    startSide = float(np.sqrt(N / model.getPackingFraction()))
    startExcess = model.getExcessEnergy()

    result = model.energySweep(minimizer = "lbfgs", annealShape = False, finishRigid = False,
                               excessEnergy = excess)
    side = float(np.sqrt(N / result.phi)) if result.packed else float("nan")
    print(f"  started at side {startSide:.4f}, excess {startExcess:.2e} "
          f"({'BELOW jamming' if startExcess < excess else 'above'})")
    print(f"  swept to  side {side:.4f}   (optimum 3.0, rigid pipeline 3.0035)   "
          f"pairOverlap {result.overlap:.2e}")
    # The sweep must end DENSER than it began, which is exactly what it could not do before, and it
    # must not have cheated its way there by leaving overlap behind.
    ok = result.packed and side < startSide and result.overlap == 0.0 and side >= 3.0
    print(f"  CHECK 5 sweep from below     {'PASS' if ok else 'FAIL'}")
    return ok


def checkWallShare(excess = 1e-6):
    """CHECK 6 -- at a held state the energy is between BODIES, not in the wall.

    The check that would have caught the whole failure in one line. The excess counts pair contact only,
    so it can always be satisfied -- but if the packing is meanwhile extruding through its boundary, the
    density it reports describes a leak. Measured at ``wallStiffness = 1``: 100.00% of the contact
    energy in the wall, pair overlap exactly 0.000e+00, 17 vertices outside the box. At 100: 0.00%,
    zero penetration, nothing outside."""
    leaky = build(wallStiffness = 1.0)
    leaky.holdExcessEnergy(excess, maxUnbalancedForce = 1e-6)
    leakyShare = 1.0 - leaky.getPairContactEnergy() / leaky.getContactEnergy()

    stiff = build(wallStiffness = 100.0)
    stiff.holdExcessEnergy(excess, maxUnbalancedForce = 1e-6)
    stiffShare = 1.0 - stiff.getPairContactEnergy() / stiff.getContactEnergy()
    starts = np.asarray(stiff.packing.startIndices, dtype = int)
    r = stiff.packing.positions.reshape(-1, 2)[:starts[int(stiff.packing.containerIndex)]]
    escaped = int(((r < 0.0) | (r > 1.0)).any(axis = 1).sum())
    print(f"  wallStiffness   1: wall carries {100 * leakyShare:6.2f}% of the contact energy")
    print(f"  wallStiffness 100: wall carries {100 * stiffShare:6.2f}%   "
          f"penetration {stiff.getWallPenetration():.2e}   {escaped} vertices outside")
    # The comparison is the check: stiffening must move the load off the boundary and onto the bodies.
    ok = stiffShare < 0.05 and stiffShare < leakyShare and escaped == 0
    print(f"  CHECK 6 load is on the bodies {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    print("excess-energy controller", flush = True)
    results = []
    for name, check in (("affine, box fixed", checkAffine),
                        ("dimensionless", checkDimensionless), ("contact only", checkContactOnly),
                        ("two-sided", checkTwoSided), ("no container", checkPeriodic),
                        ("sweep from below", checkSweepFromBelow),
                        ("load on the bodies", checkWallShare)):
        print(f"\n{name}:", flush = True)
        results.append(check())
    print(f"\n{sum(results)}/{len(results)} checks passed")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
