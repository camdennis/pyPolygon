# UNVERIFIED(Cam)
"""The ANALYTIC exact-arc gradient against the complex-step one it replaces.

``roundedContact.pairGradient`` differentiates the whole energy once per degree of freedom by complex
step. It is exact and self-consistent by construction, which is precisely what makes it the reference:
``pairGradientAnalytic`` has to reproduce it to round-off on every configuration, not merely agree with
a finite difference to a few digits.

Four layers, innermost first, so a failure localizes itself:

  1. each coefficient partial against a complex step of the value routine it belongs to;
  2. ``pairEnergyBodyGradient``'s energy against ``pairEnergy`` (same partition, same numbers);
  3. its body-array gradient against a complex step of ``frozenPairEnergy`` taken in the BODY arrays;
  4. the assembled ``pairGradientAnalytic`` against ``pairGradient``, and both against a central
     difference of the TRUE energy, which re-partitions at every step;
  5. ``packingEnergyForce`` against the per-pair complex-step assembly it used to be;
  6. the AREA tier's ``areaGradientAnalytic`` against ``areaGradient``;
  7. its packing driver against its own former assembly;
  8. both tiers through ``Model`` with a CONTAINER, against a central difference in the vertices and
     in ``rho`` -- the only layer that exercises the wall's reversed winding and ``getRhoForces``.
"""

import os
import sys
import time

import numpy as np

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root not in sys.path:
    sys.path.insert(0, root)

import roundedContact as rc


# UNVERIFIED(Cam)
def regularLoop(sides, center, size, angle):
    """A counter-clockwise regular polygon."""
    turns = angle + 2.0 * np.pi * np.arange(sides) / sides
    return np.asarray(center) + size * np.stack([np.cos(turns), np.sin(turns)], axis = 1)


# UNVERIFIED(Cam)
def randomPair(rng, sides = 4):
    """Two overlapping rounded bodies with unequal, non-degenerate radii."""
    loopA = regularLoop(sides, (0.0, 0.0), 1.0, rng.uniform(0.0, 2.0 * np.pi))
    shift = rng.uniform(-0.55, 0.55, size = 2) + np.array([1.2, 0.0])
    loopB = regularLoop(sides, shift, rng.uniform(0.8, 1.2), rng.uniform(0.0, 2.0 * np.pi))
    previousIndex = np.roll(np.arange(sides), 1)
    nextIndex = np.roll(np.arange(sides), -1)
    capA = rc.rg.maxRho(loopA, previousIndex, nextIndex)
    capB = rc.rg.maxRho(loopB, previousIndex, nextIndex)
    rhoA = capA * rng.uniform(0.25, 0.85, size = sides)
    rhoB = capB * rng.uniform(0.25, 0.85, size = sides)
    return loopA, rhoA, loopB, rhoB


# UNVERIFIED(Cam)
def complexStep(function, value, step = 1e-30):
    """``d function / d value`` at a scalar, exactly."""
    return np.imag(function(value + 1j * step)) / step


# UNVERIFIED(Cam)
def checkCoefficientPartials(rng, trials = 200):
    """Layer 1: every partial against a complex step of its own value routine."""
    worst = {"line": 0.0, "arcTo": 0.0, "harmonic": 0.0}

    for _ in range(trials):
        a, m = rng.uniform(-2.0, 2.0, size = 2)
        lo, hi = np.sort(rng.uniform(0.0, 1.0, size = 2))
        _, da, dm = rc._lineIntegralPartials(a, m, lo, hi)
        trueA = complexStep(lambda x: rc._cubicLineIntegral(x, m, lo, hi), a)
        trueM = complexStep(lambda x: rc._cubicLineIntegral(a, x, lo, hi), m)
        worst["line"] = max(worst["line"], abs(da - trueA), abs(dm - trueM))

        radius = rng.uniform(0.05, 1.0)
        height = rng.uniform(0.05, 2.0)
        lowW, highW = np.sort(rng.uniform(-2.0, 2.0, size = 2))

        def arcValue(r, h, wl, wh):
            return rc._arcToPartials(r, h, wl, wh)[0]

        analytic = rc._arcToPartials(radius, height, lowW, highW)[1:]
        reference = [
            (arcValue(radius + 1e-6, height, lowW, highW)
             - arcValue(radius - 1e-6, height, lowW, highW)) / 2e-6,
            (arcValue(radius, height + 1e-6, lowW, highW)
             - arcValue(radius, height - 1e-6, lowW, highW)) / 2e-6,
            (arcValue(radius, height, lowW + 1e-6, highW)
             - arcValue(radius, height, lowW - 1e-6, highW)) / 2e-6,
            (arcValue(radius, height, lowW, highW + 1e-6)
             - arcValue(radius, height, lowW, highW - 1e-6)) / 2e-6]
        span = max(1.0, max(abs(x) for x in reference))
        worst["arcTo"] = max(worst["arcTo"],
                             max(abs(x - y) for x, y in zip(analytic, reference)) / span)

        a, b, c = rng.uniform(-2.0, 2.0, size = 3)
        psi = rng.uniform(-2.0, 2.0)
        _, da, db, dc, dPsi = rc._harmonicPartials(a, b, c, psi)
        trueA = complexStep(lambda x: rc._cubicHarmonicIntegral(x, b, c, psi), a)
        trueB = complexStep(lambda x: rc._cubicHarmonicIntegral(a, x, c, psi), b)
        trueC = complexStep(lambda x: rc._cubicHarmonicIntegral(a, b, x, psi), c)
        truePsi = complexStep(lambda x: rc._cubicHarmonicIntegral(a, b, c, x), psi)
        worst["harmonic"] = max(worst["harmonic"], abs(da - trueA), abs(db - trueB),
                                abs(dc - trueC), abs(dPsi - truePsi))
    return worst


# UNVERIFIED(Cam)
def bodyStepGradient(loopA, rhoA, loopB, rhoB, partition, stiffness, order, step = 1e-30):
    """``dE/d(body arrays)`` by complex step, for layer 3.

    The bodies are perturbed DIRECTLY, not through the backbone, so this tests the chain rule inside
    ``pairEnergyBodyGradient`` on its own -- with the corner map's own derivative deliberately absent."""
    bodyA = rc.bodyFromBackbone(loopA, rhoA)
    bodyB = rc.bodyFromBackbone(loopB, rhoB)

    def rebuild(source, offsets):
        body = rc.RoundedBody(source.center + offsets[0], source.radius + offsets[1],
                              source.start, source.sweep + offsets[2],
                              source.tail + offsets[3], source.head + offsets[4])
        return body

    def energyOf(perturbedA, perturbedB):
        piece, low, high, kind, feature = partition
        return _frozenFromBodies(perturbedA, perturbedB, piece, low, high, kind, feature,
                                 stiffness, order)

    results = []
    for source, first in ((bodyA, True), (bodyB, False)):
        count = source.count
        shapes = [(count, 2), (count,), (count,), (count, 2), (count, 2)]
        gradient = rc.BodyGradient(count)
        arrays = [gradient.center, gradient.radius, gradient.sweep, gradient.tail, gradient.head]
        for slot, shape in enumerate(shapes):
            for index in np.ndindex(shape):
                offsets = [np.zeros(s, dtype = complex) for s in shapes]
                offsets[slot][index] = 1j * step
                zero = [np.zeros(s, dtype = complex) for s in shapes]
                perturbedA = rebuild(bodyA, offsets if first else zero)
                perturbedB = rebuild(bodyB, zero if first else offsets)
                arrays[slot][index] = np.imag(energyOf(perturbedA, perturbedB)) / step
        results.append(gradient)
    return results


# UNVERIFIED(Cam)
def _frozenFromBodies(bodyA, bodyB, piece, low, high, kind, feature, stiffness, order):
    """``frozenPairEnergy`` rewritten to take BODIES rather than backbones.

    A transcription of the same expression -- not a second law -- so that layer 3 differentiates
    exactly what ``pairEnergyBodyGradient`` claims to differentiate."""
    nodes, weights = np.polynomial.legendre.leggauss(int(order))
    n = bodyA.count
    total = 0.0 * (bodyA.center[0, 0] + bodyB.center[0, 0])
    edge = bodyB.head - bodyB.tail
    length = np.sqrt(edge[:, 0] ** 2 + edge[:, 1] ** 2)
    normal = np.stack([edge[:, 1], -edge[:, 0]], axis = 1) / length[:, None]

    for p, lo, hi, k, f in zip(piece, low, high, kind, feature):
        if hi - lo <= 0.0:
            continue
        if p >= n:
            j = p - n
            tail = bodyA.tail[j]
            vector = bodyA.head[j] - bodyA.tail[j]
            speed = np.sqrt(vector[0] ** 2 + vector[1] ** 2)
            if k == 0:
                a = normal[f] @ (bodyB.tail[f] - tail)
                m = normal[f] @ vector
                value = rc._cubicLineIntegral(a, m, lo, hi)
            else:
                offset = tail - bodyB.center[f]
                qa, qb, qc = vector @ vector, 2.0 * (offset @ vector), offset @ offset
                r = bodyB.radius[f]
                firstHi, thirdHi = rc._rootAntiderivatives(qa, qb, qc, hi)
                firstLo, thirdLo = rc._rootAntiderivatives(qa, qb, qc, lo)
                second = (qa * (hi ** 3 - lo ** 3) / 3.0 + qb * (hi ** 2 - lo ** 2) / 2.0
                          + qc * (hi - lo))
                value = (r ** 3 * (hi - lo) - 3.0 * r * r * (firstHi - firstLo)
                         + 3.0 * r * second - (thirdHi - thirdLo))
            total = total + speed * value
        else:
            radius, sweep = bodyA.radius[p], bodyA.sweep[p]
            if np.real(radius) == 0.0 or np.real(sweep) == 0.0:
                continue
            center = bodyA.center[p]
            startVector = bodyA.head[(p - 1) % n] - center
            turned = np.stack([-startVector[1], startVector[0]])
            if k == 0:
                a = normal[f] @ (bodyB.tail[f] - center)
                b = -(normal[f] @ startVector)
                c = -(normal[f] @ turned)
                value = (rc._cubicHarmonicIntegral(a, b, c, sweep * hi)
                         - rc._cubicHarmonicIntegral(a, b, c, sweep * lo))
                total = total + radius * np.sign(np.real(sweep)) * value
            else:
                s = 0.5 * (lo + hi) + 0.5 * (hi - lo) * nodes
                psi = sweep * s
                delta = center - bodyB.center[f]
                inner = (delta @ delta + radius * radius
                         + 2.0 * (np.cos(psi) * (delta @ startVector)
                                  + np.sin(psi) * (delta @ turned)))
                d = bodyB.radius[f] - np.sqrt(inner)
                total = total + (radius * sweep * np.sign(np.real(sweep)) * 0.5 * (hi - lo)
                                 * np.sum(weights * d ** 3))
    return stiffness / 3.0 * total


# UNVERIFIED(Cam)
def relative(analytic, reference):
    """Relative error against the larger of the two scales, so a zero entry cannot divide."""
    span = max(np.max(np.abs(analytic)), np.max(np.abs(reference)), 1e-300)
    return float(np.max(np.abs(analytic - reference)) / span)


# UNVERIFIED(Cam)
def wholeGradient(blocks):
    """The four gradient blocks as ONE vector.

    SCALED TOGETHER, NOT BLOCK BY BLOCK. ``dE/drhoB`` for the ordered pair is identically zero whenever
    no sub-stretch measures to an ARC of B: with the partition frozen, the distance is to the segment's
    supporting LINE, and moving B's radii slides the kiss points along that same line. Both
    implementations then return round-off, and a per-block relative error divides one 1e-18 by another
    and reports 100%."""
    return np.concatenate([np.asarray(block).ravel() for block in blocks])


# UNVERIFIED(Cam)
class SimplePacking:
    """The three attributes ``packingEnergyForce`` reads, without pulling in ``Model``."""

    # UNVERIFIED(Cam)
    def __init__(self, positions, startIndices, containerIndex = None):
        self.positions = np.asarray(positions, dtype = float)
        self.startIndices = np.asarray(startIndices, dtype = int)
        if containerIndex is not None:
            self.containerIndex = containerIndex


# UNVERIFIED(Cam)
def squareGrid(rng, across = 3, overlap = 0.12):
    """Nine jostled rounded squares on a grid, pressed together so every neighbour pair is in contact."""
    loops, radii = [], []
    # circumradius 1 at 45 degrees is an axis-aligned square of side sqrt(2), so that is the pitch that
    # just touches; anything less overlaps.
    pitch = np.sqrt(2.0) * (1.0 - overlap)
    previousIndex, nextIndex = np.roll(np.arange(4), 1), np.roll(np.arange(4), -1)
    for row in range(across):
        for column in range(across):
            center = (pitch * column, pitch * row) + rng.uniform(-0.05, 0.05, size = 2)
            loop = regularLoop(4, center, 1.0, 0.25 * np.pi + rng.uniform(-0.15, 0.15))
            cap = rc.rg.maxRho(loop, previousIndex, nextIndex)
            loops.append(loop)
            radii.append(cap * rng.uniform(0.3, 0.8, size = 4))
    starts = np.cumsum([0] + [len(loop) for loop in loops])
    return SimplePacking(np.concatenate(loops), starts), np.concatenate(radii)


# UNVERIFIED(Cam)
def stepPackingEnergyForce(packing, rho, stiffness = 1.0, wallStiffness = 1.0, quadratureOrder = 24):
    """``packingEnergyForce`` as it was before the derivation: per-pair complex-step gradients.

    Kept here rather than in the module so the library carries one assembly and the test carries the
    thing it is compared against."""
    vertices = packing.positions.reshape(-1, 2)
    starts = np.asarray(packing.startIndices, dtype = int)
    rho = np.asarray(rho, dtype = float).reshape(-1)
    container = getattr(packing, "containerIndex", None)
    count = len(starts) - 1
    loops = [vertices[starts[p]:starts[p + 1]] for p in range(count)]
    radii = [rho[starts[p]:starts[p + 1]] for p in range(count)]
    energy = 0.0
    force = np.zeros_like(vertices)
    rhoForce = np.zeros_like(rho)
    bodies = [rc.bodyFromBackbone(loops[p], radii[p]) for p in range(count)]

    for a, b in rc.candidatePairs(bodies, exterior = container):
        pairStiffness = stiffness
        if container is not None and (a == container or b == container):
            pairStiffness *= wallStiffness
        for first, second in ((a, b), (b, a)):
            value, gLA, gRA, gLB, gRB = rc.pairGradient(
                loops[first], radii[first], loops[second], radii[second],
                0.5 * pairStiffness, quadratureOrder)
            energy += value
            force[starts[first]:starts[first + 1]] -= gLA
            rhoForce[starts[first]:starts[first + 1]] -= gRA
            force[starts[second]:starts[second + 1]] -= gLB
            rhoForce[starts[second]:starts[second + 1]] -= gRB
    return float(energy), force, rhoForce


# UNVERIFIED(Cam)
def stepPackingAreaEnergyForce(packing, rho, kOverlap = 1.0, kContainer = 1.0):
    """``packingAreaEnergyForce`` as it was: per-pair complex-step shape derivatives.

    No container here -- ``squareGrid`` has none -- so the winding reversal that the real driver does
    is deliberately absent rather than duplicated wrongly."""
    vertices = packing.positions.reshape(-1, 2)
    starts = np.asarray(packing.startIndices, dtype = int)
    rho = np.asarray(rho, dtype = float).reshape(-1)
    targetArea = np.asarray(packing.targetArea, dtype = float)
    count = len(starts) - 1
    loops = [vertices[starts[p]:starts[p + 1]] for p in range(count)]
    radii = [rho[starts[p]:starts[p + 1]] for p in range(count)]
    energy = 0.0
    force = np.zeros_like(vertices)
    rhoForce = np.zeros_like(rho)
    bodies = [rc.bodyFromBackbone(loops[p], radii[p]) for p in range(count)]

    for a, b in rc.candidatePairs(bodies):
        norm = targetArea[a] + targetArea[b]
        area, gLA, gRA, gLB, gRB = rc.areaGradient(loops[a], radii[a], loops[b], radii[b])
        if area == 0.0:
            continue
        energy += 2.0 * kOverlap * (area / norm) ** 2
        weight = 4.0 * kOverlap * area / (norm * norm)
        force[starts[a]:starts[a + 1]] -= weight * gLA
        rhoForce[starts[a]:starts[a + 1]] -= weight * gRA
        force[starts[b]:starts[b + 1]] -= weight * gLB
        rhoForce[starts[b]:starts[b + 1]] -= weight * gRB
    return float(energy), force, rhoForce


# UNVERIFIED(Cam)
def main():
    rng = np.random.default_rng(0)
    stiffness, order = 1.7, 24

    print("layer 1 -- coefficient partials vs a complex step of their own value routine")
    worst = checkCoefficientPartials(rng)
    for name, value in worst.items():
        print(f"  {name:9s} {value:.3e}")

    print("\nlayer 2/3 -- energy and body-array gradient, per pair")
    print(f"  {'sides':>5s} {'energy':>10s} {'dE/dbody':>10s}")
    for sides in (3, 4, 5, 6):
        for _ in range(2):
            loopA, rhoA, loopB, rhoB = randomPair(rng, sides)
            bodyA = rc.bodyFromBackbone(loopA, rhoA)
            bodyB = rc.bodyFromBackbone(loopB, rhoB)
            piece, low, high, kind, feature, _ = rc.substretches(bodyA, bodyB)
            if not len(piece):
                continue
            partition = (piece, low, high, kind, feature)
            energy, gradA, gradB = rc.pairEnergyBodyGradient(
                bodyA, bodyB, partition, stiffness, order)
            direct = rc.pairEnergy(bodyA, bodyB, stiffness, order)
            stepA, stepB = bodyStepGradient(loopA, rhoA, loopB, rhoB, partition, stiffness, order)
            error = max(relative(gradA.flat(), stepA.flat()),
                        relative(gradB.flat(), stepB.flat()))
            print(f"  {sides:5d} {abs(energy - direct) / max(abs(direct), 1e-300):10.2e}"
                  f" {error:10.2e}")

    print("\nlayer 4 -- assembled gradient vs the complex-step gradient and vs a true-energy FD")
    print(f"  {'sides':>5s} {'energy':>10s} {'vs step':>10s} {'A vs FD':>10s} {'ref vs FD':>10s}")
    for sides in (3, 4, 5, 6):
        for _ in range(3):
            loopA, rhoA, loopB, rhoB = randomPair(rng, sides)
            reference = rc.pairGradient(loopA, rhoA, loopB, rhoB, stiffness, order)
            analytic = rc.pairGradientAnalytic(loopA, rhoA, loopB, rhoB, stiffness, order)
            if reference[0] == 0.0:
                continue
            versusStep = relative(wholeGradient(analytic[1:]), wholeGradient(reference[1:]))

            def trueEnergy(loops, radii):
                bodyA = rc.bodyFromBackbone(loops[0], radii[0])
                bodyB = rc.bodyFromBackbone(loops[1], radii[1])
                return rc.pairEnergy(bodyA, bodyB, stiffness, order)

            delta = 1e-6
            finite = []
            for slot, template in enumerate((loopA, rhoA, loopB, rhoB)):
                out = np.zeros_like(template)
                for index in np.ndindex(template.shape):
                    parts = [loopA.copy(), rhoA.copy(), loopB.copy(), rhoB.copy()]
                    parts[slot] = parts[slot].copy()
                    parts[slot][index] += delta
                    plus = trueEnergy((parts[0], parts[2]), (parts[1], parts[3]))
                    parts[slot][index] -= 2.0 * delta
                    minus = trueEnergy((parts[0], parts[2]), (parts[1], parts[3]))
                    out[index] = (plus - minus) / (2.0 * delta)
                finite.append(out)
            versusFinite = relative(wholeGradient(analytic[1:]), wholeGradient(finite))
            referenceFinite = relative(wholeGradient(reference[1:]), wholeGradient(finite))
            print(f"  {sides:5d} {reference[0]:10.2e} {versusStep:10.2e}"
                  f" {versusFinite:10.2e} {referenceFinite:10.2e}")

    print("\ntiming -- one pair, squares")
    loopA, rhoA, loopB, rhoB = randomPair(np.random.default_rng(3), 4)
    for name, call in (("complex step", rc.pairGradient), ("analytic", rc.pairGradientAnalytic)):
        start = time.perf_counter()
        for _ in range(20):
            call(loopA, rhoA, loopB, rhoB, stiffness, order)
        print(f"  {name:13s} {(time.perf_counter() - start) / 20 * 1e3:7.2f} ms")

    print("\nlayer 5 -- packingEnergyForce against the per-pair complex-step assembly")
    packing, rho = squareGrid(rng)
    start = time.perf_counter()
    energy, force, rhoForce = rc.packingEnergyForce(packing, rho, stiffness = 1.0)
    analyticTime = time.perf_counter() - start
    start = time.perf_counter()
    stepEnergy, stepForce, stepRho = stepPackingEnergyForce(packing, rho, stiffness = 1.0)
    stepTime = time.perf_counter() - start
    print(f"  bodies {len(packing.startIndices) - 1}, energy {energy:.9e}"
          f"  (delta {abs(energy - stepEnergy):.2e})")
    print(f"  force    {relative(force, stepForce):.3e}")
    print(f"  rhoForce {relative(rhoForce, stepRho):.3e}")
    print(f"  analytic {analyticTime * 1e3:8.1f} ms   complex step {stepTime * 1e3:8.1f} ms"
          f"   ({stepTime / analyticTime:.1f}x)")

    print("\nlayer 6 -- the AREA tier, per pair", flush = True)
    print(f"  {'sides':>5s} {'area':>10s} {'vs step':>10s} {'A vs FD':>10s} {'ref vs FD':>10s}")
    for sides in (3, 4, 5, 6):
        for _ in range(2):
            loopA, rhoA, loopB, rhoB = randomPair(rng, sides)
            reference = rc.areaGradient(loopA, rhoA, loopB, rhoB)
            analytic = rc.areaGradientAnalytic(loopA, rhoA, loopB, rhoB)
            if reference[0] == 0.0:
                continue

            def trueArea(parts):
                return rc.overlapArea(rc.bodyFromBackbone(parts[0], parts[1]),
                                      rc.bodyFromBackbone(parts[2], parts[3]))

            delta = 1e-6
            finite = []
            for slot, template in enumerate((loopA, rhoA, loopB, rhoB)):
                out = np.zeros_like(template)
                for index in np.ndindex(template.shape):
                    parts = [loopA.copy(), rhoA.copy(), loopB.copy(), rhoB.copy()]
                    parts[slot] = parts[slot].copy()
                    parts[slot][index] += delta
                    plus = trueArea(parts)
                    parts[slot][index] -= 2.0 * delta
                    out[index] = (plus - trueArea(parts)) / (2.0 * delta)
                finite.append(out)
            print(f"  {sides:5d} {reference[0]:10.3e}"
                  f" {relative(wholeGradient(analytic[1:]), wholeGradient(reference[1:])):10.2e}"
                  f" {relative(wholeGradient(analytic[1:]), wholeGradient(finite)):10.2e}"
                  f" {relative(wholeGradient(reference[1:]), wholeGradient(finite)):10.2e}")

    print("\nlayer 7 -- packingAreaEnergyForce against the per-pair complex-step assembly")
    packing, rho = squareGrid(rng)
    packing.targetArea = np.full(len(packing.startIndices) - 1, 2.0)
    start = time.perf_counter()
    energy, force, rhoForce = rc.packingAreaEnergyForce(packing, rho)
    analyticTime = time.perf_counter() - start
    start = time.perf_counter()
    stepEnergy, stepForce, stepRho = stepPackingAreaEnergyForce(packing, rho)
    stepTime = time.perf_counter() - start
    print(f"  energy {energy:.9e}  (delta {abs(energy - stepEnergy):.2e})")
    print(f"  force    {relative(force, stepForce):.3e}")
    print(f"  rhoForce {relative(rhoForce, stepRho):.3e}")
    print(f"  analytic {analyticTime * 1e3:8.1f} ms   complex step {stepTime * 1e3:8.1f} ms"
          f"   ({stepTime / analyticTime:.1f}x)")

    modelLayer()


# UNVERIFIED(Cam)
def modelLayer(count = 11, delta = 1e-7):
    """Layer 8: both exact tiers through ``Model``, with the container, against a central difference.

    ABSOLUTE differences, not relative. Most corners are not in contact, so their true ``dE/drho`` is
    zero and a relative error divides one piece of round-off by another -- measured, an analytic
    ``3e-19`` against a finite-difference ``8e-15`` reports 85% and means nothing. The number that
    matters is whether the disagreement sits at the difference's own floor, which at ``delta = 1e-7``
    on an energy of order ``1e-2`` is about ``1e-11``."""
    import warnings
    from model import Model

    wall = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 0.0]])
    print("\nlayer 8 -- both tiers through Model, with a container "
          f"(worst ABSOLUTE gap; the difference's own floor is ~1e-11)")
    for tier in ("depth", "area"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = Model(N = count, n = 4, seed = 0)
            model.generateEquilateralPolygons(phi = 0.72, kappa = 4.0)
            model.setMonoPerimeter()
            model.placeOnGrid()
        model.addShape(wall)
        model.pinVertices(np.arange(model.getNumVertices())[-4:])
        model.setBoundaryConditions("fixed")
        model.setModelType(tier)
        radii = 0.35 * model.getMaxRho()
        model.setGeometryType("round", rho = radii, exact = True)
        model.calcForceEnergy()
        force = model.getForces().copy()
        rhoForce = model.getRhoForces().copy()
        positions = model.packing.positions.reshape(-1, 2).copy()
        rng = np.random.default_rng(2)

        def energyAt():
            model.calcForceEnergy()
            return model.getEnergy()

        worstVertex, worstRho = 0.0, 0.0
        for _ in range(5):
            index = (int(rng.integers(0, 4 * count)), int(rng.integers(0, 2)))
            moved = positions.copy()
            moved[index] += delta
            model.packing.positions = moved.copy()
            plus = energyAt()
            moved[index] -= 2.0 * delta
            model.packing.positions = moved.copy()
            minus = energyAt()
            model.packing.positions = positions.copy()
            worstVertex = max(worstVertex, abs(-(plus - minus) / (2.0 * delta) - force[index]))

            k = int(rng.integers(0, 4 * count))
            stepped = radii.copy()
            stepped[k] += delta
            model.setRho(stepped)
            plus = energyAt()
            stepped[k] -= 2.0 * delta
            model.setRho(stepped)
            minus = energyAt()
            model.setRho(radii)
            worstRho = max(worstRho, abs(-(plus - minus) / (2.0 * delta) - rhoForce[k]))
        print(f"  {tier:6s} energy {energyAt():.6e}   dE/dvertex {worstVertex:.2e}"
              f"   dE/drho {worstRho:.2e}")


if __name__ == "__main__":
    main()
