"""Does a pinned four-trapezoid frame confine correctly under the depth-contact law?

Cam's proposal: surround a 1x1 void with four mitered trapezoids and pin them, so confinement reuses
the body-body machinery with no new term. This measures whether the answer is right, against an
INDEPENDENT reference -- dense quadrature of ``int (k/3) dist(x, void)^3 dl`` along the intruder's
boundary, which never touches ``polyContact`` at all.

The suspicion under test is the miters. The void's corners are REFLEX vertices of the wall region, and
no convex piece can contain a reflex vertex, so every convex partition of the wall must run a seam out
from each corner. The law reads any boundary as a free surface where ``d = 0``, and a seam is not a
free surface -- it is the interior of the wall. So corners should come out too soft. This says by how
much.

    python tests/wallFrameCheck.py
"""

# UNVERIFIED(Cam)

import os
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

import polyContact as pc


def trapezoidFrame(thickness = 0.5, low = 0.0, high = 1.0):
    """Four mitered trapezoids tiling the wall region around the void ``[low, high]^2``.

    Corners are cut at 45 degrees, so the four pieces tile the frame exactly: no overlap (which would
    double-count) and no gap (which would leak).

    WINDING IS NORMALIZED HERE. ``pairEnergy`` does not do it, and a clockwise wall inverts the
    membership test, turning the whole container into an attractive well -- the exact failure that
    collapsed five squares onto a point earlier in this project."""
    a, b, t = low, high, thickness
    corners = [
        [[a, a], [b, a], [b + t, a - t], [a - t, a - t]],
        [[b, a], [b, b], [b + t, b + t], [b + t, a - t]],
        [[b, b], [a, b], [a - t, b + t], [b + t, b + t]],
        [[a, b], [a, a], [a - t, a - t], [a - t, b + t]],
    ]
    return [pc.makeCounterClockwise(np.array(loop, dtype = float)) for loop in corners]


def distanceToVoid(points, low = 0.0, high = 1.0):
    """Exact distance from each point to the boundary of the square void, for points OUTSIDE it.

    Written directly from the geometry -- the standard point-to-axis-aligned-rectangle distance -- so
    it shares no code with ``polyContact`` and can serve as the reference."""
    points = np.atleast_2d(np.asarray(points, dtype = float))
    outside = np.maximum(np.maximum(low - points, points - high), 0.0)
    return np.linalg.norm(outside, axis = 1)


def referenceEnergy(loop, stiffness = 1.0, samples = 400000, low = 0.0, high = 1.0):
    """``int_{dP outside the void} (k/3) dist(x, void)^3 dl`` by dense quadrature along the boundary.

    Midpoint rule on a very fine uniform partition of each edge. The integrand is C^1 and the sample
    count is far past convergence, so this is a reference, not an estimate."""
    loop = np.asarray(loop, dtype = float)
    total = 0.0
    perimeter = np.linalg.norm(np.roll(loop, -1, axis = 0) - loop, axis = 1).sum()
    for index in range(len(loop)):
        start, end = loop[index], loop[(index + 1) % len(loop)]
        length = float(np.linalg.norm(end - start))
        count = max(2, int(samples * length / perimeter))
        s = (np.arange(count) + 0.5) / count
        points = start + s[:, None] * (end - start)
        depth = distanceToVoid(points, low, high)
        total += (stiffness / 3.0) * float(depth ** 3 @ np.full(count, length / count))
    return total


def frameEnergy(loop, walls, stiffness = 1.0):
    """What the trapezoid frame actually charges: the intruder's boundary against each wall in turn.

    Only the ``dP inside wall`` direction is summed. The reciprocal term (a wall's boundary inside the
    intruder) is the wall feeling the body, which is real physics but is NOT part of the confinement
    integral being compared, and it is carried by pinned vertices that never move."""
    return sum(pc.pairEnergy(loop, wall, stiffness) for wall in walls)


def square(centre, halfWidth):
    cx, cy = centre
    h = halfWidth
    return np.array([[cx - h, cy - h], [cx + h, cy - h], [cx + h, cy + h], [cx - h, cy + h]])


_FAILURES = []


def report(name, ours, truth, tolerance):
    error = abs(ours - truth) / max(abs(truth), 1e-300)
    ok = error < tolerance
    if not ok:
        _FAILURES.append(name)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name:34s} frame {ours:.6e}  exact {truth:.6e}"
          f"   relative {error:8.2%}")
    return error


def invertedVoid(low = 0.0, high = 1.0):
    """The void boundary wound CLOCKWISE, which makes the confining region its EXTERIOR.

    ``pairEnergy`` reads membership from the winding and does not normalize it, so a clockwise loop
    inverts the inside test: the integral runs over the part of the intruder's boundary that is
    OUTSIDE the square, weighted by the exact distance to the square's boundary. That is precisely the
    confinement integral, with no new code, no seam, and one body instead of four.

    It is also unconditionally valid, unlike body-body contact. The exterior of a CONVEX region has no
    medial axis: beyond an edge the nearest feature is that edge, beyond a corner it is that corner,
    and the seam between them is a normal line where two INCIDENT features tie -- a C^1 seam, not a
    jump in grad d. So the ``dMax/rIn << 1`` limit that caps body-body contact does not apply here and
    a wall can be pressed arbitrarily hard."""
    return np.array([[low, low], [low, high], [high, high], [high, low]], dtype = float)


def checkGradient(loop, walls, wound):
    """Analytic force against a central difference of the INDEPENDENT reference energy.

    Differencing the closed form against its own energy would only prove the bookkeeping; the reference
    here is the dense boundary quadrature, which shares no code with ``polyContact``."""
    _, analytic, _ = pc.pairGradient(loop, wound)
    step = 1e-6
    numeric = np.zeros_like(loop)
    for vertex in range(len(loop)):
        for axis in range(2):
            plus, minus = loop.copy(), loop.copy()
            plus[vertex, axis] += step
            minus[vertex, axis] -= step
            numeric[vertex, axis] = ((referenceEnergy(plus, samples = 800000)
                                      - referenceEnergy(minus, samples = 800000)) / (2.0 * step))
    scale = max(float(np.abs(numeric).max()), 1e-300)
    return float(np.abs(analytic - numeric).max()) / scale


def main():
    print("=" * 96)
    print("Pinned four-trapezoid frame vs the exact exterior distance")
    print("=" * 96)
    walls = trapezoidFrame(thickness = 0.5)

    print("\n1. FACE contact -- intruder pushed through the middle of one wall")
    for depth in (0.02, 0.05, 0.10):
        loop = square((0.5, 0.15 - depth), 0.15)
        report(f"depth {depth:.2f} at a face", frameEnergy(loop, walls),
               referenceEnergy(loop), 1e-6)

    print("\n2. CORNER contact -- intruder pushed into a corner along the diagonal")
    for depth in (0.02, 0.05, 0.10):
        offset = depth / np.sqrt(2.0)
        loop = square((0.15 - offset, 0.15 - offset), 0.15)
        report(f"depth {depth:.2f} into a corner", frameEnergy(loop, walls),
               referenceEnergy(loop), 1e-6)

    print("\n3. The same cases against the INVERTED VOID -- one clockwise loop, no seam")
    wound = invertedVoid()
    for depth in (0.02, 0.05, 0.10):
        loop = square((0.5, 0.15 - depth), 0.15)
        report(f"depth {depth:.2f} at a face", pc.pairEnergy(loop, wound),
               referenceEnergy(loop), 1e-8)
    for depth in (0.02, 0.05, 0.10):
        offset = depth / np.sqrt(2.0)
        loop = square((0.15 - offset, 0.15 - offset), 0.15)
        report(f"depth {depth:.2f} into a corner", pc.pairEnergy(loop, wound),
               referenceEnergy(loop), 1e-8)

    print("\n4. Inverted-void FORCE vs a difference of the independent reference")
    for name, loop in (("face", square((0.5, 0.10), 0.15)),
                       ("corner", square((0.15 - 0.0354, 0.15 - 0.0354), 0.15))):
        error = checkGradient(loop, walls, wound)
        ok = error < 1e-5
        if not ok:
            _FAILURES.append(f"{name} gradient")
        print(f"  [{'PASS' if ok else 'FAIL'}] {name + ' contact gradient':34s} "
              f"max relative deviation {error:.3e}")

    print("\n" + "=" * 96)
    if _FAILURES:
        print(f"DISAGREES on: {', '.join(_FAILURES)}")
        return 1
    print("every construction reproduces the exact confinement energy")
    return 0


if __name__ == "__main__":
    sys.exit(main())
