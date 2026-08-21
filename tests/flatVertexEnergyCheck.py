"""Is the DEPTH contact tier finite at an exactly collinear vertex?

The cascade died at stage 2 with `energy nan` out of the force evaluation, on geometry the minimizer's
own guards had just certified finite. The suspect is the shape the ramp is deliberately building: the
flatten stage drives selected vertices to `d / (a + b) -> 1`, which IS collinearity, and stage 2 opened
with `worstFlat 1.00000` -- every selected vertex already exactly flat before any ramping.

So the ramp's TARGET and the energy tier's domain may be in direct conflict. This measures that rather
than arguing it: build a polygon whose vertex sits exactly on the line between its neighbors, put it in
contact, and evaluate. Then sweep the offset to zero to find where finiteness actually dies.

    python tests/flatVertexEnergyCheck.py
"""

# UNVERIFIED(Cam)

import os
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(_ROOT))
sys.path.insert(0, _ROOT)

import pyPolygon as pp


def squareWithMidpoints(center, half, offset):
    """A square carrying a vertex at the middle of every edge, pushed `offset` OUT along the normal.

    `offset = 0` puts the midpoint vertex exactly on the line joining its neighbors -- flatness exactly
    1, the state the ramp aims at. Any other value is an ordinary corner, so one parameter walks
    continuously through the degeneracy."""
    corners = np.array([[-half, -half], [half, -half], [half, half], [-half, half]])
    out = []
    for k in range(4):
        a, b = corners[k], corners[(k + 1) % 4]
        mid = 0.5 * (a + b)
        normal = mid / np.linalg.norm(mid)
        out.append(a)
        out.append(mid + offset * normal)
    return np.array(out) + np.asarray(center)


def build(offset, separation, n = 8):
    """Two such polygons, overlapping, on the depth tier."""
    model = pp.Model(N = 2, n = n, seed = 3)
    model.generateEquilateralPolygons(phi = 0.30, kappa = 4.0)
    box = 1.0
    half = 0.18 * box
    left = squareWithMidpoints((0.5 * box - separation * half, 0.5 * box), half, offset)
    right = squareWithMidpoints((0.5 * box + separation * half, 0.5 * box), half, offset)
    model.packing.positions[:] = np.concatenate([left.reshape(-1), right.reshape(-1)])
    model.setDepthContact(stiffness = 1.0, wallStiffness = 100.0)
    return model


def evaluate(model):
    energy, force = model._forceEnergy(model.packing)
    force = np.asarray(force, dtype = float)
    return float(energy), int((~np.isfinite(force.reshape(-1, 2)).all(axis = 1)).sum()), \
        float(np.abs(force[np.isfinite(force)]).max() if np.isfinite(force).any() else np.nan)


def flatness(model):
    """`d / (a + b)` over every vertex, the same quantity the ramp drives."""
    positions = model.packing.positions.reshape(2, -1, 2)
    worst = 0.0
    for polygon in positions:
        count = polygon.shape[0]
        for k in range(count):
            previous, here, following = polygon[k - 1], polygon[k], polygon[(k + 1) % count]
            a = np.linalg.norm(here - previous)
            b = np.linalg.norm(following - here)
            d = np.linalg.norm(following - previous)
            worst = max(worst, d / (a + b))
    return worst


def main():
    print(__doc__.strip().splitlines()[0])
    print("\nOVERLAPPING pair, sweeping a vertex onto the line joining its neighbors")
    print(f"  {'offset':>10s}  {'flatness':>10s}  {'energy':>14s}  {'bad force':>10s}  {'max|F|':>12s}")
    separation = 1.7
    rows = []
    for offset in (0.2, 0.05, 1e-2, 1e-3, 1e-5, 1e-8, 1e-12, 0.0):
        model = build(offset, separation)
        energy, bad, peak = evaluate(model)
        flat = flatness(model)
        rows.append((offset, flat, energy, bad))
        print(f"  {offset:10.1e}  {flat:10.6f}  {energy:14.6e}  {bad:10d}  {peak:12.4e}")

    # A polygon SHRINKING is the case `equilateral` cannot forbid: it pins l_k = kappa sqrt(A) / n,
    # a RATIO, which a polygon collapsing to a point satisfies exactly. With `area = [1]` holding only
    # the MEAN, one polygon may collapse while another inflates and both constraints stay happy.
    print("\nSAME pair, shrinking ONE polygon toward zero size (shape held, ratio satisfied)")
    print(f"  {'scale':>10s}  {'area':>12s}  {'energy':>14s}  {'bad force':>10s}  {'max|F|':>12s}")
    collapse = []
    for scale in (1.0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-6, 1e-9, 0.0):
        model = build(0.0, separation)
        positions = model.packing.positions.reshape(2, -1, 2)
        center = positions[0].mean(axis = 0)
        positions[0] = center + scale * (positions[0] - center)
        polygon = positions[0]
        area = 0.5 * abs(float(np.cross(polygon, np.roll(polygon, -1, axis = 0)).sum()))
        energy, bad, peak = evaluate(model)
        collapse.append((scale, energy, bad))
        print(f"  {scale:10.1e}  {area:12.4e}  {energy:14.6e}  {bad:10d}  {peak:12.4e}")

    if any(not np.isfinite(e) or b > 0 for _, e, b in collapse):
        first = next((s, e, b) for s, e, b in collapse if not np.isfinite(e) or b > 0)
        print(f"\n  >> the tier BREAKS once a polygon shrinks to scale {first[0]:.1e}.")
        print("     `equilateral` cannot prevent this -- it pins a RATIO that a collapsing polygon")
        print("     satisfies exactly -- and `area = [1]` holds only the MEAN, so one polygon may")
        print("     collapse while another inflates with both constraints reporting success.")
    else:
        print("\n  >> the tier survives collapse too; neither degeneracy explains the stage-2 NaN.")

    exact = rows[-1]
    finiteAway = all(np.isfinite(e) and b == 0 for _, _, e, b in rows[:-1])
    brokenAtZero = (not np.isfinite(exact[2])) or exact[3] > 0

    print()
    if brokenAtZero and finiteAway:
        print("DIAGNOSIS: the tier is finite for every non-zero offset and NON-FINITE at exactly")
        print("  collinear. The ramp's goal and the energy tier's domain are in direct conflict:")
        print("  flatness 1 is a point the contact law cannot evaluate.")
    elif brokenAtZero:
        print("DIAGNOSIS: non-finite at exactly collinear, AND already broken before it -- the")
        print("  degeneracy has a WIDTH, so stopping short of 1 is not by itself enough.")
    else:
        print("DIAGNOSIS: collinearity is NOT the fault -- the tier stayed finite at offset 0.")
        print("  The stage-2 NaN comes from something else; do not spend the run here.")
    return brokenAtZero


if __name__ == "__main__":
    main()
