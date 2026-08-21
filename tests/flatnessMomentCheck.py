"""Does the DIAGONAL MOMENT family actually share the flattening work globally?

This is the gate for the overjammed cascade. ``setConstraints(diagonal = [1, 2])`` holds the mean and
variance of ``d/(a + b)`` over the selected vertices for the WHOLE packing -- two rows total -- which
is the mechanism that is supposed to let one polygon stay bent while another goes flat.

It had never been run. ``tests/flattenCascadeCheck.ramp`` is documented as walking those moments and
handing off, but its body goes straight to the per-object ``flatten`` family and prints its one result
twice, labelled "(moments)" and "(per vertex)". So every claim about the moment form -- including the
plateau and the conditioning collapse that justify the handoff -- is checked here for the first time.

Run: python tests/flatnessMomentCheck.py
"""

import os
import sys
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import Model

warnings.filterwarnings("ignore")

KAPPA = 4.0
passed, failed = 0, 0

def check(name, condition, detail = ""):
    global passed, failed
    if condition:
        passed += 1
        print(f"  PASS  {name}   {detail}")
    else:
        failed += 1
        print(f"  FAIL  {name}   {detail}")

def build(N = 5, n = 32, phi = 0.35, seed = 42, morph = 0.3):
    """Squares -> doubled -> lattice -> ROUNDED by the template, which is what gives a ramp something
    to do.

    Building at ``n`` directly returns self-intersecting polygons (|turn| 1089 deg at n = 32 against
    360). But squares-doubled is not usable either: every non-corner vertex is then EXACTLY collinear,
    so ``d/(a + b) = 1.000000`` on the whole selected set with spread 0.000000 and a flattening ramp
    has nothing left to flatten. ``setShapeTemplate`` supplies the missing seed -- a genuinely round
    polygon that is still simple by construction, because edge + diagonal pin the shape."""
    model = Model(N = N, n = 4, seed = seed)
    model.generateEquilateralPolygons(phi = phi, kappa = KAPPA)
    model.syncTargetAreas()
    model.doubleNumEdges(int(np.log2(n // 4)))
    model.placeOnGrid()
    model.setShapeTemplate(morph = 1.0, sides = 4)
    model.setConstraints(area = True, edge = True, diagonal = True)
    for step in range(1, 13):                      # walk it round in sub-steps, never in one jump
        model.setShapeTemplate(morph = 1.0 + (morph - 1.0) * step / 12, sides = 4)
        model.constraints.projectPositions(model.packing)
    model.selectFlattening(stride = 2)
    return model

def folded(model, N):
    starts = np.asarray(model.packing.startIndices, dtype = int)
    r = model.packing.positions.reshape(-1, 2)
    worst = 0.0
    for p in range(N):
        loop = r[starts[p]:starts[p + 1]]
        behind = loop - np.roll(loop, 1, axis = 0)
        ahead = np.roll(loop, -1, axis = 0) - loop
        turn = np.degrees(np.arctan2(np.cross(behind, ahead),
                                     np.einsum("ij,ij->i", behind, ahead)))
        worst = max(worst, float(np.abs(turn).sum()))
    return worst

N = 5

# ---------------------------------------------------------------- shape of the family
print("\n1. the family is TWO rows for the whole packing, not two per polygon")
model = build()
model.setConstraints(equilateral = KAPPA, edge = False, diagonal = [1, 2])
distribution = model._distributionConstraints()
J = distribution.jacobian(model.packing)
flat = model.getFlatness()
check("2 rows total", np.asarray(J).shape[0] == 2, f"jacobian rows {np.asarray(J).shape}")
check("covering every selected vertex", flat.size == N * 16,
      f"{flat.size} selected over {N} polygons at n = 32 (stride 2)")
check("and it measures d/(a+b)", np.all(flat > 0.0) and np.all(flat <= 1.0 + 1e-12),
      f"range {flat.min():.6f} .. {flat.max():.6f}   (1 = exactly flat)")

# ---------------------------------------------------------------- the Jacobian
print("\n2. the moment Jacobian against central differences")
x = model.packing.positions
base = x.copy()
h = 1e-6
numeric = np.zeros_like(np.asarray(J))
for g in range(x.size):
    x[g] = base[g] + h
    plus = distribution.residual(model.packing).copy()
    x[g] = base[g] - h
    minus = distribution.residual(model.packing).copy()
    x[g] = base[g]
    numeric[:, g] = (plus - minus) / (2.0 * h)
error = np.abs(np.asarray(J) - numeric).max() / max(1.0, np.abs(numeric).max())
check("analytic == finite difference", error < 1e-7, f"max relative error {error:.2e}")

# ---------------------------------------------------------------- THE GATE
print("\n3. THE GATE: do polygons genuinely TRADE, or does everything move together?")
model = build()
model.setConstraints(equilateral = KAPPA, edge = False, diagonal = [1, 2])
start = model.getFlatness().copy()
startSpread = float(np.std(start))
perPolygonStart = np.array([float(np.mean(start[p * 16:(p + 1) * 16])) for p in range(N)])
print(f"     start: mean {start.mean():.6f}  spread {startSpread:.6f}  "
      f"worst {start.min():.6f}")
check("the seed is genuinely UN-flat, so the ramp has work to do",
      start.mean() < 0.99 and startSpread > 1e-3,
      f"mean {start.mean():.6f}, spread {startSpread:.6f}")
print(f"     {'mean asked':>11} {'mean got':>10} {'spread':>9} {'worst':>9} {'cond':>9} {'|turn|':>9}")
history = []
for step in range(14):
    target = start.mean() + (0.995 - start.mean()) * (step + 1) / 14
    model.setFlatnessTarget(target)
    live = model.getFlatness()
    history.append((target, float(live.mean()), float(np.std(live)), float(live.min()),
                    model.constraintConditioning(), folded(model, N)))
    if step % 3 == 0 or step == 13:
        t, m, s, w, c, f = history[-1]
        print(f"     {t:>11.6f} {m:>10.6f} {s:>9.6f} {w:>9.6f} {c:>9.2e} {f:>9.2f}")

live = model.getFlatness()
perPolygonEnd = np.array([float(np.mean(live[p * 16:(p + 1) * 16])) for p in range(N)])
check("the mean advanced toward flat", live.mean() > start.mean() + 0.5 * (1.0 - start.mean()),
      f"{start.mean():.6f} -> {live.mean():.6f}")
check("the spread did NOT collapse -- the vertices are not moving in lockstep",
      float(np.std(live)) > 0.1 * startSpread,
      f"spread {startSpread:.6f} -> {float(np.std(live)):.6f}")
spread = float(np.std(perPolygonEnd))
check("and POLYGONS differ from one another -- this is the sharing",
      spread > 1e-4,
      f"per-polygon mean flatness spread {spread:.6f}, range "
      f"{perPolygonEnd.min():.5f} .. {perPolygonEnd.max():.5f}")
check("nothing folded", abs(folded(model, N) - 360.0) < 5.0,
      f"worst |turn| sum {folded(model, N):.3f} deg")

# ---------------------------------------------------------------- the plateau
print("\n4. the documented plateau, measured for the first time")
worst = float(live.min())
conditioning = model.constraintConditioning()
print(f"     after 14 rounds: mean {live.mean():.6f}  WORST selected vertex {worst:.6f}  "
      f"conditioning {conditioning:.2e}")
for extra in range(40):
    model.setFlatnessTarget(0.9999)
after = model.getFlatness()
print(f"     after 40 more at a fixed target: worst {float(after.min()):.6f}  "
      f"conditioning {model.constraintConditioning():.2e}")
check("the moments alone cannot place the WORST vertex at flat",
      float(after.min()) < 0.99999,
      f"worst plateaus at {float(after.min()):.6f} -- hence the handoff")

print("\n5. the handoff finishes what the moments cannot")
model.setConstraints(equilateral = KAPPA, edge = False, flatten = True)
begin = model.getFlatness().copy()
for step in range(12):
    blend = (step + 1) / 12
    model.setFlatTargets((1.0 - blend) * begin + blend * 0.999999)
    model.constraints.projectPositions(model.packing)
finished = model.getFlatness()
check("per-object rows drive every selected vertex flat", float(finished.min()) > 0.99999,
      f"worst {float(finished.min()):.8f} (moments left it at {float(after.min()):.6f})")
check("still simple", abs(folded(model, N) - 360.0) < 5.0,
      f"worst |turn| sum {folded(model, N):.3f} deg")
try:
    model.halveNumEdges()
    check("and halveNumEdges then ACCEPTS", True,
          f"n -> {int(np.diff(model.packing.startIndices)[0])}")
except ValueError as error:
    check("and halveNumEdges then ACCEPTS", False, str(error)[:90])

print(f"\n{passed} passed, {failed} failed")
sys.exit(1 if failed else 0)
