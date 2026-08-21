"""Does ``setContainerStiffness`` reach the contact law that is actually live?

The wall multiplier is stored twice -- ``kContainer`` for ``sharp`` / ``mollified`` / ``softDepth``,
``depthWallStiffness`` for the exact-distance ``depth`` law -- and ``setContainerStiffness`` used to
write only the first. On the depth tier that made it a SILENT no-op: the call returned ``self``, the
attribute it set was never read, and the wall stayed at 1.0. Measured before the fix, on 11 polygons,
``setContainerStiffness(100)`` moved the depth energy by 1.08e-19 and the force by 8.13e-20.

That is the worst knob in the package to lose silently. 1.0 is the value whose measured consequence
is a packing that reports itself jammed with nothing touching -- 100% of the contact energy in wall
penetration and 17 vertices outside the box.

So the checks are about ROUTING, and each one is posed on the energy rather than on the attribute:
an attribute that no kernel reads is exactly what the bug was.

Run: python tests/containerStiffnessCheck.py
"""

# UNVERIFIED(Cam)

import os
import sys
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import Model

warnings.filterwarnings("ignore")

passed, failed = 0, 0

def check(name, condition, detail = ""):
    global passed, failed
    if condition:
        passed += 1
        print(f"  PASS  {name}   {detail}")
    else:
        failed += 1
        print(f"  FAIL  {name}   {detail}")

def build(phi = 0.5, N = 11, n = 32, seed = 42):
    """Dense enough that polygons overlap the wall -- with nothing touching, every stiffness agrees."""
    model = Model(N = N, n = n, seed = seed)
    model.generateEquilateralPolygons(phi = phi, kappa = 4.0)
    model.syncTargetAreas()
    model.syncTargetPerimeters()
    model.addShape(np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 0.0]]))
    model.pinVertices(np.arange(model.getNumVertices())[-4:])
    model.setBoundaryConditions("fixed")
    return model

def energyAt(model):
    model.calcForceEnergy()
    return model.getEnergy()

print("\n1. THE DEPTH TIER -- the one the call used to miss entirely")
model = build()
model.setModelType("depth")
base = energyAt(model)
model.setContainerStiffness(100.0)
raised = energyAt(model)
check("the energy moves", abs(raised - base) > 1e-12 * max(abs(base), 1e-30),
      f"{base:.9e} -> {raised:.9e}   (x{raised / base:.3f})")
check("and it lands on depthWallStiffness, not kContainer",
      model.depthWallStiffness == 100.0 and model.kContainer != 100.0,
      f"depthWallStiffness {model.depthWallStiffness}, kContainer {model.kContainer}")

reference = build()
reference.setDepthContact(stiffness = 1.0, wallStiffness = 100.0)
check("EXACTLY what setDepthContact(wallStiffness = 100) gives",
      energyAt(reference) == raised,
      f"{energyAt(reference):.15e} vs {raised:.15e}")

print("\n2. it is LINEAR in the multiplier -- fitted, not assumed")
# THE GEOMETRY NEVER MOVES between these calls, so the true contact energy is c(k) = P + k W with BOTH
# P and W constants. Fitting that line over seven stiffnesses tests the wiring against a construction
# that never calls the split being tested -- unlike subtracting getPairContactEnergy, which returns
# c - k w by definition and would make any such ratio circular. That circularity is how the first
# version of this check "passed" a 0.53% error.
model = build()
model.setModelType("depth")
stiffnesses = np.array([1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0])
contact = []
for k in stiffnesses:
    model.setContainerStiffness(k)
    model.calcForceEnergy()
    contact.append(model.getContactEnergy())
contact = np.array(contact)
design = np.vstack([np.ones_like(stiffnesses), stiffnesses]).T
(bodyPart, wallUnit), *_ = np.linalg.lstsq(design, contact, rcond = None)
worst = float(np.abs(contact - (bodyPart + stiffnesses * wallUnit)).max())
check("the total contact energy is affine in the stiffness, to roundoff", worst < 1e-15,
      f"max |c - (P + k W)| = {worst:.2e} over k = 1 .. 100")
check("with a positive body part and a positive wall unit", bodyPart > 0.0 and wallUnit > 0.0,
      f"P = {bodyPart:.9e}, W = {wallUnit:.9e}")

print("\n3. THE PAIR / WALL SPLIT AGAINST THAT FIT -- a PRE-EXISTING defect, not the routing")
# getPairContactEnergy claims to return the body term alone, which cannot depend on the wall stiffness
# at fixed geometry. It does. The wall amount it subtracts is short of the fitted W, so the leftover
# leaks into what is reported as body contact -- and getExcessEnergy, the density controller's entire
# control variable, is that number divided by a scale. The error grows with the stiffness, so it is
# worst exactly where the stiffness is worth raising.
model.setContainerStiffness(1.0)
model.calcForceEnergy()
atOne = model.getPairContactEnergy()
model.setContainerStiffness(100.0)
model.calcForceEnergy()
atHundred = model.getPairContactEnergy()
subtracted = (model.getContactEnergy() - atHundred) / 100.0
check("the body term does not depend on the wall stiffness",
      abs(atHundred - atOne) < 1e-12 * abs(atOne),
      f"k = 1: {atOne:.9e}   k = 100: {atHundred:.9e}   "
      f"({100.0 * (atHundred - atOne) / atOne:+.2f}%)")
check("the wall amount it subtracts equals the fitted wall unit",
      abs(subtracted - wallUnit) < 1e-9 * wallUnit,
      f"subtracts {subtracted:.9e} against W = {wallUnit:.9e}, short by "
      f"{100.0 * (1.0 - subtracted / wallUnit):.3f}%")

print("\n4. THE AREA TIERS still route to kContainer")
for tier in ("sharp", "mollified"):
    model = build()
    if tier == "mollified":
        model.setSofteningFraction(0.05)
    model.setModelType(tier)
    before = energyAt(model)
    depthBefore = model.depthWallStiffness
    model.setContainerStiffness(100.0)
    after = energyAt(model)
    check(f"{tier}: the energy moves", abs(after - before) > 1e-15,
          f"{before:.6e} -> {after:.6e}")
    check(f"{tier}: lands on kContainer and leaves the depth multiplier alone",
          model.kContainer == 100.0 and model.depthWallStiffness == depthBefore,
          f"kContainer {model.kContainer}, depthWallStiffness {model.depthWallStiffness}")

print("\n5. switching the tier does NOT carry the value across")
model = build()
model.setModelType("sharp")
model.setContainerStiffness(100.0)
model.setModelType("depth")
check("depth keeps its own default after a sharp-tier setting",
      model.depthWallStiffness == 1.0,
      f"depthWallStiffness {model.depthWallStiffness} (kContainer {model.kContainer}) -- "
      f"per-tier by design, so set it AFTER choosing the law")

print("\n6. a non-positive stiffness refuses on every tier")
for tier in ("sharp", "depth"):
    model = build()
    model.setModelType(tier)
    try:
        model.setContainerStiffness(0.0)
        check(f"{tier}: setContainerStiffness(0) raises", False, "accepted it")
    except ValueError as error:
        check(f"{tier}: setContainerStiffness(0) raises", True, str(error)[:60])

print(f"\n{passed} passed, {failed} failed")
sys.exit(1 if failed else 0)
