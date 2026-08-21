"""Does the area-target feasibility guard fire when, and only when, the targets are impossible?

Reconstructed from a real failure. A cascade at N = 2 died with every position NaN; the dump showed
area targets of 0.544414 and 0.426973 inside a container of area 1 -- phi 0.971386 against a PROVED
ceiling of 0.500 for two squares. Nothing in the run said so. The retraction merely stopped converging,
which reads as a budget problem, and the traceback landed three layers away on a non-finite force.

    python tests/densityFeasibleCheck.py
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
import records


def container(model, side = 1.0):
    """Add a pinned square wall and switch to fixed boundaries."""
    half = 0.5 * side
    loop = np.array([[0.5 - half, 0.5 - half], [0.5 - half, 0.5 + half],
                     [0.5 + half, 0.5 + half], [0.5 + half, 0.5 - half]])
    model.addShape(loop)
    model.setBoundaryConditions("fixed")
    return model


def model(N = 2, n = 16, phi = 0.25):
    m = pp.Model(N = N, n = n, seed = 11)
    m.generateEquilateralPolygons(phi = phi, kappa = 4.0)
    return container(m)


def main():
    checks, ok = [], True

    print("the published ceiling has ONE definition")
    for n in (2, 5):
        side = records.bestKnownSide(n)
        expected = n / side ** 2
        got = records.maximumDensity(n)
        good = abs(got - expected) < 1e-12
        print(f"  N = {n}: maximumDensity {got:.6f}  = N / s^2 {expected:.6f}   {'ok' if good else 'FAIL'}")
        ok = ok and good
    checks.append(("maximumDensity matches N / s(N)^2", ok))

    print("\nTHE REAL FAILURE, replayed: the exact targets from the dump")
    m = model()
    m.packing.targetArea[0] = 0.544414
    m.packing.targetArea[1] = 0.426973
    with warnings.catch_warnings(record = True) as caught:
        warnings.simplefilter("always")
        asked, ceiling = m.checkDensityFeasible()
    fired = any("MORE THAN CAN EXIST" in str(w.message) for w in caught)
    print(f"  asked {asked:.6f}   ceiling {ceiling:.6f}   warned {fired}")
    good = fired and abs(asked - 0.971386) < 1e-5 and abs(ceiling - 0.5) < 1e-12
    checks.append(("infeasible targets warn, with the right numbers", good))

    print("\na FEASIBLE density must stay silent -- a guard that always fires is noise")
    m = model()
    m.packing.targetArea[0] = 0.20
    m.packing.targetArea[1] = 0.15
    with warnings.catch_warnings(record = True) as caught:
        warnings.simplefilter("always")
        asked, ceiling = m.checkDensityFeasible()
    quiet = not any("MORE THAN CAN EXIST" in str(w.message) for w in caught)
    print(f"  asked {asked:.6f}   ceiling {ceiling:.6f}   silent {quiet}")
    checks.append(("feasible targets do not warn", quiet and asked < ceiling))

    print("\nno container -> nothing to measure against, and no crash")
    m = pp.Model(N = 2, n = 8, seed = 3)
    m.generateEquilateralPolygons(phi = 0.2, kappa = 4.0)
    asked, ceiling = m.checkDensityFeasible()
    print(f"  asked {asked}   ceiling {ceiling}")
    checks.append(("no container returns (None, None)", asked is None and ceiling is None))

    print("\nthe container's own NEGATIVE area must not be counted as demand")
    m = model()
    signed = np.asarray(m.packing.targetArea, dtype = float)
    asked, _ = m.checkDensityFeasible(warn = False)
    naive = float(signed.sum())
    print(f"  wall target {signed[-1]:+.6f}   phi asked {asked:.6f}   "
          f"(naive signed sum would be {naive:+.6f})")
    checks.append(("wall excluded from the demand", asked > 0.0 and abs(asked - naive) > 1e-9))

    print()
    for name, good in checks:
        print(f"{name}: {'PASS' if good else 'FAIL'}")
    passed = sum(1 for _, good in checks if good)
    print(f"\n{passed}/{len(checks)} checks passed")
    return passed == len(checks)


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
