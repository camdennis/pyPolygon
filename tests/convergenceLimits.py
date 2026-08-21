"""Why CG costs what it costs, and how far down max|F| can meaningfully go.

Two measurements that together say when a relaxation is FINISHED:

[A] CG cost structure. CG's price is not one force evaluation per step -- the strong-Wolfe line
    search calls the objective several times per step, and under constraints EVERY trial point pays a
    SHAKE retraction plus two tangent projections. This counts the actual calls so the per-step cost
    is attributable rather than guessed.

[B] The force noise floor. The mollified energy is a near-cancellation of edge-pair panels and its
    gradient carries a 1/X1^2 conditioning floor, so the computed force has an absolute error floor
    that is independent of how well converged the configuration is. Below that floor max|F| is
    measuring arithmetic noise, not physics, and no minimizer can make progress. Two independent
    probes of it:

      - AGREEMENT: |force_cuda - force_numpy| at the same configuration. Two honest evaluations of
        the same quantity differ by roughly the error in each.
      - RESPONSE: perturb positions by eps and watch |f(x+eps) - f(x)|. For a smooth energy this
        falls linearly with eps until it hits the noise floor, then flattens. The knee is the floor.

Run:  python tests/convergenceLimits.py
"""

import os
import sys
import time
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cudaOverlap
import minimize
from energies import plummerOverlapExact, selfRepulsionEnergyForce
from model import Model
from softBody import eqSoftBodyEnergyForce

warnings.filterwarnings("ignore")


def buildModel(numPolygons = 32, numVertices = 10, phi = 1.0, softening = 0.05, seed = 42,
               constrained = True):
    model = Model(N = numPolygons, n = numVertices, seed = seed)
    model.generateEquilateralPolygons(phi = phi, kappa = 4.0)
    model.setBiPerimeter()
    model.setSpringConstants()
    model.setModelType("mollified")
    model.setSofteningFraction(softening)
    if constrained:
        model.setConstraints()
    return model


def countingForceEnergy(model, counter):
    """Wrap the model's force routine so every call is counted."""
    def wrapped(packing):
        counter[0] += 1
        return model._forceEnergy(packing)
    return wrapped


def reportCgCost(steps = 60):
    """[A] force evaluations per CG step, constrained and unconstrained."""
    print("\n[A] CG cost structure")
    print(f"      {'run':<24} {'steps':>6} {'evals':>7} {'evals/step':>11} {'wall s':>8} {'ms/step':>9}")
    for label, constrained in (("unconstrained", False), ("constrained", True)):
        model = buildModel(constrained = constrained)
        model.minimizeFIRE(maxSteps = 300, fThreshold = 1e-12,
                           dtMax = 0.30 if constrained else 0.03)
        counter = [0]
        start = time.perf_counter()
        _, taken, _ = minimize.minimizeCG(
            model.packing, countingForceEnergy(model, counter), maxSteps = steps,
            fThreshold = 1e-14, constraints = model.constraints, progress = False)
        elapsed = time.perf_counter() - start
        print(f"      {label:<24} {taken:6d} {counter[0]:7d} {counter[0] / max(taken, 1):11.2f} "
              f"{elapsed:8.2f} {elapsed / max(taken, 1) * 1e3:9.1f}")


def reportNoiseFloor():
    """[B] the absolute error floor of the computed force."""
    print("\n[B] force noise floor")
    model = buildModel(constrained = False)
    model.minimizeFIRE(maxSteps = 600, fThreshold = 1e-12, dtMax = 0.03)
    packing = model.packing
    sigma = model.sigma
    print(f"      relaxed to max|F| = {model.getMaxUnbalancedForce():.3e}")

    # Probe 1 -- two independent evaluations of the same gradient.
    gradGpu = cudaOverlap.plummerOverlapCuda(packing, sigma, packing.targetArea,
                                             packing.targetPerimeter)[1].reshape(-1)
    gradCpu = np.asarray(plummerOverlapExact(packing, sigma)[1]).reshape(-1)
    agreement = np.abs(gradGpu - gradCpu).max()
    print(f"      cuda vs numpy overlap gradient : {agreement:.3e} absolute  "
          f"({agreement / np.abs(gradCpu).max():.2e} relative)")

    # Probe 2 -- response to a shrinking perturbation.
    base = packing.positions.copy()
    _, force0 = model._forceEnergy(packing)
    rng = np.random.default_rng(0)
    direction = rng.standard_normal(base.size)
    direction /= np.linalg.norm(direction)
    print(f"      {'eps':>10} {'|f(x+eps) - f(x)|':>20} {'ratio to eps':>14}")
    for exponent in range(-6, -17, -2):
        eps = 10.0 ** exponent
        packing.positions[:] = base + eps * direction
        _, force = model._forceEnergy(packing)
        delta = np.abs(force - force0).max()
        print(f"      {eps:10.0e} {delta:20.3e} {delta / eps:14.2e}")
    packing.positions[:] = base
    print("      (linear in eps while real; the plateau at small eps IS the floor)")


def main():
    if not cudaOverlap.isAvailable():
        print("CUDA library not available -- build it with 'make -C cuda libplummer.so'")
        return 1
    print("\nConvergence limits (N=32, n=10, phi=1.0, mollified sigma=0.05*edge)")
    reportCgCost()
    reportNoiseFloor()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
