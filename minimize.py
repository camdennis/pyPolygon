"""FIRE and gradient-descent minimizers (build step 1, reused at step 7).

Both take the packing and a ``forceEnergy`` callable

    forceEnergy(packing) -> (energy: float, force: flat (2N,) array, == -dE/dr)

and relax ``packing.positions`` in place. Hiding the energy model behind this
callable lets step 1 pass eqSoftBody while the later collision model passes its
own force routine (which can refresh neighbors / intersections on each call).

FIRE is the velocity-Verlet variant of Bitzek et al. (2006), mirroring the CUDA
reference's FIRE.h: half-kick / drift / half-kick, then mix the velocity toward
the force and adapt dt / alpha.

Positions are wrapped after every step by ``box.wrapIntoCell`` -- governed by the
box, not a flag. A free-space packing (box=None, the eqSoftBody shape-build) makes
it a no-op; a periodic box wraps into the area-1 cell. So wrapping is always applied
yet correct for both the free-space build (step 1) and the periodic packing (step 7).
"""

import numpy as np

from box import wrapIntoCell


def maxForceMagnitude(force):
    """Largest per-vertex force magnitude in a flat (2N,) force array."""
    f = force.reshape(-1, 2)
    return float(np.sqrt(np.einsum("ij,ij->i", f, f)).max())

def minimizeFIRE(
    packing,
    forceEnergy,
    maxSteps = 100000,
    fThreshold = 1e-10,
    dt = 0.01,
    dtMax = 0.1,
    alphaStart = 0.1,
    fInc = 1.1,
    fDec = 0.5,
    fAlpha = 0.99,
    nMin = 5,
):
    """Relax packing.positions with FIRE. Returns (energy, steps, converged).

    ``converged`` is True when the max per-vertex force drops below fThreshold.
    On return the final force/energy are also stored on the packing.
    """
    v = packing.velocities
    v[:] = 0.0
    alpha = alphaStart
    nPos = 0
    energy, f = forceEnergy(packing)

    for step in range(maxSteps):
        if maxForceMagnitude(f) < fThreshold:
            packing.force[:] = f
            packing.energy = energy
            return energy, step, True

        # velocity Verlet: half-kick, drift, recompute force, half-kick
        v += 0.5 * dt * f
        packing.positions += dt * v
        packing.positions[:] = wrapIntoCell(packing.positions, packing.box)
        energy, f = forceEnergy(packing)
        v += 0.5 * dt * f

        # FIRE: mix velocity toward the force direction, then adapt dt / alpha
        power = float(np.dot(v, f))
        fnorm = float(np.linalg.norm(f))
        vnorm = float(np.linalg.norm(v))
        if fnorm > 1e-300:
            v[:] = (1.0 - alpha) * v + alpha * (vnorm / fnorm) * f
        if power > 0.0:
            nPos += 1
            if nPos > nMin:
                dt = min(dt * fInc, dtMax)
                alpha *= fAlpha
        else:
            nPos = 0
            dt *= fDec
            alpha = alphaStart
            v[:] = 0.0

    packing.force[:] = f
    packing.energy = energy
    return energy, maxSteps, False

def minimizeGD(
    packing,
    forceEnergy,
    maxSteps = 100000,
    fThreshold = 1e-10,
    step = 0.001,
):
    """Relax packing.positions with fixed-step gradient descent.

    Moves along the force (= -dE/dr) by ``step`` each iteration. Returns
    (energy, steps, converged); final force/energy are stored on the packing.
    """
    energy, f = forceEnergy(packing)
    for i in range(maxSteps):
        if maxForceMagnitude(f) < fThreshold:
            packing.force[:] = f
            packing.energy = energy
            return energy, i, True
        packing.positions += step * f
        packing.positions[:] = wrapIntoCell(packing.positions, packing.box)
        energy, f = forceEnergy(packing)
    packing.force[:] = f
    packing.energy = energy
    return energy, maxSteps, False