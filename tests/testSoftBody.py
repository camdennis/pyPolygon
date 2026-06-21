import numpy as np

from packing import Packing
from softBody import backboneArea, backboneEdgeLengths, eqSoftBodyEnergyForce

def testEqSoftBodyForceMatchesFiniteDifference():
    pk = Packing.fromSinglePolygon(6, rng = 0, targetEdgeLength = 0.3, targetArea = 0.2)
    kEdge, kArea = 1.5, 2.0
    _, force = eqSoftBodyEnergyForce(pk, kEdge, kArea)
    base = pk.positions.copy()
    eps = 1e-6
    numerical = np.zeros_like(force)
    for i in range(pk.positions.size):
        pk.positions[i] = base[i] + eps
        eP, _ = eqSoftBodyEnergyForce(pk, kEdge, kArea)
        pk.positions[i] = base[i] - eps
        eM, _ = eqSoftBodyEnergyForce(pk, kEdge, kArea)
        pk.positions[i] = base[i]
        numerical[i] = -(eP - eM) / (2 * eps)              # numerical force = -dE/dx
    assert np.max(np.abs(numerical - force)) < 1e-6
