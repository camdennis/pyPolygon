import numpy as np

from packing import Packing
from softBody import eqSoftBodyEnergyForce, backboneEdgeLengths, backboneArea
from minimize import minimizeFIRE, minimizeGD

def regularPolygon(n, edgeLength, center = (0.5, 0.5)):
    """A regular n-gon (the eqSoftBody global minimum) and its area."""
    R = edgeLength / (2 * np.sin(np.pi / n))
    ang = 2 * np.pi * np.arange(n) / n
    verts = np.stack([center[0] + R * np.cos(ang), center[1] + R * np.sin(ang)], axis = 1)
    area = 0.5 * n * R ** 2 * np.sin(2 * np.pi / n)
    return verts, area

def testFIREReturnsRegularHexagon():
    n, L = 6, 0.3
    verts, area = regularPolygon(n, L)
    pk = Packing(verts.reshape(-1), [0, n], targetEdgeLength = L, targetArea = area)
    pk.positions += np.random.default_rng(1).normal(0, 0.02, pk.positions.size)
    fe = lambda p: eqSoftBodyEnergyForce(p, 1.0, 1.0)
    energy, steps, converged = minimizeFIRE(pk, fe, maxSteps = 50000)
    assert converged and energy < 1e-10
    assert backboneEdgeLengths(pk).std() < 1e-5
    assert abs(backboneArea(pk)[0] - area) < 1e-6

def testFIRERandomStarToEquilateral():
    pk = Packing.fromSinglePolygon(7, rng = 0, targetEdgeLength = 0.25, targetArea = 0.15)
    fe = lambda p: eqSoftBodyEnergyForce(p, 1.0, 1.0)
    energy, steps, converged = minimizeFIRE(pk, fe, maxSteps = 50000)
    assert converged
    assert backboneEdgeLengths(pk).std() < 1e-5                     # equilateral
    assert abs(backboneArea(pk)[0] - 0.15) < 1e-5                   # area on target

def testGradientDescentDescends():
    n, L = 6, 0.3
    verts, area = regularPolygon(n, L)
    pk = Packing(verts.reshape(-1), [0, n], targetEdgeLength = L, targetArea = area)
    pk.positions += np.random.default_rng(3).normal(0, 0.02, pk.positions.size)
    fe = lambda p: eqSoftBodyEnergyForce(p, 1.0, 1.0)
    e0, _ = fe(pk)
    energy, steps, converged = minimizeGD(pk, fe, maxSteps = 20000, step = 0.02)
    assert energy < e0 * 1e-2                                       # large descent
