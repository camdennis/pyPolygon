"""Export test vectors (inputs + expected outputs) from the verified energies.py
reference, for the CUDA harnesses to validate against.

  cl2.csv          x, Cl2(x)                          -> testPlummer (Track A)
  sharpPacking.csv positions(72), area, gradient(72)  -> testSharpPacking (Track B)

Run whenever the reference changes.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import energies as E
from packing import Packing

OUT = os.path.join(HERE, "vectors")
os.makedirs(OUT, exist_ok = True)
rs = np.random.default_rng(12345)


def dump(name, cols, header):
    arr = np.column_stack([np.asarray(c, dtype = float).ravel() for c in cols])
    np.savetxt(os.path.join(OUT, name + ".csv"), arr, delimiter = ",", header = header, comments = "")
    print(f"  {name}.csv: {arr.shape[0]} rows  ({header})")


# ---- Track A: cl2 over the reduction branches (negative, >pi, >2pi, near 0/pi) ----
x = np.concatenate([
    np.linspace(-12.0, 12.0, 2001),
    np.array([0.0, np.pi, -np.pi, 2 * np.pi, -2 * np.pi, 1e-8, -1e-8, np.pi - 1e-9, 4 * np.pi + 0.3]),
])
dump("cl2", [x, E._cl2(x)], "x,expected")


# ---- Track A: tCoreReal(xi, al, be, sg) = -2 J_arcsinh, over the realistic edge-frame regime ----
# al = slope mu (any real), be = intercept sigma*nu (small), sg = sigma, xi = across-coordinate.
K = 4000
xi = rs.uniform(-2.0, 2.0, K)
al = rs.uniform(-5.0, 5.0, K)
sg = 10.0 ** rs.uniform(-2.0, -0.5, K)     # sigma in [1e-2, ~0.3]
be = rs.uniform(-0.5, 0.5, K) * sg          # so nu = be/sg stays O(1)
dump("tCoreReal", [xi, al, be, sg, E._tCoreReal(xi, al, be, sg)], "xi,al,be,sg,expected")

# master composites m1 (elementary), m2 (energy panel), m1Prime (gradient W1) over the same regime.
dump("m1", [xi, al, be, sg, E._m1(xi, al, be, sg)], "xi,al,be,sg,expected")
dump("m2", [xi, al, be, sg, E._m2(xi, al, be, sg)], "xi,al,be,sg,expected")
dump("m1Prime", [xi, al, be, sg, E._m1Prime(xi, al, be, sg)], "xi,al,be,sg,expected")

# elementary log primitives lam0/lam1: q(s)=a s^2+b s+c on s in [0,1], with a>0 and 4ac-b^2>0.
sA = rs.uniform(0.0, 1.0, K)
aq = rs.uniform(0.5, 3.0, K)
cq = rs.uniform(0.5, 3.0, K)
bq = rs.uniform(-0.9, 0.9, K) * 2.0 * np.sqrt(aq * cq)   # |b| < 2 sqrt(ac) => 4ac-b^2 > 0
dump("lam0", [sA, aq, bq, cq, E._lam0(sA, aq, bq, cq)], "s,a,b,c,expected")
dump("lam1", [sA, aq, bq, cq, E._lam1(sA, aq, bq, cq)], "s,a,b,c,expected")


# ---- Track A: single-pair mollified energy. n=6 hexagon pairs, half near-parallel (exercises the
# bridge). Row: loopA(12), loopB(12), sigma(1), energy(1) = 26 cols. Validates plummerPairEnergyExact.
N_A = 6


def hexA(cx, cy, radius, phase):
    a = phase + 2 * np.pi * np.arange(N_A) / N_A
    return np.column_stack([cx + radius * np.cos(a), cy + radius * np.sin(a)])


pairRows = []
gradRows = []
k = 0
while len(pairRows) < 200:
    phaseA = rs.uniform(0, 1)
    A = hexA(0.0, 0.0, 1.0, phaseA)
    phaseB = phaseA + rs.uniform(-0.03, 0.03) if k % 2 == 0 else rs.uniform(0, 1)  # half near-parallel
    off = rs.uniform(-1.1, 1.1, 2)
    B = hexA(off[0], off[1], rs.uniform(0.8, 1.2), phaseB)
    sigma = 10.0 ** rs.uniform(-1.5, -0.7)     # [~0.03, ~0.2]
    e = E.plummerPairEnergyExact(A, B, sigma)
    k += 1
    if abs(e) < 1e-9:
        continue
    gA = E.plummerPairGradientExact(A, B, sigma)      # dA_cap/d(A's vertices), (6, 2)
    pairRows.append(np.concatenate([A.ravel(), B.ravel(), [sigma], [e]]))
    gradRows.append(np.concatenate([A.ravel(), B.ravel(), [sigma], gA.ravel()]))

parr = np.array(pairRows)
np.savetxt(os.path.join(OUT, "plummerPair.csv"), parr, delimiter = ",",
           header = "loopA(12),loopB(12),sigma,energy", comments = "")
print(f"  plummerPair.csv: {parr.shape[0]} rows (n=6 pairs; 26 cols)")

garr = np.array(gradRows)
np.savetxt(os.path.join(OUT, "plummerGrad.csv"), garr, delimiter = ",",
           header = "loopA(12),loopB(12),sigma,gradA(12)", comments = "")
print(f"  plummerGrad.csv: {garr.shape[0]} rows (n=6 pairs; 37 cols)")


# ---- Track B: whole-packing sharp overlap. N=6 hexagons, clean overlaps ----
N_POLY, N_VERT = 6, 6
V = N_POLY * N_VERT                         # 36 vertices; startIndices [0,6,...,36] hardcoded in the harness


def hexagon(cx, cy, radius, phase):
    a = phase + 2 * np.pi * np.arange(N_VERT) / N_VERT
    return np.column_stack([cx + radius * np.cos(a), cy + radius * np.sin(a)])


def sharpPacking(centers, radii, phases):
    """positions, total sharp overlap area, and full vertex gradient for one N=6 packing, or None
    if it has no overlap or degenerate (no-follower) topology."""
    pos = np.concatenate([hexagon(centers[p, 0], centers[p, 1], radii[p], phases[p]).ravel()
                          for p in range(N_POLY)])
    pk = Packing(pos, list(range(0, V + 1, N_VERT)), box = None)
    E.updateIntersections(pk)
    if len(pk.intersections) == 0:
        return None
    E.updateFollowers(pk)
    if np.any(pk.followerIndices < 0):
        return None
    E.updateOverlapArea(pk)
    E.updateOverlapGradient(pk)
    nInter = len(pk.intersections)          # all-pairs intersection count (candidate-set reference)
    return pos, pk.overlapArea, pk.overlapGradient.ravel(), nInter


rows = []
while len(rows) < 60:
    res = sharpPacking(rs.uniform(-1.0, 1.0, (N_POLY, 2)), rs.uniform(0.6, 0.85, N_POLY),
                       rs.uniform(0, 1, N_POLY))
    if res is None or abs(res[1]) < 1e-4:
        continue
    pos, area, grad, nInter = res
    rows.append(np.concatenate([pos, [area], grad, [nInter]]))

arr = np.array(rows)
np.savetxt(os.path.join(OUT, "sharpPacking.csv"), arr, delimiter = ",",
           header = "positions(72),area,gradient(72),nIntersections", comments = "")
print(f"  sharpPacking.csv: {arr.shape[0]} rows (N=6 hexagons; 146 cols)")

print("done.")
