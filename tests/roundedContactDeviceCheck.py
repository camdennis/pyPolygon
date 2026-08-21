# UNVERIFIED(Cam)
"""Stage one of the CUDA port: device math against the numpy reference: frame, distance field, pieces, both root solvers."""
import os, subprocess, sys
import numpy as np
sys.path.insert(0, "/home/rdennis/Documents/Code/pyPolygon")
import roundedContact as rc

rng = np.random.default_rng(0)
# hard cases on purpose: random, plus near-double-root quartics built from repeated roots
cases = []
for _ in range(600):
    cases.append(rng.normal(size = 5))
for _ in range(400):
    a, b = rng.normal(size = 2)
    eps = 10.0 ** rng.uniform(-14, -4)
    poly = np.polynomial.polynomial.polyfromroots([a, a + eps, b, b - eps])[::-1]
    cases.append(poly / max(np.max(np.abs(poly)), 1e-300))
cases = np.asarray(cases)

stdin = "\n".join(" ".join(f"{x:.17g}" for x in row) for row in cases)
out = subprocess.run([os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cuda", "testRoundedContact")], input = stdin, capture_output = True,
                     text = True, timeout = 600)
if out.returncode != 0:
    print("device run failed:", out.stderr[:400]); sys.exit(1)
lines = out.stdout.split("\n")

def section(name):
    i = lines.index(name)
    j = i + 1
    while j < len(lines) and lines[j] and not lines[j].isupper():
        j += 1
    return lines[i + 1:j]

loop = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
rho = np.array([0.30, 0.15, 0.42, 0.22])
body = rc.bodyFromBackbone(loop, rho)

worst = 0.0
for row in section("FRAME"):
    k, cx, cy, r, s, tx, ty, hx, hy = (float(v) for v in row.split())
    k = int(k)
    worst = max(worst,
                abs(cx - body.center[k, 0]), abs(cy - body.center[k, 1]),
                abs(r - body.radius[k]), abs(s - body.sweep[k]),
                abs(tx - body.tail[k, 0]), abs(ty - body.tail[k, 1]),
                abs(hx - body.head[k, 0]), abs(hy - body.head[k, 1]))
print(f"corner frame   worst absolute gap {worst:.3e}")

worstD, mismatched = 0.0, 0
for row in section("DISTANCE"):
    x, y, d, kind, feature = row.split()
    point = np.array([[float(x), float(y)]])
    hostDistance = rc.signedDistance(point, body)
    _, _, hostKind, hostFeature = rc.nearestFeature(point, body)
    worstD = max(worstD, abs(float(d) - float(hostDistance[0])))
    if int(kind) != int(hostKind[0]) or int(feature) != int(hostFeature[0]):
        mismatched += 1
print(f"distance field worst absolute gap {worstD:.3e}   feature mismatches {mismatched}/121")

worstP = 0.0
for row in section("PIECE"):
    p, t, px, py = row.split()
    host = rc.evaluatePiece(body, int(p), float(t))
    worstP = max(worstP, abs(float(px) - host[0]), abs(float(py) - host[1]))
print(f"evaluatePiece  worst absolute gap {worstP:.3e}")

# root solvers: the contract is that the device must not MISS what the host keeps
missedQ = extraQ = 0
worstQ = 0.0
for row, c in zip(section("QUARTIC"), cases):
    values = [float(v) for v in row.split()[1:]]
    host = np.sort(np.asarray(rc._realQuarticRoots(c), dtype = float))
    device = np.sort(np.asarray(values))
    for r in host:
        if not len(device) or np.min(np.abs(device - r)) > 1e-6 * max(1.0, abs(r)):
            missedQ += 1
        else:
            worstQ = max(worstQ, np.min(np.abs(device - r)))
    extraQ += max(0, len(device) - len(host))
print(f"quarticRoots   host roots missed {missedQ}   extra {extraQ}   worst agreement {worstQ:.3e}")

missedT = 0
worstT = 0.0
for row, e in zip(section("TRIG"), cases):
    values = [float(v) for v in row.split()[1:]]
    host = np.sort(rc._solveTrig(tuple(e)))
    device = np.sort(np.asarray(values))
    for r in host:
        if not len(device) or np.min(np.abs(device - r)) > 1e-6:
            missedT += 1
        else:
            worstT = max(worstT, np.min(np.abs(device - r)))
print(f"solveTrig      host switches missed {missedT}   worst agreement {worstT:.3e}")


# UNVERIFIED(Cam)
def checkSpans(binary, trials = 40):
    """Stage two: the span walk, against ``roundedContact.spans``.

    The device WALKS crossings by repeated minimum-search instead of sorting them, so this is the
    check that the walk reproduces the sort -- including the dedup, where both must keep the FIRST of
    a cluster within the tolerance."""
    rng = np.random.default_rng(11)
    loopA = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    rhoA = np.array([0.30, 0.15, 0.42, 0.22])
    bodyA = rc.bodyFromBackbone(loopA, rhoA)
    worst, mismatched, compared = 0.0, 0, 0
    subCompared, subUncovered, subWrong = [0], [0], [0]
    subHostTotal, subDeviceTotal = [0], [0]
    pairEnergyGap, pairGradGap, pairCount = [0.0], [0.0], [0]
    for _ in range(trials):
        shift = rng.uniform(0.25, 1.05, size = 2)
        angle = rng.uniform(0.0, 2.0 * np.pi)
        size = rng.uniform(0.6, 1.2)
        turns = angle + 2.0 * np.pi * np.arange(4) / 4.0 + np.pi / 4.0
        loopB = shift + size * np.stack([np.cos(turns), np.sin(turns)], axis = 1)
        previousIndex, nextIndex = np.roll(np.arange(4), 1), np.roll(np.arange(4), -1)
        rhoB = rc.rg.maxRho(loopB, previousIndex, nextIndex) * rng.uniform(0.2, 0.8, size = 4)
        arguments = [f"{v:.17g}" for v in loopB.reshape(-1)] + [f"{v:.17g}" for v in rhoB]
        result = subprocess.run([binary] + arguments, input = "", capture_output = True,
                                text = True, timeout = 600)
        lines = result.stdout.split("\n")
        if "SPAN" not in lines:
            continue
        start = lines.index("SPAN") + 1
        device = []
        for row in lines[start:]:
            if not row or row.replace(" ", "").isalpha():
                break
            p, lo, hi = row.split()
            device.append((int(p), float(lo), float(hi)))
        bodyB = rc.bodyFromBackbone(loopB, rhoB)
        piece, low, high = rc.spans(bodyA, bodyB)
        host = list(zip(piece.tolist(), low.tolist(), high.tolist()))
        compared += 1
        if len(host) != len(device):
            mismatched += 1
            continue
        for (hp, hl, hh), (dp, dl, dh) in zip(sorted(host), sorted(device)):
            if hp != dp:
                mismatched += 1
                break
            worst = max(worst, abs(hl - dl), abs(hh - dh))
        subDevice = []
        if "SUBSTRETCH" in lines:
            for row in lines[lines.index("SUBSTRETCH") + 1:]:
                if not row or row.replace(" ", "").isalpha():
                    break
                sp, sl, sh, sk, sf = row.split()
                subDevice.append((int(sp), float(sl), float(sh), int(sk), int(sf)))
        hostPiece, hostLow, hostHigh, hostKind, hostFeature, _ = rc.substretches(bodyA, bodyB)
        subHost = list(zip(hostPiece.tolist(), hostLow.tolist(), hostHigh.tolist(),
                           hostKind.tolist(), hostFeature.tolist()))
        subCompared[0] += 1
        # A sub-stretch is COVERED if the device labels the host's midpoint with the same winner.
        # Comparing the lists elementwise would fail on a harmless extra breakpoint, which by this
        # module's rule costs nothing -- only a mislabelled interval matters.
        for hp, hl, hh, hk, hf in subHost:
            mid = 0.5 * (hl + hh)
            match = [d for d in subDevice if d[0] == hp and d[1] - 1e-9 <= mid <= d[2] + 1e-9]
            if not match:
                subUncovered[0] += 1
            elif match[0][3] != hk or match[0][4] != hf:
                subWrong[0] += 1
        subHostTotal[0] += len(subHost)
        subDeviceTotal[0] += len(subDevice)

        # Stages four and five: the energy and dE/d(body arrays) for the ordered pair.
        if "PAIR" in lines:
            at = lines.index("PAIR") + 1
            deviceEnergy = float(lines[at])
            deviceA = np.array([float(v) for v in lines[at + 1].split()])
            deviceB = np.array([float(v) for v in lines[at + 2].split()])
            partition = rc.substretches(bodyA, bodyB)[:5]
            hostEnergy, gA, gB = rc.pairEnergyBodyGradient(bodyA, bodyB, partition, 1.7)
            pairEnergyGap[0] = max(pairEnergyGap[0],
                                   abs(deviceEnergy - hostEnergy) / max(abs(hostEnergy), 1e-300))
            scaleBoth = max(np.max(np.abs(gA.flat())), np.max(np.abs(gB.flat())), 1e-300)
            pairGradGap[0] = max(pairGradGap[0],
                                 np.max(np.abs(deviceA - gA.flat())) / scaleBoth,
                                 np.max(np.abs(deviceB - gB.flat())) / scaleBoth)
            pairCount[0] += 1
    print(f"spans          {compared} pairs compared, {mismatched} with a differing span set, "
          f"worst endpoint gap {worst:.3e}")
    print(f"substretches   {subHostTotal[0]} host intervals, {subDeviceTotal[0]} device; "
          f"{subUncovered[0]} uncovered, {subWrong[0]} with the wrong winner")
    print(f"pair energy    {pairCount[0]} pairs, worst relative gap {pairEnergyGap[0]:.3e}")
    print(f"pair gradient  worst relative gap {pairGradGap[0]:.3e}")


checkSpans(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "cuda", "testRoundedContact"))
