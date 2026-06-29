"""Illustration: a rounded polygon with every roundify symbol labeled.

Draws a regular pentagon's backbone + rounded boundary (straight runs between kiss
points + corner arcs), and fully annotates one corner with v_k, z_k, a_k^-, a_k^+,
rho, t, theta, psi.

Run:  python examples/roundifyDiagram.py   ->  roundifyDiagram.png at the repo root.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from packing import Packing
from geometry import cornerGeometry


def cornerArc(z, rho, aMinus, aPlus, vertex, num = 80):
    """Sample the rounded-corner arc from aMinus to aPlus (the one bulging toward vertex)."""
    a0 = np.arctan2(aMinus[1] - z[1], aMinus[0] - z[0])
    a1 = np.arctan2(aPlus[1] - z[1], aPlus[0] - z[0])
    amid = np.arctan2(vertex[1] - z[1], vertex[0] - z[0])
    dccw = (a1 - a0) % (2 * np.pi)
    if (amid - a0) % (2 * np.pi) <= dccw:
        ang = a0 + np.linspace(0.0, dccw, num)
    else:
        ang = a0 - np.linspace(0.0, 2 * np.pi - dccw, num)
    return z[0] + rho * np.cos(ang), z[1] + rho * np.sin(ang)

def angleArc(center, p0, p1, radius, num = 40):
    """Sample the minor arc of given radius at ``center`` from direction p0 to p1, and
    return (x, y, midAngle) for drawing a theta / psi angle mark and placing its label."""
    a0 = np.arctan2(p0[1] - center[1], p0[0] - center[0])
    a1 = np.arctan2(p1[1] - center[1], p1[0] - center[0])
    d = (a1 - a0 + np.pi) % (2 * np.pi) - np.pi              # signed minor difference
    ang = a0 + np.linspace(0.0, d, num)
    return center[0] + radius * np.cos(ang), center[1] + radius * np.sin(ang), a0 + 0.5 * d

def main():
    n = 5
    # a vertex at the top (index 0)
    ang = 2 * np.pi * np.arange(n) / n + np.pi / 2          
    verts = np.stack([np.cos(ang), np.sin(ang)], axis = 1)
    pk = Packing(verts.reshape(-1), [0, n])
    rho = 0.28
    cg = cornerGeometry(pk, rho)
    r = pk.positions.reshape(-1, 2)

    fig, ax = plt.subplots(figsize = (7.5, 7.5))

    loop = np.vstack([r, r[0]])
    ax.plot(loop[:, 0], loop[:, 1], "--", color = "0.6", lw = 1, label = "backbone")

    for k in range(n):
        nk = pk.next[k]
        label = "rounded boundary" if k == 0 else None
        ax.plot([cg.aPlus[k, 0], cg.aMinus[nk, 0]],
                [cg.aPlus[k, 1], cg.aMinus[nk, 1]], color = "C0", lw = 2.2, label = label)
        arcX, arcY = cornerArc(cg.z[k], rho, cg.aMinus[k], cg.aPlus[k], r[k])
        ax.plot(arcX, arcY, color = "C0", lw = 2.2)
        ax.add_patch(plt.Circle(cg.z[k], rho, fill = False, color = "0.85", lw = 0.8))

    # ---- annotate the top corner (k = 0) with every symbol ----
    k = 0
    vk, zk, am, ap = r[k], cg.z[k], cg.aMinus[k], cg.aPlus[k]
    ax.add_patch(plt.Circle(zk, rho, fill = False, color = "C7", lw = 1.0))
    for pt, name, off in [(vk, r"$v_k$", (0.06, 0.06)), (zk, r"$z_k$", (0.07, -0.02)),
                          (am, r"$a_k^-$", (0.07, -0.07)), (ap, r"$a_k^+$", (-0.24, -0.07))]:
        ax.plot(pt[0], pt[1], "o", color = "k", ms = 4)
        ax.annotate(name, xy = pt, xytext = pt + np.array(off), fontsize = 14)

    ax.annotate("", xy = ap, xytext = zk, arrowprops = dict(arrowstyle = "->", color = "C3"))
    ax.text(*(0.5 * (zk + ap) + np.array([0.03, 0.03])), r"$\rho$", color = "C3", fontsize = 14)
    ax.annotate("", xy = am, xytext = vk, arrowprops = dict(arrowstyle = "->", color = "C2"))
    ax.text(*(0.5 * (vk + am) + np.array([-0.06, 0.04])), r"$t$", color = "C2", fontsize = 14)
    bHat = (zk - vk) / np.linalg.norm(zk - vk)                               # bisector unit vec
    ax.annotate("", xy = vk + 0.33 * bHat, xytext = vk,
                arrowprops = dict(arrowstyle = "->", color = "0.25", lw = 1.3))
    ax.text(*(vk + 0.20 * bHat + np.array([0.07, 0.0])), r"$\hat{b}$",
            color = "0.25", fontsize = 14)

    tx, ty, tmid = angleArc(vk, r[pk.prev[k]], r[pk.next[k]], 0.18)
    ax.plot(tx, ty, color = "C4", lw = 1.6)
    ax.text(vk[0] + 0.27 * np.cos(tmid), vk[1] + 0.27 * np.sin(tmid), r"$\theta$",
            color = "C4", fontsize = 14, ha = "center", va = "center")
    px, py, pmid = angleArc(zk, am, ap, 0.13)
    ax.plot(px, py, color = "C1", lw = 1.6)
    ax.text(zk[0] + 0.19 * np.cos(pmid), zk[1] + 0.19 * np.sin(pmid), r"$\psi$",
            color = "C1", fontsize = 14, ha = "center", va = "center")

    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Rounded polygon: backbone, corner circle, and the roundify symbols")
    ax.legend(loc = "lower right", fontsize = 9)
    out = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                        "..", "roundifyDiagram.png"))
    fig.savefig(out, dpi = 130, bbox_inches = "tight")
    print("saved", out)


if __name__ == "__main__":
    main()
