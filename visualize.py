"""Matplotlib helpers for drawing pyPolygon packings.

Phase 1 draws the straight backbone polygon(s) from raw positions (the free-space
eqSoftBody build). Rounded boundaries, corner circles, and periodic-aware drawing
across cell boundaries arrive with the later phases.
"""

import numpy as np
import matplotlib.pyplot as plt

def plotBackbone(packing, ax = None, showVertices = True, lineKwargs = None,
                 vertexKwargs = None):
    """Draw each polygon's closed straight-edge backbone on ``ax`` (raw positions).

    Creates a fresh square axes if ``ax`` is None. Returns the axes.
    """
    if ax is None:
        _, ax = plt.subplots(figsize = (5, 5))
    r = packing.positions.reshape(-1, 2)
    lineKwargs = {"color": "C0", "lw": 1.5} if lineKwargs is None else lineKwargs
    vertexKwargs = ({"s": 18, "color": "C3", "zorder": 3}
                    if vertexKwargs is None else vertexKwargs)
    for p in range(packing.numPolygons):
        a = int(packing.startIndices[p])
        b = int(packing.startIndices[p + 1])
        loop = np.vstack([r[a : b], r[a]])              # repeat first vertex to close
        ax.plot(loop[:, 0], loop[:, 1], **lineKwargs)
        if showVertices:
            ax.scatter(r[a : b, 0], r[a : b, 1], **vertexKwargs)
    ax.set_aspect("equal")
    return ax

def showPacking(packing, title = None, savePath = None):
    """Plot the backbone in a fresh figure; optionally title and save. Returns ax."""
    ax = plotBackbone(packing)
    if title is not None:
        ax.set_title(title)
    if savePath is not None:
        ax.figure.savefig(savePath, dpi = 120, bbox_inches = "tight")
    return ax

def _sampleArc(z, rho, aMinus, aPlus, vertex, num):
    """``num`` points along the corner arc from a^- to a^+ about z (the minor arc bulging
    toward the vertex)."""
    twoPi = 2.0 * np.pi
    phi0 = np.arctan2(aMinus[1] - z[1], aMinus[0] - z[0])
    phi1 = np.arctan2(aPlus[1] - z[1], aPlus[0] - z[0])
    phiMid = np.arctan2(vertex[1] - z[1], vertex[0] - z[0])
    sweep = (phi1 - phi0) % twoPi
    if (phiMid - phi0) % twoPi <= sweep:
        ang = phi0 + np.linspace(0.0, sweep, num)
    else:
        ang = phi0 - np.linspace(0.0, twoPi - sweep, num)
    return np.column_stack([z[0] + rho * np.cos(ang), z[1] + rho * np.sin(ang)])

def roundedBoundary(packing, cg, rho, polygon, arcSamples = 60):
    """Closed polyline of polygon ``polygon``'s rounded boundary: each corner arc sampled
    (``arcSamples`` points) and joined by the straight edge runs. ``rho`` is a scalar or
    per-polygon radius. Returns an (M, 2) array of raw positions."""
    from geometry import rhoPerVertex
    r = packing.positions.reshape(-1, 2)
    rhoVert = rhoPerVertex(packing, rho)
    a = int(packing.startIndices[polygon])
    b = int(packing.startIndices[polygon + 1])
    return np.vstack([_sampleArc(cg.z[k], rhoVert[k], cg.aMinus[k], cg.aPlus[k], r[k], arcSamples)
                      for k in range(a, b)])

def _imageShifts(box):
    """Lattice translations whose periodic images can enter the unit cell -- the 3x3
    neighborhood for the square box, and no shift in free space (box is None)."""
    if box is None:
        return [np.zeros(2)]
    return [np.array([i, j], dtype = float) for i in (-1, 0, 1) for j in (-1, 0, 1)]

def draw(packing, rho, ax = None, highlightIntersections = False, forces = None, arcSamples = 60,
         cmapName = "tab20", alpha = 0.85):
    """Draw the rounded polygons clipped to the periodic unit cell: each polygon AND its
    periodic images are filled with the polygon's own color and outlined in a darker shade,
    with the straight backbone dotted on top, on clean axes (no tick numbers). With
    ``highlightIntersections``, the boundary intersections (and their images) are scattered and colored
    by type (ee/ea/ae/aa) with a legend. ``forces`` (an (numVertices, 2) array) is drawn as
    per-vertex arrows, auto-scaled to the cell. Returns the axes."""
    from matplotlib.patches import Polygon as MplPolygon
    from geometry import cornerGeometry
    if ax is None:
        _, ax = plt.subplots(figsize = (6, 6))
    cg = cornerGeometry(packing, rho)
    r = packing.positions.reshape(-1, 2)
    cmap = plt.get_cmap(cmapName)
    shifts = _imageShifts(packing.box)
    for p in range(packing.numPolygons):
        fill = cmap(p % cmap.N)
        edge = tuple(0.55 * channel for channel in fill[:3])
        boundary = roundedBoundary(packing, cg, rho, p, arcSamples)
        a = int(packing.startIndices[p])
        b = int(packing.startIndices[p + 1])
        loop = np.vstack([r[a : b], r[a]])
        for shift in shifts:
            ax.add_patch(MplPolygon(boundary + shift, closed = True, facecolor = fill,
                                    edgecolor = edge, linewidth = 1.5, alpha = alpha,
                                    joinstyle = "round"))
            ax.plot(loop[:, 0] + shift[0], loop[:, 1] + shift[1], linestyle = ":",
                    color = edge, linewidth = 0.9)
    if highlightIntersections:
        _scatterIntersections(packing, rho, cg, ax, shifts)
    if forces is not None:
        f = np.asarray(forces).reshape(-1, 2)
        maxNorm = float(np.sqrt(np.einsum("ij,ij->i", f, f)).max())
        scale = maxNorm / 0.08 if maxNorm > 0.0 else 1.0
        for shift in shifts:
            ax.quiver(r[:, 0] + shift[0], r[:, 1] + shift[1], f[:, 0], f[:, 1],
                      angles = "xy", scale_units = "xy", scale = scale, color = "k",
                      width = 0.004, zorder = 6)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    if packing.box is not None:
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    return ax

def _scatterIntersections(packing, rho, cg, ax, shifts):
    """Scatter the boundary intersections of every overlapping pair (and their periodic images),
    colored by type, with one legend entry per type present."""
    from matplotlib.lines import Line2D
    from intersections import boundaryFeatures, findIntersections
    from neighbors import findNeighbors
    features = boundaryFeatures(packing, cg, rho)
    intersections = findIntersections(packing, features, findNeighbors(packing, rho))
    colorByKind = {"ee": "#d62728", "ea": "#2ca02c", "ae": "#1f77b4", "aa": "#9467bd"}
    seen = set()
    for c in intersections:
        seen.add(c.kind)
        for shift in shifts:
            ax.scatter(c.point[0] + shift[0], c.point[1] + shift[1], color = colorByKind[c.kind],
                       s = 30, zorder = 5, edgecolor = "k", linewidth = 0.4)
    if seen:
        handles = [Line2D([0], [0], marker = "o", linestyle = "", color = colorByKind[k],
                          markeredgecolor = "k", markeredgewidth = 0.4, markersize = 7, label = k)
                   for k in ("ee", "ea", "ae", "aa") if k in seen]
        ax.legend(handles = handles, title = "intersection", loc = "upper right", fontsize = 8,
                  framealpha = 0.9)
