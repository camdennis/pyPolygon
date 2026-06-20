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
