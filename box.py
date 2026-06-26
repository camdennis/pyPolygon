"""Periodic simulation box and the minimum-image ``wrap`` for pyPolygon.

A packing's boundary conditions come in two flavors (``enums.PackingType``):

  square        -- the periodic unit square [0, 1)^2. Positions live in [0, 1)^2
                   and displacements wrap into [-0.5, 0.5)^2.
  latticeVector -- a general parallelepiped set by lattice vectors (build
                   step 9). Stubbed here; ``wrap`` raises for it until then.

``wrap`` checks the box type before proceeding (prompt line 27).
"""

import numpy as np

from enums import PackingType


class Box:
    """Periodic simulation cell.

    Parameters
    ----------
    packingType : PackingType
        ``square`` -> unit square [0, 1)^2 (default); ``latticeVector`` -> the
        cell is the parallelepiped spanned by the columns of ``h`` (step 9).
    h : array_like or None
        2x2 lattice matrix whose columns are the lattice vectors [a1 | a2]. Used
        only when ``packingType is latticeVector``. For ``square`` it is left as
        ``None`` (the cell is the unit square, i.e. an implicit identity ``h``).
    """

    def __init__(self, packingType = PackingType.square, h = None):
        self.type = packingType
        self.h = None if h is None else np.asarray(h, dtype = float).reshape(2, 2)


def wrap(dr, box):
    """Minimum-image displacement under the box's periodic boundary.

    Branches on ``box.type`` (prompt line 27).

    square: positions are in [0, 1)^2, so a raw displacement is mapped to its
    nearest periodic image in [-0.5, 0.5) componentwise via
    ``dr - floor(dr + 0.5)``.

    Parameters
    ----------
    dr : array_like
        Displacement(s); any shape (operates componentwise).
    box : Box

    Returns
    -------
    np.ndarray
        Minimum-image displacement, same shape as ``dr``.
    """
    dr = np.asarray(dr, dtype = float)
    if box.type is PackingType.square:
        return dr - np.floor(dr + 0.5)
    if box.type is PackingType.latticeVector:
        raise NotImplementedError(
            "latticeVector wrap is added in build step 9 (Phase 9)."
        )
    raise ValueError(f"unknown PackingType: {box.type!r}")

def minImageShift(displacement, box):
    """Lattice translation that carries ``displacement`` to its minimum image, i.e.
    ``wrap(displacement) - displacement``. This is the rigid offset to add to a far point
    so it lands in its nearest periodic image; with single-image interactions it brings one
    polygon next to another for the crossing / overlap tests. Dispatches on the box type via
    ``wrap``; zero in free space (``box is None``).
    """
    displacement = np.asarray(displacement, dtype = float)
    if box is None:
        return np.zeros_like(displacement)
    return wrap(displacement, box) - displacement

def wrapIntoCell(positions, box):
    """Map positions into the periodic cell. Called unconditionally by the
    minimizers, so wrapping is governed entirely by the box (no per-call flag):

      box is None   -- free space (eqSoftBody shape-build): returned unchanged.
      square        -- each coordinate into [0, 1) via p - floor(p).
      latticeVector -- into the parallelepiped via fractional coords (step 9; stub).

    The area-1 cell calibration lives with the periodic box (square is the unit
    square; latticeVector will enforce |det h| = 1 at step 9).
    """
    if box is None:
        return positions
    positions = np.asarray(positions, dtype = float)
    if box.type is PackingType.square:
        return positions - np.floor(positions)
    if box.type is PackingType.latticeVector:
        raise NotImplementedError(
            "latticeVector cell-wrap is added in build step 9 (Phase 9)."
        )
    raise ValueError(f"unknown PackingType: {box.type!r}")
