"""pyPolygon -- a NumPy reference for packings of rounded polygons.

The reference is a set of flat modules (box, geometry, neighbors, intersections,
overlap, packingBuilder, visualize, validate, ...) that import one another by bare
name. Importing the package puts its own directory on sys.path so those intra-package
imports resolve whether the package is imported from inside or outside its directory,
then exposes the high-level Model facade.
"""

import os as _os
import sys as _sys

_here = _os.path.dirname(_os.path.abspath(__file__))
if _here not in _sys.path:
    _sys.path.insert(0, _here)

from model import Model

__all__ = ["Model"]
