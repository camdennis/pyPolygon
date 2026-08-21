"""pyPolygon -- a NumPy reference for packings of sharp (straight-edged) polygons.

The reference is a set of flat modules (enums, box, packing, distributions, packingBuilder,
softBody, polygonOverlap, plummerOverlap, selfRepulsion, minimize, cache) that import one
another by bare name. ``Model`` (model.py) builds the equilateral backbones; the sharp and
Plummer-mollified overlap energies are driven directly off ``model.packing``. Importing the
package puts its own directory on sys.path so those intra-package imports resolve whether the
package is imported from inside or outside its directory, then exposes the Model facade.
"""

import os as _os
import sys as _sys

_here = _os.path.dirname(_os.path.abspath(__file__))
if _here not in _sys.path:
    _sys.path.insert(0, _here)

from model import Model

__all__ = ["Model"]
