"""Enumerated types for the pyPolygon rounded-polygon MD code.

These are the three type flags referenced throughout the plan's module map:

  EnergyType    -- which energy/force model a Packing evaluates.
  PackingType   -- boundary-condition flavor; ``wrap`` branches on this.
  MinimizerType -- which minimizer the driver runs.

Checked and approved by Cam, so the definitions below carry no
``# UNVERIFIED(Cam)`` tags. (Going forward, *every* new definition -- classes and
enums included, not just ``def`` functions -- gets tagged until you check it off.)
"""

from enum import Enum, auto

class EnergyType(Enum):
    """The energy/force model evaluated for a packing.

    eqSoftBody -- equilateral spring model (edge springs + area spring) used in
                  build step 1 to relax a backbone polygon toward equal edge
                  lengths and a target area. No collisions.
    normal     -- the full rounded-polygon model: rounded overlap repulsion,
                  intra-polygon self-repulsion, adhesion, and the area/perimeter
                  springs (build step 6).
    edgeOnly   -- edge springs only.
    areaOnly   -- area spring only.
    softBody   -- collisions plus edge + area springs.
    """
    eqSoftBody = auto()
    normal = auto()
    edgeOnly = auto()
    areaOnly = auto()
    softBody = auto()

class PackingType(Enum):
    """Boundary-condition flavor of a packing.

    ``wrap`` checks this before applying the minimum-image convention:
      square        -- periodic unit square [0, 1)^2.
      latticeVector -- general parallelepiped defined by lattice vectors
                       (build step 9); the ``wrap`` branch is stubbed until then.
    """
    square = auto()
    latticeVector = auto()


class MinimizerType(Enum):
    """Which minimizer the driver runs.
    GD   -- plain gradient descent.
    FIRE -- Fast Inertial Relaxation Engine (adaptive dt / alpha).
    """
    GD = auto()
    FIRE = auto()
