# pyPolygon

A clean **NumPy reference implementation** of molecular-dynamics simulations for
**rounded polygons** — backbone polygons whose corners are replaced by radius-ρ
circular arcs, packed under periodic boundary conditions.

It is the readable, finite-difference-checkable companion to the CUDA implementation in
`pyCudaPolygonSTABLE`: the math mirrors that reference so results cross-check, and the
flat CSR array layout ports back to CUDA. At the moment, pyCudaPolygonSTABLE does not have the exact behavior we desire and may contain several bugs. It is an incomplete and incorrect implementation, but may provide some helpful references. Use this sparingly. We don't want to accidentally carry bugs and old ideas into this clean-room of code. Once this clean room is appropriately working, we can start on pyCudaPolygon ensuring it has the correct functionality.

## Status

**Phase 1 complete** — building and relaxing a single equilateral polygon:

- A random *star* polygon seed (n points scattered in the unit box, ordered CCW about
  the centre) is relaxed by the **eqSoftBody** spring model (per-edge length springs +
  an area spring) using **FIRE**, producing an equilateral polygon at a chosen perimeter
  and area. Energy is driven to ~0 and verified: every edge equal, area on target, and
  the analytic force matches a central-difference gradient to ~1e-10.

Remaining phases (rounding the corners, neighbours, intersections, the full collision
energy/force, periodic packing, stress/stiffness) follow the roadmap below.

## Module map

| Module | Responsibility |
|---|---|
| `enums.py` | `EnergyType`, `PackingType` (`square` / `latticeVector`), `MinimizerType` |
| `box.py` | Periodic cell, `wrap` (minimum image) and `wrapIntoCell` |
| `packing.py` | `Packing` flat-array container (CSR layout) and `fromSinglePolygon` |
| `softBody.py` | eqSoftBody energy/force + backbone edge-length / shoelace-area helpers |
| `minimize.py` | `minimizeFIRE` / `minimizeGD` |

## Running

Use the project's Python environment (requires `numpy`; `matplotlib` for the upcoming
plotting). Example end-to-end relaxation:

```python
from packing import Packing
from softBody import eqSoftBodyEnergyForce, backboneEdgeLengths, backboneArea
from minimize import minimizeFIRE

pk = Packing.fromSinglePolygon(7, rng=0, targetEdgeLength=0.25, targetArea=0.15)
fe = lambda p: eqSoftBodyEnergyForce(p, kEdge=1.0, kArea=1.0)
energy, steps, converged = minimizeFIRE(pk, fe)
print(converged, backboneEdgeLengths(pk).mean(), backboneArea(pk)[0])
```

## Roadmap

1. Equilateral polygon via eqSoftBody (FIRE/GD) — **done**
2. Roundify: corner circles `z_k`, kiss points `a±`, swept angle `ψ`
3. Rounded area & perimeter
4. Neighbours of each vertex within a ball
5. Intersections (edge/arc: ee, ea, ae, aa) + outersections
6. Forces, energies, overlap areas
7. Minimize the full model with FIRE
8. Critical-displacement neighbour rebuild
9. Lattice vectors
10. Stress & stiffness tensors
11. Set / modify per-element area

## Conventions

- `camelCase` for all identifiers; spaces around every `=` (including kwargs / defaults).
- New definitions are tagged `# UNVERIFIED(Cam)` until reviewed; the tag is removed once
  the definition has been checked.
