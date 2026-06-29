# Plan: `pyPolygon` — MD simulation of rounded polygons (NumPy reference build)

## Context

We are building a **clean NumPy reference implementation** of a molecular-dynamics
model for *rounded polygons* — backbone polygons whose corners are replaced by
radius-ρ circular arcs. A working but backbone-only CUDA predecessor (the "STABLE"
reference) preceded this; it had the data layout, FIRE minimizer, neighbor/intersection
machinery, and MC overlap check, but **does not implement ρ / rounded corners**, nor the
`latticeVector`
boundary mode. This Python build adds those, in a form that (a) is readable and
finite-difference-checkable line by line, and (b) mirrors the reference's flat-array
layout so numbers cross-check directly and the math ports back to CUDA later.

This is being built **incrementally, gated on Cam's per-function sign-off** — the
explicit goal (prompt line 1) is that Cam does not fall behind on what the code does.
The plan front-loads execution-ready detail for steps 1–3 (the new geometry) and leaves
steps 4–11 as outlines we tighten as each is reached.

### Model decisions confirmed with Cam this session
1. **Overlap is between ROUNDED polygons.** Energy term #1 (overlap area) is the area
   where the two **rounded** shapes overlap — boundary = straight edge runs + corner
   arcs — so `ee/ea/ae/aa` intersections all bound this **one** overlap region. (Line
   5's "backbone" wording was loose; it caused a miscommunication.) The radius-ρ corner
   circles are **intra-polygon self-repulsion only**: a corner/vertex repels *other
   vertices of its own polygon* to keep the boundary from self-intersecting (topological
   safety). This self-repulsion **never acts between different polygons**.
2. **All energy terms use the ROUNDED polygon**, not the backbone — overlap (#1),
   adhesion (#2), and the `K_A` area / `K_P` perimeter springs (#3/#4) all read the
   rounded area/perimeter and rounded boundary.
3. **Step 1 minimizes the BACKBONE.** Cam plans to start from **star polygons** with
   potentially very sharp interior angles that can violate ρ-feasibility. So the
   equilateral build (step 1) runs on the straight backbone; rounding is applied *after*
   and must tolerate sharp corners (Phase 2).
4. **Adhesion chord.** Term #2 sums the **straight-line chord over every consecutive
   intersection↔outersection pair**, including cross-feature pairs (a non-convex pair
   may cross at >2 points). ⚠️ Memory `project_phase2_walkarea_incomplete` records the
   reference's `walkArea` *missing cross-feature chord pairs* — the Python build must
   handle cross-feature pairing correctly from the start.

## Working agreement (how we proceed)

- **Naming.** All identifiers are `camelCase` — functions, methods, variables, fields
  (Cam's hard requirement). Module files too where multi-word (`softBody.py`).
- **Marking.** Every definition I write — functions, methods, and classes/enums — gets
  `# UNVERIFIED(Cam)` on the line directly above it. You delete the tag when you've
  checked it. `grep -rn UNVERIFIED pyPolygon/` is your live checklist. Definitions you
  explicitly approve are written without the tag.
- **Gating / "scold me".** I will not start a build step until the previous step's
  functions are signed off and FD/visual-verified. If you ask to jump ahead while a
  prior step is still tagged `UNVERIFIED`, I'll say so and push back.
- **Verification is mandatory, not optional** (memory `feedback_empirical_verification`):
  every geometry/force function ships with a finite-difference or Monte-Carlo or visual
  check before it counts as done. No acting on theory alone.
- **No rewrites of approved code** (memory `feedback_no_rewrites`): once you sign a
  function off, later steps make minimal targeted edits, never wholesale rewrites.

## Stack & layout

- **Pure NumPy**, flat arrays in the reference's CSR style; SciPy + matplotlib for
  validation/visualization only. (No Numba/JAX in this reference build.)
- Mirror reference names: `positions` (shape `2N`), `shapeId` (`N`), `startIndices`
  (`P+1`), `next`/`prev` (`N`), per-polygon `targetArea`/`targetPerimeter`/
  `targetEdgeLength` (`P`), `wrap()` for minimum image. CSR indexing supports variable
  vertex count per polygon (assumed; matches reference).

### Module map (`pyPolygon/`)
| Module | Responsibility | Build step |
|---|---|---|
| `enums.py` | `EnergyType{eqSoftBody, normal, edgeOnly, areaOnly, hybrid}`, `PackingType{normal, latticeVector}`, `MinimizerType{GD, FIRE}` | 1 |
| `box.py` | Box/cell holding `PackingType` + lattice; `wrap(dr)` **branches on type** (prompt line 27) | 1, 9 |
| `packing.py` | `Packing` container: flat state arrays, targets, ρ, energy/force/velocity, box ref | 1 |
| `geometry.py` | Backbone edge lengths, shoelace area + gradient; **rounding** (`z_k`, `a±`, `t`, `ψ`, convex flag); **rounded** area & perimeter + gradients | 2, 3 |
| `softBody.py` | `eqSoftBody` energy/force (edge spring + area spring) to minimize the backbone (equilateral / star) | 1 |
| `neighbors.py` | Brute-force per-vertex neighbor lists within ball `D` (**no spatial hash**), critical-displacement rebuild | 4, 8 |
| `intersections.py` | `ee/ea/ae/aa` intersections; intersection records; **outersection** CCW-walk partner | 5 |
| `energy.py` | Full `normal` energy+force (all on the **rounded** polygon): rounded overlap (boundary integral over edges+arcs), intra-polygon self-repulsion, adhesion chord-sum, `K_A` area spring, `K_P` perimeter spring | 6 |
| `minimize.py` | FIRE (adaptive `dt`/`α`, velocity-bend, energy rollback) + GD; per-step neighbor/intersection refresh | 1, 7 |
| `observables.py` | Stress tensor, stiffness tensor; set/modify per-element area | 10, 11 |
| `validate.py` | MC overlap/area (cell size **0.5**), finite-difference force/gradient checks | all |
| `visualize.py` | matplotlib: backbone, rounded boundary, corner circles, intersection points | all |

---

## Phase 1 — Equilateral backbone via eqSoftBody (build step 1)  *[execution-ready]*

Goal: minimize the **backbone** polygon (straight edges) with `eqSoftBody` — edge
springs drive all edges to equal length, an area spring pins area to target — producing
the intended equilateral backbone. **Cam plans to start from star polygons** (non-convex,
possibly very sharp interior angles); the minimization runs on the backbone precisely
because those sharp corners can violate ρ-feasibility (Phase 2), so rounding is deferred
until the backbone is set. For a convex target the regular `n`-gon is the unique
max-area equilateral; for a star the area spring still selects the regular star.

Functions (`enums.py`, `box.py`, `packing.py`, `softBody.py`, `minimize.py`):
- `wrap(dr, box)` — minimum image; branches on `box.type` (`normal` → unit square
  `[0,1)²`; `latticeVector` stubbed until step 9).
- `Packing.__init__` / `fromSinglePolygon(n, ...)` — allocate flat arrays, CSR indices,
  `next`/`prev`, set `targetEdgeLength`, `targetArea`.
- `backboneEdgeLengths(packing)`, `backboneArea(packing)` — shoelace; per-vertex area
  gradient `∂A/∂r_k = ½(y_{k+1}−y_{k−1}, x_{k−1}−x_{k+1})`.
- `eqSoftBodyEnergyForce(packing, kEdge, kArea)` —
  `E = ½·kEdge·Σ(l_e − l₀)² + ½·kArea·(A − A₀)²`; force `= −∂E/∂r`.
- `minimizeFIRE(packing, ...)` / `minimizeGD(...)` — port FIRE.h loop: half-step
  position+velocity update, force eval, `P=v·f`, velocity bend toward force, adaptive
  `dt`/`α`, energy rollback on uphill step.

Verify: `validate.checkGradient` (central-difference −dE/dx vs analytic force, like
`tests/ForceEnergyDiagnostic.py`); `visualize.plot` shows the equilateral backbone
(regular `n`-gon, or regular star); assert edge-length variance ≈ 0 and area ≈ target.

## Phase 2 — Roundify: z_k, a±, ψ (build step 2)  *[execution-ready]*

For CCW vertex `v_k` with `dPrev = (v_{k−1}−v_k)/|·|`, `dNext = (v_{k+1}−v_k)/|·|`,
interior angle `θ = acos(dPrev·dNext)`:
- Convex test (CCW): `cross(v_k−v_{k−1}, v_{k+1}−v_k) > 0`.
- Tangent length `t = ρ·cot(θ/2)`; kiss points `aMinus = v_k + t·dPrev`,
  `aPlus = v_k + t·dNext`.
- Center distance `ρ/sin(θ/2)` along bisector `b̂ = (dPrev+dNext)/|·|`; `z_k` toward
  interior if convex, exterior if reflex (sign from convex test).
- Swept angle `ψ_k = π − θ` (arc length `ρψ_k`).
- **ρ feasibility:** the offset `t ≤ ½·min(adjacent edge lengths)` keeps neighboring
  arcs from colliding. **Sharp star corners can violate this** — so do not hard-fail:
  flag offending corners and handle per an agreed strategy (open item below), e.g.
  locally cap `t` at the half-edge / shrink local ρ, or reject the configuration.

Functions (`geometry.py`): `cornerGeometry(packing, rho)` returning `z`, `aMinus`,
`aPlus`, `psi`, `t`, `convex` arrays. Verify: `visualize` overlays circles + kiss points
on the backbone (visual continuity/tangency); assert each circle is tangent to both
edges (distance from `z_k` to each edge == ρ).

## Phase 3 — Rounded area & perimeter (build step 3)  *[execution-ready]*

- **Perimeter:** straight runs `Σ_e (l_e − t_{e,start} − t_{e,end})` + arcs `Σ_k ρψ_k`.
- **Area:** `A_backbone − Σ_k s_k · cut_k`, `s_k = +1` convex / `−1` reflex, with
  `cut_k = ½t²sin θ − ½ρ²(ψ − sin ψ)` (corner triangle minus circular segment).
- Analytic gradients of both w.r.t. vertex positions (needed by `K_A`/`K_P` springs).

**Confirmed (Cam):** the energy's area/perimeter terms use these **rounded** quantities,
not the backbone. Step 3 computes exactly them and their gradients for Phase 6.

Verify: FD-check rounded area/perimeter gradients; cross-check rounded area against MC
point sampling (cell size **0.5**, memory `feedback_mc_cell_size`).

## Phase 4 — Neighbors & ball size (build step 4)  *[outline]*
**No spatial hash** (Cam) — direct brute-force pairwise search: for each vertex, collect
the vertices within a **per-polygon** ball `D_i` (Cam):
`D_i = globalPercentage·(maxEdgeLength + edgeLength[i]) + 2ρ`. The global `maxEdgeLength`
term guarantees coverage of the largest possible partner regardless of how small `i` is;
the `+2ρ` margin covers the rounded-feature reach for inter-polygon crossings that the
edge-length terms miss. Record a pair if *either* endpoint finds the other within its own
`D` (symmetric). Reduces to a global constant when monodisperse.
(Critical-displacement rebuild deferred to step 8.)

**Revised (2026-06-25):** the neighbor list is **inter-polygon only** — self-repulsion is
no longer routed through it (it is intra-polygon and needs no spatial search; see Phase 6).
The current `neighbors.py` still keeps/flags same-polygon pairs (`sameShape`); that is
**transitional** and will be stripped to inter-polygon pairs only when Phase 6 lands,
dropping the `sameShape` flag.

## Phase 5 — Intersections ee/ea/ae/aa + outersections (build step 5)  *[outline]*
Segment–segment (`ee`), segment–arc (`ea/ae`), arc–arc (`aa`) intersection tests;
record `(shapeI, shapeJ, feature ids, point, parameter)`. **Outersection:** walk CCW to
the partner intersection of the same pair (reference `updateOutersectionsKernel`,
binary-search + cyclic-distance pairing). Must support cross-feature pairing (see
adhesion note).

## Phase 6 — Forces, energies, overlap areas (build step 6)  *[broken into sub-phases]*
Assemble U (prompt line 11) **plus the intra-polygon self-repulsion**, everything on the
**rounded** polygon. `K_adh`, `K_A`, `K_P` are **global scalars** (Cam); `targetArea`/
`targetPerimeter` are per-polygon. The overlap term is the heavyweight (and where the known
landmines live — the `walkArea` cross-feature bug and topology micro-jumps), so it is split
out. Build straightforward first, optimize later (memory `project_cuda_python_architecture`).

- **6a — Overlap area via the walk (energy; serial baseline).** Per pair, order crossings by
  σ_A and σ_B, store the `entering` flag, trace intersection→outersection runs (∂A-inside-B
  alternating with ∂B-inside-A), integrate `½ε_{αβ}∮X^α dX^β` over edge runs + corner-arc runs
  (arcs add circular-segment terms). This serial walk is the **correct reference baseline**.
  Validate vs **MC** (cell size 0.5, memory `feedback_mc_cell_size`); stress-test harder later.
- **6b — Overlap area, parallel method.** A `pyCudaPolygonSTABLE`-style reformulation
  (interior/exterior per-feature passes, no serial walk) for CUDA portability — structure
  borrowed from the reference, **correctness cross-checked against the 6a walk** (do not import
  its bugs; memory `project_pycuda_reference_caveat`).
- **6c — Overlap force.** `∂(overlap area)/∂vertex`, including the moving intersection points.
  FD-validate.
- **6d — Intra-polygon self-repulsion (energy + force).** Radius-ρ corner circles repelling
  *other vertices of the same polygon* (self-avoidance for floppy shapes; never inter-polygon),
  **scanned DIRECTLY per polygon** over own non-adjacent vertex pairs (not the neighbor list;
  also lets `neighbors.py` drop `sameShape`). Pin exact pairing (circle↔circle / circle↔vertex
  / circle↔edge) and functional form with Cam. (Two-circle overlap is a standard lens formula,
  derived fresh in a `notes/*.tex` then.) FD-validate.
- **6e — Adhesion (energy + force).** `−(K_adh/2)·Σ(2·chord/(P_i^t+P_j^t))²` over consecutive
  intersection↔outersection pairs incl. cross-feature. FD-validate.
- **6f — K_A / K_P springs (energy + force).** Rounded area & perimeter springs reusing the
  Phase-3 gradients. FD-validate.
- **6g — Assemble the full model.** Combine all terms into the energy/force entry point;
  FD-validate the total. (The shrink→minimize→reset-φ protocol lives with Phase 7.)

⚠️ Known risk areas (memory): instability at dense configs / short edges
(`project_phase2_dual_status`); intersection-topology micro-jumps
(`project_normal_topology_followup`); cross-feature pairing in the walk
(`project_phase2_walkarea_incomplete`, `project_phase5_crossing_completeness`). Budget FD/MC
checks heavily here.

## Phase 7 — Minimize the full model (build step 7)  *[outline]*
FIRE loop refreshing neighbors + intersections each step.

## Phase 8 — Critical-displacement neighbor rebuild (build step 8)  *[outline]*
Verlet skin: rebuild only when max vertex displacement since last build > skin/2.

## Phase 9 — Lattice vectors (build step 9)  *[outline]*
`PackingType.latticeVector`: box = parallelepiped `h=[a₁ a₂]`; `wrap` via fractional
coords `s=h⁻¹r` (the branch stubbed in Phase 1). Targets/positions follow lattice.

## Phase 10 — Stress & stiffness tensors (build step 10)  *[outline, math TBD with you]*

## Phase 11 — Set / modify per-element area (build step 11)  *[outline]*
API to set and rescale each polygon's `targetArea` (and re-derive `targetEdgeLength`).

---

## Verification (end-to-end)
- `validate.checkGradient(packing, eps=1e-7)` — analytic force vs central-difference
  −dE/dx, run after **every** new energy term.
- `validate.mcOverlap(packing, cellSize=0.5)` — MC cross-check of **rounded** overlap
  and rounded area (memory `feedback_mc_cell_size`).
- `visualize.plot(packing)` — backbone, rounded boundary, corner circles, intersection &
  outersection points; the primary "are we drawing the right shape" check.
- Cross-check backbone scalars (areas, edge lengths) against `pyCudaPolygonSTABLE`;
  rounded overlap only matches the reference's backbone overlap in the ρ→0 limit.

## Reference material (in-repo, self-contained)
- `notes/roundedDefinitions.pdf` / `.tex` — definitions, perimeter, area, gradients.
- `notes/intersections.pdf` / `.tex` — crossings ee/ea/ae/aa, outersections, periodic shift.

## Open items I'll confirm at the relevant step (not blocking step 1)
- ρ-feasibility strategy for sharp star corners (Phase 2): cap `t` / shrink local ρ vs
  reject the configuration.
- Exact intra-polygon self-repulsion pairing + functional form (Phase 6).
- ρ: global vs per-polygon — **still open**. (Resolved: `K_adh`/`K_A`/`K_P` are global
  scalars; `D` is per-polygon `globalPercentage·(maxEdgeLength + edgeLength[i]) + 2ρ`;
  targets are per-polygon.)
