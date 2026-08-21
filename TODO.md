# pyPolygon — Master TODO

## ▶▶ CURRENT: squares by shrinking every other PAIR of edges (2026-08-14)

Cam's replacement for the flattening cascade. A 16-gon at kappa 4 whose edges run in consecutive
PAIRS, alternating short-pair / long-pair, with each LONG pair bent to a right angle. `u = s/(2P/n)`
walks 1/2 → 0; shorts gone plus corners square IS a square. `alternating.py`, `tests/
alternatingEdgesCheck.py` (42/42), `tests/alternatingSquares.ipynb`.

**The corner is the FLATTEN family repointed.** For a pair the two edges are equal, so `d/(a+b)` is
`d/(2l) = cos(theta/2)` and Cam's `d = sqrt(2) l` is `d/(a+b) = 1/sqrt(2) = 0.70710678`. No new
Jacobian: `selectPairCorners` + `setConstraints(flatten = True)` + `setCornerTargets(RIGHT_ANGLE)`.
Unlike a flatten-to-straight ramp this target is an INTERIOR point, so it is met exactly rather than
approached — the bound-has-no-gradient rule finally works FOR us.

### Three corrections the measurements forced

1. **The right angle goes on the LONG pairs only.** On every pair the target set is empty at every
   u > 0: the polygon becomes a closed chain of 8 rigid right-angled Vs, and the shortfall against
   `A0 = 4p^2` is exactly `4(2 - sqrt2) s l` — at u = 1/2 the best such 16-gon encloses
   `(2 + sqrt2) p^2 = 3.414 p^2`, **15% short**.
2. **The corner is PINNED at 90, not ramped — this overturned my own recommendation.** 30-round ramps,
   worst max|C|: pinned 5.2e-13 (u reached 0.003); pinned without springs 1.4e-01; open-to-0.93-and-
   back 7.1e+01; 0.85→90 9.2e+01; straight→90 2.4e+01. The seed already IS a square, so any corner
   ramp makes the four sharp vertices migrate away and back — a rearrangement, not a ramp.
3. **The spring step is load-bearing.** Cam's "minimize each shape with a spring potential" between
   rounds is what keeps SHAKE near the manifold; without it the same schedule is lost halfway.

**Compliance comes from u and withdraws itself.** Max area vs the target at corners = 90: +2.9% at
u = 0.5, +0.20% at 0.2, +0.025% at 0.1, +0.003% at 0.05 — about `u^3/4`. The polygon keeps 8 shape
DOF throughout; the area row pinches them as u falls. Nothing goes infeasible: the square satisfies
the set at every u, with the short pairs straight along its sides.

**One ramp, two decimations.** A short pair merges THREE vertices, so `halveNumEdges(criterion =
"short")` drops two of every four (16 → 8); what is left is a square with four collinear midpoints,
which the EXISTING flat test removes exactly (8 → 4). Free-space chain: area lost 0.0000%, final
`|turn - 90| = 0.0000` deg, kappa 4.000000.

### Fixed on the way

- `halveNumEdges` merges targets over the general cyclic RUN between kept vertices (a pair collapse
  merges 3 edges then 1); reduces to the old adjacent-pair sum for a stride-2 drop.
- `halveNumEdges` raises the merged perimeter to the isoperimetric floor, the guard `resampleEdges`
  already had. Without it the last rung was refused: "shape index 4.000000 against a floor of
  4.000000".
- `diagonalMask` now follows the vertices through a decimation. It is indexed by vertex, so a stale
  one does not raise — the ragged gather silently holds the WRONG vertices.
- `flatTarget` is DROPPED at a decimation, not carried. Carrying it broke the old flatten cascade at
  its last rung, because an unset target CAPTURES the live flatness while a carried one COMMANDS.
- The parity is chosen once (`alternating.ensureParity`) by whichever call gets there first. When the
  two halves disagreed, each corner had one long edge and one short one — `d/(a+b)` is only
  `cos(theta/2)` when they are EQUAL, so the rows read satisfied at 8e-16 with corners at 95.68 deg.

### ‼ Open, and in the way

- **NOTHING KEEPS A POLYGON SIMPLE, and it bites from two directions.** Edge lengths plus the SIGNED
  shoelace area are satisfied just as well by a folded star, so neither the springs nor the constraint
  set object to one.
  - `generateEquilateralPolygons(kappa = 4)` returns self-intersecting polygons at every n > 4:
    measured 5/5 folded, total |turn| 468 deg at n = 8, 719 at n = 16, **1089 at n = 32** against 360.
    Worked around by building at n = 4 (unique solution) and calling `doubleNumEdges`. **This affects
    `annealNgonMorph.ipynb`, which builds 32-gons this way.**
  - `relaxShapes` folds them too if the u target moves far in one go. Opening the spread from u = 1/2
    to the drawn distribution in ONE jump gave |turn| sum **1080 deg** — with `max|C| = 4.2e-15` and
    every corner at exactly 90, so every downstream number looked healthy. Four or more sub-steps land
    on the identical u at 364.8 deg. The notebook opens it over 6.
  - Watch `sum |turn|` per round. It is the only thing that says the packing is still made of polygons.
### ‼‼ ROOT CAUSE of the density controller "bug": the contact law is NON-MONOTONIC past half overlap

Penetration depth is the shortest translation that separates two shapes, so once one has passed more
than halfway through its neighbour the shorter way out is the FAR side. The energy therefore peaks
near half overlap and **falls** toward full interpenetration. Two 16-gons at kappa 4, one walked
across the other:

| offset / side | 0.00 | 0.25 | 0.50 | 0.75 |
|---|---|---|---|---|
| pair overlap | 0.0833 | 0.0706 | 0.0467 | 0.0000 |
| pair energy | 4.07e-06 | 2.71e-05 | **1.23e-04** | 3.84e-05 |

**A fully stacked pair is a genuine force-balanced MINIMUM, not a violation.** That is the whole
mechanism behind everything that looked like a density bug: a run sat at `max|F| = 7e-10` with a third
of the box double-covered, read its excess as ~0, and let `holdExcessEnergy` compress to phi 1.2963
hunting energy that never arrives. The controller was reading the state correctly; the state was on
the wrong side of a barrier.

**`buildEquilateralPacking` places polygons at RANDOM CENTRES**, so at any interesting density they
start past that barrier and the relaxation pulls them further in. `Model.placeOnGrid()` (new) lays the
centroids on a lattice. Measured at phi 0.45 on 5 polygons:

| start | pair overlap before | after relax | `holdExcessEnergy` → phi |
|---|---|---|---|
| random centres | 0.134 | **0.259** (worse) | 0.216, excess never shed |
| grid | 0.0045 | **0.000006** | **0.698** (ceiling 0.682), excess 5.1e-05 on target |

A small residual overlap is fine — only crossing HALF matters — so `placeOnGrid` warns rather than
raises when the polygons overhang their cells, and the caller checks `getPairOverlapArea()`. Log it
every round: a climbing pair overlap means the barrier has been crossed and nothing after is real.

**This supersedes "the density controller is broken" as the open item.** Whether the controller has a
separate fault is now untestable until a run stays on the right side of the barrier.
- **`ShapeConstraints.projectPositions` gives up silently** at `maxIter` with no warning; only the
  composite version warns. Every diagnosis above had to read `maxResidual` by hand.
- Nothing computes the `u^3/4` feasibility law from a live packing; `max|C|` is the honest guard.

## ▶ PREVIOUS: squares by flattening every other vertex, then removing it (2026-08-12)

Cam's protocol. Every polygon EQUILATERAL at kappa = 4 with its SIZE FREE; compliance withdrawn by
removing vertices, and each removal made exact by flattening the vertices first.

**New constraint primitives** (`constraints.py`), all three Jacobians FD-verified at the central
difference floor — 8.8e-10, 9.7e-10, 2.4e-10:

- `equilateral = kappa` — pins every edge to `kappa sqrt(A) / n` from the polygon's OWN live area, so
  the shape is held and the size is not. A 1.6x rescale moves the residual 4e-16. Rank n, so `n - 4`
  shape DOF. Generalizes `idealEdgeLengths`, which was already this with kappa at the regular floor.
- `flatten = True` + `Model.setFlatTargets` — one row per SELECTED vertex holding `d / (a + b)` against
  its own rampable target. Scale-free, so it works while sizes move.
- a vertex MASK (`packing.diagonalMask`, set by `Model.selectFlattening`) shared by the per-object and
  moment forms and by `halveNumEdges`, so all three act on the same vertices by construction.

**The collective row does NOT work, and the reason is worth keeping.** A single row on the SUM of
`d/(a+b)` is either degenerate or ineffective and cannot be both:

- aimed at exactly `count` it does force every term to 1 (each is at most 1, so the sum reaches count
  only if all do) — but that is a MAXIMUM of the quantity, so its gradient VANISHES there. Measured:
  conditioning 5.97e-01 -> 2.02e-10, then LAPACK failed with "SVD did not converge".
- backed off to `0.99 count` it is well conditioned but holds only the MEAN: 0.99000 exactly with the
  worst vertex at 0.83, and in a packing it moved the worst from 0.71 to 0.80 over eight rounds.

Per-vertex targets have neither problem: 0.999999 everywhere, conditioning 1.0. `tests/
flattenCascadeCheck.py` 6/6.

**A constraint aimed at a quantity's BOUND has no gradient there.** This bit twice in one session —
once on the flatness bound, once on the shape budget's own documented vanishing at the regular polygon.
Always target just short of a bound and let the residual be the price.

**Removal is no longer exact**, and that is the price of staying off the boundary: ~4e-04 of a
polygon's area against 29% for blind decimation. The test threshold moved 1e-06 -> 1e-03 with the
justification written in — it guards inflation pushing polygons through the container (19% linear broke
it; this is 0.02%), not tidiness.

**Two bugs found on the way.** `setConstraints(flatten = True)` projects immediately, and a target
defaulting to exactly 1 collapsed the polygons on activation (kappa infinite) — enabling a constraint
now captures the current state and is inert. And `_orthonormalRows` no longer dies with `LinAlgError`
when rows go parallel.

### ‼ FIXED — QR was LAUNDERING a NaN, and the warning blamed the wrong constraint (2026-08-13)

Stage 2 of the cascade reported `1 of 1 constraint rows are NOT FINITE` on the AREA moment row — the
one part of the composite that divides by nothing geometric. It was not the culprit. The chain:

`ShapeConstraints._qrFactor` never checked its input. **`np.linalg.qr` propagates a NaN silently, and
every comparison against a NaN is False, so the rank test `diagonal <= tol` reports FULL RANK on a
block that is entirely NaN** — verified directly: an all-NaN block returns `np.any(diagonal <= 1e-12)
== False`. The poisoned factor then left as a normal basis, the innocent moment row was projected
through it in `CompositeConstraints.normalBasis`, and only THAT row reached `_orthonormalRows`, which
is the only place that warns. The block's own `FloatingPointError` in `_decompose` — which names the
polygon — was never reached, because the fast path never fell through to it.

Fixed with a finiteness check at the top of `_qrFactor` (return None → `_decompose` raises). Cost
measured: 0.121 ms against the QR's own 9.224 ms at N = 128, n = 32, i.e. **0.9%**.

**‼ MY FIRST DIAGNOSIS WAS WRONG AND THE SWEEP KILLED IT.** I read `worstFlat 1.00000` together with
the non-finite row, concluded "collapsed polygon → area → 0 → division by zero", and had written that
into the error message before testing it. `tests/degenerateJacobianCheck.py` walks a polygon to
degeneracy two ways and every layer stays FINITE at exactly zero area:

| collapse | `|A|` | `max|blockJ|` | any layer non-finite |
|---|---|---|---|
| squashed onto a line | 0.000e+00 | 4.239e+57 | no |
| shrunk to a point | 0.000e+00 | 3.142e+01 | no |

Every geometric divisor in `constraints.py` is floored (`_MIN_AREA`, `_MIN_EDGE_LENGTH`), so degenerate
geometry gives enormous-but-finite rows. **Only two inputs are NOT floored: the POSITIONS, and the
stored TARGETS** (`targetArea` / `targetPerimeter`, which the area and perimeter rows divide by raw).
Both now reproduce the fault and are named by the error. The message says all of this, so the next
person does not go looking for a collapsed polygon.

**The run then hit the new error: all five polygons at once**, inside FIRE's retraction immediately
after the transient step. Every polygon simultaneously rules out a per-polygon geometry fault and means
a GLOBAL input went bad. Two more layers of instrumentation went in so the next run is conclusive
rather than another round trip:

- `ShapeConstraints.poisonedInputReport` — prints, at the raise, how many positions are non-finite,
  whether each stored target array is non-finite or `<= 0`, and **which FAMILY owns the broken row**.
  Verified on both injected faults: a NaN position reports `POSITIONS: 1 of 48 ... families:
  ['equilateral']`, a zeroed `targetArea` reports `targetArea: 1 <= 0 ... families: ['area']`.
- Four guards in `minimizeFIRE` splitting the step into **FIRE step / transient / retraction / force
  evaluation**, each raising where the NaN is PRODUCED. Cost 0.0093 ms per guard against a 74.8 ms
  FIRE step at N = 64 — 0.05%.

`_checkForce` exists because `max|F| < fThreshold` cannot double as a validity test — the same NaN
comparison trap as `_qrFactor`: a completely broken force reads as "not yet converged" and the loop
keeps using it.

**Latent bug found on the way: `ShapeConstraints.families()` was STALE.** It listed
`edge, perimeter, area` — from before `diagonal`, `flatten` and `equilateral` existed — so it both
omitted three families and named them in an order that did not match the Jacobian's actual row
assembly. Nothing had noticed because it was only ever printed. Replaced by `rowFamilies()`, one source
of truth returning `(name, slice)` in true row order, which is what makes "which family owns the broken
row" trustworthy.

**The guards answered it, and it is NEITHER candidate.** The run stopped at `_checkForce`:

```
the force evaluation returned a non-finite result at step 3: energy nan,
84 vertices with a non-finite force.
```

Positions clean, retraction clean — so not the integrator and not a stored target. **The energy tier
itself returns NaN at that configuration.** And 84 is every vertex there is: 80 polygon + the 4-vertex
container, so the whole array is poisoned, not one bad spot.

Two degeneracy hypotheses were then MEASURED and both are dead (`tests/flatVertexEnergyCheck.py`):

| geometry | depth-tier energy | non-finite |
|---|---|---|
| vertex exactly collinear (flatness 1) | 1.849600e-02, smooth as offset -> 0 | 0 |
| polygon collapsed to exactly zero area | 5.092480e-01 | 0 |

So the ramp's goal is NOT in conflict with the tier's domain, and a collapsed polygon is not it either.
`degenerateJacobianCheck` covers the CONSTRAINT Jacobian under collapse; this covers the ENERGY, which
nothing had.

## ROOT CAUSE FOUND (2026-08-13): the area targets were geometrically impossible

The dump from Cam's N = 2 run settled it. Every position was NaN, but the TARGETS survive independently
of the coordinates and are the informative half:

| shape | n | target area | pinned |
|---|---|---|---|
| 0 | 16 | +0.544414 | 0/16 |
| 1 | 16 | +0.426973 | 0/16 |
| container | 4 | -1.000000 | 4/4 |

Two polygons asking for **0.971386** inside a container of area 1. The PROVED ceiling for two squares
is **0.500** (`s(2) = 2`), so the targets were **1.94x impossible**. The implied square sides are
0.737844 and 0.653432, summing to 1.391 inside a container of side 1 — they cannot fit under any
arrangement.

So the NaN was never a geometry fault, and none of the five mechanism hypotheses could have been right:
the whole chain hangs off an infeasible target set. The retraction cannot converge because there is
nothing to converge to; the geometry is driven somewhere impossible; the energy tier eventually returns
NaN three layers from the cause. **The warning text had been saying this the entire time** — "the
targets are UNREACHABLE, not that the budget is short, no iteration count fixes that" — and it was
correct on every firing.

This also reclassifies the deferred density item: sitting near phi 1.0 is not an endgame problem to
clean up later, it is the PROXIMATE CAUSE of every failure in this cascade.

**Guard added so it cannot recur silently:** `Model.checkDensityFeasible()` compares the phi the area
targets ask for against `records.maximumDensity(N)` and warns in one line at the point the targets are
set. `tests/densityFeasibleCheck.py` (5/5) replays the exact failing numbers, and equally checks that a
feasible density stays SILENT — a guard that always fires is noise. `records.maximumDensity(n)` gives
`n / s(n)^2` one definition.

- [ ] **Fix the density controller.** `holdExcessEnergy` drove phi to 0.971 against a ceiling of 0.500.
  The step-0 measurement in the plan — sweep excess against phi on a jammed packing — is now the
  critical path, not a preliminary.
- [ ] **Re-run the cascade with feasible targets** and confirm the NaN does not return. Everything
  below was diagnosis; none of it has been shown to make the cascade work.

- [ ] ~~Still unfound~~ **FOUND — see above.** Kept for the method: the dump is what ended it — five mechanism predictions in a row have
  been wrong here (see the pattern note below), each killed by a cheap measurement. So the next step is
  evidence, not another hypothesis:
  `_dumpState` now writes the failing configuration to `data/failure-<tag>-<stamp>.npz` at the raise,
  and `tests/inspectFailureDump.py` reloads it offline — per-polygon area and edge extremes off the CSR
  boundaries, closest vertex pair, containment, and every non-finite quantity. Verified end to end with
  an injected NaN. Reproducing the fault costs an hour of cascade; this makes every further hypothesis
  free to test against the real geometry.
  The failing force is saved separately as `failingForce`, because `packing.force` still holds the last
  GOOD one — a dump without that shows a healthy force beside a NaN energy and reads as no fault.
- [ ] **Separately anomalous: `worstFlat 1.00000` at the START of stage 2**, before the ramp did any
  work. Free space starts stage 2 at 0.61915 (`flattenCascadeCheck`), so this is notebook- or
  physics-specific and is a DIFFERENT fault from the NaN. A run reaching the new error will show
  whether the two share a cause.

### Open

- **The density endgame, untouched.** Five squares cannot exceed `phi = 0.6823`; runs sit near 1.0.
  `holdExcessEnergy` compresses when it should decompress once polygons escape — measured, phi 1.271 at
  1.86x the maximum possible. Cam chose "fix the controller itself", which is a MEASUREMENT first:
  sweep excess against phi on a jammed n = 4 packing to test whether the dimensionless excess goes flat
  under the affine density move. Not yet run.
- **No end-to-end packing run has completed.** Everything verified is free space + numpy.
- `excess = 5e-3` is crushing rather than pressing (own notes: 2e-5 is already crushing); 1e-6 to 1e-5
  is likely to flatten far more easily.
- Nothing is committed; everything carries `# UNVERIFIED(Cam)`.


## ✓ Stopping on EVIDENCE instead of maxSteps (2026-08-11)

Cam: *"I sometimes hit a wall with my maxSteps. I need a better way of determining when a minimizer
stops giving lucrative steps."*

**The immediate cause, found in his own run.** `annealNgonMorph.ipynb` cell 5 asked
`minimizeLBFGS(maxUnbalancedForce = 1e-12)`. The force noise floor is **3e-12**, so that target is
unreachable by construction — but it does not merely fail to converge, it GRINDS. Below the floor the
energy differences along the search direction are themselves roundoff, so the strong-Wolfe conditions
are tested against noise and never met, and `zoom` burns its full 40-bisection cap every step. Cam's
own `evalsPerStep` instrumentation caught it: **41.0 evaluations per step** against a healthy 1.08, at
1.16 steps/s. Fixed in the notebook (1e-10).

**Two mechanisms, in `minimize.py`:**

- `checkReachable(threshold)`, called from `Model._convergenceThreshold`, so EVERY minimizer warns at
  call time when the tolerance is at or under the floor. No stopping rule can rescue that case; the
  only useful moment to say so is before the first step.
- `_Stall` — `patience` steps must divide max|F| by `stallFactor`, else stop. On the FORCE, not the
  energy: near a minimum `dE ~ |F|^2`, so an energy test goes quiet a decade of |F| early, and `dE/E`
  is useless on the contact tiers where `E ~ d^3`. Best-so-far within the window, because FIRE's
  residual is not monotone by construction. Mirrors `anneal._STUCK_DENSITY_DROP` — a fixed budget must
  buy a fixed factor.

**The diagnosis is the point.** `Model.getStopReason()` returns which wall was hit, and the three want
opposite responses: `noise` (converged — the tolerance was under the floor), `flat` (a floor of the
ENERGY, not the minimizer: the C1 kink that `_SHARP_TOLERANCE = 1e-4` hardcodes per tier, now detected
generically), `slow` (real convergence, and it prints how many more steps the target needs), `search`
(the line search exhausted its bisections on over half the window — roundoff, as above).

`tests/stallCheck.py` — 7/7. Synthetic traces deliberately, since a detector has to be judged against
inputs whose right answer is known independently; the 'slow' projection is exact (24,600 against a true
24,600), a 40x-overshoot FIRE-like trace is never interrupted, and check 6 confirms the wiring on a
real numpy FIRE run.

**Defaults are still `patience = None` (off).** Turning it on globally would change every existing call
including the suite, and a wrong stall-stop is worse than a slow run. The notebook opts in explicitly.
Calibrating a default wants the depth-tier traces, which needs the card.

### Three corrections the RUN made to the design

Running it honestly found three things arguing from the design did not.

1. **The `slow` verdict had to become BUDGET-AWARE.** The first faithful run stopped cell 5's FIRE
   after it reported "7,029 more steps needed" — against a budget of 10,000, i.e. it was going to
   make it. A fixed `stallFactor` per `patience` demands a fixed RATE (2x per 500 steps = 1.4
   decades/1000); this run does 0.35. Now a still-converging run stops only when the projection
   exceeds the REMAINING budget, which is what "lucrative" actually means, and `patience` is just the
   measurement window instead of a per-tier tuning knob. `stallCheck` check 4 improved with it: short
   patiences now wait for the real floor at step ~1050 instead of firing at step 100 mid-decay.

2. **The `search` diagnosis was WRONG, and confidently so.** It asserted the arithmetic floor. The
   real stop came at max|F| = 4.9e-10 — **135x above** the 3e-12 floor — so the arithmetic was fine
   and the constrained LANDSCAPE was the limit: directional energy differences go under roundoff long
   before the residual does. It now splits the two cases and gives opposite advice. A wrong
   explanation is worse than none; this one would have sent the reader chasing precision.

3. **`Model.getReachableWidth()`**, because the notebook's ramp aimed at
   `getEdgePolydispersity()['between']` — the same number only to ~2e-10, which is under
   `setTargetPolydispersity`'s `floor * (1 - 1e-9)` guard. Every final round printed two "targets are
   UNREACHABLE" warnings that were true by two parts in ten billion and entirely misleading.

Also `minimizeFireLBFGS` now forwards `patience` / `stallFactor` to the FIRE leg instead of letting
`**lbfgsKwargs` swallow them — FIRE is the leg that stalls, so guarding only L-BFGS guarded the wrong
half.

### The run itself: sections 1-3 work end to end at N = 5

```
  as built           within 0.1734  between 0.0932  pooled 0.1969   kappa 4.0000
  cell 5 held        excess 9.939e-07 (asked 1e-6)  phi 0.98641  side 2.2514
                     wall 0.00%   0 vertices outside
  ramp round  1      within 0.1616  kappa 4.0000  side 2.2383  excess 1.03e-06
  ramp round 10      within 0.0000  kappa 4.0000  side 2.2436  excess 1.01e-06
```

The compliance is fully withdrawn — 0.1734 to 0.0000 — with kappa pinned at 4.0000 and the excess held
at ~1e-6 every round, ten rounds in 167 s. That is the protocol doing what it was designed to do, and
it leaves exactly the equilateral state section 4's closure argument needs.

Both stop mechanisms paid for themselves in that run: FIRE stopped `slow`, and L-BFGS stopped `search`
after 200 steps instead of grinding 4000 at ~41 evaluations each — about 150,000 force evaluations not
spent.

**Section 4 (cell 12) is still unverified** — the template plus the final decompression sweep.

## ‼ FIXED — a moment constraint was FIGHTING every density move (2026-08-11)

Found while auditing `annealNgonMorph.ipynb`. Pre-existing, nothing to do with the edge-draw work, and
it was live in cells 5, 7 and 11 — every one of them runs a density controller under `edge = [1, 2]`.

`DistributionConstraints` holds ABSOLUTE sums (`sum l`, `sum l^2`). `Model.setPackingFraction` scales
every polygon about its own centroid, which moves those sums by `factor^k` while leaving the
distribution's SHAPE untouched — so the constraint read a pure size change as a violation and SHAKE
retracted against the compression. Measured at N = 5, n = 8 under `area = True, perimeter = True,
edge = [1, 2]`:

```
  x1.02   residual 1.30e-02   retraction pulled phi back 0.083%
  x1.10   residual 4.06e-01   RETRACTION DID NOT CONVERGE, phi back 0.117%
  control: the same moves with edge = True (per object)  ->  3.55e-15, no drift
```

The residual came out at exactly `factor - 1` on the first moment — the signature of the family being
carried by the geometry while its reference stayed behind. Every density controller reaches this:
`holdExcessEnergy`, `energySweep`, `bisectJamming`.

**Fixed** with `DistributionConstraints.rescale(factor)`, called from `setPackingFraction` next to the
`targetDiagonal` line that maintains the same coupling for the same reason. The exponent is the
family's LENGTH DIMENSION, not a uniform `factor^k`: areas `factor^2`, lengths `factor`, and the direct
shape distortion is dimensionless so it must not move at all.

`tests/momentRescaleCheck.py` — 2/2 over six constraint sets including deviation mode, compression and
decompression. The strong check compares the carried reference against one recomputed from scratch on
the rescaled geometry, so it tests the definition rather than restating the formula.

Residuals after the fix: **4.11e-15**, phi drift exactly **0**, CV moved 7.8e-16.

## ✓ FIRE-then-L-BFGS minimizer, and where the time ACTUALLY goes (2026-08-11)

`Model.minimizeFireLBFGS(..., coarseness = 1e3, fireSteps = None, fireTolerance = None)` plus
`minimizer = "fireLbfgs"` in `anneal._relax`, so `holdExcessEnergy` and `energySweep` take it too. FIRE
runs to `coarseness` x the final tolerance, then L-BFGS finishes; the polish is skipped if FIRE already
made the target. Wired through `annealNgonMorph.ipynb` with `coarseness` as a single knob in cell 3.

The two minimizers fail in opposite places, which is the reason to pair them: FIRE is damped dynamics
and does not care how rough the landscape is or how far the start is, but converges LINEARLY; L-BFGS is
superlinear NEAR a minimum but far from one its curvature memory describes a landscape the iterate has
already left.

### ‼ MEASURED: adding FIRE does NOT help on the DEPTH tier, in either regime

Ten runs, N = 11, n = 16, and L-BFGS alone wins every one on BOTH time and energy, monotonically in how
much FIRE is added:

```
  WARM start (pre-relaxed with FIRE to 1e-2)          RAW start (nothing relaxed)
  lbfgs      2.572e-07    357     5.4s               seed 42  lbfgs      2.406e-05    74    1.3s  7 out
  fireLbfgs  3.262e-07    481     6.9s               seed 42  fireLbfgs  3.247e-05   194    2.9s  6 out
  fireDeep   3.516e-07   1738    13.1s               seed 42  fireDeep   3.675e-05  2040   24.5s  8 out
  fire       4.353e-06  20000   135.7s NOT CONVERGED seed  7  lbfgs      2.772e-07    25    0.5s  0 out
                                                     seed  7  fireLbfgs  2.912e-07   141    2.1s  0 out
                                                     seed  7  fireDeep   3.268e-07  2019   24.5s  1 out
```

My hypothesis that a RAW start would reverse the ordering is refuted -- it does not.

**And there is a reason, which also says where FIRE DOES belong.** The depth tier's energy is C2: closed
form, no quadrature, so L-BFGS's curvature memory is meaningful even far out and its line search has
real curvature to exploit. The SHARP tier's energy is only C1 -- the contact set changes discontinuously
as a vertex crosses an edge -- so strong-Wolfe has nothing smooth to work with, which is why
`anneal._relax` already FORCES FIRE there regardless of the caller. The split is by TIER, not by how far
the start is, and the code was already organised that way.

▶ **Recommendation, Cam's call:** put the depth-tier calls in the notebook back to `lbfgs` and keep
`minimizeFireLBFGS` for C1 tiers. Left wired as-is pending his say, since he asked for it and may be
seeing something at a scale these tests do not reach.

One useful accident: `coarseness` spans the whole range -- set it large and FIRE hands over immediately,
so `minimizeFireLBFGS` degenerates to pure L-BFGS. The spectrum is reachable from the one knob.

### ‼ A single-sample "result" I reported and had to withdraw

The first comparison showed `fireLbfgs` reaching a 4.5% LOWER energy and I described it as a quality
win. A rerun of the SAME seed gave 27% worse. CUDA is not bit-reproducible here (~3e-12 over ~120 steps,
already on record), which is enough to flip basin selection. Energy differences at the few-percent level
between these minimizers are noise; only the TIME differences (consistent, 2-50x) and the pure-FIRE
non-convergence (25x, 17x worse energy) are real.

### ‼‼ THE CONSTRAINT MACHINERY IS 0.4% OF A STEP — do not optimize it

I proposed exploiting the block-diagonal (disjoint-support) structure of the constraint Jacobian, and
was wrong twice over.

**First, it already exists.** `ShapeConstraints` extends `_RaggedBlocks`; its jacobian is documented as
"shape (P, m, 2 maxN), **block diagonal by polygon**", `_decompose` is a BATCHED SVD over those blocks,
and `_qrFactor` is a QR fast path used whenever the blocks are full rank. Only
`DistributionConstraints` is dense, and it is a handful of rows ("a moment couples the WHOLE packing").

**Second, it would not have mattered.** Measured per call on the same relaxed configuration:

```
  N    n    force(GPU)   project   retract    basis    CPU share
  5   16      62.42ms     0.45ms    0.09ms   0.39ms       0.9%
 11   16     168.64ms     0.64ms    0.09ms   0.53ms       0.4%
 11   32     370.07ms     1.20ms    0.11ms   1.07ms       0.4%
 32   32     905.55ms     3.16ms    0.17ms   2.91ms       0.4%
```

A 100x win on 0.4% is 0.4%.

### ▶ THE REAL LEVER — the force evaluation swings 20x with how OVERLAPPED the state is

Same N = 11, n = 32, same kernel, same machine: **17.4 ms well-relaxed against 370 ms under-relaxed.**
The per-edge work is a walk over boundary crossings and deep overlap makes many of them, so cost tracks
overlap depth. That argues for exactly what Cam is doing by hand — relax deeper before compressing —
and it means protocol beats kernel tuning here.

▶ Still unmeasured: whether the ~17 ms floor at a relaxed state is kernel or HOST setup. ~40 pairs x 32
edges is microseconds of arithmetic on this card, so 17 ms is suspicious; each call rebuilds a BodySet
and recomputes centroids and circumradii in numpy before launching. `scratchpad/overhead.py` is written
to split it and has not been run.

### ▶ WART — the minimizers disagree about what they return

`Model.minimizeFIRE` returns a bare `steps` int; `minimizeLBFGS` and `minimizeCG` return
`(energy, steps, converged)`. This caught me TWICE in one session — once in `minimizeFireLBFGS` itself
and again in the test script written minutes later. Anything composing the three walks into it.

## ‼ OPEN — `annealNgonMorph.ipynb`: the SQUISHY premise is not switched on (2026-08-11)

Running the notebook headless at N = 5 (its own cells, only N and the reference substituted), cell 3
opens with:

```
  edge width 0.0000 -> 0 over 10 rounds, at fixed excess 1.0e-06
```

`width0` is EXACTLY zero, so section 3's whole "withdraw the compliance gradually" ramp is 0 -> 0: ten
rounds of re-holding an excess with nothing changing, which is most of its 33-minute runtime.

The cause is one line in cell 1:

```python
  # SQUISHY: kappa is locked at 4 by pinning area and perimeter; the shape is otherwise free.
  packing.setConstraints(area = True, edge = True)
```

`edge = True` pins every edge length per object, so the edge-length distribution has zero width BY
CONSTRUCTION. The markdown's headline says "Squishy means `setConstraints(area = True, perimeter =
True)`" — a different constraint set from the one the code applies. The notebook's central mechanism
has never been active in this version.

### ‼‼ AND THE RAMP TARGET IS INFEASIBLE — the full causal chain

Cam ran it too and got further into the wreckage. The chain, in order:

1. `edge = True` pins edges per object, so `width0 = 0.0000`
2. the ramp is therefore `0 -> 0` and round 1 calls `setTargetPolydispersity(0.0)`
3. **zero edge width is geometrically infeasible.** `_reachableWidth` is ~half the area CV; the
   notebook's `setLogNormalScale(0.25)` puts the floor at 0.0488
4. the moment retraction chases it: *"residual 8.525e-02 after 796 passes ... the targets are
   UNREACHABLE"*
5. the failed retraction mangles the geometry until **a polygon is 2.57 EDGE LENGTHS outside the
   container** (caught by the new `holdExcessEnergy` entry warning)
6. every density and energy after that is meaningless

Same family as [[project_isoperimetric_infeasibility]]. **Fixed in the library**:
`setTargetPolydispersity` now clamps to the reachable floor with a warning, as `energySweep` has always
done for its own ramp — the setter did not, so any hand-written ramp walked straight in. Verified on the
notebook's own setup: asking 0.0 clamps to 0.0488 and the constraint residual stays at 4.19e-13 instead
of diverging to 8.5e-02.

### ▶ AND SECTION 3 IS ANNEALING THE WRONG QUANTITY

Measured under option A (`area = True, perimeter = True`): the edge width as built is **0.0488, exactly
the floor**. So even with the compliance switched on, the edge ramp has nowhere to go — and that is
structural, not tuning. The packing-wide edge CV is dominated by the SIZE spread between polygons and
cannot fall below ~half the area CV whatever the shapes do, so `getPolydispersity()['edge']` is a size
measure, not a compliance measure.

The compliance section 3 means to withdraw -- polygons moulding while keeping kappa = 4 -- lives in the
SHAPE DISTORTION (`getMaxShapeDistortion`, the `setShapeBudget` / `distortion` machinery that
`energySweep`'s `annealShape` path already ramps). Narrowing the edge moments cannot withdraw it,
because the edge moments were never holding it. Option A is needed to CREATE the compliance; it is not
sufficient on its own.

**Cam's call, because the two options are different physics:**

- **A, genuinely squishy** — `setConstraints(area = True, perimeter = True)`. Locks kappa at 4 while
  leaving 61 shape DOF free at n = 32, the edge distribution acquires real width, and section 3 has
  something to withdraw. This is what the prose describes.
- **B, accept held shapes** — keep `edge = True` and DELETE section 3, making the run a search over
  arrangements of near-regular polygons. Legitimate, but then the ten rounds are dead weight.

What it cannot stay is the current mix, where the code pins the edges while the prose and section 3
both assume they are free.

### ✔ RESOLVED at the source — 2026-08-11, Cam's call: draw the edges from a distribution

Cam cut through the A/B choice by attacking the BUILD rather than the constraint set: *"instead of
equilateral polygons, let's start off with polygons that have edges drawn from a distribution (but with
the same shape ratio 4). This way we can use it as a degree of freedom properly."*

That is the right move, because the diagnosis above was one level too shallow. The edge CV had no room
not because the constraints held it but because an EQUILATERAL BUILD HAS NO WITHIN-POLYGON SPREAD AT
ALL: the pooled CV is then entirely the size spread between polygons, which is exactly `_reachableWidth`.
Option A would have let width accumulate as the packing moulded, but the run would still have STARTED on
the floor.

`Model.generatePolygons(phi, kappa, edgePolydispersity = ...)` draws each polygon's n edge targets
log-normally at fixed perimeter and fixed area, so every polygon keeps kappa EXACTLY (measured 1.8e-15
on the targets, 5.7e-07 on the geometry after the relax) while its edges differ.
`generateEquilateralPolygons` is now the `edgePolydispersity = 0` case of it.

`getEdgePolydispersity()` splits the pooled CV into `within` and `between` in quadrature (exact variance
decomposition, vertex-count weighted for ragged n) — the diagnostic whose absence hid this. Reading only
the pooled number is what let the run look healthy while sitting pinned at its floor.

Measured at N = 11, n = 32 with `setLogNormalScale(0.25)`:

```
  build              pooled   floor    headroom   within
  equilateral        0.1171   0.1171    +0.00%    ~0       <- the ramp is a no-op
  edgePoly = 0.2     0.2104   0.0881   +138.86%   0.1911   <- and the ramp takes it to 0.0008
```

Note the floor MOVES between the two rows (0.1171 -> 0.0881) because it is computed from the target
areas, which the two builds draw identically but which `setLogNormalScale` renormalizes differently.

**The feasibility limit is not the binding one.** Unequal edges enclose less area than equal ones, so
they raise the shape-index floor (`build.minShapeIndex`, from the cyclic polygon's area) — but at
n = 32, kappa = 4 that floor only reaches 4 at an edge CV near **2.07**. The refusal is a backstop, not
a design limit. `build.cyclicArea` is verified to machine precision against the regular n-gon's closed
form and to ~1e-12 against Heron on 2000 random triangles, including the obtuse branch where the
circumcenter falls outside.

`tests/edgePolydispersityCheck.py` — 8/8.

**Still open, and it is a genuine obstruction:** the endpoint cannot keep the drawn edges. A
`sides`-cornered template prescribes the turning angles, and turning angles PLUS unequal edge lengths
over-determine a closed polygon — a 4-cornered template forces opposite runs of edges to have equal
total length. Measured: a 0.2 draw at n = 32 misses closure by 1.60% of the perimeter.
`setShapeTemplate(keepEdges = True)` refuses it with that number rather than handing it to the springs.
So section 3's ramp is not optional — the within-spread has to reach ~0 before section 4 can impose a
square. What the drawn build buys is a compliant START and a ramp that does real work, not a
polydisperse finish.

#### ‼ AND THE JUSTIFICATION ABOVE IS CONDITIONAL — measured, not assumed

The claim "an equilateral build leaves the ramp with nothing to withdraw" is true AT BUILD TIME and
**depends entirely on what section 2 does next**. Measured end to end at N = 5, n = 32, seed 42:

```
  section 2 with edges FREE (perimeter = True, edge = False)
    equilateral   as built within 0.0000  ->  at excess 0.9283  ->  after ramp 0.0418   side 2.2429
    0.2 draw      as built within 0.1734  ->  at excess 0.8832  ->  after ramp 0.1015   side 2.2523
```

A compliant relax MANUFACTURES within-spread near 0.93 by itself, from an equilateral build slightly
more readily than from a drawn one. In that protocol `edgePolydispersity` buys nothing, and the two
arms land within basin-selection noise of each other.

With section 2 settling under `edge = True` — every edge pinned per object through the minimizers, then
`edge = [1, 2]`, which only ever narrows — the relax cannot manufacture any, and the drawn spread is the
only source. That is the protocol the notebook now uses, and it is the reading under which the build
feature earns its place.

So the honest statement is: the feature does exactly what it says, and whether the run NEEDS it is set
by one line in section 2. Both readings are written into the notebook.

#### The notebook, updated (2026-08-11)

`annealNgonMorph.ipynb` now runs **N = 5**, whose optimum is PROVEN at `2 + 1/sqrt(2) = 2.7071` —
a better target than 11's conjectured 3.877, because a run can be wrong against it. `bestSide` and
`rigidBaseline` are per-N dicts; the hardcoded 4.0057 baseline line only draws when it is N = 11's.

Cell 1 gains `edgeSpread` and a `widths()` helper; cell 2's ramp runs `width0 -> floor` (measured via
`getEdgePolydispersity()['between']`) instead of `-> 0`, prints within/between per round, and warns
loudly when it starts on the floor. The plot panel shows within and between separately, since the
pooled curve cannot distinguish a working ramp from a stuck one.

Cell 11 keeps `setShapeTemplate(morph = 1.0, sides = 4)` — `keepEdges` defaults to False, which is the
only feasible endpoint (see the closure obstruction above). Section 3's ramp must run first.

### Cells 0-2 do work

```
  as built: phi 1.03069  side 2.2025  excess 2.537e-02   (asking for 1.0e-06)
  held at excess 1.020e-06:  phi 0.99535  side 2.2413  kappa 4.0000  pairOverlap 6.443e-04
    wall carries 0.34% of the contact energy   wallDepth 0.0008 edges
```

The controller pulled 2.5e-02 down onto 1.02e-06 in 33 s with the load on the BODIES (0.34% wall), so
the leak-instead-of-jam failure is gone in the live path and not only in the suite.

### ‼ My error, not the code's

That first run died in cell 3 with `CUDA error 719 during neighborPairsCuda`. I had run
`polyContactCheck.py` twice on the same 4 GB card while the notebook was mid-cell, after saying I would
keep to one job at a time. Concurrent CUDA here produces 719 launch failures — already recorded once
this project — and the run is void rather than evidence of a defect.

## ✓ DONE — EXCESS ENERGY replaces density as the control parameter of an anneal (2026-08-10)

`holdExcessEnergy(excess)` moves the density until the RELAXED contact energy sits a fixed amount above
jamming. `energySweep(excessEnergy = ...)` re-establishes it after every anneal round, and the full
decompression happens once, at the end.

**Why a fixed density cannot work here.** It has to be a density above jamming, which is the answer
being searched for; worse, that density MOVES during the run. Compliant 32-gons at `kappa = 4` mould
around each other and jam far denser than the rigid squares they are morphing into, so a phi picked to
overlap the squares leaves the squishy shapes rattling in free space with nothing in contact — the
anneal does no work and the sweep hands back the lower bound it was given. Each round of stiffening
then raises jamming again underneath the packing.

```
  Model.getContactEnergy()   contact term ALONE, per tier -- getEnergy() carries live shape springs
  Model.getExcessEnergy()    the above / (N k meanEdge^4) -- dimensionless, tier-aware
  Model.holdExcessEnergy()   two-sided ladder onto a target; compresses OR decompresses
  anneal.energyScale()       the unit; anneal._setDensity() moves the box, or the polygons if no box
```

### Measured on the depth tier (6 equilateral kappa-4 octagons, seed 42)

```
  excess 1e-9    the NOISE FLOOR, unreachable    residual settling, no trend in phi at all
  excess 1e-6    ~3% areal overlap               the intended "just above jamming"
  excess 2.4e-5  13% areal overlap               crushing, not pressing
```

The floor is real and was the first thing this got wrong. The contact energy below jamming is not zero
in practice: across `phi = 0.43 .. 0.93` it read 3.8e-11, 1.8e-11, 8.7e-12, 1.8e-11, 4.6e-11, 1.7e-11,
2.2e-12, 2.4e-11, 2.4e-11 — wandering, **no trend** — then jumped four decades to 2.3e-07 at
`phi = 1.03`. A free-flight test against exactly zero therefore never fires, and the controller crawls
the whole journey at its fine step. The test is now against a fraction of the target.

### ▶ Path dependence is LARGE and its sign is not fixed

`tests/excessEnergyCheck.py` check 3 drives the same packing to `excess = 1e-6` from both directions.
After the wall and pair-only fixes:

```
  compressing up from loose        excess 1.0018e-06   phi 0.406670
  decompressing down from 1.0e-03  excess 1.0027e-06   phi 0.962556
```

**137% apart, the DECOMPRESSING run the denser.**

‼ An earlier reading of this same check gave 4.3% with compression the denser, and that figure was
quoted into the `holdExcessEnergy` docstring, this file and memory. **It does not stand** — it was taken
before the excess counted body contact only and before the wall could be stiffened, i.e. from the
configuration where the packing was held by its boundary rather than by itself. Corrected everywhere.

What survives is qualitative: the state reached depends on the route as much as on the target, so a
density reported here is a property of the history too. Which direction is BETTER is NOT established,
and the 4.3% episode is the reason not to pick one on a single configuration.

✓ **RE-MEASURED in the corrected configuration** (pair-only excess, `wallStiffness = 100`, FIRE pinned
at 2000 steps for every row so the comparison is about the starting density and nothing else):

```
  scale    phi0     final phi   side     wall%   outside   time
  0.975   1.0519    0.96831    3.3705   0.00%      0        74s
  0.990   1.0203    0.93154    3.4363   0.00%      0        78s
  1.150   0.7561    0.74565    3.8409   0.14%      2        10s
  1.350   0.5487    0.54870 -> 0.54647  0.00%      0         2s
```

The conclusion survives and is MONOTONE: denser start, better packing, every step. `0.975` beats
`0.99`, so Cam's change to it was the right direction, and going further may keep paying — bounded by
the contact law's `dMax/rIn << 1` limit, which `_MAX_USEFUL_DENSITY` already caught being crossed at
phi = 1.478 elsewhere.

The `1.35` row reproduces the old broken-configuration number EXACTLY (4.4865 both times), which is
also the explanation: a loose start never engages the wall, so there was nothing there for the wall fix
to change. And the loose rows still finish BELOW their starting phi, which was the mechanism claim.

▶ Still NOT re-measured from that configuration: the 8000-vs-500 FIRE-step comparison (2% of side for
19x the time). The rows above hold FIRE fixed at 2000, so they say nothing about it either way.

### ‼ Cam's correction — the density move must be AFFINE, and the box must never move

The first version scaled the CONTAINER when there was one, leaving the polygons where they sat. Wrong,
and worst exactly where it was needed: decompressing that way retreats the walls from a cluster whose
interior never expands, so the contact energy collapses to the noise floor and the controller reads
"unjammed" off a packing that has not moved. Compounded by the step cap, an overjammed start crawled at
1% per round and exhausted `maxRounds` without recognising anything.

`_setDensity` now always calls `setPackingFraction`, and **the box stays 1 x 1 for the whole run**.
That IS the affine protocol, written in the box's frame: shrink the box by 1/f about its centre, carry
every centroid with it, leave the polygons their own size, then rescale by f to restore the box —
centroids land back where they started and the polygons come out f times larger, which is one call.
(`compressToJamming` still moves the box, legitimately: there the box carries a degree of freedom under
an applied pressure, so the wall does mechanical work rather than teleporting past the packing.)

Two consequences folded in at the same time:

- **The step cap no longer applies inside a bracket.** The bracket is never wider than one stride and
  is itself the constraint; capping on top of it pinned the secant to the cap every round.
- **The approach is SYMMETRIC.** Whichever side it starts on, it strides while more than a decade from
  the target and ladders for the last decade. A version that deliberately overshot below so the last
  leg was always a compression was written and then backed out: its only justification was the 4.3%
  figure retracted above, and with that gone there is nothing saying either direction is better.
- **A runaway compression now stops and says why** (`_MAX_USEFUL_DENSITY = 1.5`). Above phi = 1 the
  polygons must overlap by (phi-1)/phi of their area, so an unresponsive energy up there is a broken
  tier, not a dense packing.

Measured afterwards from Cam's own overjammed start (`sideLength = sqrt(N)*0.99`, N = 11, n = 12,
phi = 1.0203, excess 3.15e-04 — two and a half decades over target):

```
  phi 1.020304  excess 3.1522e-04    strides down at 5%
  phi 0.971718  excess 1.6350e-05    strides again
  phi 0.925446  excess 4.5571e-07    crossed -> bracket [0.9254, 0.9717]
  phi 0.935411  excess 1.1547e-06    secant
  phi 0.933863  excess 1.0059e-06    converged; side 3.4321
  boxArea 1.000000000  unchanged = True
```

Five relaxations, under a second, box exact to the last bit.

### ▶ THE SHARP-TIER PRE-RELAX IS THE WHOLE COST, and it does not converge

Same run, staged:

```
  spring (sharp tier, FIRE 8000 steps)   maxF 7.44e-03   331 s   <- asked for 1e-3, never got there
  depth  (LBFGS)                         maxF 9.06e-07     0 s
```

331 seconds burnt on a relaxation that misses its tolerance by 7x, on a configuration the depth tier
then settles in under a second. It is a C1 energy at phi = 1.02, so FIRE converges linearly to a floor —
`_SHARP_TOLERANCE` already documents exactly this and puts the floor at 1e-4. The pre-relax exists only
to pull polygons INSIDE the box (`generateEquilateralPolygons` places them periodically, so some
straddle the wall, and an affine density move cannot rescue a centroid that is already outside).

### ‼ REFUTED — "start loose, it is cheaper": Cam's overjammed start is doing the real work

The obvious saving is to do the containment relax at a loose density, since the excess and not the
initial phi is what puts the run above jamming. Measured (N = 11, n = 12, excess 1e-6), it is 30% worse:

```
  start     phi0     spring   outside   final excess   final phi   side
  0.99     1.0203     331 s      -        1.006e-06     0.93386   3.4321
  1.15     0.7561      10 s      6        1.028e-06     0.74322   3.8471
  1.35     0.5487       2 s      1        1.016e-06     0.54647   4.4865
```

The final phi of the two loose runs is BELOW where they started. A loose build is already jammed at its
own density: the shapes are rigid, the minimizer is local, and affine compression only makes contact
within the arrangement it was handed — nothing rearranges. Relieving a large initial overlap is what
forces the global rearrangement, and it is worth 3.4321 against 4.4865. **Keep `sqrt(N) * 0.99`.**

(Side is not comparable to the 3.877 square figure here — these are n = 12 kappa-4 shapes.)

### ✓ ANSWERED — the pre-relax cannot be skipped, but 8000 steps buys 2% over 500

```
  sharp FIRE 8000  pre 308s  outside 25->19  excess 1.006e-06  phi 0.93406  side  3.4317  wall 4.32e-03
  sharp FIRE 500   pre  16s  outside 24->17  excess 1.044e-06  phi 0.89823  side  3.4995  wall 4.16e-03
  no sharp FIRE    pre   0s  outside 19-> 0  excess 1.563e-03  phi 0.02059  side 23.1155  wall 0.00e+00
```

Without it the controller decompressed **50x** (phi 1.0203 -> 0.0206) and the excess never fell. That is
the signature of overlap an affine move cannot relieve: an affine map takes coincident points to
coincident points, so two polygons sharing a centroid stay concentric at every density. Only
rearrangement clears it, and that is what the FIRE leg is actually buying — not containment, which it
does not even achieve (25 vertices still outside after 308 s).

**`_STUCK_DENSITY_DROP` / `_STUCK_ENERGY_DROP` now catch this**: if halving phi has not even halved the
excess, the controller stops and names the tangle instead of decompressing to nothing. The old warning
misdiagnosed it as "the target is under the noise floor", which is the opposite advice.

The 19x time saving from 8000 -> 500 steps costs 2% in side (3.4995 against 3.4317). Cam's call; a
middle value has not been measured.

### ‼‼ ROOT CAUSE — the excess was being satisfied ENTIRELY by escaping the box

Cam's picture at the end of the compliant stage: "completely unjammed with large gaps between edges".
Measured at a state the controller reported as held at excess 1.043e-06, N = 11, n = 12:

```
  total contact energy   9.225282e-10
  WALL (confinement)     9.225282e-10   = 100.00%
  PAIR (polygon-polygon) 1.827535e-19   =   0.00%
  pair overlap AREA      0.000e+00
  wall penetration       4.159e-03  (0.0439 edges)
  vertices outside box   17 of 132
```

**Nothing was touching anything.** Pair overlap exactly zero, the whole "excess" being 17 vertices
extruded through the wall. The controller stopped at a completely unjammed packing that was leaking out
of its container, and the stiffening loop then re-held that same satisfiable-by-escape target every
round, so the interior never compressed at all.

The wall and the body contacts are not merely different terms — they are ALTERNATIVES the packing
chooses between, and whichever is softer wins. **`getExcessEnergy` is now written on
`getPairContactEnergy`**, polygon-polygon only, with the confinement term subtracted via
`confinementEnergyGradient` (agrees with the batched path to 1e-19; ~77 ms, charged once per controller
round, not per force evaluation). Containment stays a separate verdict, as a DEPTH against
`wallTolerance`, which is what `energySweep` already did.

### ‼ MY BUG — bracketing and extrapolating are not the same thing

Refusing a noise-floor point as a bracket end (to keep it out of the secant) also stopped it bounding
the answer. Whenever the onset is steeper than one stride the controller then held a `high` with no
`low`, strode 5% down, 5% up, and oscillated until it ran out of rounds — every row of the wall sweep
ended at exactly 1.0203/1.05^2 = 0.92545, which is two strides and then nothing.

A point below the target brackets the answer however small its energy is; what it cannot do is anchor a
SECANT, since a slope measured off the noise floor spans the whole onset knee. Now every point brackets,
geometric bisection is the always-valid fallback, and the secant is gated separately on BOTH endpoints
carrying signal. Measured immediately after: `kWall = 1` reached excess 1.003e-06 in 1 s, where the
same run had stalled at 2.266e-08 after 80 rounds.

### ✓ FIXED — `setDepthContact(wallStiffness = ...)`; 10 removes the escape entirely

With BOTH fixes in (pair-only excess, and bracketing separated from extrapolation), N = 11, n = 12,
target excess 1e-6, all reaching the target in ~1 s:

```
  kWall    excess     phi      side    wall%   pairOverlap   wallDepth   outside
      1   1.003e-06  0.93850  3.4236  94.17%   1.164e-03   0.0674 ed      28
     10   1.051e-06  0.95144  3.4002   0.28%   7.272e-04   0.0040 ed       6
    100   1.000e-06  0.95456  3.3946   0.00%   9.404e-04   0.0000 ed       0
```

The energy moves from 94% wall to 0% wall, the packing jams against ITSELF instead of the boundary, and
it lands denser for it — phi 0.9385 -> 0.9546. Default stays 1.0 so nothing changes unless asked;
**100 is the value to use** for a confined packing (10 already removes most of it, and is what the
mollified tier reached independently as `_DEFAULT_CONTAINER_STIFFNESS`).

### Verification — `tests/excessEnergyCheck.py`, **7/7**

```
  0  affine, box fixed      box area drift 0.00e+00, centroid drift 1.1e-16, areas exact to 1.1e-15
  1  dimensionless          the POWER; residual non-homogeneity wanders 5e-13 .. 1e-07 (see below)
  2  contact term only      spring 5.442177e-02, total - contact matches to EXACTLY 0.00e+00
  3  two-sided              both converge; 136.7% apart in phi, reproducible to 0.03% across runs
  4  no container           periodic path: excess 1.0194e-06 at phi 0.430091
  5  sweep from below       six unit squares, started BELOW jamming -> side 3.0479, overlap 0.00e+00
  6  load is on the bodies  wall share 14.69% -> 0.00%, zero penetration, zero escapees
```

Check 3's spread reproduces to 0.03% across runs (136.692%, 136.656%), so the path dependence is a
real property of the landscape and not scatter.

Check 5 is the only one measured against something outside this code. Check 6 is the one that would
have caught the original failure in a single line.

### ✓ ANSWERED — check 1's "non-homogeneity" was my own test's arithmetic

The drift wandered 5e-13 .. 1e-07 across runs of the identical test and I gated it loosely at 1e-6,
blaming absolute tolerances in the geometric predicates. Wrong. Sweeping the rescale factor against the
energy level:

```
  factor 2.00, 4.00    drift EXACTLY 0.00e+00 at every energy level (1e-4 .. 1e-7 excess)
  factor 3.00, 1.10    drift 1e-14 .. 8e-12, implied |dE| only 1e-20 .. 1e-22
```

The discriminator is not the energy — it is whether the factor is a **power of two**. 2 and 4 move a
binary exponent and nothing else, so the whole computation reproduces bit for bit; 3 and 1.1 perturb
every coordinate at the ULP level and the cubic law carries it through. My predicted `1/E` scaling is
refuted too: the implied absolute error is 1e-20 .. 1e-22, four orders below the 3e-16 I had assumed.

`energyScale`'s dimensional analysis is therefore EXACT, and check 1 now asserts that — `drift == 0.0`
and `rawRatio == factor**4` with `factor = 4.0`, a bit-for-bit test instead of a tolerance.

Three defects the suite itself found, all fixed:

- `_WALL_DOMINANCE` was referenced in the "did not reach" warning and never defined, so a run died with
  a NameError at the exact moment it was trying to explain a failure.
- Check 3 crushed its "from above" leg with `setPackingFraction(x3.3)`, which grows every polygon 1.8x
  linearly about its own centroid and carries the system past `dMax/rIn << 1`, where the repulsion
  REVERSES and the energy reads small at phi = 1.478. `_MAX_USEFUL_DENSITY` caught it. The leg now sets
  itself up with `holdExcessEnergy(excess * 1000)`: naming an ENERGY cannot leave the regime the law is
  valid in, because the energy is what the law reports. Naming a DENSITY can.
- `_depthWallSurplus` returned a zero force array even at the default `wallStiffness = 1`, reallocating
  the whole force buffer on every evaluation to add nothing. Returns None now.

### ✓ DONE — per-body wall stiffness in the KERNEL; it is now free

`contactKernel` takes an `exterior` index and a `wallStiffness` multiplier and applies it per work
item. Exact rather than approximate: energy and gradient are both linear in k, so there is no second
code path for the wall, only a different k. Mirrored in the numpy loop so the two stay the same
algorithm. `Model._depthWallSurplus` and its `confinementEnergyGradient` detour are deleted.

```
  N=11 n=12   wallStiffness 1.0   4.20 ms      wallStiffness 100.0   4.13 ms
  N=11 n=32   wallStiffness 1.0  17.40 ms      wallStiffness 100.0  17.29 ms
```

Identical within noise, against ~6x for the surplus path it replaces.

`tests/wallStiffnessCheck.py`, **4/4**. The absolute CUDA-vs-numpy gap is itself the proof the
multiplier is exact — the same roundoff discrepancy scaled by k and nothing else:

```
  wallStiffness    1.0   |dE| 7.28e-19        linearity gap at k=1: EXACTLY 0.00e+00
  wallStiffness   10.0   |dE| 7.28e-18        mixed vertex counts:  2.3e-17 / 1.2e-16
  wallStiffness  100.0   |dE| 7.28e-17
  wallStiffness 1000.0   |dE| 7.28e-16
```

### ‼ WORTH KNOWING GENERALLY — compare this law's energies ABSOLUTELY, never relatively

The first version of those checks gated on RELATIVE agreement and failed a correct kernel. Sweeping the
penetration depth of one square through one wall:

```
  depth/edge      energy          relE        |dE| absolute
  1.0e-01       5.067e-05       5.43e-12        2.8e-16
  1.0e-02       5.307e-08       9.53e-09        5.1e-16
  1.0e-03       5.331e-11       9.26e-06        4.9e-16
  1.0e-04       5.331e-14       2.27e-03        1.2e-16
  1.0e-05      -9.336e-17       2.92e+00        ~1e-16      <- NEGATIVE
```

The paths agree to ~3e-16 absolute at EVERY depth; only the denominator moves, because `E ~ d^3`. So a
relative tolerance fails a correct kernel on any packing whose wall contacts are shallow — which is
every packing near jamming, i.e. exactly the regime the knob exists for.

▶ The last row is its own finding: **below about 1e-05 of an edge the contact energy is entirely
roundoff and can come out NEGATIVE.** That is the floor underneath the ~1e-9 excess noise measured
earlier, and it says the law has a smallest meaningful indentation.

### ▶ SUPERSEDED — `wallStiffness > 1` used to cost ~6x per force evaluation

The surplus runs through `confinementEnergyGradient`, the slow per-body reference: 76.9 ms of a 92.3 ms
evaluation at N = 11. Correct, but the batched kernel carries ONE stiffness for the whole system, so
making this cheap means a per-body stiffness in `cuda/polyContactKernels.cu` and its driver. Given that
1.0 is demonstrably wrong for any confined packing and 10 is demonstrably right, that change looks
worth doing. Cam's contact law, Cam's call.

Both surviving rows above end with 17-19 vertices OUTSIDE the container and `wallDepth ~ 4.2e-03`,
about 4.4% of an edge, while typical pair contacts at the same excess sit around 1.8e-05 of depth —
some 240x shallower. So the reported side is optimistic by however much is poking out.

**Hypothesis, UNVERIFIED.** The depth tier has no wall stiffness at all:
`polyContactSystem.packingEnergyForce(packing, stiffness)` takes ONE stiffness and applies it to the
exterior body along with everything else, and `Model.setContainerStiffness` / `kContainer` are simply
not consulted on this tier. `_DEFAULT_CONTAINER_STIFFNESS = 10.0` exists because of exactly this
failure on the mollified tier — measured there, at equal stiffness "an overjammed packing relieves
stress by escaping through the wall rather than by overlapping its neighbours, since escape lowers the
confinement for everyone" — and it was worth +5.4% in density.

Testing it means giving the exterior body its own stiffness in `polyContactSystem`. Cam's call.

Note the final verdict is still guarded: `energySweep`'s `wallTolerance` would reject a state at
4.3e-03 (the notebook's own tolerance is 4.9e-04) and keep decompressing. It is the INTERMEDIATE held
states, and the "side at excess" printed during the stiffening rounds, that carry the escape.

### ‼ FIXED on the way — `mean(targetEdgeLength)` was averaging in the CONTAINER's edges

The wall is stored as one more polygon and its edges are the size of the whole system. At `N = 6, n = 8`
that turns a true mean edge of 0.133 into 0.200; `energyScale` raises it to the fourth power, so the
unit would have been **5x wrong and would move whenever the box was scaled** — exactly what a control
parameter must not do. Now `anneal._meanEdge`, container excluded. The same expression was already
present in `energySweep` and `bisectJamming` (setting `wallTolerance` and the sigma floor) and both are
switched over: tolerances there tighten by ~10% at `N = 11`, which is immaterial but correct.

## ✓ DONE — `distortion` moment family; and three infeasible constraint sets it exposed (2026-08-09)

`setConstraints(distortion = [1, 2])` takes moments of the DIMENSIONLESS distortion
`d_i = P/(g sqrt(A)) - 1`. Not a new quantity: `quantity(packing, 'shape')` in direct mode already
returned exactly this. What blocked a moment list was `familyMoments` force-collapsing shape to `[1]`
and the jacobian's direct-shape branch being hardcoded to one row. Both lifted, with `k d^(k-1)`
weighting. FD-verified: **1.7e-08 / 4.6e-08 / 4.1e-08** for `[1]`, `[1,2]`, `[1,2,4]`.

Kept separate from `shape` because they are different quantities and the names should say which:
`distortion` is dimensionless, `shape` with `deviation = True` is the isoperimetric DEFICIT, a LENGTH.
With polydisperse areas a moment of the deficit weights a big polygon more than a small one at equal
RELATIVE distortion, so it measures size as much as shape.

```
  distortion = [1]        cond 7.0e-01     comfortable
  distortion = [1, 2]     cond 1.1e-02     usable
  distortion = [1, 2, 4]  cond 2.5e-05     marginal
  anything containing -1  retraction diverges, EVERY combination tried
```

**A negative exponent on the DIRECT distortion is not usable, and it is structural.** A barrier exists
to repel zero, but `d = 0` here is the regular polygon -- reachable, desirable, and the anneal's
destination. Unlike the deviation families, whose barrier diverges before their quantity can reach
zero, this one is fighting where it is going.

### Three infeasible sets, all reported as something else

1. **`edge = True` (the DEFAULT) contradicts any `distortion` constraint.** Per-object edges plus fixed
   area pin the shape outright -- for a quadrilateral, equal edges at fixed area IS a square, `d = 0` --
   while the distortion rows hold `d` at its current value. `spreadShapes` moves geometry and NOT the
   edge targets, so after seeding, targets sat **89%** from the geometry. The retraction diverged inside
   `setConstraints` itself, conditioning 2.0e-26, distortion blown from 0.26 to **3.5e+06**. Now raises.
2. **`_enableShapeBudget` returned early on the DIRECT shape family**, leaving it in place, then
   `setShapeDeficit` demanded the deviation barrier it had skipped. Now checks for the barrier and
   upgrades, with a warning.
3. **Area moments cannot ride the deviation upgrade.** `deviation` is a flag on the whole SET, not per
   family, so a shape barrier makes area one too -- and area deviation is shrink-only `A0 - A`,
   measured at **-8.6e-03** after an ordinary relaxation, singular under `k = -1`. Areas are pinned
   per-object for the sweep instead.

### ▶ OPEN — `deviation` should be PER FAMILY, not per set

Item 3 costs a real capability: during a sweep the polygons can no longer TRADE size, which is one of
the two annealing freedoms Cam asked for. Making the flag per-family would let area moments stay direct
while shape runs the barrier. This is the first thing to try for a better packing.

### End to end on `tests/anneal.ipynb` (N = 11, seed 42, depth tier, L-BFGS)

```
  cell 6   phi 0.84876   overlap 2.845e-03   maxDistortion 0.2509
  cell 9   21.2 s, 76 steps -> phi 0.668533, overlap 0.00e+00, maxDistortion 5.773e-15
```

Regular to 5.8e-15 and exactly zero overlap. Equivalent to 11 unit squares in a box of side **4.056**;
the best known is around 3.877 (UNVERIFIED -- check against a reference), so roughly 9% short in phi.
That gap is a SEARCH shortfall, consistent with [[project_optimum_is_a_minimum]].

## ‼‼ BUG FOUND + FIXED — vertex-nearest energy was 3x too big, in ALL THREE implementations (2026-08-09)

`pairGradient`'s vertex-nearest branch was missing the `/3` that the edge branch carries, so every
vertex-nearest sub-stretch contributed **three times** its energy to the returned energy and to the
MEASURE group of the gradient. Fixed in `polyContact.py` and `cuda/polyContactKernels.cu`.

**The vendored reference has the same bug.** `polyContactReference.grad_pair` returns
`7.999681e-04` where the truth is `4.447851e-04` on the L-vs-cross medial-axis configuration, and its
own gradient misses a finite difference of `pairEnergy` by `3.55e-04`. This is the SECOND bug found in
the handoff's ground truth (after the `march` prefilter). It is no longer a trustworthy oracle for this
branch — check 4a compares against it and will now disagree wherever the branch fires.

`pairEnergy` was always right, and is what proved it: its finite difference matches a dense
boundary-quadrature reference sharing no code with `polyContact` to **2.4e-13**, while `pairGradient`
disagreed by 2.3e-2. The two now return identical energies.

### Why it survived 14 green checks — the coverage hole

**Every case in `GRADIENT_CASES` has ZERO vertex-nearest sub-stretches**, including both labelled
`[nonconvex]`:

```
  parallel faces |e|=1 / |e|=2 / rotated 30 / no vertex of A in B / vertex-on-face / crossed bars
  L vs cross [nonconvex] / L vs cross rotated [nonconvex]        ALL ZERO
```

The branch fires only when the LOOP body presents a **reflex vertex at the contact**, which needs the
obstacle nonconvex *where contact happens*, not merely nonconvex somewhere. New check `4c` in
`tests/polyContactCheck.py` covers it (grad==FD to 2.7e-13) and **asserts the branch was reached** —
an FD check that silently exercises nothing is the thing being guarded against.

### ▶ NEEDS CAM'S CALL — check 4b was passing for the wrong reason

`4b` asserts the gradient BREAKS at a medial-axis configuration (`max|dg| > 1e-6`). It was detecting
this factor-of-3 bug; with the bug fixed the gradient agrees with FD at **2.38e-11** and 4b now fails.
Its premise, "an invalid state has no gradient there", looks wrong: the medial-axis failure is that the
energy stops being a sensible penalty, NOT that the derivative is miscomputed — so FD agreeing is
consistent with the documented breakdown. Probing it directly, energy still *increases* with deeper
penetration (4.45e-04 -> 7.16e-04 over shifts 0 -> 0.30 at dMax/rIn = 3.12), so I could not reproduce a
sign reversal there either. Left failing rather than rewritten: this is your law's documented property.

## ✓ DONE — `anneal.ipynb` runs: confinement implemented, and three things it exposed (2026-08-09)

`packingEnergyForce` no longer raises on a container; `energySweep` runs the depth tier.

**1. The wall rides the ordinary pair loop.** Inverted winding makes it just another body, so it goes
through the same batched path and the same CUDA kernel. Writing it as a per-body Python loop first cost
**76.9 ms of a 92.3 ms** force evaluation at N=11 against 12.7 ms for the whole body-body term; batching
it gave **92.31 -> 18.88 ms**. `confinementEnergyGradient` keeps the slow spelling as the reference the
fast path is checked against (they agree to 2e-13). `BodySet.__init__` normalizes winding and would
undo the inversion, so the set is built via `__new__` — and that normalization is a real guard: a
counter-clockwise wall gives a **250x-too-large attractive well**.

**2. `dtMax = 0.03` throttles FIRE on the depth tier.** Same 4000 steps:

```
  dtMax      0.03        0.10        0.30
  max|F|   1.34e-07    1.05e-07    5.85e-09      and E 1.8e-09 -> 2.6e-11
```

Nearly identical wall time. That default was tuned for the mollified tier, whose contact force goes as
`1/sigma`; the cubic law's stiffness is `phi'' = 2kd -> 0` at contact, so the stable step is far larger.
NOT changed globally — the default is shared with tiers that need it. Open question: make it
tier-dependent?

**3. `energySweep` takes `minimizer`** — `"cg"` (default, Cam's call), `"lbfgs"`, `"fire"` for the old
FIRE-then-CG-polish pair. Also `anneal.py` gated the anneal on `modelType == "softDepth"`; the `depth`
tier belongs in the same branch (it has no regulator at all) and crashed on `model.sigma` being None.

### ▶ OPEN — the convergence threshold is not tier-neutral

`maxUnbalancedForce = 1e-5` is met immediately on the depth tier at N=11 (max|F| = 8.1e-07 straight out
of the spring stage, FIRE runs 0 steps). The cubic law's force is `k d^2`, so at `d ~ 1e-3` the forces
are ~1e-6 REGARDLESS of how badly the packing is resolved. A force threshold carried over from the
spring or mollified tiers does not mean the same thing here. Worth a tier-aware default or a
normalization by `k`.

## ‼ FIXED — two constraint families that cannot both hold, and a warning that hid it (2026-08-09)

`setConstraints(area = True, edge = [1, 2, 4, -1])` followed by `energySweep` produced hundreds of
"moment retraction did not converge" warnings, residual stuck at **5.13e-01 across 126 passes** with
conditioning collapsing to 1.18e-04. Not a tolerance problem — an EMPTY FEASIBLE SET.

`_enableShapeBudget` replays the caller's constraints with `shape = [1, -1], deviation = True` added,
keeping the edge family but collapsing its moment list to `[1, -1]`. Under `deviation = True` the edge
rows stop measuring lengths and start measuring `|l_ik - l0_i|`, held AWAY from zero by the `k = -1`
barrier — so no edge can ever be ideal. The shape ramp drives the isoperimetric deficit to zero, which
is exactly the state where every edge IS ideal. One family forbids the point the other aims at.
`constraints.py` already said so: *"for driving a packing to regular polygons prefer shape;
edge deviations are for holding a SPREAD away from zero."*

Fixed by dropping the edge family in `_enableShapeBudget` with a warning that names the reason. Nothing
is lost — the caller's moment list was already being discarded and replaced by `[1, -1]`.

### The warning was the real cost — it defeated Python's own de-duplication

Both moment alarms interpolate the live residual and pass count, and Python keys duplicate suppression
on the message TEXT. "5.130e-01 after 179 passes" and "5.129e-01 after 89 passes" are different strings,
so **every single occurrence printed**. Now keyed on the condition, once per constraint set.

The text was also misleading: "it is the moment TARGETS that were not reached" reads as a convergence
shortfall and sends you looking for a bigger iteration budget. Now says outright that a residual which
barely moves means the targets are unreachable and no iteration count fixes it.

**Rule worth keeping: a warning whose text carries live numbers will never de-duplicate.** Put the
varying quantities in the message only if the alarm is guarded by its own once-flag.

## ✓ SETTLED — finite boundaries need NO new code: wind the box CLOCKWISE (2026-08-09)

Cam proposed four pinned trapezoids forming a 1x1 void. Measured against a dense-quadrature reference
(`tests/wallFrameCheck.py`), that construction is **exact at faces (0.00%)** and wrong at corners:

```
  corner depth   0.02      0.05      0.10
  frame error    7.4%     17.5%     32.2%     always UNDER-reading
```

The miters are the cause, and no convex partition avoids them: the void's corners are **reflex vertices
of the wall region**, and no convex piece can contain a reflex vertex, so a seam must run out from every
corner. The law reads any boundary as free surface where `d = 0`, and a seam is wall interior, so
corners come out too soft — worst exactly where a compression protocol loads hardest. (Slab half-planes
fail comparably: they charge `dx^3 + dy^3` against the true `(dx^2+dy^2)^{3/2}`, 29% low on the diagonal.)

**The fix is one line of geometry.** `pairEnergy` reads membership from the winding and does not
normalize it, so passing the box boundary as a single CLOCKWISE loop makes the confining region its
exterior. Exact everywhere — faces and corners — to `1e-10`, which is the reference quadrature's own
convergence limit, and the force matches an independent FD to `7.6e-10`. One body, not four, no seam,
no pins needed.

It is also **unconditionally valid**, unlike body-body contact: the exterior of a convex region has no
medial axis, so the `dMax/rIn << 1` cap does not apply and a wall can be pressed arbitrarily hard.

Note the trap this rides on — a clockwise wall inverting the membership test is the same thing that
collapsed five squares onto a point on 2026-08-01. Here it is the mechanism, so it must be *asserted*,
not left implicit; an explicit `exterior = True` flag would be safer than relying on winding.

## ✓ DONE — L-BFGS minimizer; CG was the bottleneck, not the kernel (2026-08-09)

`minimize.minimizeLBFGS` / `Model.minimizeLBFGS`, spelled exactly like `minimizeCG` so a notebook call
swaps in place. On the `penetrationDepth.ipynb` configuration (N=32, n=32, seed 42, depth tier):

```
                200 steps    max|F|       E
  CG              736.6 s    4.72e-10    4.52e-11
  L-BFGS           54.2 s    2.67e-10    3.58e-11      13.6x, and wins on BOTH
```

The win is entirely in evaluations per step: CG needs `c2 = 0.1` to keep a Polak-Ribiere direction
meaningful and so almost never accepts its first trial (~20 evals/step measured), whereas L-BFGS scales
its own direction, admits `c2 = 0.9`, and takes alpha = 1 outright — **1.18 evals/step**. This is what
`contact.tex`'s status section meant by "the minimiser matters more than expected".

Verified in `tests/lbfgsCheck.py` (6 groups, green): analytic minimizer of a well-conditioned quadratic
to 8e-12; **parity with scipy L-BFGS-B** at equal iterations and equal memory on a kappa=1e4 quadratic
(6.9e-07 vs 7.9e-07); monotone energy; evals/step; constraint residual held at SHAKE tolerance; pins
unmoved. `tests/minimizerCompare.py` reruns the timing.

Two things to know about the checks. A stiff quadratic is NOT a correctness test — L-BFGS with memory 10
genuinely crawls on kappa=1e4, and reading that as a bug cost an hour; the two-loop recursion was
verified directly against an explicitly assembled BFGS matrix (4e-17) before scipy confirmed the
behavior was normal. And an earlier "L-BFGS beats gradient descent" check passed only because GD had
DIVERGED to 1e117 — a check that cannot fail for the right reason is worse than no check.

### ✗ RETRACTED — the "3.6x CUDA regression" from staging `LoopFrame` in shared memory

It does not reproduce. On the same notebook configuration the current staged-frame kernel evaluates in
**35.07 ms**, better than the 41.2 ms it was supposed to have regressed from, and the reported 150.2 ms
does not appear at any configuration tried. The two numbers were also read at different states (E
1.14e-06 against E 3.86e-09), so they were never comparable. Keep the staged frames.

Static analysis backs that up: `-Xptxas -v` reports **98 registers, 0 spill stores, 4096 bytes smem**,
so the frames cost no spill, and on a GTX 1650 (1:32 fp64) the kernel is transcendental-bound — staging
removes a normalize and a sqrt per access, which cannot be a loss. The remaining lever is the O(M^2)
`nextBreakpoint`, not the frame layout.

### ‼ MACHINE — `/dev/nvidia-uvm` returned EIO; every CUDA call failed as error 999

`cudaGetDeviceCount` returned 999 while `nvidia-smi` was perfectly healthy, and `pyPolygon` fell back to
the numpy tier. `strace` located it: `openat("/dev/nvidia-uvm") = -1 EIO`. The kernel module needs a
reload (`sudo modprobe -r nvidia_uvm && sudo modprobe nvidia_uvm`) — Cam did this and it cleared.
Worth recognizing on sight, because the symptom is "everything got 100x slower", not an error.

## ✓ DONE — depth-contact law on the GPU, 19-67x (2026-08-08)

`cuda/polyContact.cuh` + `cuda/polyContactKernels.cu`, bound through `cudaOverlap.polyContactCuda`.
Matches the numpy assembly to **relE 2e-16 and max|dg| ~1e-17** on every case tried.

```
              numpy      cuda
n= 6 N= 9    68.1 ms    1.59 ms   43x
n=12 N= 9    99.0 ms    2.89 ms   34x
n=32 N= 9   125.9 ms    6.76 ms   19x
n= 6 N=25   193.2 ms    2.90 ms   67x
n=12 N=49   525.0 ms   10.21 ms   51x
```

**This law suits a GPU far better than softDepth did.** There the depth was a softmin over ALL of B's
half-planes, so every quadrature node's gradient touched every vertex of B — nB atomics per node,
forcing a shared-memory accumulator and a block-per-pair decomposition just to survive. Here the
nearest feature is a SINGLE edge or vertex, so a sub-stretch's gradient touches at most two vertices of
A and two of B. Atomics are cheap.

**Nothing is stored per thread.** Sorted crossings and envelope breakpoints are both walked by repeated
minimum-search rather than materialized, so a thread needs no local arrays and never spills. B is staged
in shared memory with its periodic shift folded in.

**One bug, and it was a clean factor of two.** The law is `E = 1/2 sum over ORDERED pairs`; the cull
emits both orders and the kernel integrates each, so the raw sum is `E_AB + E_BA` and needed halving.
Worth recording because a factor of exactly 2 reads like a stiffness convention rather than a missing
term — it was caught only by comparing against the host on a two-square case with a known answer.

Contracted as `polyContactSystemCheck` check 8: n = 4, 6, 12, 13, 32, 33, 64, **mixed n = 4/7/13 in one
system**, and `POLYCONTACT_MAXN` reported rather than truncated. The mixed case is the one that matters
— a uniform-n packing cannot detect a kernel that assumes uniform n.

- [ ] Uniform edge grid — still argued premature; the M-scaling is not biting.
- [ ] HELD at Cam's request: fixed-boundary confinement.
- [ ] `Model`'s depth tier still routes through the numpy assembly, not the kernel.

## ‼ BUG IN THE HANDOFF'S REFERENCE — `march` misses feature switches (2026-08-08)

`reference/polycontact_ref.py`'s `march()` prunes candidates before selecting the winner, using a
bound computed on each candidate's window CLIPPED to the interval. That understates the true maximum of
a candidate that is invalid over part of the interval, so the bound is too tight and GENUINE WINNERS
GET PRUNED. The winner is then wrong, crossings are computed for the wrong curve, and the switch is
never found.

Measured in `polyContact` before the fix:
- **5 genuine feature switches missed across 107 spans** of a 9-body 12-gon packing, moving that
  packing's total energy by **0.7%**;
- **23 missed over 120 random chords** on cross / flower(16) / L shapes. Zero after.

**The reference's own suite cannot catch this.** `E_pair_closed` partitions by `feature_partition`
(sample-and-bisect), not by `march`, so the production path never exercises it; and its march test
compares only two hand-picked chords of one shape. Worth reporting back upstream.

Fixed here by dropping the prefilter entirely: in the batched form the arrays stay full-size and it was
only a mask, so removing it costs no arithmetic. The window test is exact and is the only restriction
needed. Guarded by `polyContactCheck` check 7b — a dense nearest-feature scan asserting nothing is
missed.

**How it was found, which is the transferable part:** batching `march` changed the system energies in
the 3rd digit. The natural assumption was that batching had broken something. Checking BOTH against the
reference showed the new values were right to 1e-16 and the old ones wrong. A discrepancy appearing
after a refactor is not evidence about which side is wrong.

### State of the polygon-contact work

| file | contract | status |
|---|---|---|
| `polyContact.py` | `tests/polyContactCheck.py`, 13 groups | green; matches reference to 6.68e-17 |
| `polyContactSystem.py` | `tests/polyContactSystemCheck.py`, 8 groups | green |
| `model.py` | `setModelType("depth")` / `setDepthContact(k)` | wired; periodic/free only |

System energy timings (jittered lattice, energy+gradient): n=6 N=9 65 ms, n=12 N=9 88 ms,
n=32 N=9 130 ms, n=32 N=25 358 ms. All match the reference to ~1e-16.

- [ ] Uniform edge grid (handoff step 4) — still argued PREMATURE: it targets `spans` +
  `nearestFeature` while the M-scaling is not yet biting (n=12 -> n=32 is 88 -> 130 ms, sub-linear).
- [ ] CUDA port of the pair law.
- [ ] HELD at Cam's request: fixed-boundary confinement — slab walls (what the spec's test 10 uses) vs
  a genuine inverted domain in `spans`.

## ▶ depth tier wired into Model; system layer built, one collapse trap found (2026-08-08)

`setModelType("depth")` / `setDepthContact(stiffness)` now select the exact-distance contact law.
`polyContactSystem.py` carries the many-body layer: CSR bodies, body-level broad phase, assembled
energy/gradient, rigid-body relaxation, validity monitor, and the initialization protocol.

Verified: assembled gradient vs FD 1.03e-12 (free) and 2.15e-12 (periodic); rigid gradient vs FD
1.41e-11; `relaxRigid` took E 4.34e-04 -> 8.95e-07 in 368 L-BFGS iterations with body area preserved
to 5.55e-16 and dMax/rIn 0.0262.

**‼ TRAP — an unconstrained relaxer COLLAPSES the bodies.** The law is purely repulsive, so with every
vertex free its global minimum is every body shrunk to a POINT. Measured: 16 hexagons at phi 0.9755
reached E = 0 **in one L-BFGS iteration** by shrinking to 68% of their area, and reported
`dMax/rIn = 0.000` and a perfect force balance while doing it. It looks exactly like a successful
relaxation. `relaxRigid` (three DOF per body) is the fix for the protocol; `relax` (two per vertex) is
only meaningful with a caller-supplied shape term, which is what the `Model` path provides.

**Two more test-configuration traps, both of which produced convincing-looking nonsense:**
- a body larger than HALF the box makes the single minimum-image shift flip discontinuously at the
  half-box. Four hexagons in a box of 1.4688 broke the assembled gradient's FD at 50% while net force
  sat at 3e-18. Now warned by `candidatePairs`.
- a perfect square lattice is symmetric, so every body's rigid gradient vanishes by symmetry (1.16e-17)
  and any relaxation test on it is vacuous. Jitter before testing.

- [ ] **No container.** A wall polygon is not an obstacle under this law: a body correctly INSIDE the
  box has its whole boundary inside the wall, so the law would push every body OUT. `packingEnergyForce`
  RAISES on a container rather than miscomputing. Two ways forward, Cam's call: half-space slabs (what
  the spec's own test 10 uses -- four of them make a box, no new math), or a genuine inverted domain in
  `spans`. HELD OFF at Cam's request 2026-08-08.
- [ ] `systemValidity` is expensive (a 400x400 inradius probe per body) -- diagnostic only, not per step.
- [ ] Uniform edge grid (handoff step 4), CUDA, adhesion, analytic Hessian.

## ▶▶ NEW LAW — polygon contact supersedes softDepth (2026-08-08)

`notes/files(3).zip` delivers a verified spec + reference for a contact law on simple polygons,
convex **or not**:

    E = 1/2 sum over ordered pairs (P,Q) of  int_{dP cap Q} (k/3) d_Q(x)^3 dl(x)

with `d_Q` the EXACT distance to the boundary. Closed-form energy AND gradient, no quadrature anywhere,
no regularization length anywhere, nonconvex handled with NO decomposition. Spec now in
`notes/polygonContact/`; reference vendored at `tests/polyContactReference.py`.

**It independently rejects two things built here for softDepth, with numbers:**

- *vertex-sampled quadrature* — blind to shallow face-on-face contact and INVERTS the face/vertex
  contrast (true ratio 2.00, reported 0.001). Found independently here on 2026-07-31.
- *convex decomposition* — low by a factor of **19** on a convex control case, because the distance to
  a piece's boundary is not the distance to the body's boundary. **That is what `convexDifference.py`
  does.** The new law needs no decomposition at all.

Also rejected: vertex-only softmin/softmax features (*impossible*, not merely inaccurate — for two
crossed bars every vertex of each body lies outside the other, so the vertex-depth vector is identically
zero on a 4% overlap), smoothed minimum-translation distance, and spheropolygons.

### Done — `polyContact.py`, contract green

`tests/polyContactCheck.py`, 13 groups, all PASS. Port matches the reference **entry by entry to
6.68e-17** on every gradient case; energy matches independent quadrature to 1e-13; gradient matches its
own finite difference to 1e-11. 21x faster than the reference on L-vs-cross, from replacing
sample-and-bisect (~3000 nearest-feature queries per span) with the closed-form output-sensitive march.

**One port bug found, and only the finite difference caught it.** `march` returns exact breakpoints but
an unreliable WINNER list — it identifies the winner at `t + 1e-13`, which does not separate two
candidates crossing shallowly. Trusting it gave (E3,E3) where truth was (E3,E1) on crossed bars
perturbed by 1e-5, inflating the energy EIGHTFOLD while breakpoints stayed correct to 1e-9. Features are
now re-identified at sub-stretch midpoints, which is what the reference does. Conservation passed
throughout — as the handoff warns, it vanishes structurally and is nearly worthless as a test.

### THE ONE HARD CONSTRAINT — validity

`d_B` has a ridge (the medial axis) at ~the inradius. Crossing it is not an accuracy loss but a **sign
reversal**: past the ridge the leading edge's depth decreases and the bodies are PULLED THROUGH.

    REQUIRED:  max over overlap components of  dMax / rIn  <<  1

For a limbed shape `rIn` is the **limb half-width**, not the particle size. `polyContact.validityRatio`
computes it. Persistent rather than transient, so checking the converged state suffices for packing
generation — but **not for AQS**, where the trajectory is the observable and a pass-through injects a
spurious irreversible event, contaminating exactly the signal being measured.

- [ ] Uniform grid over all edges of all bodies (cell ~2x mean edge) — serves crossings, nearest
  feature and membership; O(M) total. Handoff build step 4.
- [ ] L-BFGS on the analytic gradient. Do NOT use FIRE alone: from a 1e-4 perturbation of a minimum,
  L-BFGS reached 9.8e-11 in 41 iterations where FIRE stalled at 5.5e-6 after 150.
- [ ] Initialization protocol: certified-disjoint lattice, adaptive compression, reject any step with
  `dMax/rIn > 0.35`. Disjoint init is necessary but NOT sufficient — one large step reproduced 1.0000.
- [ ] CUDA port.
- [ ] Wire into `Model` as a tier.
- [ ] Decide what happens to `softDepth.py` / `convexDifference.py` — Cam's call, not mine.

## ✓ DONE — softDepth walled packings 30x faster; and a tier-switching trap named (2026-08-01)

`packingEnergyForce` on Cam's walled 5-square config went **1242 ms -> 41 ms** per force evaluation,
and now barely grows with N because the pair cost is on the device:

```
N= 5   41 ms       N=11   57 ms       N=32   85 ms
```

Four changes, in order of size:

1. **The GPU now takes the pairs even when a container is present.** The dispatch gate required
   `containerIndex is None`, so every walled packing ran entirely on numpy. The kernel already excludes
   the container from its pair loop, so the split is exact: pairs on the device, confinement on the
   host. This was the single biggest factor.
2. **The container is batched.** Every polygon's boundary against the wall now goes through ONE
   `edgeSetAgainstLoop` call instead of one per polygon — 93% of the remaining time was N independent
   root solves of ~50 numpy calls each, on four-element arrays. `contactIntervals` and
   `edgeLoopEnergyForce` are now thin wrappers over edge-list forms.
3. **The loop frame is hoisted out of the root finder.** It was rebuilt 1419 times per force
   evaluation of five squares; it is constant per pair.
4. **The Gauss rule is cached** (`_gaussRule`); `leggauss` was re-solving for its nodes every call,
   ~18% of an evaluation. And `_ROOT_STEPS` 24 -> 16: the safeguard's worst case is bisection, so 16
   guarantees 1.5e-05, inside check 8's tolerance. 12 was tried and fails that check at 2.4e-04 even
   though its ENERGY error is 1.2e-13 — the energy is insensitive to the root, the interval is not.

Verified throughout: numpy vs hybrid `relE 1.7e-08` (the order-16 quadrature residual), `sum F 1.1e-16`,
and all three suites green (12 pair checks, 7 device checks, 3 packing checks).

## ‼ PATTERN — setters that silently change `modelType`. Bitten three times in one day.

- `setSofteningFraction` sets `modelType = "mollified"` on its first line, so
  `setModelType("softDepth"); setSofteningFraction(...)` silently runs the mollified tier. Cost: a whole
  session of "why is CG not converging" on a packing that was never on soft depth.
- `setMollification` does the same (model.py:1434).
- **`energySweep` has no softDepth path at all.** `anneal.py:588` calls `setMollification`, flipping the
  tier back; `anneal.py:647` then switches to `"sharp"` for the verdict; `anneal.py:160/280/320` branch
  on `modelType` with only `"sharp"` and `"mollified"` cases. And `setSoftDepth` leaves `sigma = None`,
  which `anneal.py:520` and `:582` do arithmetic on. So `annealRecommended.ipynb` cell 12
  (`setModelType("softDepth"); setSoftDepth(...); energySweep(...)`) cannot work as written.

- [ ] Decide: either give `energySweep` a soft-depth finishing phase, or document that soft depth is a
  POST-sweep polish (`minimizeFIRE`/`minimizeCG` after `energySweep` returns) — which is what Cam said
  he wants, and what `softDepthPackingCheck` check 2 verifies works.
- [ ] Consider making tier changes explicit rather than a side effect of a width setter.

## ‼‼ STANDING DEBT — softDepth under-reads contact energy at REFLEX CORNERS. Cam wants reminding.

**Accepted deliberately on 2026-08-01, NOT fixed. Cam: "please please add a note and don't let this get
forgotten. Remind me often."** Do not close this without his say-so.

Outside a convex piece the half-plane softmin reads the largest single half-plane VIOLATION, not the
Euclidean distance. At a corner of interior angle `theta` the depth is short by `sin(theta/2)`, so with
`phi ~ h^(5/2)` the contact energy is short by `sin(theta/2)^(5/2)`. Only COMPLEMENTED nodes (pockets)
are evaluated from outside, so only their corners carry it — and those are exactly the reflex vertices
of the shape.

| builder shape (phi=0.8, seed 42) | reflex corners | pocket angle med / min | energy factor med / worst |
|---|---|---|---|
| n=8 kappa=4 | 18 | 133.9 / 125.4 deg | 0.812 / 0.745 |
| n=16 kappa=4 | 59 | 154.9 / 55.2 deg | 0.941 / 0.146 |
| n=32 kappa=4 | 115 | 158.5 / 2.2 deg | 0.957 / **0.0001** |
| n=32 kappa=20 | 312 | 82.6 / 0.6 deg | **0.354** / 0.0000 |

The median is benign; the TAIL is not. A 2.2-degree notch reads 1e-4 of its true contact energy and is
effectively invisible to contact. At kappa=20 even the median is 2.8x too weak. The error is
single-signed, so it biases rather than averaging out — **and this project turns on corners finding
small gaps, which is exactly the geometry it suppresses.**

Accepted to keep `ell_i` affine along an edge, which the exact envelope walk, the single-contact-interval
guarantee, and sec:chord's closed-form stretch integral all depend on.

- [ ] **The fix when it matters: exterior-exact distances on POCKET NODES ONLY** —
  `min_i dist(x, segment_i)`, which coincides with `min_i ell_i` inside (sec:conservation proves the
  argmin's foot lies on its own segment). Leaves the convex fast path, the affine structure and the
  closed form untouched.
- [ ] Do NOT add vertex terms to the general softmin instead. The obvious form `-|x - v_j|` breaks the
  interior: at the centroid it is roughly minus the circumradius, dominates the min, and drives `h`
  negative INSIDE.
- [ ] `convexDifference.warnOnSharpPockets` surfaces it at runtime (bucketed by 5 degrees). **Do not
  silence it.**


## ‼ BLOCKER — softDepth does not minimize. Everything below it is on hold (2026-08-01)

Cam looked at the DRAWING and said "this isn't minimizing at all." He was right, and the numbers had
been saying "converged" the whole time.

**Two separate faults, found in that order.**

**1. The convexity precondition was never enforced — FIXED (guard only).** Lemma 1
(`dist = min_i ell_i`) holds only for convex loops. With a reflex vertex the supporting line of an
adjacent edge cuts the interior, `min_i ell_i` goes negative INSIDE, so `h` does, `[h]_+` never fires,
and the energy silently collapses to ~0 however deep the overlap.

`generateEquilateralPolygons(N=32, n=32, kappa=4)` gives **197 reflex corners across 1024 vertices** —
`h` at one polygon's own CENTROID reads **-4.29e-02**. softDepth scored that packing at E = 5.2e-13,
max|F| = 1.9e-10 ("converged") where the sharp tier scores E = 3.907, max|F| = 0.658. The builder
**cannot produce convex polygons above n=4 at any kappa** (measured at kappa = 4, 20, 100, 1000).

`isConvex` existed from the start and was never called on the energy path. `requireConvex` now runs on
every evaluation and raises. The real remedy is sec 15's convex decomposition + softMAX (46), NOT built.

**2. Even fully convex, a quench from a random start lands badly — but the FORCE IS CORRECT.**
16/16 convex squares, phi = 0.8, identical start, identical constraints:

```
SHARP:      real overlap 6.70e-01 -> 1.48e-09    finds the valid packing
softDepth:  real overlap 6.70e-01 -> 9.9996e-01  and reports max|F| = 1.1e-08
```

Two hypotheses tested and BOTH DISPROVEN, recorded so they are not retried:

- *dropped contacts.* Brute force over every ordered pair and all 9 periodic images reproduces
  `packingEnergyForce` to a ratio of **1.000000**. The covering-radius cull and the first-vertex
  minimum-image shift drop nothing.
- *wrong force.* Started from the zero-overlap packing the sharp tier finds, softDepth evaluates it at
  E = 7.77e-11 and FIRE **holds it there** (E -> 4.49e-11, real overlap 7.1e-09). The energy recognizes
  the valid packing as its minimum and the force does not push off it.

So this is a SEARCH failure on the softDepth landscape, not a model or implementation fault — the same
shape of conclusion as [project_optimum_is_a_minimum] for the sharp tier.

**The mechanism is intrinsic to a depth law and worth stating.** softDepth penalizes penetration DEPTH,
never overlap AREA. For a contact of chord `L` and depth `d`, `E ~ L d^(5/2)` while `area ~ L d`, so at
FIXED area `a` the energy is `~ a d^(3/2)` — strictly decreasing as the overlap is spread thinner and
wider. A quench can therefore lower softDepth energy monotonically while the real overlap area GROWS,
which is exactly what the run above did. The area-squared sharp law has no such direction. This is the
flip side of the boundary-area trade: choosing the exponent freely is what buys C2 contact, and the
price is that area is not in the functional at all.

- [ ] Needed regardless: a PACKING-level check that a descent decreases an INDEPENDENT measure of real
  overlap, not just the tier's own energy. The sharp tier is that measure and is already available. The
  pair-level suite cannot see any of this.
- [ ] The note's own "free consistency guard" (sec:chord) is the cheap version: assert
  `chord * delta / area` lies in [1/2, 2] on every overlapping pair, as a permanent debug-mode check.

**Process failure worth recording.** Both faults were reported by me as successes because I read
`max|F|` and never checked the configuration was physical. The CUDA suite compounded it: checks 1 and 3
ran on the non-convex builder packings, so numpy and CUDA agreed to 1e-12 on a meaningless quantity —
that validated the PORT against itself, not the model. Only the hand-built `ring()` cases (check 2) and
the square cases were ever valid. Cross-check against the sharp tier on the working configuration, not
against the same tier in a different language.


## ▶ BUG — CG is a NO-OP on the mollified tier (2026-07-31)

Found while diagnosing why `tests/penetrationDepth.ipynb` "converged slowly". At N=32, n=32,
periodic, constrained, sigma = 2.18e-04:

```
FIRE   393 steps   20.1 s -> max|F| 3.958e-06
CG    1000 steps  185.4 s -> max|F| 3.958e-06   converged = False
```

`max|F|` is **bit-identical** before and after 1000 CG steps and 185 seconds. That is not slow
convergence, it is no convergence — CG burns its entire budget and moves nothing. The same packing on
the soft-depth tier reaches 1.9e-10 in 67 FIRE steps and CG confirms it in 7, so the minimizer is fine
and this is specific to the mollified tier.

Possibly the same root as [project_cg_no_transient_targets] / [project_newton_still_broken], but those
were about co-evolving targets and this is a plain constrained packing. Not yet investigated.

## ✓ FIXED — `setSofteningFraction` silently steals the softDepth tier (2026-07-31)

Cam's notebook cell read:

```python
packing.setModelType("softDepth")
packing.setSofteningFraction(fraction = 1e-2)
```

`setSofteningFraction` sets `self.modelType = "mollified"` as its FIRST action, so it overwrites the
line above it. The notebook had been running the mollified tier throughout — none of the softDepth work
was being exercised, and the "slow convergence" was the bug above.

He reached for it because there was no fraction-based softDepth setter, which was flagged and not acted
on. `setSoftDepth(fraction = ...)` now exists and is the preferred spelling, since an absolute
`epsilon` silently becomes nonsense when edges are short (at n=32 an innocent `epsilon = 1e-2` is 46%
of an edge and rounds the corners to 9.3x the polygon). Giving both raises.

- [ ] The one-line notebook change is `setSofteningFraction(fraction = 1e-2)` ->
  `setSoftDepth(fraction = 1e-2)`. Cam drives the notebook; not touched.

## ✓ DONE — softDepth on the GPU: 122–302x, and a periodicity bug found on the way (2026-07-31)

`cuda/softDepth.cuh` + `cuda/softDepthKernels.cu` (kernels and driver in one file, as `selfRepulsion.cu`
does), bound through `cudaOverlap.softDepthCuda`, dispatched automatically from
`softDepth.packingEnergyForce`. **Pair interactions only** — adhesion and the container stay on numpy,
since silently dropping either would be worse than being slow.

| | numpy | cuda | |
|---|---|---|---|
| N=32 n=4 | 2215 ms | 7.3 ms | **302x** |
| N=32 n=32 | 3186 ms | 26.1 ms | **122x** |
| N=64 n=8 | 5307 ms | 21.0 ms | **253x** |

End to end, the exact call in `tests/penetrationDepth.ipynb` —
`minimizeFIRE(maxSteps = 10_000, fThreshold = 1e-3)` at N=32, n=32 — now converges in **2.8 s** (67
steps, 42.5 ms/step). It was ~6.4 hours.

**Three facts from the note did the work.** `h_eps` is concave, so `{h >= 0}` on an edge is ONE
interval — at most two roots, no enumeration. The root finding is fixed-count and branch-free, so the
dominant cost has zero warp divergence. And the panels are consumed as produced, so the ragged panel
count never reaches memory.

**The device envelope walk is EXACT and better than numpy's.** Each `ell_i` is affine along an edge, so
a switch is one divide; numpy probes `max(16, 2 nB)` points instead and can miss a segment shorter than
the probe spacing. Deliberately NOT ported back: numpy runs 24 root steps and a probe, the device runs
12 and an exact walk, so agreement between them is a genuinely independent check rather than two
spellings of one loop. `tests/softDepthCudaCheck.py` (7 checks) leans on exactly that.

Agreement, hand-built overlapping polygons: **relE 0 to 8.9e-16, max|dF| ~1e-17** at n = 12, 13, 32,
33, 64 and mixed n = 4/7/13. On builder packings the gap is 2.4e-08 at order 16 — that is QUADRATURE,
not port error, and check 3 proves it by collapsing the gap to 7.6e-12 at order 32.

**Cam's edge cull is in and is exact**, not heuristic: `h = min_i ell_i - eps log(total)` with
`total >= 1`, so `h > 0` forces the point strictly inside the loop, hence within `radius` of its
centroid and within `|e|/2` of the edge midpoint. Measured, it removes **68%** of edges at N=32 n=32
(6208 -> 1991, of which only 277 actually contact). It did NOT move the wall clock, so the time is
elsewhere — see below. Kept because it is free and matters more at lower density.

NOT routed through `candidateEdgePairs`: edge-EDGE candidacy is not conservative here, since an edge
lying wholly inside the loop has `h > 0` along its length while crossing none of the loop's edges.

### Found on the way: periodicity was silently OFF in `packingEnergyForce` — FIXED

The pair shift was applied to the boundary AND the loop, which is a rigid translation that cancels,
while the cull still selected pairs by their minimum image. A pair overlapping only across the seam
measured **exactly 0.0** against 8.2e-05 of real contact. `penetrationDepth.ipynb` is periodic, so this
was live. Fixed (only the loop moves) and guarded by check 11 in `tests/softDepthCheck.py`, which is
also the first test to exercise the assembly at all — checks 8-10 all drive `edgeLoopEnergyForce`
directly, which is how it survived.

- [ ] **The next 2-4x is the loop-side deposit, not the root finding.** Every quadrature node issues 4
  shared-memory atomics per plane, and all 64 threads of a block contend on the same `nB` slots —
  ~1.4M serialized shared atomics at N=32 n=32. The fix is one WARP per edge with lanes striding over
  planes, so each lane owns a distinct slot and the contention disappears (the `selfRepulsion.cu`
  shape). Worth it only if 26 ms/eval starts to bind.
- [ ] Order is still coupled to epsilon with no automatic choice (order 16 -> 3.9e-06 at
  `eps/edge = 1e-2`, 7.7e-05 at 1e-3). Only 16 and 32 are tabulated on the device.
- [ ] `generateEquilateralPolygons` cannot be pushed to real contact at large `n`: phi = 1.6 leaves
  |F| ~ 7e-05 at n=64 and phi >= 2.0 gives no contact at all. The CUDA checks hand-build their
  configurations to dodge this, but the builder is worth a look.

## ▶ NEW — softDepth is a BOUNDARY-AREA law, and now actually integrates its boundary (2026-07-31)

Three families of overlap law, and why we sit where we sit:

| family | energy | verdict |
|---|---|---|
| boundary–boundary | `int_dA int_dB K(\|x-y\|)` (Plummer panels) | **rejected** — the kernel has no notion of *inside*, so it is not a penetration law: it is non-monotone in depth and admits pass-through. Also locks stiffness to geometric fidelity through one `lambda`. |
| boundary–area | `int_dA phi(h_eps^B) dl` (**softDepth**) | **pursuing** — the measure and the contact law decouple, so the exponent is ours to choose. Hertz gives `d^(3/2)` on faces (C2 at onset) and `d^(5/2)` at corners, against area–area's `d^1` (C1 only) and `d^3`. No crossing topology anywhere. |
| area–area | `\|A ∩ B\|` (sharp / mollified) | **rejected** — the exponent is not ours: `a^2` is harmonic (C1), and Hertz would need `a^(5/4)`, whose second derivative blows up at `a = 0`. Corner response is two powers weaker no matter what. And evaluation is combinatorial, which is where the tangency degeneracy lives. |

**The bug that was blocking it.** `packingEnergyForce` passed `pointLoopEnergyForce` the polygon's
VERTICES and nothing else — zero samples in any edge interior. That is not a low-order quadrature of
`int_dA phi(h) dl`, it is a different and wrong law. Two squares meeting face to face have no vertex of
either inside the other, so it returned exactly **0.0** against 7.7e-02 of real overlap. It was also
wrong where it was nonzero: 2.563e-02 on a corner contact whose true value is 7.272e-03.

**Fixed** by `edgeLoopEnergyForce` — Gauss-Legendre per edge, with two splits:

- **at `h = 0`**, killing the `5/2` branch point in `phi`. Safe and cheap because `h_eps` is CONCAVE
  (`-eps` times a log-sum-exp of affine functions), so `{h >= 0}` on an edge is a single interval —
  at most two roots, no crossing enumeration. `contactIntervals`.
- **at the softmin's envelope switches**, where the active half-plane changes and `h` turns over on the
  scale of `eps`. These are EXACT linear solves, not root finds, since each `ell_i` is affine along the
  edge. `envelopeCuts`. Not optional: at `eps/edge = 1e-3` a single panel is wrong by **2.2e-03 at
  order 32 and non-monotone in the order** — worse than being wrong, since the error gives no signal.

The moving limits contribute nothing to the gradient (`phi(h) = 0` at a crossing, so the Leibniz
boundary terms vanish) and are held fixed when differentiating; check 9's FD confirms it. Forces gained
two terms: the node's barycentric split back onto its edge, and a TANGENTIAL force from `d\|e\|/dv`,
since `dl = \|e\| dt` moves with the geometry. Both are torque-free by construction.

Verified in `tests/softDepthCheck.py` checks 8–10: energy matches an independent uniform boundary walk
to 0.0 / 3.7e-11 / 1.5e-10 / 1.3e-11 on four contact types; forces and torques sum to ~1e-17; FD to
1e-11..1e-13.

- [ ] **Order is coupled to epsilon and there is no automatic choice.** Order 16 gives 3.9e-06 at
  `eps/edge = 1e-2` but only 7.7e-05 at 1e-3 (needs 32–64 there). Corners are easy at any order
  (7.0e-10 at 16). Check 10 prints the table; picking from it is currently manual.
- [x] **CONTAINER FIXED (2026-08-01).** It had contributed exactly zero, always: the confinement path
  reversed a loop's winding on the reasoning that this negates `h`, but `h` is a softMIN and
  `min(-ell) = -MAX(ell)`, so the reversed loop's `h` was negative both inside AND outside. Now
  `edgeLoopEnergyForce(..., confine = True)` penalizes `[-h]_+` on the UNREVERSED loop. `h` is concave,
  so `{h >= 0}` is one interval and `{h <= 0}` is its complement, `[0, t0]` and `[t1, 1]` — the same
  roots bound both, nothing new is solved for. Verified: inward force scaling with excursion, forces
  summing to 1e-18, FD 3.2e-11, zero energy for a polygon wholly inside. Check 12.

  `-h` measures the largest single half-plane violation: exact through a FACE, short by `sin(theta/2)`
  past a CORNER (0.707 at a square's). Euclidean exterior distance would need the vertex terms that
  have no working construction — same accepted bias as the reflex corners above.

- [x] **WINDING NORMALIZED IN `loopFrame` (2026-08-01).** `n_i = J t_i` is outward only for a CCW loop.
  The wall in `tests/squaresInASquareArea-Boundary.ipynb` is written `[[0,0],[0,1],[1,1],[1,0]]`, signed
  area **-1**, i.e. clockwise — so every normal pointed inward, `h` read **-0.5139 at the box CENTRE**
  and -1.5 outside, `-h` was MINIMAL at the centre, and confinement became an ATTRACTIVE WELL. Measured:
  all five squares collapsed onto a single point at [0.5, 0.5]. Cam spotted it from the picture.
  `loopFrame` now flips the normals on a negative shoelace sum, and the CUDA kernel does the same so the
  tiers cannot diverge on a clockwise loop. Guarded by check 12.

- [x] **CONVEX-ONLY BY DECISION (2026-08-01).** `packingEnergyForce` raises on non-convex loops unless
  `allowNonConvex = True`. The convex differences tree still exists and is verified, but it is ~12.7 s
  per force evaluation with no CUDA path and carries the reflex bias, so it is opt-in. See
  `notes/penetrationDepthReview.md`.

## ✓ SETTLED — the optimum IS a minimum; the shortfall is a SEARCH failure (2026-07-27)

Cam, from the notebook: the 11-square packing has "this little gap that one of the squares pokes its
sharp corner into. Even if things go well otherwise, it doesn't appear to be sufficiently motivated to
find that little gap." `tests/knownOptimumCheck.py` tests exactly that, on the closed-form optimal
packing of 5 unit squares (side `2 + 1/sqrt(2)`, phi = 0.68227 — four in the corners, one tilted 45
degrees with its corners in the gaps):

| check | result |
|---|---|
| valid | pair overlap -2.0e-17, shape index exactly 4.000000, phi correct to 3.9e-10 |
| critical point | max\|F\| 5.4e-08, energy 1.8e-16 |
| survives relaxation | phi held to 9 digits, max\|F\| polished to 8.6e-14 |
| attracting | perturbed 1e-6 / 1e-4 / 1e-3, phi returns to 0.682274643 every time |

**The landscape holds the optimum.** So `energySweep` landing ~4% short is descent failing to reach it,
not the energy failing to reward it — and no contact-law or protocol tuning fixes that.

**The mechanism, quantified.** Perturbing the optimum by delta leaves residual overlap 4.1e-12 /
4.1e-08 / 2.1e-06 for delta = 1e-6 / 1e-4 / 1e-3 — exactly `delta^2`. That is the corner-into-face law
(area `delta^2`, energy `delta^4`, force **`delta^3`**) measured independently earlier the same day at
the wall. A corner approaching a gap feels a force vanishing CUBICALLY while the overlap it must
resolve falls only quadratically, and the contact energy is purely repulsive so nothing pulls it in.

- [ ] The protocol is a QUENCH, not an anneal. Barrier crossing is what is missing: basin hopping,
  explicit rotational moves, or a genuine thermal anneal.
- [ ] Revisit `sharpDecompress = True` in this light. It was justified (the Plummer tail does not vanish
  on a valid packing and settles looser), but it switches OFF the mollification's barrier smoothing
  before the descent that picks the final configuration.
- [ ] The seed study changes meaning: it is no longer "is this basin dependence" but "how far does a
  quench spread", which is worth knowing but is not the interesting question any more.

## ▶ BUG — CUDA 719 (launch failure) on search-generated configurations

`search.basinHop` reliably kills the CUDA context with `CUDA error 719 during sharpOverlapCuda`
(unspecified launch failure -- usually an out-of-bounds write). The numpy tier handles the same
configurations fine, so this is a device-path robustness problem, not a geometry one. It is STICKY:
once it fires the context is dead and the process must be restarted.

The search stresses the sharp kernel in a way relaxation never does -- trial moves produce heavily
overlapping, rotated, and displaced configurations that a quench would never visit.

**HYPOTHESIS TESTED AND REJECTED: it is not NaN.** I guessed non-finite positions were reaching
`computeCellKernel`, where `(int)floor((NaN - xmin)/cellSize)` is undefined and would give a garbage
cell index and an out-of-bounds `atomicAdd`. Measured over 40 search moves on the numpy tier: zero
non-finite configurations, `max|position| = 1.008`. So the mechanism is something else and I do not
yet know what.

Ruled out so far:
- intersection buffer overflow -- `maxInter = numVert^2` (576 for this case) against at most
  `C(24,2) = 276` possible edge-pair crossings, and writes are bounds-checked with an overflow report
- non-finite positions (above)
- degenerate edges from the moves -- rotate and translate are rigid, so edge lengths are invariant

- [ ] Reproduce with `cuda-memcheck` / `compute-sanitizer`, which will name the offending access
  directly rather than requiring more guessing.
- [ ] Regardless of cause, `sharpOverlapCuda` should validate its input and raise a Python error rather
  than launching a kernel that can corrupt the context for the rest of the session.

## ▶ BUG — exactly-tangent packings break the overlap routine, and the two paths disagree

Found while building the test above. On the exactly-tangent optimum (every contact a corner precisely
on an edge):

| | reported pair overlap |
|---|---|
| all-to-all `updateIntersections` | 5.82e-02 |
| neighbor-candidate path | 1.68e-01 |
| truth | 0 |
| maximum possible (one square's whole area) | 0.1365 |

Both are wrong and they disagree. `polygonPairIntersections` requires a strict `0 < s < 1`, so a vertex
exactly on an edge is the boundary case and floating point decides arbitrarily whether it counts; the
follower/area assembly then works from a malformed crossing sequence.

Unreachable in normal use -- 1e-9 of noise and both paths agree on exactly 0, and relaxed packings are
never exactly tangent. But every KNOWN OPTIMUM is tangent by construction, so this blocks precisely the
configurations worth evaluating. Worked around with Gaussian noise in the test.

- [ ] Real fix: decide what a tangent contact SHOULD report and make both paths agree — probably an
  epsilon-consistent crossing test rather than a strict inequality.
- [ ] `tests/neighborCheck.py` has no tangency case, which is why the candidate/all-to-all divergence
  went unnoticed. Add one once the behavior is defined.

## ▶ NEW — anneal the SHAPE, don't confiscate it (2026-07-27)

**The diagnosis.** `energySweep`'s displayed `polydispersity` is std/mean of the target AREAS — it says
whether the polygons are the same SIZE, and nothing about whether they are SQUARE. Under
`setConstraints(area = True, edge = [1, 2, 4, -1])` the two are decoupled by construction: each area is
pinned per polygon, the edges are held only in their global moments, so a polygon is free to become a
kite. `setSizePolydispersity` rescales each polygon about its own centroid, which drives the size
spread to 1e-16 while touching no shape at all. Hence "polydispersity 2e-16 but the shapes are far from
square" — the readout was answering a different question. New: `Model.getShapeDistortions()` /
`getMaxShapeDistortion()`, shown next to polydispersity in the sweep title, bar and history.

**The real problem it exposed.** The rigid handoff was giving back everything the anneal won:

| | overlap |
|---|---|
| before the anneal | 2.593e-03 |
| after the anneal | 9.207e-04 |
| after `_rigidify` (one-shot projection) | **2.905e-03** |

Slightly WORSE than the start. The anneal optimizes over equal-area quadrilaterals and the
configuration it likes is not near the good configuration for squares.

**The fix — a third moment family.** `DistributionConstraints(shape = True)` holds the SHAPE BUDGET
`Phi = sum_i d_i` with `d_i = P_i / (sqrt(A_i) g_i) - 1 >= 0`, `g_i = sqrt(4 n_i tan(pi/n_i))`.
- Nonnegativity is the whole trick: the sum can only reach zero with every polygon regular, so ONE row
  does the squeeze. No second moment — higher moments of a nonnegative quantity add nothing but
  degenerate rows.
- Normalizing by the n-dependent floor is what makes a mixed-n packing summable at all.
- NOT "shape-index polydispersity": the shape index is one-sided (floored at 4 for a square), so
  std/mean reads zero for eleven identical rhombi. The mean is the control variable, not the width.
- `Model.setShapeBudget` walks it down; the ramp is interleaved with decompression
  (`energySweep(annealShape = True)`), so shapes straighten as the packing opens rather than being
  straightened first and opened afterwards.

**FINDING — the shape row degenerates at its own target.** `d_i` is MINIMIZED at the regular polygon,
so `dd_i/dr` vanishes exactly where the ramp is headed. Measured row norm: 1.55e-01 → 7.64e-02 →
4.18e-02 → 2.24e-02 as the budget fell 0.030 → 0.0009. The ramp must hand off to per-object
constraints while still transverse — same lesson as the mean/variance degeneracy, different mechanism.
Note `conditioning()` CANNOT see it (a ratio of singular values is identically 1 for a single row);
`DistributionConstraints.rowNorms()` is the diagnostic that can.

**BUG (fixed) — undamped Newton in both moment retractions.** `DistributionConstraints.projectPositions`
and `CompositeConstraints.projectPositions` took the full step unconditionally. Fine for area/edge rows
(near-linear), fatal for the shape row: dividing by a vanishing singular value threw the packing far
enough that the next block SHAKE could not recover — measured, the hard areas came back wrong by a
factor of **860**. Both now backtrack on the residual 2-norm; the well-behaved rows still accept the
full step first try.

**FINDING — the builder's polygons are not regular to machine precision.**
`generateEquilateralPolygons` seeds random stars and relaxes them with FIRE, so its output carries a
shape distortion of **1.7e-06** (from a 4e-08 spread in edge length). That is the floor a fresh packing
can honestly be annealed to; an analytically-placed regular polygon reads 8.9e-16.

**The mollification now comes off before decompressing** (`sharpDecompress = True`). Phase A needs the
smooth landscape; phase B decides the answer, and there the Plummer contact is a liability — it does
not vanish on a valid packing (8.2e-04 measured where the true overlap was exactly zero), so it keeps
pushing separated polygons apart and settles looser than the shapes admit. `maxSigmaRatio = 2.0` also
caps how fast sigma may fall per round, lengthening phase A rather than outrunning the relaxation.

Verified: `tests/shapeBudgetCheck.py`, 6/6 — distortion vs an independent construction 2.2e-16, exact
regular n-gon 8.9e-16 for n = 3..8, Jacobian vs central differences 5.1e-10 on rows of size 5.7e-02,
retraction lands on the requested budget with area error 8.9e-16, row degeneracy as above, and the
worst polygon following the sum (1.52e-02 → 3.33e-04).

**RESULT — the interleaved anneal wins, by less than the shortfall.** Same build, same seed (42), only
the two flags differing:

| | phi | % of 0.7318 | overlap | distortion |
|---|---|---|---|---|
| `annealShape = True`, `sharpDecompress = True` | **0.664942** | 90.9% | 0 | 2.0e-15 |
| `annealShape = False`, `sharpDecompress = False` | 0.655473 | 89.6% | 0 | 8.0e-15 |

+1.4% density; the packing point moved 0.6535 -> 0.6595 before refinement. Both end on genuine squares
at exactly zero overlap. NOTE the 0.6895 quoted earlier in the session is NOT a baseline for this — it
came from a different build and is not comparable to either row.

So the shape freedom was worth having but is NOT what costs the remaining ~10%. Still open, cheapest
first:

### ✓ RESOLVED — the verdict was rejecting every jammed state (2026-07-27)

**What it was.** `energySweep` accepted on `totalOverlap <= finalEnergy` with `finalEnergy = 1e-12`, on
the premise that the exact overlap is identically zero below jamming. Splitting the overlap into its
two parts across a density sweep:

| phi | interior (polygon-polygon) | wall | worst single wall cap |
|---|---|---|---|
| 0.665692 | 0.000000e+00 | 0.00e+00 | 0.00e+00 |
| 0.667692 | 0.000000e+00 | 3.58e-10 | 3.58e-10 |
| 0.671692 | 0.000000e+00 | 3.64e-10 | 3.64e-10 |
| 0.673692 | 0.000000e+00 | 3.45e-10 | 3.45e-10 |

**Polygon-polygon overlap IS a perfect sign change** — identically zero at every valid density, no
floor to calibrate. The whole residual is ONE polygon whose CORNER clips the wall.

**It is physics, not a bug.** Cam's question — doesn't the wall behave like any other polygon? — is
what cracked it. It does: same normalized-squared functional, and for equal-size polygons literally the
same normalizer (`norm_S = 2 A0` against `norm_AB = A0 + A0`). Measured contact scaling by translating
the offending polygon:

| slope over the last decade | measured | corner predicts |
|---|---|---|
| area vs delta | 1.978 | 2 |
| energy vs delta | 3.955 | 4 |
| force vs delta | 2.966 | 3 |

That is the corner-into-face law already predicted in the energy-sweep section above. With `F ~ δ³` the
minimizer stops when the contact force sinks into the ~3e-12 force noise, not when the geometry is
clean. Resting penetration fitted from `a = C(δ₀+s)²`, `C = 1.0124` (predicts the small-δ rows to 4
digits): **δ₀ = 1.88e-05**, i.e. 7.5e-05 of an edge.

**MY ERROR, recorded.** I first told Cam this was a soft-wall equilibrium and separately that a 2.4e-08
force "must" exist — both asserted without checking. The force is 3.63e-12, and it closes exactly once
the real contact chord is measured (3.7e-05, not the full edge 0.25 I assumed). Cam's pushback is what
forced the measurement.

**The fix — a two-part verdict.** `finalEnergy` (now default **0.0**) applies to
`getPairOverlapArea()`, tested against zero. `wallTolerance` (default 1e-4 of the mean edge) applies to
`getWallPenetration()`, a geometric DEPTH. Depth not area, because the two differ by a square: 3.56e-10
of area is 1.88e-05 of depth, and a 1e-9 area tolerance silently grants ~2e-05 of depth.

New: `Model.getPairOverlapArea`, `getContainerOverlapArea`, `getWallPenetration`, `_distanceToLoop`.
Verified by `tests/wallPenetrationCheck.py` (5/5): zero when contained; a known displacement read back
to 1e-16; a corner excursion measured to the corner POINT (5.000000e-03 for a 3-4-5 offset) where a
point-to-line distance would understate it; area/depth differing by a square; and the split summing to
the old total.

**Quad precision** would help, since the stall is at the force noise floor — but `F ~ δ³` means each
factor of ten in δ costs a thousand in precision, `np.float128` on x86-64 is only 80-bit extended
(~10x in δ), and true quad has no CUDA. Cam has a proprietary 128-bit CUDA library and may revisit on a
later rewrite. The geometric criterion sidesteps it entirely.

### ✓ DONE — neighbor list, sorted followers, and a profile that redirects the next work (2026-07-27)

`updateIntersections` was all-to-all and self-described as "a reference / testing version". It ran on
every sharp force evaluation (via `sharpContainerEnergyForce`) and every verdict (via
`getOverlapArea`), so it was the hot loop of a sweep.

**`neighbors.py`** — two levels of geometry-attached ball, no grid (Cam: large loops have big voids,
and a grid's cell size would be set by the container's box-spanning edges). Polygon balls (centroid,
covering radius) cull whole pairs; edge balls (midpoint, half the edge length) cull within survivors.
Exact at `skin = 0`: a crossing puts each midpoint within its own half-length of the crossing point.
Verlet skin with `rebuild when max displacement > skin/2`.

**Sorted followers** — Cam's scheme, ported from the CUDA `followersKernel` (which already had it) to
the CPU, which was still running the O(M^2) reference scan. The tiers now share a complexity.

| | before | after |
|---|---|---|
| intersections, n=16 N=128 | 9383.91 ms | 0.62 ms (15135x; 924x fewer pairs + one vectorized pass) |
| followers, n=16 N=128 | 41.28 ms | 0.72 ms (57x) |
| FIRE 3000 steps, N=11 | 64.3 s | 28.0 s |
| full energySweep, N=11 | 172.9 s | 129.1 s |

`result.phi = 0.699489080606` both ways — identical to 12 digits, which is the gate for a pure
performance change.

**Two bugs of mine, found by profiling and by Cam's questions:**
- `_refreshNeighbors` ran unconditionally in `_forceEnergy`, rebuilding a list the mollified periodic
  path never reads: 6.1% of a run wasted. Now gated on sharp-tier-or-container.
- The default skin averaged the covering radius over ALL polygons INCLUDING the container, whose radius
  is the box half-diagonal — 0.2343 against the polygons' 0.1913, a 22% inflation. Same mistake
  `_RaggedBlocks` warns about. Now excluded via `meanPolygonRadius`.

**Adaptive skin.** Edge length was the wrong scale (`R/edge` runs 0.71 at n=4 to 5.1 at n=32, so one
fraction means seven different things); covering radius is right for the pair-COST side but says
nothing about the BENEFIT side, which is `skin / (2 x per-step displacement)` — dynamics, not geometry.
So the skin is now solved from a target reuse using the displacement the staleness test already
measures. **Scanning `targetReuse` 5/10/20/40 moved the pair count 9x and the runtime not at all**
(28.3-29.2 s, inside scatter) — an earlier report of "5% slower" was noise I over-read from a single
pair of timings.

**Where the time actually goes now** (32 32-gons, mollified, hard constraints, 300 FIRE steps at
101.84 ms/step):

| term | per step | % |
|---|---|---|
| cuda plummerOverlap | 1.00 | 43.0% |
| ShapeConstraints.projectPositions | 1.00 | 40.1% |
| ShapeConstraints.normalBasis | 1.00 | 10.8% |
| cuda selfRepulsion | 1.00 | 3.3% |

- [x] **SHAKE — done, and the answer was NOT the GPU (2026-07-27).**

  The cost was one routine: at n=32, N=32 a SHAKE call spent 5.79 ms in `np.linalg.svd` against 0.30 ms
  assembling the Jacobian, 0.09 ms on the residual and 0.05 ms on the einsums. **93% was the thin SVD.**

  **The device port was benchmarked and REJECTED.** cuSOLVER `gesvdaStridedBatched` on the real shapes
  (33x64 transposed to 64x33, batch 32-64) runs 23.66 ms against numpy's 5.79 ms -- 2.6-4x SLOWER --
  and scales perfectly linearly in batch size, so it is looping rather than batching.
  `gesvdjBatched` caps at 32x32 and cannot take the n=32 shape at all. One benchmark, ~80 lines, saved
  building a slower SHAKE.

  **The fix was the factorization.** SHAKE never needed an SVD -- it needs an orthonormal basis of J's
  row space and a minimum-norm solve, and a QR of `J^T` gives both:

      J^T = Q R    ->    normalBasis = Q^T,    J^+ C = Q R^-T C

  since `J J^T = R^T R` at full row rank. It never FORMS `J J^T`, so `R` carries J's condition number
  rather than its square -- the objection that rules out the normal equations does not apply here.
  `np.linalg.solve` was then replaced by actual forward substitution (it was LU-factorizing a matrix
  already known triangular: 0.989 -> 0.570 ms, agreeing to 7.9e-31).

  Measured on a quiet machine, 32 32-gons, 300 FIRE steps:

  | | before | after |
  |---|---|---|
  | step | 101.84 ms | **55.55 ms** (1.83x) |
  | `projectPositions` | 12.255 s (40.1%) | 2.085 s (12.5%) — 5.9x |
  | `normalBasis` | 3.299 s (10.8%) | 0.449 s (2.7%) — 7.3x |
  | SHAKE total | 50.9% of a step | **15.2%** |

  The SVD is retained as the fallback: `_qrFactor` returns None on rank deficiency, which is REAL for
  triangles under area+edge (24/32 rows kept, cond 2.4e16), for perimeter+edge, and for ragged packings
  whose blocks are zero-padded. `tests/shakeFactorCheck.py` pins all of it -- projectors agreeing to
  5e-14, Newton steps to 1e-16, identical iteration counts and residuals, and the guard refusing all
  three deficient cases.

- [ ] `cuda plummerOverlap` is now 75.7% of a step and already on the device — the next target if
  minimization speed matters again.
- [ ] Host<->device traffic is 32 bytes per vertex per force evaluation, 9.9 MB over 300 steps —
  latency, not bandwidth, and it scales LINEARLY while the physics scales superlinearly, so the ratio
  improves with size. It is NOT the reason to go device-resident. The reason is that half the time is
  host-side linear algebra that cannot move to the GPU while the state lives on the host.
- [x] **`cuda/neighbors.cu` — device broad phase, done.** Identical candidate sets to numpy across all
  18 configurations (n = 4/16/32, walled and periodic, skin = 0 / 0.05 / 0.2, up to 1586 pairs), now
  check 5 of `tests/neighborCheck.py`. Build cost per rebuild:

  | n | N | verts | numpy | cuda | speedup |
  |---|---|---|---|---|---|
  | 4 | 32 | 128 | 5.81 ms | 0.38 ms | 15.1x |
  | 16 | 32 | 512 | 6.36 ms | 0.91 ms | 7.0x |
  | 32 | 32 | 1024 | 7.33 ms | 1.47 ms | 5.0x |
  | 16 | 128 | 2048 | 76.53 ms | 2.52 ms | 30.4x |

  **A WRONG DERIVATION OF MINE, caught only by the comparison.** The kernel was first written WITHOUT
  the polygon-level prefilter, on the argument that an edge ball sits inside its polygon's covering
  ball so the coarse test is implied by the fine one. False. The parallelogram law gives
  `d^2 + h^2 <= R^2` for an edge whose endpoints are within R of the centroid, hence
  `d + h <= sqrt(2) R`, NOT `<= R` -- at `d = h = R/sqrt(2)` the sum is 1.41 R. So the covering-ball
  test can reject a pair whose edge balls overlap, and the device found 33 candidates where the host
  found 29. Neither filter is incorrect (both are necessary conditions for a crossing, so the
  INTERSECTION sets agreed either way), but two broad phases that disagree cannot be checked against
  each other. The device now applies both levels. The corrected derivation is in the file header.

### ✓ CONFIRMED — stiffening the wall recovers most of the shortfall (2026-07-27)

Same build and seed (42), five points on one curve:

| kContainer | phi | % of optimum | pairOverlap | wallDepth | max\|F\| |
|---|---|---|---|---|---|
| 1 | 0.663723 | 90.7% | 0 | 2.09e-05 | 1.41e-12 |
| 3 | 0.683598 | 93.4% | 0 | 2.84e-05 | 9.78e-12 |
| 10 | **0.699817** | **95.6%** | 0 | 0 | 0 |
| 30 | (running) | | | | |
| 100 | 0.694380 | 94.9% | 0 | 0 | 0 |

Monotone rise to 10 then a fall, so the peak is real rather than sampling noise. **+5.4% density** over
the default, landing essentially on the phi = 0.7015 where the polygons had stopped overlapping each
other in the leaking run -- the gap closing exactly where the diagnosis said it would.

NOT a mollification artifact: `sharpDecompress = True` is the default, so the descent and every number
above were measured with the mollification already off ("sharp mollification off (sigma was 3.310e-03)"
appears before the descent in every log).

**No CUDA work needed.** I claimed the container kernel took no stiffness argument, added a guard in
`_forceEnergy` forcing the numpy path whenever `kContainer != 1.0`, and framed the default choice around
that cost. All of it was wrong and unchecked: `cuda/container.cu` has carried `kContainer` from the
start, using it in both the energy (`2 kContainer (a/norm)^2`) and the weight (`4 kContainer a/norm^2`),
and the Python wrapper always passed it. Verified numpy against CUDA at k = 1, 10, 100: agreement 4.8e-15
relative in energy, 5.7e-13 absolute in force. The guard is removed.

- [x] Default set to **10.0** (`_DEFAULT_CONTAINER_STIFFNESS`), adjustable via `setContainerStiffness`;
  pass 1.0 for the old equal-stiffness behavior. One seed so far -- the seed study would also say how
  much the peak moves.

### ✓ ADDED — Phase D, compress the found packing back up

The descent's answer is PATH DEPENDENT: the configuration reached by descending to a density is not the
one reached by compressing into it, since each density gets its own relaxation. Measured, the packing
returned at 0.665692 stayed valid compressed back up to at least 0.673692 -- 0.008 given away.

`energySweep` now finishes by compressing the found configuration in `phiStep/4` steps until overlap
first appears, then bisecting `compressRounds` times (default 6), capped at the starting density.
Snapshots at every accepted step, so a failed trial cannot leave the model invalid. This does not
reopen the decompress-rather-than-compress choice: the BRANCH is still the one decompression selected.

Also settled: `max|F| == 0` at the answer is NOT a symptom of stopping early. Below jamming the relaxed
state generically has no contacts at all -- both springs are dropped and the energy is zero, so the
polygons drift apart until nothing touches. Requiring contact as a verdict would be satisfiable only
exactly at phi_J.

### ▶ OPEN — is the remaining 4.4% just basin dependence?

0.6998 against the 0.7318 GLOBAL optimum. That optimum is the single best arrangement of 11 unit
squares; a protocol quenching from one random start finds a LOCAL jamming density, and the two are not
the same quantity. Treating "not at the optimum" as evidence of a defect was probably the wrong frame
past the point where the leak was fixed.

- [ ] THE instrument, still not run: the same protocol from ~12 seeds. Scatter across 0.68-0.72 with
  the best approaching 0.73 means basin, and the answer is "run many, keep the best". A tight pile at
  0.70 with nothing above means something systematic still caps it.

### ▶ SUPERSEDED — the packing LEAKS OUT of the box rather than jamming (2026-07-27)

Found while checking whether the two-part verdict raised the density. It did not (0.663536 against
0.664942), and the descent trace says why:

```
rigidify     phi 0.725505   pairOverlap 5.921e-05   wallDepth 7.928e-03
decompress   phi 0.713505   pairOverlap 4.959e-06   wallDepth 6.425e-03
decompress   phi 0.701505   pairOverlap 0.000e+00   wallDepth 4.909e-03
decompress   phi 0.689505   pairOverlap 0.000e+00   wallDepth 3.380e-03
decompress   phi 0.677505   pairOverlap 0.000e+00   wallDepth 1.838e-03
decompress   phi 0.665505   pairOverlap 0.000e+00   wallDepth 2.818e-04
decompress   phi 0.659505   pairOverlap 0.000e+00   wallDepth 0.000e+00   <- packs
```

**The polygons stop overlapping EACH OTHER at phi = 0.7015** — 96% of the 0.7318 optimum. The
remaining 0.04 of density is spent waiting for the packing to climb back INSIDE the box, which it is
sticking out of by 5e-03 on a unit box. Not a marginal corner contact: those polygons are substantially
outside the wall.

**Why.** Both contacts obey the same normalized-squared law with the same normalizer for equal-size
polygons, but escaping through the wall lowers the effective confinement for EVERY polygon while
overlapping a neighbor relieves nothing globally. With equal stiffness, leaking is the cheaper route.
This is the "packings peeling away from the walls" symptom from earlier in the session, now measured
rather than guessed at.

New: `Model.setContainerStiffness(k)` — wall stiffness relative to the pair contact, default 1.0 so
nothing changes until it is used. Note a non-default value puts the wall term on the numpy path, since
the CUDA container kernel takes no stiffness argument.

- [x] Ran, five points: the leak WAS what binds. See the confirmed section above.
- [ ] If it does, decide between the stiffness ratio and HARD containment (a constraint rather than a
  penalty). Stiffness is one parameter but shortens FIRE's timestep as 1/sqrt(k); a constraint cannot
  leak at any energy but is real work and interacts with the pinned wall.
- [ ] Separately: `_forceEnergy` computes the container term on the SHARP path and then discards it
  (the `eW`/`fW` block runs before the model-type branch, and the sharp branch recomputes `eC`/`fC`).
  Wasted work on every force evaluation of a sharp run.

### ✓ RESOLVED — how many moments can the deviation form carry? (2026-07-27)

Cam asked why not `[1, 2, -1, 4]`. Answer: that works; the ceiling is conditioning, and two real bugs
were in the way.

**More exponents are useful.** `+1` sets the scale, `+2` the variance, `-1` the barrier at zero and the
harmonic mean, `+4` the tail (as k grows `sum delta^k` approaches `max delta`, clamping the single worst
polygon). With only `+1, -1` one polygon can stay bent while the sum and inverse sum balance.

**Feasibility is the trap, and self-similar scaling removes it.** Moments of a nonnegative sequence
obey Cauchy-Schwarz, AM-HM and the Stieltjes conditions above them, so moving targets independently
eventually asks for a combination nothing realizes. `setShapeDeficit` scales the whole distribution,
`delta_i -> lambda delta_i` so `Phi_k -> lambda^k Phi_k`, which is realizable for any exponent list by
construction.

**Measured ceiling** (8 squares, ramping the deficit down 1000x):

| rows | exponents | conditioning | relative error |
|---|---|---|---|
| 2 | `[1, -1]` | 8.6e-02 | 6.5e-13 |
| 3 | `[1, 2, -1]` | 3.9e-02 | 2.1e-12 |
| 4 | `[1, 2, -1, 4]` | 2.7e-02 | 7.3e-13 |
| 6 | `[1, 2, 3, -1, -2, 4]` | 5.6e-04 | 1.1e+01 |

Four rows is comfortable, six is past the cliff, and N does not rescue it (8, 16, 32 all alike). The
collapse is self-inflicted: the more of the distribution the rows determine, the closer they drive it
to monodisperse, and a monodisperse quantity makes every moment row parallel.

**Bug 1 — retraction budget did not scale with row count.** `[1, 2, -1, 4]` silently under-converged at
24 passes (deficit off 35%, hard areas off 3.5e-06) and was exact at 80. `momentMaxIter` now defaults
to `max(24, 20 * numRows)`; it exits on tolerance, so a generous budget is free (80 and 200 identical).

**Bug 2 — `setShapeDeficit` compounded its own drift.** It scaled the STORED REFERENCE by `lambda^k`
instead of the geometry's moments, so any retraction that landed slightly off was multiplied by the
next call and the `+1` target quietly stopped being what was asked. This produced the confusing symptom
of a 1.59e-12 residual alongside a deficit 6.2x from target -- the residual was truthful, the geometry
really was on the drifted reference. Now anchored to the geometry, so the `+1` target is exactly
`total` and a miss shows up as residual.

**New warnings.** The composite retraction now says so when it stops without meeting tolerance, via BOTH
its exit paths (budget exhausted and backtracking-failed; the degenerate set takes the latter, which is
why the first attempt at this warning never fired). Two separate alarms: residual above `100 *
momentTol` (a healthy run lands just over a 1e-12 tolerance routinely, so warning at the tolerance
itself cries wolf), and conditioning below `_MOMENT_CONDITIONING_FLOOR = 1e-3` in its own right.

### ✗ SUPERSEDED — "the sharp tolerance is exonerated"

- [x] **`_SHARP_TOLERANCE = 1e-4` RULED OUT.** Compressing the packed state back up, relaxing at the
  sweep's budget and then 20000 steps further:

  | phi | short: overlap, max\|F\| | long: overlap, max\|F\| |
  |---|---|---|
  | 0.665692 | 0.000e+00, **0.0e+00** | 0.000e+00, 0.0e+00 |
  | 0.666692 | 1.642e-08, 3.1e-10 | 7.106e-09, 8.8e-11 |
  | 0.667692 | 1.523e-07, 8.7e-09 | 3.591e-10, 1.0e-12 |

  The ceiling never binds near jamming: the FORCE SCALE collapses as phi -> phi_J (3e-10, 9e-09 —
  six orders under the tolerance, reached inside the short relax) because the contacts carrying it are
  marginal. It only binds far above jamming, which is where the plateau measurement already showed it
  costs nothing. The long relax buys 1-3 orders on the residual overlap but never crosses zero, so no
  verdict flips.

  **Also: at the reported density max\|F\| is EXACTLY zero — nothing is touching.** The returned
  configuration is a valid arrangement with slack, and it overlaps 0.001 higher in phi, so the branch
  really does jam at ~0.6662 and the sweep resolves it to within its phiStep.

- [ ] So the ~10% shortfall is BASIN, not measurement: the protocol finds a poor branch rather than
  mis-measuring a good one. Next moves are on the anneal path — the seed study (12 sweeps, ~15 min
  each), slower ramps, or allowing phi back UP once the shapes have straightened.
- [ ] `tests/energySweepCheck.py` green.
- [ ] `cuda/testSharpPacking.cu` vectors are stale against the normalized-squared functional;
  regenerate with `cuda/genVectors.py`.

## ▶ IN PROGRESS — energy sweep toward the jamming density (2026-07-26)

Goal: measure `E ~ (phi - phi_J)^alpha`, then use the power law to approach phi_J by extrapolation
rather than by bisecting on a threshold. New: `Model.getPackingFraction` / `setPackingFraction`
(scales polygons AND their targets together, so hard area constraints stay satisfied to 4e-15 with no
SHAKE -- that is what makes single-branch compression possible) and `tests/energySweep.py`.

**FIRST RESULT IS NOT TRUSTWORTHY — recorded so the mistake is not repeated.** Sweeping phi = 0.70
to 0.90 on 5 rigid squares gave R^2 = 0.99999 and a meaningless answer:

| sigmaFrac | phi_J | alpha |
|---|---|---|
| 0.080 | 0.329 | 3.004 |
| 0.040 | 0.493 | 2.885 |
| 0.020 | 0.573 | 2.644 |

The window spans **less than one decade of E and sits far above jamming**, so `alpha` and `phi_J` trade
off almost perfectly: assuming alpha = 2 outright ALSO fits (R^2 = 0.9985, phi_J = 0.583), and the
pressure exponent came out 2.07 where the energy fit demands 1.885. Three inconsistent answers, all
with excellent R^2. LESSON: for a 3-parameter power law with a free offset, R^2 measures smoothness,
not correctness -- bound the decades of E before believing a fit.

Useful by-product: phi_J drifts +0.164 then +0.080 as sigma halves, a geometric sequence extrapolating
to **phi_J(sigma -> 0) ~ 0.65**, below the 0.6823 optimum, which is where a random compressed branch
should jam.

**STRUCTURAL OBSTACLE.** Approaching phi_J the overlaps vanish, so `delta << sigma` ALWAYS near the
transition -- the mollified energy can never show the sharp contact law in the limit of interest.
sigma sets a crossover and the power law only lives in `sigma << delta << L`. Fix under test: relax
with the MOLLIFIED energy (smooth, converges) but MEASURE with the sharp one (exact area, no sigma in
it), so the measured quantity carries no mollification.

**CONTACT-LAW PREDICTION to test.** Polygons have no single contact law: face-face overlap area goes
as `w delta` so `U ~ delta^2` (alpha = 2), while corner-into-face goes as `delta^2` so `U ~ delta^4`
(alpha = 4). A jammed state is a mixture, and since the corner term dies faster, the effective exponent
should DECREASE toward 2 as phi_J is approached. That is a sharp prediction the redesigned sweep can
test, and it means "the" exponent may not be universal for polygons the way it is for spheres.

- [ ] Redesign: locate phi_J by decompression first, then sweep a narrow window just above it spanning
  3+ decades of E; verify sigma-independence of the measured alpha.
- [ ] Deferred at Cam's request: pressure and the stress tensor (thermodynamic `P = phi dE/dphi` is
  already in the sweep; the independent check is the WALL VIRIAL, which needs the container's own
  gradient that `energies.containerEnergyForce` skips when the wall is fully pinned).

## ▶ NEW — distribution (moment) constraints; two real bugs found (2026-07-26)

**The design.** Hard per-object AREAS + edge lengths held only by the global moments of their
distribution: `setConstraints(area = True, edge = [1, 2])`. Cam's idea, and it is better than the
transient-target route it replaces — moments of the ACTUAL geometry, so there are no target variables,
no springs, and nothing chasing anything. That also sidesteps `targetForces`'s known blind spot (the
overlap normalizer's `targetArea` dependence) because there is no target force left to be incomplete.
New in `constraints.py`: `_RaggedBlocks` (shared padded block tables), `DistributionConstraints`,
`CompositeConstraints`. Anneal with `setTargetPolydispersity`.

Why hard areas rather than area moments: the φ=0.70 false "packs" was squares SHRINKING 1%. Free
individual areas re-admit exactly that. Free edges only reshape a square into a same-area rectangle,
which cannot cheat on φ.

**BUG 1 (fixed) — `targetEdgeLength` indexed as per-polygon.** `constraints.py` sliced it `[:numPolygons]`
after the per-edge refactor made it per-VERTEX, so every polygon got the first few vertices' targets.
Silent whenever the packing is uniform+monodisperse; catastrophic otherwise — `constraintCheck [5]`
(ragged) SHAKE stalled at max|C| = 7.1e-01 instead of 1.1e-15. Now gathered via `edgeTargets`.

**BUG 2 (fixed) — the container was inside the transient moment sums.** `TransientTargets` included the
wall. Its area is SIGNED and negative, so for 5 squares at φ=0.8 the k=1 reference was `sum A0 = -0.2`
(0.8 of polygons minus a 1.0 wall); the restore then drove polygon targets NEGATIVE chasing it
(measured -227), after which the constraint Jacobian divided by it and the run died as an opaque
`LinAlgError: SVD did not converge`. This is the NaN in the square search and the crash in
`transientSquares.ipynb` cell 7. Fixed by excluding the container (`free`/`setFree`/`numFree`);
`_decompose` now also raises a message naming the cause instead of letting LAPACK do it.

**FINDING — `area = True` + `edge = True` is generically INFEASIBLE after independent target draws.**
Not a bug, a geometry fact worth remembering: for an n-gon, `A <= P^2 / (4 n tan(pi/n))`, so a quad with
four edges of length l can never enclose more than l^2. `generateEquilateralPolygons` lands exactly ON
that bound (p = P/sqrt(A) = 4 for a square), so perturbing area and perimeter INDEPENDENTLY —
`setLogNormalArea` then `setLogNormalPerimeter` — puts roughly half the polygons a hair over it. Then
SHAKE has nowhere to converge to and FIRE grinds forever: `transientSquares.ipynb` cell 4 burned all
100000 steps at max|C| = 3.88, and one square wanted 1.0223x the maximum possible area.
`ShapeConstraints.infeasibleReason` now REFUSES this with the offending polygon and shape index.

**FINDING — the mean/variance degeneracy is gradual, not a cliff.** Measured `sigma2/sigma1 = 0.40 CV`
across seven decades. So there is no rank drop to detect (both rows stay formally independent at every
width the geometry can represent) while noise amplification grows as 1/CV. The anneal handoff to
per-object constraints must be decided on the WIDTH; `constraintConditioning()` is the diagnostic,
`constraintRank()` will not warn in time.

**FINDING — the freedom only pays when it is DRIVEN.** The rigid configuration satisfies the moment
constraints, so the moment feasible set strictly contains the rigid one and its global optimum cannot
be worse. But FIRE is local: simply switching the freedom on leaves it in the same basin (measured at
φ=0.75, undriven moments came out marginally WORSE than rigid). Widening the distribution and closing
it back down is what moves between basins — see `tests/momentConstraintCheck.py [4]`.

- [ ] Point `tests/squarePackingSearch.py`'s `relaxAt` at `setConstraints(area = True, edge = [1, 2])`
  with a CV ramp, replacing the springs + transient targets; springs let the squares shrink, which
  invalidated the bisection criterion.
- [ ] Still open: `doubleEdgesCheck [2]` assertion needs `np.repeat(edgeBefore, 2) / 2`.

## ✓ DECIDED — Plummer kernel, fully-analytic result only; code cleaned (2026-07-22)

Committed to the original plan: the **Plummer-mollified overlap, fully-analytic closed-form tier**, as
THE overlap energy. Alternatives explored and dropped: the radial/gauge energy (rejected, below),
"rule as model" quadrature-as-Hamiltonian (`ruleAsModel.pdf`; low-n breaks — n=1 gives an *attractive*
force at contact onset), and dropping J_arcsinh (breaks force = −∇U). J_arcsinh is load-bearing (~68% of
U, wrong-signed force without it) and is finished + verified to 1e-16.

**Cleanup done (energies.py, model.py):**
- Deleted the SEMI-ANALYTIC (outer-quadrature) tier: `plummerPairEnergy`, `plummerPairGradient`,
  `plummerOverlap`, and the `_OUTNODES`/`_OUTWTS` nodes. `plummerMeasure` KEPT (the analytic Hessian
  uses it).
- Deleted the dead complex-arithmetic core `_tCore` + `_li2` (superseded by the real Clausen
  `_tCoreReal`); dropped the now-unused `scipy.special.spence` import.
- model.py: single `_forceEnergy` on `plummerOverlapExact` (was semi); removed `_forceEnergyExact` and
  the `exact=` flag on minimizeCG/minimizeNewton. **FIRE now runs on the exact tier (~9× slower:
  ~62ms→567ms/call at N=6) — accepted; speed is the CUDA port's job.**
- Kept the SHARP unmollified area+gradient (`updateOverlapArea`/`updateOverlapGradient` + the
  intersection/follower machinery) as the σ→0 reference, per Cam.
- Fixed `notes/verify_gradient_masters.py` (`_tCore`→`_tCoreReal`, still ~1e-14).
- Tests green (4 passed).
- Debris removed: `radialOverlap.py` + `tests/radialOverlapCheck.py` (rejected radial idea),
  `tests/jArcsinhWeightCheck.py` + `tests/relaxedN6.npy` (settled "is J needed?" check), stray
  `texput.log`. The findings live in this file and in memory; the scripts are gone.

## ✗ CLOSED — radial "ray through the center" energy, rejected (2026-07-21)

Evaluated as a replacement for the mollified overlap: for p on ∂B inside A, shoot the ray from z_A
through p, exit ∂A at p_A, density (|p_A−p|/|p_A−z_A|)². This is the **Minkowski gauge** of A about
z_A. Built (`radialOverlap.py`), validated, **rejected**; both the module and its check script have
since been removed (the measured numbers below are the record).

- The reduction is as clean as hoped: |p| cancels, so g = 1 − max_k ψ_k with ψ_k affine ⇒ the
  integrand is a **quadratic polynomial** in arclength, exact with no quadrature and no special
  functions. Verified against literal ray casting to **1e-9**.
- **But it is only C¹.** g vanishing at the crossings kills the moving-limit terms only for a
  TRANSVERSAL contact. At a **flat face-to-face** contact an O(1) arc switches on at once with
  g = ε/h, so E = w ε²/h² above onset and 0 below — **E″ jumps 0 → 2.0004** (measured, w = h = 1),
  against E ~ ε³ and continuous E″ at a corner. Face-to-face is the *common* contact in a dense
  packing, so this is the mollifier's discontinuity relocated, not removed.
- **The lock (keep this).** For any compactly supported geometric potential with density gᵐ, a flat
  contact gives E ~ εᵐ ⇒ E″ ~ ε^(m−2). C² needs m ≥ 3, and m ≥ 3 sends **stiffness → 0 at onset**.
  Smoothness order and stiffness exponent are one knob; an area integral instead of a boundary
  integral only shifts m by one. Mollification escapes only by deleting the onset event — that is
  the real reason σ is necessary, and it belongs in the notes.
- Secondary costs, all real: z_A couples every vertex of A ⇒ dense pair-Hessian block (loses the
  edge-pair-pass Hessian); 1/h_k makes stiffness direction-dependent *and* drifting as the springs
  deform h_k; star-shapedness fails exactly when self-repulsion is earning its keep; on device it
  trades a log + arctan (cheap on an SM) for clipping, a max-reduction and warp divergence.
- Unverified pointer worth checking before relying on it: LS-DEM surface-node penetration schemes
  reportedly show force discontinuities / discretization sensitivity, with VLS-DEM returning to
  overlap-volume forces.

**Decision: finish the CUDA port of the mollified tier.** `energies.py` is the reference; what's left
is transcribing J_arcsinh, Λ₀/Λ₁, the master table and the near-parallel bridge. Cl₂ on device is a
smooth function on [0,π] — a degree-12 Chebyshev fit holds 1e-15; it is not the hard part.

## ▶ CURRENT FOCUS — notes derivation + forces rework (2026-07-20)

**Where we are.** `notes/definitions.tex` §7 (energy J_arcsinh) is reworked and finished:
- [x] Deprecated the 4-pole partial-fraction path; cut it out (backup in
  `scratchpad/section7_removed_Jarcsinh_reduction.tex`, and math still lives in code + `.nb`).
- [x] Wrote the **fully-real route** (7.27–7.37): sinh-t → IBP → R(y) def (7.29) → poles → ±i/4
  residues → two-arctangent V → ∫V dt → −½[Im G(η+)−Im G(η−)] → Bloch–Wigner → 8-Cl₂ recipe.
  Verified: `notes/verify_realroute.py` (V′≈1e-10, J≈1e-16), poles/residues/Cl₂ in
  `partial_fractions_arcsinh.py` + `arcsinhClausen.nb`.
- [x] Defined everything (R=N/D, poles, residues), renamed the quadratic root `s`→`ζ`, and renamed
  the poles/residues off the edge indices: **β→η, α→c**.
- Note recorded: β+β−=−1 does NOT fold the two Clausen terms (checked); 7.37 is minimal.

**Forces rework (§8 Mollified gradient) — DONE 2026-07-20.**
- [x] Full fully-real re-derivation of the gradient, unified with the energy via ONE master form
  **𝕄[V] = ∫V′(ξ)Θ dξ = V·Θ − ∫V·Θ′ dξ**: energy panel (V=v₊, weight √) = +J_arcsinh, gradient W0
  (V=√, weight ξ/√) = elementary, gradient W1 (V=v₋, weight ξ²/√) = −J_arcsinh — all sharing the single
  transcendental T = −2·J_arcsinh, closed by §7's fully-real Cl₂ recipe (7.27–7.37).
- [x] Fixed the α/β slope/intercept clash → **μ, σν**; edge indices untouched; θ_αβ→θ_mk in the bridge.
- [x] Verified: `notes/master_form.py` (the three reductions +J/0/−J to ~1e-16), notes compile clean.
- [x] Code: added `_masterM(xi, al, be, sg, v, kappa)` in `energies.py`; `_m2` (v=_vPlus, κ=−1) and
  `_m1Prime` (v=_vMinus, κ=+1) now go through it — **bitwise-identical** refactor (diff 0.0), so the
  exact tier and exact-Newton are unchanged (still ~9e-13 at N=6).

**New: clean standalone write-up.** `notes/mollifiedDerivation.tex` (12 pp) — self-contained, very
detailed derivation of the mollified area + gradient + the exact-Newton scheme, written from scratch
(not a transcription; no blue/black split). Supersedes nothing; `definitions.tex` stays as the
transcription of your handwriting. Compiles clean, verification table in its \S9.

**Standing decisions / small stuff.**
- [ ] `r_i = √(ζ²+1)` and Lewin's `D(re^{iθ})` both use `r` (local, distinct) — rename if it bugs you.
- [ ] Equation numbers past 7.28 are mine (auto), not your handwriting — renumber when you fold it in.
- [ ] The `notes/definitions-*.pdf` handwriting still has the deprecated 4-pole path + the 7.22 sign
  (reads `+`, should be `−σ²/(|e_α|e_β^y)`) — fix in the handwriting when convenient.

---

Sharp (straight-edged) polygon model with a **Plummer (softened-log) mollified overlap** for
machine-precision minimization. The rounded model and the exploratory Gaussian overlap have been
retired to `scrap/` (recoverable). Keep this list updated as items land.

## Module map (post-consolidation, 2026-07-04)
`enums` · `packing` (+box/PBC) · `build` (area dist + equilateral backbone build) · `softBody`
(backbone geometry + eqSoftBody) · `energies` (SHARP overlap + PLUMMER overlap [semi- & fully-analytic]
+ self-repulsion) · `minimize` (FIRE/GD/CG) · `model` (facade + save/load). `Model` only BUILDS the
packing; energies are driven directly off `model.packing`.

## Fully analytic Plummer tier — remaining
Derivation + 25-digit mpmath oracle: `notes/plummerOverlap.tex` + `notes/verify_plummer_analytic.py`
(tests T0–T7 pass ~1e-24). numpy/scipy port (`plummer*Exact`, Li2 via `scipy.special.spence`) matches
the semi-analytic tier + oracle.
- [x] **Vectorized** the exact tier (2026-07-04): batched over all edge-pairs (np.where for the
  parallel branch), matches scalar to 1e-10. NB the tier is dilog-bound, so it's ~8x SLOWER per pair
  than the semi-analytic (accuracy, not speed — as §6.7 predicted).
- [x] **Near-parallel bridge (§6.7) — LANDED 2026-07-19**: the panel arctan term and the gradient
  moments all divide by X1 = |e_a|sin(theta_ab), which vanishes near-parallel; the closed forms then
  lose precision to cancellation (al = P1/X1 -> inf), though the underlying integrals are finite. Fix:
  with xi = X0 + u X1 they are the regular u-integrals int_0^1 sqrt(xi^2+sg^2) Theta du (energy) and
  the xi/sqrt(xi^2+sg^2) * {1,u} weightings (gradient), Theta = arctan((U0+P1 u)/sqrt(xi^2+sg^2)) --
  no 1/X1. Peak-split at u* = -U0/P1 (width ~sg), 2x24 Gauss, ~1e-15 down to sg~1e-3; applied for
  |X1| <= 1e-2*|e_a|, closed forms elsewhere (`energies.py` `_ceBridge`, `_wBridge`; notes bridge
  block at end of definitions.tex §8). BOTH panels needed it (the earlier "gradient is healthy" read
  was FD-of-energy contamination): at N=6 (min|sin|=9.1e-6) exact ENERGY self-consistency 5e-2->7e-9,
  GRADIENT 5e-3->3e-9, both at the semi/FD floor. Verified vs a dense 2D ln-integral reference to
  ~1e-16 across the near-parallel sweep. Cosmetic: closed ceGen is still computed (then np.where-
  discarded) for near pairs, so huge al/be raise benign RuntimeWarnings -- wrap in np.errstate later.
- [x] **Exact-Newton unblocked 2026-07-19**: with the bridge in, Newton on the fully-analytic force
  reaches max|F|=9.3e-13 at N=6 in 4 steps (clean quadratic: 9e-7 -> 3.7e-8 -> 1.7e-12), where it
  stalled at ~1e-2 before. `Model.minimizeNewton(exact=True)` now works; update its docstring (still
  says "leave it off until the near-parallel bridge lands").
- [x] **Newton-to-precision proof of concept (2026-07-05)**: FIRE + Newton on the SEMI force reaches
  max|F|=5.7e-12 at N=6 (`pyPolygon/pocPrecision.py`) -- mollification beats the sharp 3.6e-3 floor by
  ~9 decades. The semi force is self-consistent, which is what Newton needs.
- [ ] **Analytic Hessian** (= Phase-10 stiffness matrix) → Newton at N=32. Newton-on-semi already hits
  ~1e-12 (correctness proven); the only blocker at scale is COST -- the FD Hessian is 2*(2N) semi evals
  (~14 min/step at N=32). The analytic Hessian removes that, and is also the Phase-10 stiffness anyway.
  - [x] **DERIVED 2026-07-20** in `notes/mollifiedDerivation.tex` §9. Key result: the one new object is
    grad Psi_B = -oint K_sigma n_B dl (the mollifier smeared on dB, verified vs FD to 3e-10), whose
    per-edge form int dr/q^2 is rational+arctan. The outer moments add a 4th master-form row
    (weight (X^2+1)^-3/2 -> V=X/sqrt(X^2+1)), whose V*Theta' is RATIONAL -> **no new transcendentals**;
    the dilog never reappears. Cost = one edge-pair pass (~4*Nv cheaper than FD). Sparse + symmetric.
  - [x] **IMPLEMENTED + TESTED 2026-07-20** in `energies.py`: `_plummerQuadInts` (I0,I1),
    `_gradPlummerMeasure`, `_dPlummerMeasureDvB`, `_pairHessianOneSided`, `plummerPairHessian`,
    `plummerOverlapHessian`; `minimize.minimizeNewton(hessian=...)` hook. Verified: pair Hessian vs FD
    5e-11, symmetry EXACT 0 (A-B and B-A come from different code paths), packing Hessian rel 2.5e-7
    (= the semi/exact TIER floor, not an impl error: 3e-9 gradient gap / sigma; our H is closer to
    FD(exact) than the semi-tier H is). **Newton at N=6: 2 steps, 6 s, max|F|=6.2e-13** vs FD-Hessian
    4 steps, 345 s, 9.3e-13 -> **~57x end-to-end**, Hessian build 0.9 s vs 83 s (~110x), and the gap
    grows linearly with Nv. Written up in `notes/mollifiedDerivation.tex` §10 + verification tables.
  - LESSON (cost me a 30-step non-convergence first): the Hessian needs a FINER outer rule than the
    force -- grad Psi_B is sharper than Psi_B. Reusing the force's 32-pt rule => H only 8e-7 accurate
    and Newton loses its quadratic tail. Dedicated 96-pt `_HNODES` fixes it. Refining past 64 pts does
    nothing (tier floor).
  - [ ] Next: time it at N=32 (where the ~4*Nv advantage should really show), and apply the
    near-parallel bridge inside S^(pq) for grazing pairs.
- [ ] **Exact self-repulsion FORCE**: `S_closed` (oracle T7) is the energy integral only; the analytic
  force is "differentiate w.r.t. the four endpoints" (mechanical, no new transcendentals). Current
  self-rep is the Gaussian-bump barrier (analytic force, works) — refine kSelf/delta / kernel later.
- [ ] **DELETE the semi-analytic stuff** (Cam, 2026-07-04): the O(G) outer-quadrature evaluators in
  `energies.py` — `plummerMeasure`, `plummerPairGradient`, `plummerPairEnergy`, `plummerOverlap` (and
  the `_OUTNODES/_OUTWTS` Gauss-Legendre nodes) — once the fully-analytic `*Exact` tier is signed off
  and vectorized. Keeping them as the cross-check oracle for now.
- [ ] All energy functions carry `# UNVERIFIED(Cam)` — awaiting sign-off.

## Periodicity / scaling
- [ ] Image offsets are integer (dx,dy) = SQUARE box only; `latticeVector` (box mode) needs box.h
  lattice-vector columns.
- [ ] Self-image overlap (a polygon vs its OWN periodic image) is not summed. Harmless while
  diameter < box side.
- [x] Neighbor list + smooth switch (2026-07-04): `_assembleOverlap` prunes far polygon pairs on the
  centroid separation (radius = targetPerimeter/4, a fixed a-priori bound so the C2 switch stays
  smooth) and multiplies each pair by S(|c_A-c_B|); excluded pairs are EXACTLY zero. FD-clean (2e-8),
  ~20x faster at N=32 (410 ms vs 8100), FIRE ~0.6 s/step. Removes the 1/r^4 tail past rOff (an
  intended O(sigma^2) model modification, §7.2).
- [ ] Neighbor-list follow-ups: (a) a proper VERLET list with skin + rebuild triggers (rebuilt each
  call now); (b) targetPerimeter/4 is loose (0.174 vs measured 0.121 at N=32) -- a tighter fixed
  radius or measured-radius-with-skin prunes more; (c) edge-pair-level switch (§7.2) instead of the
  coarser polygon-pair switch, if intra-pair edge pruning is needed.

## Sharp overlap (energies.py — the exact geometric area / sigma->0 reference)
- [ ] `updateIntersections` is ALL-TO-ALL O(N^2) (reference oracle); `updateFollowers` is O(M^2) —
  replace with neighbor-derived pairs / sort+binary-search (definitions.tex sec 5); fold the two.
- [ ] Degeneracies: strict `0<s<1` skips exactly-collinear edges / crossings landing on a vertex
  (measure-zero). Containment (one polygon inside another, no crossing) returns 0.
- Confirmed to floor at max|F|~3.6e-3 (the vertex-edge contact kink) — the gradient is discontinuous
  there, so no minimizer reaches machine precision; the mollified route is the answer.

## Minimizers (validated end-to-end on the Plummer energy, no folds)
- FIRE→CG→Newton ladder: FIRE ~200 steps/decade (robust), CG ~10-15/decade (conditioning-limited),
  Newton (FD Hessian) 5 iters to 2.4e-9. Hybrid: FIRE to ~1e-3, then Newton. Self-repulsion keeps
  loops simple, which is what makes CG/Newton safe (they folded polygons before).

## Notes
- [ ] `definitions.tex` (sharp reference): sign flags in blue — eq 1.3 area factors; eq 2.2/2.3
  numerator sign (`v_j - v_i` for s). The CODE uses the correct signs.
- Live: `plummerOverlap.tex` (production derivation), `definitions.tex`, `overlapImplementation.tex`,
  `verify_plummer_analytic.py`, `mollificationDemo.png`.

## Retired to scrap/ (recoverable)
Rounded model (`geometry, neighbors, overlap, overlapForce, springs, intersections, visualize,
validate` + examples + rounded notes); Gaussian overlap (`mollifiedOverlap.py`, `smoothOverlap.tex`).
`tests/overlapAreaDemo.ipynb` is rounded-era and now errors — a fresh sharp/Plummer demo is Cam's to write.
