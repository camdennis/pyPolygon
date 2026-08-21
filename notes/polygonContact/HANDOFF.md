# Polygon–Polygon Contact for Nonconvex Deformable Particles — Handoff

## What this is

A verified specification plus reference implementation of a contact law for
**simple polygons, convex or not**, intended for deformable-particle simulation
of jamming and the reversible–irreversible transition under cyclic shear.

The law is an **interfacial integral**, not a pairwise overlap function:

```
E = 1/2 * sum over ordered pairs (P,Q) of   ∫_{∂P ∩ Q}  (k/3) d_Q(x)^3  dl(x)
```

where `d_Q(x)` is the distance from `x` to `∂Q`. Everything — closed forms,
gradients, scaling, regularity — follows from that one expression.

**Your job is to port `reference/polycontact_ref.py` to fast code (CUDA /
vectorized) without changing what it computes.** The reference is slow and
deliberately literal. `tests/test_reference.py` is the contract.

## Read in this order

| File | What for |
|---|---|
| `HANDOFF.md` | this file — orientation, build order, traps |
| `spec/derivation.pdf` | the mathematics: every step from geometry to force. **Read before writing code.** |
| `spec/contact.pdf` | the design record: why these choices, what was rejected, validity limits |
| `reference/polycontact_ref.py` | ground truth implementation, ~450 lines |
| `tests/test_reference.py` | the contract; must stay green |
| `figures/` | four annotated figures of the tests |

## Current status

**Verified and settled.** Closed-form energy (no quadrature anywhere).
Closed-form analytic gradient, agreeing with finite differences to ~1e-12 and
54× faster than differencing. Nonconvex handled with no convex decomposition.
`O(M)` scaling in vertex count. Corner scaling exponents 3 / 7-2 / 4.
`C²` regularity through contact onset, feature switches, and
crossing-through-vertex events. Machine-precision force balance (`|F|∞ =
1.7e-14`) demonstrated from a certified-disjoint lattice under adaptive
compression.

**Open.**
1. Machine-precision force balance on a valid **and stressed** state. The two
   contacting configurations reached in the compression run got to `8.6e-9`
   with the L-BFGS iteration cap binding, not the tolerance. Raise the cap.
2. Reaching a genuinely jammed state for the cross shape family needs much more
   compression — the crosses interdigitated into an unstressed configuration at
   φ ≈ 0.55.
3. Adhesion. Deferred by design; see `contact.pdf` §12 for the argument and the
   two structural constraints (whole-boundary domain, signed distance).
4. Analytic Hessian. Derivable by the same three-group Leibniz route in
   `derivation.pdf` §3. Needed for normal modes, not for relaxation.

## Build order

1. **Geometry kernel.** `edges`, `is_reflex`, `nearest_feature`, `membership`,
   `crossings`, `spans`. Get tests 1, 6, 9, 11 green.
2. **Energy, closed form.** `E_pair_closed`. Tests 2, 3, 10.
3. **Gradient, closed form.** `grad_pair`. Tests 4, 4b, 5. This is where the
   bugs are; see traps below.
4. **Uniform grid.** One grid over *all* edges of *all* bodies, cell size ≈ 2×
   mean edge length. Serves crossings, nearest-feature, and (via `membership`)
   inside/outside. `O(M)` total. `contact.pdf` §11.
5. **Output-sensitive march** (`march`) to replace `feature_partition`'s
   sample-and-bisect. Test 7 pins them together.
6. **Relaxer.** L-BFGS on the analytic gradient. Do **not** use FIRE alone.
7. **Protocol.** Certified-disjoint lattice placement, adaptive compression with
   the `d_max/r_in` rejection test.

## Conventions — fixed, do not re-derive

Four sign errors during development all came from re-deriving the outward
normal. Define these once, in one place:

```
CCW vertices, signed area > 0
edge j:            g_j   = V[j+1] - V[j],   tau_j = g_j/|g_j|
OUTWARD normal:    n_j   = (tau_j.y, -tau_j.x)
signed line dist:  ell_j(x) = n_j . (V[j] - x)      > 0 inside
perpendicular foot on line j:   q = x + ell_j * n_j      NOT  x - ell_j * n_j
```

## Traps

Each of these was hit during development and each is invisible to the obvious
test. Do not simplify the tests that catch them.

**T1 — Missing arclength factor `L_i`.** The integrand group of the gradient
carries a factor `L_i = |e_i|`. Omitting it is **exact** whenever the contacting
edge has unit length. A suite built on unit squares will not detect a 48% error.
*Test:* gradient case `parallel faces, |e|=2`.

**T2 — `1/m` antiderivative.** `∫d³dt` has a compact antiderivative that
divides by `m = n_j·e_i`, which vanishes **exactly** for face-parallel contact —
the dominant configuration — and cancels catastrophically near it. Use the
**polynomial** form in the moments `M_q` instead; no division, no branch, no
tolerance. *Test:* energy on an exactly parallel pair and one perturbed off
parallel by 1e-8.

**T3 — `ρ̂` instead of `ρ⃗`.** In the vertex-nearest gradient branch the
perpendicular offset enters **undivided**. Using the unit vector gives ~1e-3
relative errors that conservation does not catch.

**T4 — Quadrature paired with a Leibniz gradient.** The Leibniz three-group
formula differentiates the *exact* integral. A fixed-node quadrature sum has
its own dependence on moving nodes, so the formula is not its derivative.
Mixing them gave ~1e-2 relative gradient errors. **Whatever you evaluate for
the energy is what you must differentiate.**

**T5 — Conservation is a nearly worthless test.** Net force and torque vanish
*structurally*, independent of whether `P0, P1, V0, V1` are correct. They passed
on every buggy intermediate version, including one with a 48% error. Only
finite differencing localizes anything.

**T6 — Cached reflex flags.** `membership()` depends on per-vertex reflex flags,
and a deforming body can flip a vertex convex↔reflex. Caching them with a
neighbour list silently inverts the inside/outside test for exterior points near
a flipped vertex — a sign error in `d` with no crossing and nothing raised.
Recompute every step: one cross-product sign per vertex.

**T7 — Wrapping individual vertices under PBC.** A body straddling a boundary
acquires edges spanning the box, after which crossings, parity and
nearest-feature all return nonsense on a well-formed body. Store vertices
unwrapped relative to a body reference point; wrap only the reference point;
apply minimum image at the **body-pair** level.

**T8 — Testing only on symmetric configurations.** Slabs, unit squares and
parallel faces hide sign errors by coincidence: for a slab, `x - ell*n` lands on
the *other* face's correct foot. Always include a non-unit edge length, a
rotated pair, and a vertex-versus-edge tie.

## The validity limit — the one hard constraint

`d_B` has a **ridge** (the medial axis) at distance ≈ the inradius from the
boundary. Crossing it is not an accuracy loss, it is a **sign reversal**: past
the ridge the leading edge's depth *decreases*, and for a thin limb (little
side-edge chord to compensate) `dE/dδ` goes negative and the bodies are pulled
through. Measured: `dE/dδ` positive up to `δ = r_in`, negative beyond.

```
REQUIRED:   max over overlap components of d_max / r_in  <<  1
```

For a limbed shape `r_in` is the **limb half-width**, not the particle size.
Assert this per overlap component, every step, and treat a violation as a failed
step. Margin is comfortable in the target regime — the barrier is 1185× a
contact at 10% of `r_in` — so the risk is **discrete jumps**, not creep:
initialization at target density, an oversized compression or strain increment,
and avalanche transients.

Good news for packing generation: the signature is **persistent**, not
transient. Crossed limbs report `d/r_in ≈ 1` for as long as they stay crossed,
so a check on the converged state suffices. For **AQS it does not suffice** —
there the trajectory *is* the observable, and a pass-through injects a spurious
irreversible event, i.e. contaminates exactly the signal being measured. Check
every relaxation iteration in that mode.

## Two facts about `d_B` that make the whole thing work

**Feature switches are harmless; the medial axis is not.** Two features are tied
at *every* switch. The criterion is whether they share a **nearest point**:

- incident features (edge and its own endpoint) share it → `∇d` continuous → `C¹`
- non-incident features (medial axis) do not → nearest point jumps → `∇d` jumps

Measured: benign case, two tied distances both 0.1000000 realized at the *same*
point, separation 0. Medial axis, 0.159991 and 0.160009 realized at points
0.320000 apart (the full limb width), slope of `d_B` turning `+1 → −1`.
A run-time medial-axis check must test **realizing-point separation**; counting
near-tied features does not distinguish the cases.

**Integration buys back one derivative.** `d_B`'s Hessian jumps at every feature
switch, which would land directly in the stiffness matrix for a *pointwise*
depth potential. Under `∫ dl` the adjacent boundary terms cancel exactly, so the
switch costs nothing and the bisection locating it never needs differentiating.
That is why no regularization length appears anywhere in this formulation.

## Initialization protocol

```
1. place bodies on a lattice with spacing > 2 * circumradius   (certifies E = 0)
2. assert E == 0 and d_max == 0 exactly
3. compress in small increments; after each, relax and check d_max/r_in
4. if d_max/r_in > 0.35, reject the step, halve it, retry
```

Disjoint initialization is **necessary but not sufficient**: one large
compression step from a certified-disjoint lattice (container half-size 1.218 →
0.90) reproduced `d_max/r_in = 1.0000`.

If you inherit a bad configuration, the **overlap-area** potential
`½ k_A (area)²` repairs it where the depth potential cannot — it drove a
maximal-depth state to exactly zero overlap in ~120 iterations while the depth
potential stalled, lowering its own energy by making overlaps shallower and
wider. `overlap_area()` is available for free from the same span set via Green's
theorem. But it is **not** guaranteed: for two perpendicular bars fully crossing,
both the area and the depth energy are *exactly* translation-invariant, so
neither supplies an escape gradient. Prevention by construction is the only
guarantee.

Note that **interdigitation is not interpenetration**. Bodies whose convex hulls
overlap while the bodies themselves are disjoint are valid and are the natural
dense state for nonconvex shapes.

## Minimiser

Use **L-BFGS on the analytic gradient**. From a 1e-4 perturbation of a minimum,
with the identical gradient: L-BFGS reached `9.8e-11` in 41 iterations; FIRE
stalled at `5.5e-6` after 150. FIRE is fine for a far-from-minimum rough phase.
The energy is `C²`, so Newton converges quadratically on a fixed contact set,
but the Hessian is **not** guaranteed positive definite (corner contacts
contribute negative transverse terms), so any Newton variant needs a trust
region or modified factorization.

## Rejected alternatives — do not revisit

Each was tested and failed; `contact.pdf` §10 has the numbers.

- **Vertex-only features** (softmin/softmax over vertex penetrations) —
  *impossible*, not merely inaccurate. For two crossed bars every vertex of each
  body lies outside the other, so the vertex-depth vector is identically zero and
  any function of it returns the separated-pair value on a 4% overlap.
- **Vertex-sampled quadrature** — blind to shallow face-on-face contact; inverts
  the face/vertex contrast (true ratio 2.00, reported 0.001).
- **Smoothed minimum-translation distance** — reports the same depth for
  face-on-face and vertex-on-face; gradient vanishes identically for symmetric
  crossed bars.
- **Convex decomposition** — low by a factor of **19** on a convex control case,
  because distance to a piece's boundary is not distance to the body's boundary.
- **Disk-decorated boundaries (spheropolygons)** — removes nearly every
  difficulty, but corrugation gives an artificial friction of 1.8e-3 even at 8
  disks per contact radius, decaying only as (s/r)². That manufactures hysterons
  at the mesh scale, which is fatal for a study whose entire point is that
  adhesion is the *only* source of hysteresis. Eq. (1) is exactly
  translation-invariant along a flat face, by construction rather than by
  convergence.

## Running the tests

```bash
python tests/test_reference.py        # ~2 min, all PASS expected
```

If a gradient test fails, **check `d_max/r_in` first**. A gradient failure on a
configuration with a contact on a medial axis is the formulation working
correctly — the gradient does not exist there. Test 4b is a negative control
that asserts exactly this.
