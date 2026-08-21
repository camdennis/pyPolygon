# The penetration-depth model: what it is, what works, and how it compares

Status review as of 2026-08-01. Everything quoted here was measured in this codebase; nothing is
estimated. Companion to `notes/softDepth.tex` (the derivation) and `TODO.md` (the live task list).

---

## 1. Where it sits among the alternatives

Three families of overlap law, distinguished by which pair of geometric objects the energy integrates
over:

| family | energy | status |
|---|---|---|
| boundary–boundary | `int_dA int_dB K(\|x-y\|)` (Plummer panels) | rejected |
| **boundary–area** | `int_dA phi(h_eps^B) dl` | **this model** |
| area–area | `\|A ∩ B\|` (sharp / mollified) | in use, and still the workhorse |

**boundary–boundary** was rejected because the kernel has no notion of *inside*. `K(|x−y|)` sees only
the distance between two curves, which is unsigned: once A is inside B the two boundaries are far apart
again, so the repulsion is non-monotone in penetration and admits pass-through. Separately, one `lambda`
sets both geometric fidelity and stiffness, so they cannot be chosen independently.

**area–area** cannot choose its contact exponent. The measure is already `d^1` on faces and `d^2` at
corners, so with `E ~ a^2` the force is harmonic — vanishing at onset but with a jump in `phi''`, hence
C1 only. Getting Hertz would need `E ~ a^(5/4)`, whose second derivative blows up as `a -> 0`. And
evaluation is combinatorial: `|A ∩ B|` needs the crossing topology, which is where the exact-tangency
degeneracy lives (both code paths disagreed *and* both were wrong: 5.82e-02 vs 1.68e-01 against a truth
of 0).

**boundary–area** decouples the measure from the contact law. The geometry supplies `dl`, the physics
supplies `phi`, and the exponent is free. That is the entire argument for it.

### The contact-law exponents this buys

```
                       face-on-face        corner-into-face
  boundary-area        E ~ L d^(5/2)       E ~ d^(7/2)
                       F ~ L d^(3/2)       F ~ d^(5/2)
  area-area (E=k a^2)  E ~ L^2 d^2         E ~ d^4
                       F ~ 2k L^2 d        F ~ 4k d^3     (measured 3.955 / 2.966)
```

Face contact is C2 at onset (`F' ~ d^(1/2) -> 0`) where the area law's `F'` jumps to `2kL^2`. At a
corner the force is `d^(5/2)` instead of `d^3` — at `d = 1e-3`, 32x stronger.

---

## 2. What is built

| component | file | state |
|---|---|---|
| soft depth, gradient, Hessian, contact law | `softDepth.py` | verified |
| boundary integral with contact + envelope splitting | `softDepth.py` | verified |
| packing assembly, periodic, culled | `softDepth.py` | verified |
| CUDA convex kernel | `cuda/softDepth.cuh`, `cuda/softDepthKernels.cu` | verified, 99–548x |
| non-convex support (convex differences tree) | `convexDifference.py` | verified in numpy only |
| container / confinement | `softDepth.py` | fixed 2026-08-01, verified |
| adhesion (chord form, sec:chord) | — | not built |

### Verified numbers

Pair level (`tests/softDepthCheck.py`, 11 checks):

```
eikonal identity            6.661e-16      forces / torques   3.9e-16 / 1.8e-16
equilibrium cubic residual  1.617e-15      work of adhesion   2.19e-06
apex = eps log 2            n = 4, 6, 8    dE/dx, dE/dv_j     1.1e-10, 1.9e-10
boundary integral vs an independent uniform boundary walk:
    face-to-face 0.0   face-to-face sharp 3.7e-11   corner 1.5e-10   deep 1.3e-11
conservation  ~1e-17          periodic translation invariance  6.7e-16 energy, 1.8e-18 force
```

Device level (`tests/softDepthCudaCheck.py`, 7 checks). The two tiers are deliberately *different
algorithms* — numpy runs 24 root steps with a probe-based envelope, the device runs 12 with an exact
envelope walk — so agreement is an independent check, not two spellings of one loop:

```
N=32 n=32 convex   relE 3.33e-15   max|dF| 7.62e-14
MIXED n = 4, 7, 13 relE 6.08e-14   max|dF| 1.93e-13
n past every stride: 4, 12, 13, 32, 33, 64      SOFTDEPTH_MAXN reported, never truncated
speed   N=32 n=4  548x     N=32 n=32  99x     N=64 n=8  295x
```

Non-convex (`convexDifference.py`): composed depth matches true distance to `eps log 2` in the mean;
energy matches an independent walk to 9.9e-09; conservation 7.2e-18; FD 1.25e-08.

---

## 3. What is broken or deliberately accepted

**Container — fixed 2026-08-01.** It had contributed exactly zero, always: the confinement path
reversed a loop's winding on the reasoning that this negates `h`, but `h` is a soft*min* and
`min(-ell) = -MAX(ell)`, so the reversed loop's `h` came out negative everywhere. It now penalizes
`[-h]_+` on the unreversed loop, over the complement of the contact interval — `h` is concave so
`{h >= 0}` is one interval and its complement is `[0, t0]` and `[t1, 1]`, bounded by the same roots.
`-h` reads the largest single half-plane violation: exact through a face, short by `sin(theta/2)` past a
corner, the same accepted bias as the reflex corners.

**Winding — normalized 2026-08-01.** `n_i = J t_i` is outward only for a CCW loop. A clockwise wall (as
written in `squaresInASquareArea-Boundary.ipynb`) put `h` at −0.5139 in the box's own centre, making
`-h` minimal there and turning confinement into an attractive well: five squares collapsed onto a point.
`loopFrame` now normalizes, and the CUDA kernel matches.

**Reflex-corner bias — accepted, unfixed.** Outside a convex piece the half-plane softmin reads the
largest single half-plane violation, not the Euclidean distance, so at a corner of interior angle
`theta` the depth is short by `sin(theta/2)` and the energy by `sin(theta/2)^(5/2)`:

```
                 reflex corners   pocket angle med / min   energy factor med / worst
n= 8 kappa= 4          18            133.9 / 125.4 deg          0.812 / 0.745
n=16 kappa= 4          59            154.9 /  55.2 deg          0.941 / 0.146
n=32 kappa= 4         115            158.5 /   2.2 deg          0.957 / 0.0001
n=32 kappa=20         312             82.6 /   0.6 deg          0.354 / 0.0000
```

Median benign, tail catastrophic, and single-signed so it biases rather than averaging out. Accepted to
keep `ell_i` affine along an edge, which the exact envelope walk, the single-contact-interval guarantee
and sec:chord's closed-form stretch integral all depend on. Surfaced at runtime by
`convexDifference.warnOnSharpPockets`.

**Quadrature is removable and has not been removed.** sec:chord states the stretch integral is
closed-form (`Psi_lambda(z) = (z + sqrt(z^2+lambda^2))/2`) and that raising Gauss order is the wrong
repair. The same applies to the repulsive term: with `h ~ alpha − beta u` affine per stretch,
`int (alpha − beta u)^(5/2) du` is elementary. The current code spends Gauss nodes where an exact
formula exists.

### Four silent failures found in one day

Worth recording as a class, because all four reported success:

1. **Vertex sampling.** `packingEnergyForce` evaluated the integrand only at polygon vertices. Two
   squares meeting face-to-face have no vertex of either inside the other, so it returned exactly `0.0`
   against 7.749e-02 of real overlap — and was wrong where it was non-zero (2.563e-02 against a true
   7.272e-03).
2. **Periodicity.** The pair shift was applied to both bodies, cancelling, while the cull still used the
   minimum image. A pair overlapping only across the seam measured `0.0` against 8.2e-05.
3. **Convexity.** Lemma 1 requires convex loops; the builder produces 197 reflex corners across 1024
   vertices, and `h` at one polygon's own centroid read −4.29e-02. Energy 5.2e-13, `max|F|` 1.9e-10
   ("converged") on a configuration the sharp tier scores at 3.907 with `max|F|` 0.658. `isConvex`
   existed from the start and was never called.
4. **CG on the mollified tier.** Not soft depth, but found alongside: `max|F|` is bit-identical after
   1000 CG steps and 185 seconds.

The common thread is that every one of them was read as success from `max|F|`. A converged gradient
proves the force is consistent with the energy, and says nothing about whether the energy models
anything. That is now guarded by `tests/softDepthPackingCheck.py`, which measures against the sharp
tier rather than against the model's own gradient.

---

## 4. The comparison that matters

Identical start, identical constraints, 16 convex squares, `phi = 0.8` — a density where a zero-overlap
arrangement provably exists:

```
SHARP (area-area):   real overlap 6.70e-01 -> 1.48e-09    finds the valid packing
softDepth:           real overlap 6.70e-01 -> 9.9996e-01  reports max|F| = 1.1e-08
```

Two hypotheses were tested and both disproven, so this is not an implementation fault:

- *dropped contacts* — brute force over every ordered pair and all 9 periodic images reproduces the
  assembled energy to a ratio of **1.000000**;
- *wrong force* — started from the zero-overlap packing the sharp tier finds, soft depth scores it
  7.77e-11 and **holds it** (overlap stays at 7.1e-09).

So the energy recognises the valid packing as its minimum and the force does not push off it. This is a
landscape property, and the mechanism is intrinsic to a depth law:

> Soft depth penalizes penetration DEPTH and never AREA. For a contact of chord `L` and depth `d`,
> `E ~ L d^(5/2)` while `area ~ L d`, so at **fixed area** the energy is `~ a d^(3/2)` — strictly lower
> the thinner and wider the overlap is spread. A quench can lower the energy monotonically while the
> real overlap area grows.

The area-squared law has no such direction: its functional is minimised exactly when area is zero, which
is the goal. This is the flip side of the trade in §1 — choosing the exponent freely is what buys C2
contact, and the price is that area is not in the functional at all.

`tests/softDepthPackingCheck.py` check 3 reproduces this on a third configuration, N=16 n=4 at
phi = 0.8 over 1500 FIRE steps: the energy falls **84x** (2.2288e-03 -> 2.6578e-05) while the real
overlap **grows 27%** (6.4285e-01 -> 8.1334e-01). The check reports this rather than asserting against
it, deliberately — asserting that overlap must fall would encode an expectation the functional does not
support.

### Performance, for completeness

```
N=32 n=32, one force evaluation
  sharp                   3 ms
  softDepth CUDA (convex) 26 ms      99-122x over its own numpy
  softDepth numpy         3186 ms
  softDepth numpy, NONCONVEX      12677 ms   (CUDA port not written)
```

---

## 5. Assessment

**The two models are good at different jobs, and the evidence now separates them cleanly.**

For *finding dense packings* — the actual goal — area–area wins on today's evidence, and not
marginally. Its functional is aligned with the objective; the depth law's is not. No amount of tuning
fixes that, because it is the definition rather than the implementation.

For *smooth mechanics* — Hessians, AQS trajectories, contact make/break without stiffness jumps, bond
hysteresis — soft depth gives what area–area structurally cannot: C2 at onset, real-analytic everywhere,
closed-form gradient and Hessian, and no combinatorial topology (hence no tangency degeneracy, which is
a live defect on the sharp tier). Those were the reasons for building it and they still hold.

**Is it a mess?** The convex path is not: it is clean, verified against an independent construction, and
99–548x on the GPU. The mess is the non-convex extension, and it is a chain of forced consequences —
the depth field needs convexity, which needs decomposition, which needs the R-function, which breaks the
eikonal identity, which removes the cheapest validation, which is how three of the four silent failures
survived. Each link is sound; the chain is fragile.

### Recommendation

1. **Keep the convex tier.** It is finished, fast and verified. Convex shapes (n=4, or any convex loop
   supplied directly) get the full benefit at 99–548x.
2. **Do not invest further in the non-convex CUDA port yet.** It is the largest remaining piece of work,
   and §4 says the model it would accelerate is the wrong tool for packing generation. The numpy
   reference exists and is verified if the physics is wanted.
3. **Use area–area for packing generation**, which is what it is already doing.
4. **Revisit soft depth when the question is mechanics rather than search** — a jamming/AQS study, where
   the C2 property is the whole point and where the sharp tier's Hessian jumps and tangency degeneracy
   are the actual blockers.
5. Two cheap items worth doing regardless: fix the container (a modelling decision, then small), and
   replace the quadrature with sec:chord's closed form (removes a whole class of accuracy question).

### Caveat on the strength of §4

One seed, three configurations (N=16 n=4 and N=8 n=8 at phi = 0.8, N=16 n=4 at phi = 1.0), convex and
non-convex, all showing the same direction. Consistent, but it is not a seed study. Before treating
"area–area finds packings and depth does not" as settled it should be repeated across seeds and
densities. The *mechanism* (`E ~ a d^(3/2)` at fixed area) is analytic and does not depend on the
sample, which is why it is the part worth trusting.
