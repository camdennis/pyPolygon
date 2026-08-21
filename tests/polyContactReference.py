# VENDORED VERBATIM from notes/files(3).zip -- reference/polycontact_ref.py.
# GROUND TRUTH for tests/polyContactCheck.py. Do not optimise or edit; port from it.
# The handoff is explicit: the fast implementation must reproduce this, not replace it.
"""
polycontact_ref.py -- REFERENCE implementation of the polygon-polygon contact law.

Slow, plain-numpy, deliberately literal. This is the ground truth a fast
implementation must reproduce. Do not optimise this file; port from it.

Conventions (fixed once, never re-derived -- four sign errors in development
all came from re-deriving the outward normal):
    polygons are simple, vertices counterclockwise, signed area > 0
    edge j of V:      g_j = V[j+1] - V[j],  tau_j = g_j/|g_j|
    outward normal:   n_j = (tau_j.y, -tau_j.x)
    signed line dist: ell_j(x) = n_j . (V[j] - x)     > 0 on the interior side
    perpendicular foot on line j:  q = x + ell_j * n_j      (NOT x - ell_j n_j)

Energy (repulsion only; adhesion is deferred, see spec section 12):
    E = 1/2 sum_{(P,Q)} int_{dP cap Q} (k/3) d_Q(x)^3 dl
"""
import numpy as np

# ---------------------------------------------------------------- geometry


def edges(V):
    """(g, G, tau, n) for a CCW polygon. n is the OUTWARD unit normal."""
    Q = np.roll(V, -1, axis=0)
    g = Q - V
    G = np.linalg.norm(g, axis=1)
    tau = g / G[:, None]
    n = np.stack([tau[:, 1], -tau[:, 0]], 1)
    return g, G, tau, n


def signed_area(V):
    Q = np.roll(V, -1, axis=0)
    return 0.5 * np.sum(V[:, 0] * Q[:, 1] - V[:, 1] * Q[:, 0])


def make_ccw(V):
    return V if signed_area(V) > 0 else V[::-1].copy()


def is_reflex(V):
    """Per-vertex reflex flag. MUST be recomputed every step: a deforming body
    can flip a vertex convex<->reflex, and membership() depends on this."""
    out = np.zeros(len(V), bool)
    for i in range(len(V)):
        a, b, c = V[i - 1], V[i], V[(i + 1) % len(V)]
        out[i] = ((b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0])) < 0
    return out


def inside_parity(P, V):
    """Ray-cast parity. Exact for any simple polygon. O(M) -- used only as a
    reference; production membership should use membership() below, O(1)."""
    P = np.atleast_2d(P)
    Q = np.roll(V, -1, axis=0)
    y, x = P[:, 1][:, None], P[:, 0][:, None]
    y0, y1 = V[:, 1][None, :], Q[:, 1][None, :]
    x0, x1 = V[:, 0][None, :], Q[:, 0][None, :]
    c = (y0 > y) != (y1 > y)
    with np.errstate(all="ignore"):
        xi = x0 + (y - y0) * (x1 - x0) / (y1 - y0)
    r = np.sum(c & (x < xi), axis=1) % 2 == 1
    return r if np.ndim(P) > 1 and len(r) > 1 else r[0]


def nearest_feature(x, V):
    """(kind, j, d) with kind in {'E','V'}: nearest feature of dV to x.
    Brute force over 2M candidates, min-reduce. Branch-free on a GPU."""
    g, G, tau, n = edges(V)
    best = ("?", -1, np.inf)
    for j in range(len(V)):
        s = tau[j] @ (x - V[j])
        if -1e-15 <= s <= G[j] + 1e-15:                 # foot inside the edge
            d = abs(n[j] @ (V[j] - x))
            if d < best[2]:
                best = ("E", j, d)
        d = np.linalg.norm(x - V[j])
        if d < best[2]:
            best = ("V", j, d)
    return best


def realizing_point(V, kind, j, x):
    """The actual nearest POINT of feature (kind,j) to x. Whether two tied
    features share this point decides C^1 vs C^0 (spec eq. 5)."""
    g, G, tau, n = edges(V)
    return x + (n[j] @ (V[j] - x)) * n[j] if kind == "E" else V[j].copy()


def membership(x, V, refl=None):
    """Inside/outside from the nearest-feature query alone -- no ray cast.
        nearest = edge j    -> inside iff n_j.(V[j]-x) > 0
        nearest = vertex j  -> inside iff V[j] is REFLEX
    Verified against inside_parity on 4000 samples x 5 shapes, zero mismatch."""
    if refl is None:
        refl = is_reflex(V)
    g, G, tau, n = edges(V)
    best = (None, -1, np.inf, None)
    for j in range(len(V)):
        s = tau[j] @ (x - V[j])
        if -1e-15 <= s <= G[j] + 1e-15:
            lj = n[j] @ (V[j] - x)
            if abs(lj) < best[2]:
                best = ("E", j, abs(lj), lj > 0)
        d = np.linalg.norm(x - V[j])
        if d < best[2]:
            best = ("V", j, d, bool(refl[j]))
    return best                                          # (kind, j, d, inside)


def d_signed(x, V, refl=None):
    k, j, d, ins = membership(x, V, refl)
    return d if ins else -d


# ---------------------------------------------------------------- spans


def crossings(A, B, i):
    """Crossings of edge i of A with dB: list of (t, j, s).
        t = ((B[j]-A[i]) x g_j) / (e_i x g_j),   s = ((B[j]-A[i]) x e_i) / (e_i x g_j)"""
    eA, _, _, _ = edges(A)
    eB, _, _, _ = edges(B)
    p, ei = A[i], eA[i]
    out = []
    for j in range(len(B)):
        fj = eB[j]
        D = ei[0] * fj[1] - ei[1] * fj[0]
        if abs(D) < 1e-14:                               # parallel; see spec sec. 12
            continue
        r = B[j] - p
        t = (r[0] * fj[1] - r[1] * fj[0]) / D
        s = (r[0] * ei[1] - r[1] * ei[0]) / D
        if 1e-12 < t < 1 - 1e-12 and -1e-12 <= s <= 1 + 1e-12:
            out.append((t, j, s))
    return out


def spans(A, B):
    """Maximal stretches of dA inside B, as (i, p, e_i, t0, t1). Exact for any
    simple polygon: no convexity, no decomposition, no root finding."""
    eA, LA, thA, _ = edges(A)
    out = []
    for i in range(len(A)):
        ts = sorted({0.0, 1.0} | {c[0] for c in crossings(A, B, i)})
        for a, b in zip(ts[:-1], ts[1:]):
            mid = A[i] + 0.5 * (a + b) * eA[i]
            if membership(mid, B)[3]:
                out.append((i, A[i].copy(), eA[i].copy(), a, b))
    return out


def feature_partition(A, B, i, p, ei, t0, t1, nsamp=60, nbis=52):
    """Subdivide a span at nearest-feature switches. Reference version uses
    sampling + bisection for robustness. Production should use the closed-form
    output-sensitive march (see march() below) -- both are tested to agree."""
    ts = np.linspace(t0, t1, nsamp)
    fs = [nearest_feature(p + t * ei, B)[:2] for t in ts]
    cuts = [t0]
    for m in range(nsamp - 1):
        if fs[m] != fs[m + 1]:
            lo, hi = ts[m], ts[m + 1]
            for _ in range(nbis):
                mid = 0.5 * (lo + hi)
                if nearest_feature(p + mid * ei, B)[:2] == fs[m]:
                    lo = mid
                else:
                    hi = mid
            cuts.append(0.5 * (lo + hi))
    cuts.append(t1)
    return np.array(cuts)


def _quad_coeffs(p, ei, V):
    """Squared distance to every candidate is a QUADRATIC in t: (A,B,C,window,id).
    Edge candidates carry a validity window from the affine foot parameter."""
    g, G, tau, n = edges(V)
    L2 = ei @ ei
    out = []
    for j in range(len(V)):
        al = n[j] @ (V[j] - p)
        m = n[j] @ ei
        s0, s1 = tau[j] @ (p - V[j]), tau[j] @ ei
        if abs(s1) < 1e-14:
            win = (-np.inf, np.inf) if 0 <= s0 <= G[j] else (1.0, 0.0)
        else:
            r0, r1 = (0 - s0) / s1, (G[j] - s0) / s1
            win = (min(r0, r1), max(r0, r1))
        out.append((m * m, -2 * al * m, al * al, win, ("E", j)))
    for j in range(len(V)):
        c = p - V[j]
        out.append((L2, 2 * (c @ ei), c @ c, (-np.inf, np.inf), ("V", j)))
    return out


def march(p, ei, V, t0, t1, maxstep=64):
    """Output-sensitive forward march on the lower envelope. Closed-form roots
    only. Cost scales with the number of ACTUAL switches, not with M^2.
    Returns (breakpoints, winners)."""
    cand = _quad_coeffs(p, ei, V)

    def val(c, t):
        A, B, C, win, _ = c
        return np.inf if not (win[0] - 1e-15 <= t <= win[1] + 1e-15) else A * t * t + B * t + C

    lo, hi = [], []
    for A, B, C, win, _ in cand:                          # prefilter
        aa, bb = max(t0, win[0]), min(t1, win[1])
        if aa > bb:
            lo.append(np.inf); hi.append(np.inf); continue
        ts = [aa, bb] + ([-B / (2 * A)] if A > 0 and aa < -B / (2 * A) < bb else [])
        vs = [A * t * t + B * t + C for t in ts]
        lo.append(min(vs)); hi.append(max(vs))
    bh = min(hi)
    cand = [c for c, l in zip(cand, lo) if l <= bh + 1e-12]

    t, bps, wins = t0, [t0], []
    for _ in range(maxstep):
        vs = [val(c, t + 1e-13) for c in cand]
        w = int(np.argmin(vs))
        wins.append(cand[w][4])
        A0, B0, C0, win0, _ = cand[w]
        tnext = t1
        for q, c in enumerate(cand):
            if q == w:
                continue
            A, B, C, win, _ = c
            dA, dB, dC = A0 - A, B0 - B, C0 - C
            roots = []
            if abs(dA) < 1e-14:
                if abs(dB) > 1e-14:
                    roots = [-dC / dB]
            else:
                disc = dB * dB - 4 * dA * dC
                if disc >= 0:
                    sq = np.sqrt(disc)
                    roots = [(-dB + sq) / (2 * dA), (-dB - sq) / (2 * dA)]
            for r in roots + list(win):
                if t + 1e-12 < r < tnext:
                    tnext = r
        bps.append(min(tnext, t1))
        if tnext >= t1 - 1e-15:
            break
        t = tnext
    return np.array(bps), wins


# ---------------------------------------------------------------- energy


def _F_vert(w, r):
    """Phi_r(w) = int (w^2+r^2)^{3/2} dw. Verified symbolically, residual 0."""
    s = np.hypot(w, r)
    return w * (2 * w * w + 5 * r * r) * s / 8 + 3 * r ** 4 * np.arcsinh(w / r) / 8


def E_pair_closed(A, B, k=1.0):
    """int_{dA cap B} (k/3) d_B^3 dl in CLOSED FORM. No quadrature."""
    _, _, _, nB = edges(B)
    cB = np.einsum("ij,ij->i", nB, B)
    tot = 0.0
    for (i, p, ei, t0, t1) in spans(A, B):
        L = np.linalg.norm(ei)
        th = ei / L
        cuts = feature_partition(A, B, i, p, ei, t0, t1)
        for ta, tb in zip(cuts[:-1], cuts[1:]):
            if tb - ta < 1e-14:
                continue
            kind, j, _ = nearest_feature(p + 0.5 * (ta + tb) * ei, B)
            if kind == "E":
                al = nB[j] @ (B[j] - p)
                m = nB[j] @ ei
                # integer exponent => POLYNOMIAL in t. no division by m, no
                # cancellation, no near-parallel branch. see spec eq. (10).
                M0 = tb - ta
                M1 = (tb ** 2 - ta ** 2) / 2
                M2 = (tb ** 3 - ta ** 3) / 3
                M3 = (tb ** 4 - ta ** 4) / 4
                I = al ** 3 * M0 - 3 * al * al * m * M1 + 3 * al * m * m * M2 - m ** 3 * M3
            else:
                tj = (th @ (B[j] - p)) / L
                rv = (p - B[j]) + (th @ (B[j] - p)) * th
                r = max(np.linalg.norm(rv), 1e-14)
                wa, wb = L * (ta - tj), L * (tb - tj)
                I = (_F_vert(wb, r) - _F_vert(wa, r)) / L
            tot += L * (k / 3.0) * I
    return tot


def E_pair_quad(A, B, k=1.0, ng=32):
    """Same integral by fixed-node Gauss-Legendre. Independent cross-check of
    E_pair_closed. WARNING: if you evaluate the energy this way you must
    differentiate THIS expression -- pairing quadrature with the Leibniz
    gradient is wrong and gave ~1e-2 relative errors."""
    GX, GW = np.polynomial.legendre.leggauss(ng)
    tot = 0.0
    for (i, p, ei, t0, t1) in spans(A, B):
        cuts = feature_partition(A, B, i, p, ei, t0, t1)   # integrand is only C^1
        for ta, tb in zip(cuts[:-1], cuts[1:]):           # across a switch, so
            if tb - ta < 1e-14:                           # subdivide first
                continue
            hw, ct = 0.5 * (tb - ta), 0.5 * (tb + ta)
            pts = p + np.outer(ct + hw * GX, ei)
            d = np.array([nearest_feature(x, B)[2] for x in pts])
            tot += np.linalg.norm(ei) * hw * np.dot(GW, (k / 3.0) * d ** 3)
    return tot


def E_contact(A, B, k=1.0):
    """Symmetrised contact energy for the ordered pair."""
    return 0.5 * (E_pair_closed(A, B, k) + E_pair_closed(B, A, k))


# ---------------------------------------------------------------- gradient


def grad_pair(A, B, k=1.0):
    """(E, dE/dA, dE/dB) for int_{dA cap B} (k/3) d_B^3 dl, all closed form.

    Leibniz gives three groups (spec eq. 16):
      measure   : dL_i/dv times the span's own integral
      domain    : IDENTICALLY ZERO here (phi(0)=0 at crossings, dt/dv=0 at vertices)
      integrand : needs only P0 = int phi' dt and P1 = int phi' t dt
    Sub-stretch breakpoints contribute nothing: d_B is continuous across a
    feature switch, so the two adjacent boundary terms cancel exactly.
    """
    nA, nB_ = len(A), len(B)
    gA = np.zeros((nA, 2))
    gB = np.zeros((nB_, 2))
    E = 0.0
    _, _, thB, nB = edges(B)
    _, LB, _, _ = edges(B)
    for (i, p, ei, t0, t1) in spans(A, B):
        L = np.linalg.norm(ei)
        th = ei / L
        ip = (i + 1) % nA
        Ispan = 0.0
        cuts = feature_partition(A, B, i, p, ei, t0, t1)
        for ta, tb in zip(cuts[:-1], cuts[1:]):
            if tb - ta < 1e-14:
                continue
            kind, j, _ = nearest_feature(p + 0.5 * (ta + tb) * ei, B)
            M0 = tb - ta
            M1 = (tb ** 2 - ta ** 2) / 2
            M2 = (tb ** 3 - ta ** 3) / 3
            M3 = (tb ** 4 - ta ** 4) / 4
            if kind == "E":
                al = nB[j] @ (B[j] - p)
                m = nB[j] @ ei
                Iphi = k * (al ** 3 * M0 - 3 * al * al * m * M1
                            + 3 * al * m * m * M2 - m ** 3 * M3) / 3.0
                P0 = k * (al * al * M0 - 2 * al * m * M1 + m * m * M2)
                P1 = k * (al * al * M1 - 2 * al * m * M2 + m * m * M3)
                ddx = -nB[j]
                gA[i] += L * (P0 - P1) * ddx
                gA[ip] += L * P1 * ddx
                s0 = (thB[j] @ (p - B[j])) / LB[j]
                s1 = (thB[j] @ ei) / LB[j]
                Sg = s0 * P0 + s1 * P1                    # int phi' sigma dt
                gB[j] += L * (P0 - Sg) * nB[j]
                gB[(j + 1) % nB_] += L * Sg * nB[j]
            else:
                tj = (th @ (B[j] - p)) / L
                rv = (p - B[j]) + (th @ (B[j] - p)) * th   # NOTE: full vector, NOT unit
                r = max(np.linalg.norm(rv), 1e-14)
                wa, wb = L * (ta - tj), L * (tb - tj)
                sa, sb = np.hypot(wa, r), np.hypot(wb, r)
                J0 = lambda w, s: (w * s + r * r * np.arcsinh(w / r)) / 2
                J1 = lambda w, s: s ** 3 / 3
                J2 = lambda w, s: w * s ** 3 / 4 - r * r * w * s / 8 - r ** 4 * np.arcsinh(w / r) / 8
                Q0 = lambda w: w ** 3 / 3 + r * r * w
                Q1 = lambda w: w ** 4 / 4 + r * r * w * w / 2
                Iphi = k * (_F_vert(wb, r) - _F_vert(wa, r)) / L
                dJ0 = J0(wb, sb) - J0(wa, sa)
                dJ1 = J1(wb, sb) - J1(wa, sa)
                dJ2 = J2(wb, sb) - J2(wa, sa)
                V0 = k * (th * dJ1 + rv * dJ0) / L
                V1 = k * (tj * (th * dJ1 + rv * dJ0) + (th * dJ2 + rv * dJ1) / L) / L
                gA[i] += L * (V0 - V1)
                gA[ip] += L * V1
                gB[j] += -L * V0
            Ispan += Iphi
        E += L * Ispan
        gA[ip] += Ispan * th                               # measure term
        gA[i] += -Ispan * th
    return E, gA, gB


def grad_contact(A, B, k=1.0):
    """Symmetrised energy and gradients."""
    E1, gA1, gB1 = grad_pair(A, B, k)
    E2, gB2, gA2 = grad_pair(B, A, k)
    return 0.5 * (E1 + E2), 0.5 * (gA1 + gA2), 0.5 * (gB1 + gB2)


# ---------------------------------------------------------------- diagnostics


def overlap_area(A, B):
    """Exact overlap area from the span set via Green's theorem:
       Area = 1/2 sum over span segments p->q of (p x q). Order-independent."""
    tot = 0.0
    for X, Y in ((A, B), (B, A)):
        eX, _, _, _ = edges(X)
        for (i, p, ei, t0, t1) in spans(X, Y):
            a, b = p + t0 * ei, p + t1 * ei
            tot += a[0] * b[1] - a[1] * b[0]
    return 0.5 * tot


def d_max(A, B, ns=200):
    """Max penetration depth over the pair. THE validity monitor: assert
    d_max / r_in << 1. Past the medial axis the repulsion can reverse sign."""
    best = 0.0
    for X, Y in ((A, B), (B, A)):
        for (i, p, ei, t0, t1) in spans(X, Y):
            for t in np.linspace(t0, t1, ns):
                best = max(best, nearest_feature(p + t * ei, Y)[2])
    return best


def inradius(V, ngrid=400):
    """Largest inscribed-circle radius (grid estimate). Denominator of the
    validity ratio. For a limbed shape this is the LIMB half-width, not the
    particle size."""
    lo, hi = V.min(0), V.max(0)
    gx = np.linspace(lo[0], hi[0], ngrid)
    gy = np.linspace(lo[1], hi[1], ngrid)
    X, Y = np.meshgrid(gx, gy)
    P = np.stack([X.ravel(), Y.ravel()], 1)
    ins = inside_parity(P, V)
    best = 0.0
    for x in P[ins][:: max(1, ins.sum() // 4000)]:
        best = max(best, nearest_feature(x, V)[2])
    return best


# ---------------------------------------------------------------- shapes


def regular(M, R=1.0, phase=0.0):
    a = np.linspace(0, 2 * np.pi, M, endpoint=False) + phase
    return make_ccw(np.stack([R * np.cos(a), R * np.sin(a)], 1))


def cross_shape(arm=0.55, w=0.16):
    V = np.array([[arm, w], [w, w], [w, arm], [-w, arm], [-w, w], [-arm, w],
                  [-arm, -w], [-w, -w], [-w, -arm], [w, -arm], [w, -w], [arm, -w]], float)
    return make_ccw(V)


def flower(M, R=1.0, amp=0.28, k=5, phase=0.0):
    th = np.linspace(0, 2 * np.pi, M, endpoint=False) + phase
    r = R * (1 + amp * np.cos(k * th))
    return make_ccw(np.stack([r * np.cos(th), r * np.sin(th)], 1))


def L_shape():
    return make_ccw(np.array([[0., 0.], [2, 0], [2, 1], [1, 1], [1, 2], [0, 2]]))


def rect(x0, y0, x1, y1):
    return make_ccw(np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], float))


def place(V0, x, y, th):
    c, s = np.cos(th), np.sin(th)
    return V0 @ np.array([[c, s], [-s, c]]) + np.array([x, y])
