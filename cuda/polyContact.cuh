// Polygon-polygon contact, device math. CUDA C++ port of ../polyContact.py, which is itself the
// vectorized port of the handoff reference in ../notes/polygonContact.
//
//   E = 1/2 sum over ordered pairs (P,Q) of  int_{dP cap Q} (k/3) d_Q(x)^3 dl(x)
//
// with d_Q the EXACT distance to the boundary. Closed form throughout; no quadrature anywhere.
//
// CONVENTIONS ARE FIXED AND MUST NOT BE RE-DERIVED. Four sign errors during the reference's
// development all came from re-deriving the outward normal.
//
//     CCW vertices, signed area > 0
//     edge j:            g_j = V[j+1] - V[j],  tau_j = g_j/|g_j|
//     OUTWARD normal:    n_j = (tau_j.y, -tau_j.x)
//     signed line dist:  ell_j(x) = n_j . (V[j] - x)      > 0 inside
//     perpendicular foot on line j:  q = x + ell_j n_j     NOT x - ell_j n_j
//
// NOTHING IS STORED PER THREAD. The host walks sorted crossings and envelope breakpoints into arrays;
// here both are walked by repeated minimum-search over the candidates, so a thread needs no local
// arrays at all and never spills. Cost is O(M^2) per edge in the worst case, which at M <= 64 is a few
// thousand flops and far cheaper than the spill would be.
//
// THE PREFILTER IS DELIBERATELY ABSENT. The reference prunes candidates before choosing the envelope
// winner, using a bound computed on each candidate's window clipped to the interval; that understates
// the true maximum of a candidate invalid over part of it, so genuine winners are pruned and feature
// switches are missed -- measured on the host, 5 across 107 spans of one packing, a 0.7% energy error.
// See ../TODO.md. The window test below is exact and is the only restriction needed.
#pragma once
#include <math.h>

// UNVERIFIED(Cam)
namespace polyContact {

// One edge's frame, staged ONCE in shared memory. Recomputing it inside candidateQuadratic -- which
// runs ~2 x 2M times per breakpoint -- costs a normalize and a sqrt each time and dominated the kernel:
// 37 ms of a 39 ms evaluation at N=32, n=32.
struct LoopFrame {
    double2 base;
    double2 tangent;
    double2 normal;
    double length;
};

__device__ __forceinline__ double2 vertexAt(const double2* verts, int count, int index, double2 shift) {
    double2 v = verts[index % count];
    return make_double2(v.x + shift.x, v.y + shift.y);
}

// Outward frame of edge j. Orientation is NOT normalized here: the driver guarantees CCW input, and
// re-deriving the normal is exactly the mistake the convention block exists to prevent.
__device__ __forceinline__ void edgeOf(const double2* verts, int count, int j, double2 shift,
                                       double2* base, double2* tangent, double2* normal,
                                       double* length) {
    double2 a = vertexAt(verts, count, j, shift);
    double2 b = vertexAt(verts, count, j + 1, shift);
    double ex = b.x - a.x, ey = b.y - a.y;
    double len = sqrt(ex * ex + ey * ey);
    *base = a;
    *tangent = make_double2(ex / len, ey / len);
    *normal = make_double2(ey / len, -ex / len);
    *length = len;
}

__device__ __forceinline__ bool isReflexAt(const LoopFrame* frames, int count, int j) {
    double2 previous = frames[(j + count - 1) % count].base;
    double2 here = frames[j].base;
    double2 following = frames[(j + 1) % count].base;
    double backX = here.x - previous.x, backY = here.y - previous.y;
    double forwardX = following.x - here.x, forwardY = following.y - here.y;
    return (backX * forwardY - backY * forwardX) < 0.0;
}

// Nearest feature of a loop to a point, plus membership, in ONE pass.
//   kind 0 = edge, 1 = vertex.
//   nearest is edge j    -> inside iff n_j . (V[j] - x) > 0
//   nearest is vertex j  -> inside iff V[j] is REFLEX      <- the whole content of the rule
// Candidates are visited INTERLEAVED (E_0, V_0, E_1, V_1, ...) with a strict <, matching the host's
// tie-break exactly. Ties are measure-zero in the distance, but the INSIDE flag can differ between a
// tied edge and a tied vertex, so the ordering is part of what is being ported.
__device__ __forceinline__ void nearestFeature(double2 x, const LoopFrame* frames, int count,
                                               int* kind, int* index,
                                               double* distance, bool* inside) {
    double best = 1.0e300;
    int bestKind = 0, bestIndex = 0;
    bool bestInside = false;
    for (int j = 0; j < count; ++j) {
        LoopFrame f = frames[j];
        double toX = x.x - f.base.x, toY = x.y - f.base.y;
        double foot = f.tangent.x * toX + f.tangent.y * toY;
        if (foot >= -1e-15 && foot <= f.length + 1e-15) {
            double signedLine = -(f.normal.x * toX + f.normal.y * toY);
            double d = fabs(signedLine);
            if (d < best) { best = d; bestKind = 0; bestIndex = j; bestInside = signedLine > 0.0; }
        }
        double d = sqrt(toX * toX + toY * toY);
        if (d < best) { best = d; bestKind = 1; bestIndex = j; bestInside = isReflexAt(frames, count, j); }
    }
    *kind = bestKind; *index = bestIndex; *distance = best; *inside = bestInside;
}

// Squared distance from base + t*dir to one candidate feature, as a quadratic a t^2 + b t + c, with
// the validity window over which that candidate's perpendicular foot lies on its own segment.
__device__ __forceinline__ void candidateQuadratic(const LoopFrame* frames, int count, int slot,
                                                   double2 base, double2 dir,
                                                   double* a, double* b, double* c,
                                                   double* windowLow, double* windowHigh) {
    int j = slot >> 1;
    LoopFrame f = frames[j];
    if ((slot & 1) == 0) {
        double2 vertexBase = f.base, tangent = f.tangent, normal = f.normal;
        double length = f.length;
        double offX = vertexBase.x - base.x, offY = vertexBase.y - base.y;
        double alpha = normal.x * offX + normal.y * offY;
        double slope = normal.x * dir.x + normal.y * dir.y;
        double footStart = -(tangent.x * offX + tangent.y * offY);
        double footRate = tangent.x * dir.x + tangent.y * dir.y;
        *a = slope * slope; *b = -2.0 * alpha * slope; *c = alpha * alpha;
        if (fabs(footRate) < 1e-14) {
            bool always = (footStart >= 0.0) && (footStart <= length);
            *windowLow = always ? -1.0e300 : 1.0;
            *windowHigh = always ? 1.0e300 : 0.0;
        } else {
            double low = (0.0 - footStart) / footRate;
            double high = (length - footStart) / footRate;
            *windowLow = fmin(low, high);
            *windowHigh = fmax(low, high);
        }
    } else {
        double2 v = f.base;
        double toX = base.x - v.x, toY = base.y - v.y;
        *a = dir.x * dir.x + dir.y * dir.y;
        *b = 2.0 * (toX * dir.x + toY * dir.y);
        *c = toX * toX + toY * toY;
        *windowLow = -1.0e300; *windowHigh = 1.0e300;
    }
}

// The next envelope breakpoint strictly after `at`, clamped to `upper`. Walks every candidate against
// the current winner; no prefilter (see the header note).
__device__ __forceinline__ double nextBreakpoint(const LoopFrame* frames, int count,
                                                 double2 base, double2 dir, double at, double upper) {
    int slots = 2 * count;
    double probe = at + 1e-13;
    int winner = 0;
    double bestValue = 1.0e300;
    for (int slot = 0; slot < slots; ++slot) {
        double a, b, c, lo, hi;
        candidateQuadratic(frames, count, slot, base, dir, &a, &b, &c, &lo, &hi);
        if (probe < lo - 1e-15 || probe > hi + 1e-15) continue;
        double value = a * probe * probe + b * probe + c;
        if (value < bestValue) { bestValue = value; winner = slot; }
    }
    double aw, bw, cw, lw, hw;
    candidateQuadratic(frames, count, winner, base, dir, &aw, &bw, &cw, &lw, &hw);

    double best = upper;
    for (int slot = 0; slot < slots; ++slot) {
        double a, b, c, lo, hi;
        candidateQuadratic(frames, count, slot, base, dir, &a, &b, &c, &lo, &hi);
        double dA = aw - a, dB = bw - b, dC = cw - c;
        double roots[4];
        int found = 0;
        if (fabs(dA) < 1e-14) {
            if (fabs(dB) > 1e-14) roots[found++] = -dC / dB;
        } else {
            double disc = dB * dB - 4.0 * dA * dC;
            if (disc >= 0.0) {
                double root = sqrt(disc);
                roots[found++] = (-dB + root) / (2.0 * dA);
                roots[found++] = (-dB - root) / (2.0 * dA);
            }
        }
        roots[found++] = lo;
        roots[found++] = hi;
        for (int r = 0; r < found; ++r)
            if (roots[r] > at + 1e-12 && roots[r] < best) best = roots[r];
    }
    return best;
}

// int (w^2 + r^2)^(3/2) dw -- the vertex-nearest branch's exact antiderivative.
__device__ __forceinline__ double vertexAntiderivative(double w, double r) {
    double s = sqrt(w * w + r * r);
    return w * (2.0 * w * w + 5.0 * r * r) * s / 8.0 + 3.0 * r * r * r * r * asinh(w / r) / 8.0;
}

}  // namespace polyContact
