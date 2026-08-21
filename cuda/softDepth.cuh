// Soft penetration depth -- device helpers. CUDA C++ port of the MATH in ../softDepth.py, which
// implements ../notes/softDepth-2.pdf. Equation numbers refer to that document.
//
//   h_eps(x) = -eps log sum_i exp(-ell_i(x)/eps),      ell_i(x) = c_i - n_i . x      (7)
//   grad h   = -nbar,   nbar = sum_i w_i n_i                                         (10)
//   E        = int_{dA} phi(h_eps^B(x)) dl(x),  phi = (2/5) k [h]_+^(5/2)            (20), (37)
//
// THE SHIFT IS NOT OPTIONAL (48). Writing m = min_i ell_i, every exponent is <= 0 and the sum lies in
// [1, N]. The unshifted form overflows as soon as ell_i/eps exceeds a few hundred, which at
// eps ~ 1e-3 in units of the particle size is immediate.
//
// Three structural facts from the note make this port simple, and each is load-bearing:
//
//   1. h_eps is CONCAVE -- minus eps times a log-sum-exp of affine functions. So along an edge the
//      contact set {h >= 0} is a single interval: at most two roots, bracketed by the peak. No
//      candidate enumeration.
//   2. ell_i restricted to an edge is AFFINE in the parameter, so an envelope switch is the crossing
//      of two lines and costs one divide -- no root finding, and no probe count.
//   3. phi(h) = 0 at a crossing, so the Leibniz boundary terms vanish and the integration limits are
//      NOT differentiated. That is what lets the quadrature panels be consumed as they are produced
//      instead of being stored.
//
// Built into libplummer.so; called from ../cudaOverlap.py via ctypes. double/double2, sm_75.
#pragma once
#include <math.h>

// UNVERIFIED(Cam)
namespace softDepth {

// One outward half-plane of a loop, ell_i(x) = c - n . x, with n = J t and c = n . v_i (eqs 1, 2).
// Staged in SHARED memory once per block so every edge thread reads B's geometry from there rather
// than re-fetching vertices from global memory at each of the ~120 depth evaluations it performs.
struct Plane {
    double nx, ny, c;
};

// Restriction of ell_i to the edge p0 -> p0 + d: affine, ell_i(t) = a_i + m_i t. Kept as a function
// rather than a cached array because a_i, m_i for 64 planes would be 128 registers per thread; two
// fused multiply-adds are cheaper than the spill.
__device__ __forceinline__ double ellAt(const Plane& p, double2 base, double2 dir, double t) {
    return p.c - (p.nx * (base.x + t * dir.x) + p.ny * (base.y + t * dir.y));
}

__device__ __forceinline__ double ellSlope(const Plane& p, double2 dir) {
    return -(p.nx * dir.x + p.ny * dir.y);
}

// h_eps and nbar at one point, in the shifted form (48). Two passes over the planes: the first finds
// the shift, the second accumulates the normalizer and the weighted normal together. grad h = -nbar,
// so the derivative along an edge is dh/dt = -nbar . dir.
__device__ __forceinline__ void depthAndNormal(const Plane* planes, int nB, double2 x, double eps,
                                               double* h, double* nbarX, double* nbarY) {
    double lowest = 1.0e300;
    for (int i = 0; i < nB; ++i) {
        double ell = planes[i].c - (planes[i].nx * x.x + planes[i].ny * x.y);
        if (ell < lowest) lowest = ell;
    }
    double total = 0.0, sumX = 0.0, sumY = 0.0;
    double invEps = 1.0 / eps;
    for (int i = 0; i < nB; ++i) {
        double ell = planes[i].c - (planes[i].nx * x.x + planes[i].ny * x.y);
        double weight = exp(-(ell - lowest) * invEps);
        total += weight;
        sumX += weight * planes[i].nx;
        sumY += weight * planes[i].ny;
    }
    double inv = 1.0 / total;
    *h = lowest - eps * log(total);
    *nbarX = sumX * inv;
    *nbarY = sumY * inv;
}

// h and dh/dt at parameter t along the edge base -> base + dir. The root finding needs nothing else.
__device__ __forceinline__ void depthAlongEdge(const Plane* planes, int nB, double2 base, double2 dir,
                                               double t, double eps, double* h, double* slope) {
    double2 x = make_double2(base.x + t * dir.x, base.y + t * dir.y);
    double nbarX, nbarY;
    depthAndNormal(planes, nB, x, eps, h, &nbarX, &nbarY);
    *slope = -(nbarX * dir.x + nbarY * dir.y);
}

// Scalar contact law (23), purely repulsive branch (20): E = (2/5) k [h]_+^(5/2), phi' = k [h]_+^(3/2).
// Adhesion is deliberately absent from the device tier for now -- the numpy path carries it and this
// kernel is the repulsive workhorse.
__device__ __forceinline__ void contactLaw(double h, double stiffness, double* energy, double* first) {
    if (h <= 0.0) { *energy = 0.0; *first = 0.0; return; }
    double root = sqrt(h);
    double h32 = h * root;
    *energy = 0.4 * stiffness * h32 * h;
    *first = stiffness * h32;
}

// The concave peak of h along the edge, by bisection on the SLOPE. h is concave so dh/dt is
// decreasing, which makes this an honest bracket; the peak only has to separate the two roots, so it
// is deliberately given fewer steps than the roots themselves.
__device__ __forceinline__ double peakOf(const Plane* planes, int nB, double2 base, double2 dir,
                                         double eps, double slopeLo, double slopeHi, int steps) {
    if (slopeLo <= 0.0) return 0.0;
    if (slopeHi >= 0.0) return 1.0;
    double lo = 0.0, hi = 1.0, h, slope;
    for (int s = 0; s < steps; ++s) {
        double mid = 0.5 * (lo + hi);
        depthAlongEdge(planes, nB, base, dir, mid, eps, &h, &slope);
        if (slope > 0.0) lo = mid; else hi = mid;
    }
    return 0.5 * (lo + hi);
}

// Root of h inside a bracket, by Newton with a bisection safeguard: the Newton step is taken only when
// it lands inside the live bracket, so the iteration inherits bisection's guarantee at Newton's rate.
//
// The accuracy demanded is very weak. Misplacing the crossing by d changes the energy by the integral
// over a sliver where h is itself O(d), i.e. by O(d^(7/2)) -- the same vanishing that kills the
// Leibniz boundary term. This is headroom, not a knife edge.
__device__ __forceinline__ double bracketedRoot(const Plane* planes, int nB, double2 base, double2 dir,
                                                double eps, double lo, double hi, bool positiveAtLo,
                                                int steps) {
    double t = 0.5 * (lo + hi);
    for (int s = 0; s < steps; ++s) {
        double h, slope;
        depthAlongEdge(planes, nB, base, dir, t, eps, &h, &slope);
        bool atLoSide = (h > 0.0) == positiveAtLo;
        if (atLoSide) lo = t; else hi = t;
        double next = (fabs(slope) > 1.0e-300) ? t - h / slope : lo - 1.0;
        t = (next > lo && next < hi) ? next : 0.5 * (lo + hi);
    }
    return t;
}

// The next envelope breakpoint strictly after t, clamped to tEnd -- the parameter at which the softmin's
// ACTIVE half-plane changes and h turns over on the scale of eps.
//
// EXACT, and cheaper than the numpy version. Each ell_i is affine along the edge, so a switch is the
// crossing of two lines. The envelope is concave, so only a plane falling FASTER than the current one
// can take over: with p the current argmin, the candidates are j with m_j < m_p, crossing at
// t* = (a_p - a_j) / (m_j - m_p). numpy locates these by probing max(16, 2 nB) points and solving
// where the argmin changes, which misses any segment shorter than the probe spacing; this does not.
__device__ __forceinline__ double nextEnvelopeCut(const Plane* planes, int nB, double2 base, double2 dir,
                                                  double t, double tEnd) {
    int active = 0;
    double lowest = 1.0e300;
    for (int i = 0; i < nB; ++i) {
        double ell = ellAt(planes[i], base, dir, t);
        if (ell < lowest) { lowest = ell; active = i; }
    }
    double slopeActive = ellSlope(planes[active], dir);
    double interceptActive = ellAt(planes[active], base, dir, 0.0);
    double best = tEnd;
    for (int j = 0; j < nB; ++j) {
        if (j == active) continue;
        double slope = ellSlope(planes[j], dir);
        double gap = slope - slopeActive;
        if (gap >= 0.0) continue;
        double cut = (interceptActive - ellAt(planes[j], base, dir, 0.0)) / gap;
        if (cut > t && cut < best) best = cut;
    }
    return best;
}

}  // namespace softDepth
