// Sharp (unmollified) overlap -- device helpers (Track B).
//
// Faithful CUDA C++ port of the sharp-tier MATH in ../energies.py (the chord h(),
// the edge-edge intersection test + orientation, and the hat-function gradient deposit),
// factored as small device functions reused by the separated-sum kernels in sharpKernels.cu:
// one kernel parallel OVER INTERSECTIONS (U_ex), one parallel OVER VERTICES (U_int).
//
// The math follows energies.py (verified); STABLE is a STRUCTURAL reference only (its
// overlap math is the old/buggy model). double/double2, sm_75.
#pragma once
#include <math.h>
#include <stdint.h>

// double2 arithmetic (self-contained header; global scope so double2 ADL finds them).
__device__ __forceinline__ double2 operator-(double2 a, double2 b) { return make_double2(a.x - b.x, a.y - b.y); }
__device__ __forceinline__ double2 operator+(double2 a, double2 b) { return make_double2(a.x + b.x, a.y + b.y); }
__device__ __forceinline__ double2 operator*(double s, double2 a) { return make_double2(s * a.x, s * a.y); }

namespace sharp {

// ---- packed intersection key: (si<<48 | sj<<32 | edgeI<<16 | edgeL), 16 bits each ----
// si = leave-polygon shape, sj = other shape, edgeI = leave edge (local), edgeL = arrival edge on
// the OTHER polygon (local). Fractional coords carried alongside as tu = (sI on si, sJ on sj).
__host__ __device__ __forceinline__ uint64_t packKey(int si, int sj, int edgeI, int edgeJ) {
    return ((uint64_t)(si & 0xFFFF) << 48) | ((uint64_t)(sj & 0xFFFF) << 32)
         | ((uint64_t)(edgeI & 0xFFFF) << 16) | (uint64_t)(edgeJ & 0xFFFF);
}
__host__ __device__ __forceinline__ int keySi(uint64_t k)    { return (int)((k >> 48) & 0xFFFF); }
__host__ __device__ __forceinline__ int keySj(uint64_t k)    { return (int)((k >> 32) & 0xFFFF); }
__host__ __device__ __forceinline__ int keyEdgeI(uint64_t k) { return (int)((k >> 16) & 0xFFFF); }
__host__ __device__ __forceinline__ int keyEdgeJ(uint64_t k) { return (int)(k & 0xFFFF); }

// Chord-triangle area h(P, Q) = 1/2 (P - R) x (Q - R)  (energies._h; notes eq 3.2).
__device__ __forceinline__ double hDevice(double2 P, double2 Q, double2 R) {
    return 0.5 * ((P.x - R.x) * (Q.y - R.y) - (P.y - R.y) * (Q.x - R.x));
}

// Edge-edge intersection of segment [p0, p0+dA] with [q0, q0+dB] (energies.polygonPairIntersections):
// returns true with (sA, sB) in (0,1) at a genuine crossing, false otherwise.
__device__ __forceinline__ bool segCross(double2 p0, double2 dA, double2 q0, double2 dB,
                                         double* sA, double* sB) {
    double denom = dA.x * dB.y - dA.y * dB.x;
    if (fabs(denom) < 1e-14) return false;
    double2 w = q0 - p0;
    double a = (w.x * dB.y - w.y * dB.x) / denom;
    double b = (w.x * dA.y - w.y * dA.x) / denom;
    if (a > 0.0 && a < 1.0 && b > 0.0 && b < 1.0) { *sA = a; *sB = b; return true; }
    return false;
}

// Hat-function deposit of one boundary segment on an edge (energies deposit()): accumulates the
// overlap-area GRADIENT dA/dv (matching energies.overlapGradient; the physical force is its negative).
// (eps e) = (e_y, -e_x); vertex `gEdge` gets ds*(1-sbar)*(eps e), vertex z = gNext gets ds*sbar*(eps e).
// Atomically scattered into the flat grad array (2 doubles/vertex) at the GLOBAL vertex ids.
// ``weight`` is the pair's chain-rule factor dU/da. It defaults to 1, which gives the bare
// d(area)/dv; the normalized-squared contact law passes 4 k a / norm^2 so each pair's segments are
// deposited with its own stiffness.
__device__ __forceinline__ void depositGrad(double* grad, double2 v0, double2 v1,
                                            int gEdge, int gNext, double l0, double lf,
                                            double weight = 1.0) {
    double2 e = v1 - v0;
    double2 epsE = make_double2(e.y, -e.x);
    double sbar = 0.5 * (l0 + lf);
    double ds = weight * (lf - l0);
    atomicAdd(&grad[2 * gEdge],     (1.0 - sbar) * ds * epsE.x);
    atomicAdd(&grad[2 * gEdge + 1], (1.0 - sbar) * ds * epsE.y);
    atomicAdd(&grad[2 * gNext],     sbar * ds * epsE.x);
    atomicAdd(&grad[2 * gNext + 1], sbar * ds * epsE.y);
}

}  // namespace sharp
