// Fixed-boundary (container) confinement on the GPU -- the port of energies.containerEnergyForce.
//
// For each ordinary polygon S against the wall C, the penalised quantity is the area of S lying
// OUTSIDE the wall,
//
//     a_S = area(S) + sign * A_cap(S, C),      U = 2 k sum_S (a_S / norm_S)^2,  norm_S = 2 A_tgt[S]
//
// with no covering-radius switch (the wall spans the whole cell, so a cutoff would disable it
// exactly where it is needed) and ``sign`` read from the wall's winding.
//
// Three kernels, because a_S cannot be formed until every panel of S has landed:
//   1. panelKernel  -- one thread per (shape edge, wall edge). fusedPanelMomentA gives the cap term
//                      and BOTH hat moments from one evaluation, sharing tCoreReal.
//   2. shapeKernel  -- one thread per shape: shoelace area, a_S, the energy, and the chain-rule
//                      weight w_S = 4 k a_S / norm^2.
//   3. depositKernel-- one thread per shape VERTEX: gathers the moments of the two edges meeting
//                      there plus the area gradient, scales by w_S, writes the force.
//
// The wall's OWN gradient is not computed. It is pinned in every use so far, and the host routine
// falls back to numpy for the rare free-wall case rather than carrying a fourth kernel.
#include "plummer.cuh"
#include <math.h>

#define CONTAINER_BLOCK 128
#define INV_FOUR_PI 0.07957747154594766788

__device__ __forceinline__ void containerFrame(double2 a0, double2 ea, double2 b0, double2 eb,
                                               double* P0, double* P1, double* X0, double* X1,
                                               double* LA, double* LB) {
    double lb = sqrt(eb.x * eb.x + eb.y * eb.y);
    double bhx = eb.x / lb, bhy = eb.y / lb;
    double w0x = a0.x - b0.x, w0y = a0.y - b0.y;
    *P0 = w0x * bhx + w0y * bhy;
    *P1 = ea.x * bhx + ea.y * bhy;
    *X0 = w0x * bhy - w0y * bhx;
    *X1 = ea.x * bhy - ea.y * bhx;
    *LA = sqrt(ea.x * ea.x + ea.y * ea.y);
    *LB = lb;
}

// Kernel 1 -- one thread per (shape edge, wall edge).
__global__ void containerPanelKernel(const double2* pos, const int* nextIdx, const int* shapeId,
                                     int numShapeEdges, int wallStart, int wallCount,
                                     double sigma, double* capBuf, double* momBuf) {
    long t = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= (long)numShapeEdges * wallCount) return;
    int e = (int)(t / wallCount), c = (int)(t % wallCount);

    double2 a0 = pos[e], a1 = pos[nextIdx[e]];
    double2 ea = make_double2(a1.x - a0.x, a1.y - a0.y);
    double2 b0 = pos[wallStart + c], b1 = pos[wallStart + (c + 1) % wallCount];
    double2 eb = make_double2(b1.x - b0.x, b1.y - b0.y);

    double P0, P1, X0, X1, LA, LB, I, W0, W1;
    containerFrame(a0, ea, b0, eb, &P0, &P1, &X0, &X1, &LA, &LB);
    plummer::fusedPanelMomentA(P0, P1, X0, X1, LA, LB, sigma, &I, &W0, &W1);

    // Outward unit normals of CCW loops: (tau.y, -tau.x), matching energies._edges.
    double nax = ea.y / LA, nay = -ea.x / LA;
    double nbx = eb.y / LB, nby = -eb.x / LB;
    atomicAdd(&capBuf[shapeId[e]], -(nax * nbx + nay * nby) * LA * LB * I * INV_FOUR_PI);
    atomicAdd(&momBuf[2 * e], W0);
    atomicAdd(&momBuf[2 * e + 1], W1);
}

// Kernel 2 -- one thread per shape: area, a_S, energy, chain-rule weight.
__global__ void containerShapeKernel(const double2* pos, const int* starts, int numShapes,
                                     const double* Atgt, const double* capBuf, double sign,
                                     double kContainer, double* weightBuf, double* energyOut) {
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= numShapes) return;
    int base = starts[p], n = starts[p + 1] - base;
    double area = 0.0;
    for (int k = 0; k < n; ++k) {
        double2 v = pos[base + k], vn = pos[base + (k + 1) % n];
        area += v.x * vn.y - vn.x * v.y;
    }
    area *= 0.5;
    double a = area + sign * capBuf[p];
    double norm = 2.0 * Atgt[p];
    atomicAdd(energyOut, 2.0 * kContainer * (a / norm) * (a / norm));
    weightBuf[p] = 4.0 * kContainer * a / (norm * norm);
}

// Kernel 3 -- one thread per shape VERTEX. Each vertex owns edge v and also receives the far-end
// moment of edge prev[v], plus its shoelace area gradient.
__global__ void containerDepositKernel(const double2* pos, const int* nextIdx, const int* prevIdx,
                                       const int* shapeId, int numShapeEdges, const double* momBuf,
                                       const double* weightBuf, double sign, double* force) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= numShapeEdges) return;
    int nv = nextIdx[v], pv = prevIdx[v];

    double2 c = pos[v], n = pos[nv], q = pos[pv];
    double ex = n.x - c.x, ey = n.y - c.y;
    double length = sqrt(ex * ex + ey * ey);
    double nax = ey / length, nay = -ex / length;
    double px = c.x - q.x, py = c.y - q.y;
    double plen = sqrt(px * px + py * py);
    double pax = py / plen, pay = -px / plen;

    double m0 = momBuf[2 * v], m1 = momBuf[2 * v + 1];
    double pm1 = momBuf[2 * pv + 1];
    // dA_cap/dv: (L (W0 - W1)) n_hat from this vertex's own edge, plus (L W1) n_hat from the one
    // arriving at it.
    double gx = sign * (length * (m0 - m1) * nax + plen * pm1 * pax);
    double gy = sign * (length * (m0 - m1) * nay + plen * pm1 * pay);
    // Shoelace area gradient: 0.5 (y_next - y_prev, x_prev - x_next).
    gx += 0.5 * (n.y - q.y);
    gy += 0.5 * (q.x - n.x);

    double w = weightBuf[shapeId[v]];
    force[2 * v] = -w * gx;
    force[2 * v + 1] = -w * gy;
}

// ---- C API. Returns 0 on success. ----
static double2* g_cPos = nullptr; static int *g_cStarts = nullptr, *g_cNext = nullptr;
static int *g_cPrev = nullptr, *g_cShape = nullptr;
static double *g_cAtgt = nullptr, *g_cCap = nullptr, *g_cMom = nullptr;
static double *g_cWeight = nullptr, *g_cForce = nullptr, *g_cEnergy = nullptr;
static int g_cNumVert = 0, g_cNumPoly = 0;

extern "C" int containerEnergyForceCuda(const double* positions, int numVert,
                                        const int* startIndices, int numPoly, int containerIndex,
                                        const int* nextIdx, const int* prevIdx, const int* shapeId,
                                        const double* Atgt, double sigma, double kContainer,
                                        double sign, double* energyOut, double* forceOut) {
    int numShapes = containerIndex;
    int wallStart = startIndices[containerIndex];
    int wallCount = startIndices[containerIndex + 1] - wallStart;
    int numShapeEdges = wallStart;                 // shape vertices occupy [0, wallStart)
    if (numShapes < 1 || wallCount < 3) return 1;

    if (numVert != g_cNumVert || numPoly != g_cNumPoly) {
        if (g_cPos) { cudaFree(g_cPos); cudaFree(g_cStarts); cudaFree(g_cNext); cudaFree(g_cPrev);
                      cudaFree(g_cShape); cudaFree(g_cAtgt); cudaFree(g_cCap); cudaFree(g_cMom);
                      cudaFree(g_cWeight); cudaFree(g_cForce); cudaFree(g_cEnergy); }
        cudaMalloc(&g_cPos, numVert * sizeof(double2));
        cudaMalloc(&g_cStarts, (numPoly + 1) * sizeof(int));
        cudaMalloc(&g_cNext, numVert * sizeof(int));
        cudaMalloc(&g_cPrev, numVert * sizeof(int));
        cudaMalloc(&g_cShape, numVert * sizeof(int));
        cudaMalloc(&g_cAtgt, numPoly * sizeof(double));
        cudaMalloc(&g_cCap, numPoly * sizeof(double));
        cudaMalloc(&g_cMom, 2 * numVert * sizeof(double));
        cudaMalloc(&g_cWeight, numPoly * sizeof(double));
        cudaMalloc(&g_cForce, 2 * numVert * sizeof(double));
        cudaMalloc(&g_cEnergy, sizeof(double));
        g_cNumVert = numVert; g_cNumPoly = numPoly;
    }
    cudaMemcpy(g_cPos, positions, numVert * sizeof(double2), cudaMemcpyHostToDevice);
    cudaMemcpy(g_cStarts, startIndices, (numPoly + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(g_cNext, nextIdx, numVert * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(g_cPrev, prevIdx, numVert * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(g_cShape, shapeId, numVert * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(g_cAtgt, Atgt, numPoly * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemset(g_cCap, 0, numPoly * sizeof(double));
    cudaMemset(g_cMom, 0, 2 * numVert * sizeof(double));
    cudaMemset(g_cForce, 0, 2 * numVert * sizeof(double));
    cudaMemset(g_cEnergy, 0, sizeof(double));

    long nPanel = (long)numShapeEdges * wallCount;
    containerPanelKernel<<<(int)((nPanel + CONTAINER_BLOCK - 1) / CONTAINER_BLOCK), CONTAINER_BLOCK>>>(
        g_cPos, g_cNext, g_cShape, numShapeEdges, wallStart, wallCount, sigma, g_cCap, g_cMom);
    containerShapeKernel<<<(numShapes + CONTAINER_BLOCK - 1) / CONTAINER_BLOCK, CONTAINER_BLOCK>>>(
        g_cPos, g_cStarts, numShapes, g_cAtgt, g_cCap, sign, kContainer, g_cWeight, g_cEnergy);
    containerDepositKernel<<<(numShapeEdges + CONTAINER_BLOCK - 1) / CONTAINER_BLOCK, CONTAINER_BLOCK>>>(
        g_cPos, g_cNext, g_cPrev, g_cShape, numShapeEdges, g_cMom, g_cWeight, sign, g_cForce);

    cudaMemcpy(energyOut, g_cEnergy, sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(forceOut, g_cForce, 2 * numVert * sizeof(double), cudaMemcpyDeviceToHost);
    return 0;
}
