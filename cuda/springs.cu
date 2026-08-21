// eqSoftBody shape springs on the GPU -- the port of softBody.eqSoftBodyEnergyForce (relative form).
//
//   E = (1/2) kEdge sum_k (l_k/l0 - 1)^2  +  (1/2) kArea sum_p (A_p/A0 - 1)^2
//
// Cheap in absolute terms (~0.2 ms in numpy), so this exists not for its own speed but so a force
// evaluation never has to leave the GPU: with the overlap, self-repulsion and springs all resident,
// the whole term set is one upload and one download instead of three round trips.
//
// Two kernels, the same separation the other drivers use:
//   1. areaKernel  -- one thread per VERTEX; accumulates each polygon's shoelace area (a reduction
//                     that must complete before any area force can be formed).
//   2. springKernel -- one thread per VERTEX; the edge springs of the two edges meeting at the
//                     vertex, plus the area-spring term using the completed area.
//
// Built into libplummer.so; called from ../cudaOverlap.py via ctypes.
#include <math.h>

#define SPRING_BLOCK 128

// Kernel 1 -- shoelace area per polygon. Thread per vertex, atomic into areaBuf[shape].
__global__ void areaKernel(const double2* pos, const int* shapeId, const int* nextIdx, int numVert,
                           double* areaBuf) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= numVert) return;
    double2 p = pos[v], q = pos[nextIdx[v]];
    atomicAdd(&areaBuf[shapeId[v]], 0.5 * (p.x * q.y - q.x * p.y));
}

// Kernel 2 -- energy + force. Thread per vertex.
// Edge spring: with stretch_k = w (l_k - l0), w = 1/l0^2, the force on vertex k is
//   kEdge * (stretch_k uhat_k - stretch_{k-1} uhat_{k-1}).
// Area spring: force = -kArea * w_A (A - A0) * dA/dr, dA/dr_k = 0.5 (y_next - y_prev, x_prev - x_next).
__global__ void springKernel(const double2* pos, const int* shapeId, const int* nextIdx,
                             const int* prevIdx, int numVert,
                             const double* l0, const double* A0, const double* areaBuf,
                             double kEdge, double kArea, double* force, double* energyOut) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= numVert) return;
    int p = shapeId[v], nv = nextIdx[v], pv = prevIdx[v];
    // l0 is PER EDGE (indexed by the vertex the edge leaves), not per polygon: each edge carries its
    // own target, so a polygon need not be equilateral.
    double rest = l0[v], invRest2 = 1.0 / (rest * rest);
    double restPrev = l0[pv], invPrev2 = 1.0 / (restPrev * restPrev);

    double2 c = pos[v], n = pos[nv], q = pos[pv];
    double ex = n.x - c.x, ey = n.y - c.y;
    double len = sqrt(ex * ex + ey * ey);
    double safe = (len > 1e-15) ? len : 1.0;
    double stretch = invRest2 * (len - rest);

    double px = c.x - q.x, py = c.y - q.y;
    double plen = sqrt(px * px + py * py);
    double psafe = (plen > 1e-15) ? plen : 1.0;
    double pstretch = invPrev2 * (plen - restPrev);

    double fx = kEdge * (stretch * ex / safe - pstretch * px / psafe);
    double fy = kEdge * (stretch * ey / safe - pstretch * py / psafe);

    // Each vertex owns one edge, so summing (l_k - l0)^2 over vertices counts every edge exactly once.
    double e = 0.5 * kEdge * invRest2 * (len - rest) * (len - rest);

    double target = A0[p], invA2 = 1.0 / (target * target);
    double residual = invA2 * (areaBuf[p] - target);
    double gAx = 0.5 * (n.y - q.y), gAy = 0.5 * (q.x - n.x);
    fx -= kArea * residual * gAx;
    fy -= kArea * residual * gAy;

    force[2 * v] += fx;
    force[2 * v + 1] += fy;
    atomicAdd(energyOut, e);
}

// The area energy is per POLYGON, so it is emitted by a separate thread-per-polygon kernel rather
// than being divided among the vertices.
__global__ void areaEnergyKernel(int numPoly, const double* A0, const double* areaBuf, double kArea,
                                 double* energyOut) {
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= numPoly) return;
    double target = A0[p], diff = areaBuf[p] - target;
    atomicAdd(energyOut, 0.5 * kArea * diff * diff / (target * target));
}

// Device-side entry point reused by the fused driver: assumes buffers are already resident and
// ``force`` / ``energyOut`` already zeroed (or holding another term's contribution).
void springsLaunch(const double2* pos, const int* shapeId, const int* nextIdx,
                   const int* prevIdx, int numVert, int numPoly,
                   const double* l0, const double* A0, double* areaBuf,
                   double kEdge, double kArea, double* force, double* energyOut) {
    cudaMemset(areaBuf, 0, numPoly * sizeof(double));
    int vBlocks = (numVert + SPRING_BLOCK - 1) / SPRING_BLOCK;
    int pBlocks = (numPoly + SPRING_BLOCK - 1) / SPRING_BLOCK;
    areaKernel<<<vBlocks, SPRING_BLOCK>>>(pos, shapeId, nextIdx, numVert, areaBuf);
    springKernel<<<vBlocks, SPRING_BLOCK>>>(pos, shapeId, nextIdx, prevIdx, numVert, l0, A0, areaBuf,
                                            kEdge, kArea, force, energyOut);
    areaEnergyKernel<<<pBlocks, SPRING_BLOCK>>>(numPoly, A0, areaBuf, kArea, energyOut);
}

// ---- Host-array C API (standalone use / validation). ----
static double2* g_spPos = nullptr; static int* g_spShape = nullptr; static int* g_spNext = nullptr;
static int* g_spPrev = nullptr; static double* g_spL0 = nullptr; static double* g_spA0 = nullptr;
static double* g_spArea = nullptr; static double* g_spForce = nullptr; static double* g_spEnergy = nullptr;
static int g_spNumVert = 0, g_spNumPoly = 0;

extern "C" void springsCuda(const double* positions, int numVert,
                            const int* shapeId, const int* nextIdx, const int* prevIdx, int numPoly,
                            const double* l0, const double* A0, double kEdge, double kArea,
                            double* energyOut, double* forceOut) {
    if (numVert != g_spNumVert || numPoly != g_spNumPoly) {
        if (g_spPos) { cudaFree(g_spPos); cudaFree(g_spShape); cudaFree(g_spNext); cudaFree(g_spPrev);
                       cudaFree(g_spL0); cudaFree(g_spA0); cudaFree(g_spArea); cudaFree(g_spForce);
                       cudaFree(g_spEnergy); }
        cudaMalloc(&g_spPos, numVert * sizeof(double2));
        cudaMalloc(&g_spShape, numVert * sizeof(int));
        cudaMalloc(&g_spNext, numVert * sizeof(int));
        cudaMalloc(&g_spPrev, numVert * sizeof(int));
        cudaMalloc(&g_spL0, numVert * sizeof(double));
        cudaMalloc(&g_spA0, numPoly * sizeof(double));
        cudaMalloc(&g_spArea, numPoly * sizeof(double));
        cudaMalloc(&g_spForce, 2 * numVert * sizeof(double));
        cudaMalloc(&g_spEnergy, sizeof(double));
        g_spNumVert = numVert; g_spNumPoly = numPoly;
    }
    cudaMemcpy(g_spPos, positions, numVert * sizeof(double2), cudaMemcpyHostToDevice);
    cudaMemcpy(g_spShape, shapeId, numVert * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(g_spNext, nextIdx, numVert * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(g_spPrev, prevIdx, numVert * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(g_spL0, l0, numVert * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(g_spA0, A0, numPoly * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemset(g_spForce, 0, 2 * numVert * sizeof(double));
    cudaMemset(g_spEnergy, 0, sizeof(double));
    springsLaunch(g_spPos, g_spShape, g_spNext, g_spPrev, numVert, numPoly, g_spL0, g_spA0,
                  g_spArea, kEdge, kArea, g_spForce, g_spEnergy);
    cudaMemcpy(energyOut, g_spEnergy, sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(forceOut, g_spForce, 2 * numVert * sizeof(double), cudaMemcpyDeviceToHost);
}
