// Intra-polygon self-repulsion on the GPU -- the port of energies.selfRepulsionEnergyForce.
//
//   U = (1/2) kSelf sum_{non-adjacent edge pairs} int_0^1 int_0^1 exp(-|y(s) - z(t)|^2 / 2 delta^2)
//
// evaluated with a 12-point Gauss-Legendre rule on each parameter, so every edge pair carries a
// 12x12 = 144 node-pair grid. The numpy version materializes the whole (K, 12, 12, 2) tensor, which
// makes it memory-bandwidth-bound and, after the overlap moved to CUDA, the single largest term in a
// force evaluation (~75% at N=64).
//
// Parallel structure: ONE WARP PER EDGE PAIR, each lane striding over the 144 node pairs, then a
// warp-level shuffle reduction. This is the natural form of the separated sum here -- the pairwise
// unit is the NODE PAIR and the assembly is over the four endpoint vertices, but unlike the overlap
// those four vertices are private to the edge pair, so the "vertex kernel" collapses into the warp
// reduction instead of needing a staging buffer. Only 7 scalars survive the reduction (see below),
// and lane 0 issues the handful of global atomics.
//
// Built into libplummer.so alongside plummerDriver.cu; called from ../cudaOverlap.py via ctypes.
#include <math.h>

#define SELFREP_G 12                       // Gauss nodes per parameter
#define SELFREP_NODES (SELFREP_G * SELFREP_G)
#define WARPS_PER_BLOCK 4
#define SELFREP_BLOCK (32 * WARPS_PER_BLOCK)

// Gauss-Legendre on [0, 1]: nodes 0.5*(x+1), weights 0.5*w. Must match numpy leggauss(12) exactly.
__device__ __constant__ double SELFREP_SX[12] = {
    9.21968287664037822e-03, 4.79413718147626011e-02, 1.15048662902847654e-01,
    2.06341022856691259e-01, 3.16084250500909936e-01, 4.37383295744265543e-01,
    5.62616704255734401e-01, 6.83915749499090064e-01, 7.93658977143308686e-01,
    8.84951337097152346e-01, 9.52058628185237454e-01, 9.90780317123359566e-01
};
__device__ __constant__ double SELFREP_SW[12] = {
    2.35876681932560110e-02, 5.34696629976594423e-02, 8.00391642716730550e-02,
    1.01583713361532824e-01, 1.16746268269177320e-01, 1.24573522906701345e-01,
    1.24573522906701345e-01, 1.16746268269177320e-01, 1.01583713361532824e-01,
    8.00391642716730550e-02, 5.34696629976594423e-02, 2.35876681932560110e-02
};

__device__ __forceinline__ double warpSum(double v) {
    for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffffu, v, off);
    return v;
}

// One warp per edge pair. The four endpoint gradients are recovered from three reduced vectors:
//   sumAll = sum dU/dd,  sumI = sum dU/dd * s_i,  sumJ = sum dU/dd * s_j
// since y = (1-s_i) a0 + s_i a1 and z = (1-s_j) b0 + s_j b1 with d = y - z, giving
//   gA1 = sumI,  gA0 = sumAll - sumI,  gB1 = -sumJ,  gB0 = -(sumAll - sumJ).
// Force is -dU/dr, so the deposits are negated.
__global__ void selfRepulsionKernel(const double2* pos, const int* pairs, int numPairs,
                                    double kSelf, double delta, double* force, double* energyOut) {
    int warpId = threadIdx.x >> 5, lane = threadIdx.x & 31;
    int k = blockIdx.x * WARPS_PER_BLOCK + warpId;
    if (k >= numPairs) return;

    int iA0 = pairs[4 * k], iA1 = pairs[4 * k + 1];
    int iB0 = pairs[4 * k + 2], iB1 = pairs[4 * k + 3];
    double2 a0 = pos[iA0], a1 = pos[iA1], b0 = pos[iB0], b1 = pos[iB1];
    double eax = a1.x - a0.x, eay = a1.y - a0.y;
    double ebx = b1.x - b0.x, eby = b1.y - b0.y;

    double invTwoDelta2 = 1.0 / (2.0 * delta * delta), invDelta2 = 1.0 / (delta * delta);
    double energy = 0.0;
    double allx = 0.0, ally = 0.0, ix = 0.0, iy = 0.0, jx = 0.0, jy = 0.0;

    for (int t = lane; t < SELFREP_NODES; t += 32) {
        int i = t / SELFREP_G, j = t % SELFREP_G;
        double si = SELFREP_SX[i], sj = SELFREP_SX[j];
        double dx = (a0.x + si * eax) - (b0.x + sj * ebx);
        double dy = (a0.y + si * eay) - (b0.y + sj * eby);
        double phi = exp(-(dx * dx + dy * dy) * invTwoDelta2);
        double base = 0.5 * kSelf * SELFREP_SW[i] * SELFREP_SW[j] * phi;
        energy += base;
        // dU/dd = base * (-d / delta^2)
        double gx = -base * invDelta2 * dx, gy = -base * invDelta2 * dy;
        allx += gx;      ally += gy;
        ix   += gx * si; iy   += gy * si;
        jx   += gx * sj; jy   += gy * sj;
    }

    energy = warpSum(energy);
    allx = warpSum(allx); ally = warpSum(ally);
    ix = warpSum(ix);     iy = warpSum(iy);
    jx = warpSum(jx);     jy = warpSum(jy);
    if (lane != 0) return;

    atomicAdd(energyOut, energy);
    atomicAdd(&force[2 * iA1],     -ix);
    atomicAdd(&force[2 * iA1 + 1], -iy);
    atomicAdd(&force[2 * iA0],     -(allx - ix));
    atomicAdd(&force[2 * iA0 + 1], -(ally - iy));
    atomicAdd(&force[2 * iB1],     jx);
    atomicAdd(&force[2 * iB1 + 1], jy);
    atomicAdd(&force[2 * iB0],     (allx - jx));
    atomicAdd(&force[2 * iB0 + 1], (ally - jy));
}

// ---- C API. Persistent buffers, sized on the first call and reused (the pair list is topology and
// changes only when the packing does, but it is small, so it is refreshed every call for safety). ----
static double2* g_srPos = nullptr; static int* g_srPairs = nullptr;
static double* g_srForce = nullptr; static double* g_srEnergy = nullptr;
static int g_srNumVert = 0, g_srNumPairs = 0;

extern "C" void selfRepulsionCuda(const double* positions, int numVert,
                                  const int* pairs, int numPairs,
                                  double kSelf, double delta,
                                  double* energyOut, double* forceOut) {
    if (numPairs <= 0) {
        *energyOut = 0.0;
        for (int i = 0; i < 2 * numVert; ++i) forceOut[i] = 0.0;
        return;
    }
    if (numVert != g_srNumVert || numPairs != g_srNumPairs) {
        if (g_srPos) { cudaFree(g_srPos); cudaFree(g_srPairs); cudaFree(g_srForce); cudaFree(g_srEnergy); }
        cudaMalloc(&g_srPos, numVert * sizeof(double2));
        cudaMalloc(&g_srPairs, 4 * (size_t)numPairs * sizeof(int));
        cudaMalloc(&g_srForce, 2 * numVert * sizeof(double));
        cudaMalloc(&g_srEnergy, sizeof(double));
        g_srNumVert = numVert; g_srNumPairs = numPairs;
    }
    cudaMemcpy(g_srPos, positions, numVert * sizeof(double2), cudaMemcpyHostToDevice);
    cudaMemcpy(g_srPairs, pairs, 4 * (size_t)numPairs * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(g_srForce, 0, 2 * numVert * sizeof(double));
    cudaMemset(g_srEnergy, 0, sizeof(double));

    int blocks = (numPairs + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    selfRepulsionKernel<<<blocks, SELFREP_BLOCK>>>(g_srPos, g_srPairs, numPairs, kSelf, delta,
                                                   g_srForce, g_srEnergy);
    cudaMemcpy(energyOut, g_srEnergy, sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(forceOut, g_srForce, 2 * numVert * sizeof(double), cudaMemcpyDeviceToHost);
}
