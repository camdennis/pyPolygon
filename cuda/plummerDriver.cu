// Whole-packing mollified overlap driver (Track A) -- the GPU replacement for
// energies.plummerOverlapExact / _assembleOverlap. Computes the normalized-squared overlap energy
// U = 2 sum_{A<B} (a/norm)^2,  a = sum_images S(d) A_cap,  and its vertex gradient dU/dv.
//
// SEPARATED SUM, mirroring the sharp protocol (sharpKernels.cu). The sharp model separates into one
// kernel OVER INTERSECTIONS and one OVER VERTICES; the mollified model has no intersections, and its
// pairwise unit is instead the EDGE-PAIR PANEL, so it separates the same way:
//
//   1. panelKernel   -- one thread per (pair, image, edgeA, edgeB). The panel is the mollified
//                       analogue of an intersection: it is where the pair interaction actually lives.
//                       Each thread evaluates ONE panel and accumulates its cap contribution and its
//                       two edge moments. This is the parallelism the old code threw away by looping
//                       nA*nB panels serially inside a per-(pair,image) thread -- at N=32, n=10 that
//                       is 9216 threads doing 100 panels each, versus 1.3M threads doing one.
//   2. switchKernel  -- one thread per (pair, image). The switch S and its derivative need the
//                       image's COMPLETED cap sum, so they cannot live in the panel kernel; this is
//                       the join point where a = sum_images S A_cap is formed.
//   3. vertexKernel  -- one thread per (pair, vertex). Converts accumulated edge moments into
//                       per-vertex gradient, applies the normalized-squared chain rule, scatters.
//                       Directly parallel to the sharp interior kernel.
//
// The per-edge-pair math is plummer.cuh (validated). Built as a shared lib; called from
// ../cudaOverlap.py via ctypes. double/double2, sm_75.
#include "plummer.cuh"
#include <math.h>

#define NUM_IMAGES 9
// Largest vertices-per-polygon this build will accept. The per-pair buffers scale as
// numPairs * 4 * maxN, so this caps memory rather than correctness -- the strides themselves are
// measured from the packing. Exceeding it is REPORTED, never silently truncated: a previous build
// hard-coded 12 and quietly dropped the gradient on every vertex past the 24th of a pair while the
// energy stayed correct, which is exactly the kind of failure that survives a test suite. Raise it
// here and rebuild (make -C cuda libplummer.so) if you need larger polygons.
#define PLUMMER_MAXN 64
// Per-pair buffer stride, in doubles: (nA + nB) entries x 2 components, with nA, nB <= maxN, so
// 4 * maxN. maxN is measured from the PACKING at call time and passed in -- it used to be a
// compile-time PLUMMER_MAXN = 12, which silently corrupted every packing with more than 12 vertices
// per polygon: vertexKernel indexes vert = t % (2 * maxN), so vertices past the 24th of a pair were
// never assigned a thread and received NO gradient, while the moment writes ran off the end of the
// pair's slot into its neighbour's. The ENERGY stayed correct (it is accumulated per work item, not
// per vertex), so the failure was invisible unless the gradient was checked directly.
#define PANEL_BLOCK 128                   // threads per surviving (pair, image); must be a power of 2

// C2 smoothstep switch S(d): 1 for d<=rOn, 0 for d>=rOff (energies._switch).
__device__ __forceinline__ void switchDev(double d, double rOn, double rOff, double* S, double* dSdd) {
    double t = (d - rOn) / (rOff - rOn);
    if (t <= 0.0) { *S = 1.0; *dSdd = 0.0; return; }
    if (t >= 1.0) { *S = 0.0; *dSdd = 0.0; return; }
    double t2 = t * t, t3 = t2 * t, t4 = t3 * t, t5 = t4 * t;
    *S = 1.0 - (6.0 * t5 - 15.0 * t4 + 10.0 * t3);
    *dSdd = -(30.0 * t4 - 60.0 * t3 + 30.0 * t2) / (rOff - rOn);
}

// Edge-pair frame (P0,P1,X0,X1,LA,LB) for outer edge (a0,ea) and inner edge (b0,eb) (energies._pairGrid).
__device__ __forceinline__ void frameOf(double2 a0, double2 ea, double2 b0, double2 eb,
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

// Per-(pair, image) geometry: the periodic shift, centroid distance and switch. Recomputed by every
// panel thread of the image -- a few flops each, far cheaper than staging it through memory.
__device__ __forceinline__ bool imageSetup(const double2* cent, const double* rad, int A, int B,
                                           int img, double sigma, double gOn, double gOff,
                                           bool periodic,
                                           double* sx, double* sy, double* d, double* S, double* dSdd,
                                           double* deltax, double* deltay) {
    double rOn = rad[A] + rad[B] + gOn * sigma, rOff = rad[A] + rad[B] + gOff * sigma;
    // FREE SPACE is a single self-image: no minimum-image shift and no 3x3 neighborhood. Summing the
    // periodic offsets with box = None invents copies one unit away and two shapes exactly one unit
    // apart land on top of each other.
    if (periodic) {
        double bsx = -floor(cent[B].x - cent[A].x + 0.5), bsy = -floor(cent[B].y - cent[A].y + 0.5);
        *sx = bsx + (img / 3 - 1); *sy = bsy + (img % 3 - 1);
    } else {
        *sx = 0.0; *sy = 0.0;
    }
    *deltax = cent[B].x + *sx - cent[A].x; *deltay = cent[B].y + *sy - cent[A].y;
    *d = sqrt(*deltax * *deltax + *deltay * *deltay);
    if (*d >= rOff) return false;
    switchDev(*d, rOn, rOff, S, dSdd);
    return true;
}

// One surviving (pair, image): everything the panel and switch kernels need, computed ONCE instead
// of redundantly by all nA*nB panel threads of the image.
struct ImageWork {
    int pairFlat;
    double sx, sy, S, dSdd, d, deltax, deltay;
};

// ---------------------------------------------------------------------------
// Kernel 0 -- CULL. One thread per (pair, image); the survivors are compacted into a dense work
// list. Nearly everything dies here: at N=32 only ~210 of 9216 (pair, image) slots clear the cutoff,
// so the panel launch below shrinks by ~44x. Purely a launch-geometry change -- the surviving set is
// identical, so the arithmetic is untouched.
// ---------------------------------------------------------------------------
__global__ void cullKernel(const double2* cent, const double* rad, int numPoly,
                           double sigma, double gOn, double gOff, int numImages, bool periodic,
                           ImageWork* work, int* workCount) {
    long t = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= (long)numPoly * numPoly * numImages) return;
    int img = (int)(t % numImages); int pairFlat = (int)(t / numImages);
    int A = pairFlat / numPoly, B = pairFlat % numPoly;
    if (B <= A) return;
    ImageWork w;
    if (!imageSetup(cent, rad, A, B, img, sigma, gOn, gOff, periodic, &w.sx, &w.sy, &w.d, &w.S,
                    &w.dSdd, &w.deltax, &w.deltay))
        return;
    w.pairFlat = pairFlat;
    work[atomicAdd(workCount, 1)] = w;
}

// ---------------------------------------------------------------------------
// Kernel 1 -- OVER PANELS. One BLOCK per surviving (pair, image); its threads stride over that
// image's nA*nB panels. Striding rather than one-thread-per-padded-slot means a 10-gon launches 100
// panels, not PLUMMER_MAXN^2 = 144, and ragged vertex counts need no padding at all.
// Accumulates the panel's cap term into capBuf[workIndex] and its two edge moments (A as outer
// frame, then B as outer frame) into momBuf. S is folded in here so moments sum straight over images.
// ---------------------------------------------------------------------------
__global__ void panelKernel(const double2* pos, const int* starts, int numPoly, double sigma,
                            const ImageWork* work, const int* workCount, int stride,
                            double* capBuf, double* momBuf) {
    int w = blockIdx.x;
    if (w >= *workCount) return;
    ImageWork job = work[w];
    int pairFlat = job.pairFlat;
    int A = pairFlat / numPoly, B = pairFlat % numPoly;
    int baseA = starts[A], nA = starts[A + 1] - baseA;
    int baseB = starts[B], nB = starts[B + 1] - baseB;
    double sx = job.sx, sy = job.sy, S = job.S;

    double capLocal = 0.0;
    for (int p = threadIdx.x; p < nA * nB; p += blockDim.x) {
        int ia = p / nB, ib = p % nB;
        double2 a0 = pos[baseA + ia], a1 = pos[baseA + (ia + 1) % nA];
        double2 ea = make_double2(a1.x - a0.x, a1.y - a0.y);
        double2 bp = pos[baseB + ib], bpn = pos[baseB + (ib + 1) % nB];
        double2 b0 = make_double2(bp.x + sx, bp.y + sy);
        double2 eb = make_double2(bpn.x - bp.x, bpn.y - bp.y);

        // A as the outer edge: the fused evaluator shares tCoreReal between the energy term and the
        // gradient moments, so the cap and the A-side moments come from one call.
        double P0, P1, X0, X1, LA, LB, I, W0, W1;
        frameOf(a0, ea, b0, eb, &P0, &P1, &X0, &X1, &LA, &LB);
        plummer::fusedPanelMomentA(P0, P1, X0, X1, LA, LB, sigma, &I, &W0, &W1);

        capLocal += (ea.x * eb.x + ea.y * eb.y) * I;

        double* mom = momBuf + (size_t)pairFlat * stride;
        atomicAdd(&mom[2 * ia],     S * W0);
        atomicAdd(&mom[2 * ia + 1], S * W1);

        // B as the outer edge: a genuinely different frame, so a second evaluation.
        double Q0, Q1, Y0, Y1, LB2, LA2, V0, V1;
        frameOf(b0, eb, a0, ea, &Q0, &Q1, &Y0, &Y1, &LB2, &LA2);
        plummer::wClosedDevice(Q0, Q1, Y0, Y1, LB2, LA2, sigma, &V0, &V1);
        atomicAdd(&mom[2 * (nA + ib)],     S * V0);
        atomicAdd(&mom[2 * (nA + ib) + 1], S * V1);
    }
    // Block-reduce the cap so the image contributes ONE atomic instead of nA*nB of them.
    __shared__ double capShared[PANEL_BLOCK];
    capShared[threadIdx.x] = capLocal;
    __syncthreads();
    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) capShared[threadIdx.x] += capShared[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) atomicAdd(&capBuf[w], capShared[0]);
}

// ---------------------------------------------------------------------------
// Kernel 2 -- OVER (pair, image). The join: the image's cap sum is complete, so form a = sum_images
// S A_cap and deposit the switch-derivative term (which is uniform over each polygon's vertices and
// so has no panel to belong to).
// ---------------------------------------------------------------------------
__global__ void switchKernel(const int* starts, int numPoly, const ImageWork* work,
                             const int* workCount, const double* capBuf, int stride,
                             double* aBuf, double* dsBuf) {
    int w = blockIdx.x * blockDim.x + threadIdx.x;
    if (w >= *workCount) return;
    ImageWork job = work[w];
    int pairFlat = job.pairFlat;
    int A = pairFlat / numPoly, B = pairFlat % numPoly;

    const double INV4PI = 0.07957747154594766788;
    double eP = -INV4PI * capBuf[w];
    atomicAdd(&aBuf[pairFlat], job.S * eP);
    double dSdd = job.dSdd;
    if (dSdd == 0.0) return;

    int nA = starts[A + 1] - starts[A], nB = starts[B + 1] - starts[B];
    double dhatx = job.deltax / job.d, dhaty = job.deltay / job.d;
    double cA = eP * dSdd / nA, cB = eP * dSdd / nB;
    double* ds = dsBuf + (size_t)pairFlat * stride;
    for (int i = 0; i < nA; ++i) {
        atomicAdd(&ds[2 * i], -cA * dhatx); atomicAdd(&ds[2 * i + 1], -cA * dhaty);
    }
    for (int i = 0; i < nB; ++i) {
        atomicAdd(&ds[2 * (nA + i)], cB * dhatx); atomicAdd(&ds[2 * (nA + i) + 1], cB * dhaty);
    }
}

// ---------------------------------------------------------------------------
// Kernel 3 -- OVER VERTICES. One thread per (pair, vertex). A vertex collects the moments of the two
// edges meeting at it (its own edge k and the preceding edge k-1), adds the switch-derivative
// deposit, applies the normalized-squared weight w = 4a/norm^2, and scatters to the global gradient.
// The energy is emitted once per pair, by its vertex 0.
// ---------------------------------------------------------------------------
__global__ void vertexKernel(const double2* pos, const int* starts, const double* Atgt, int numPoly,
                             const double* aBuf, const double* momBuf, const double* dsBuf,
                             int maxN, int stride, double* grad, double* energyOut) {
    long t = (long)blockIdx.x * blockDim.x + threadIdx.x;
    int vert = (int)(t % (2 * maxN)); int pairFlat = (int)(t / (2 * maxN));
    if (pairFlat >= numPoly * numPoly) return;
    int A = pairFlat / numPoly, B = pairFlat % numPoly;
    if (B <= A) return;

    double a = aBuf[pairFlat];
    if (a == 0.0) return;
    int baseA = starts[A], nA = starts[A + 1] - baseA;
    int baseB = starts[B], nB = starts[B + 1] - baseB;
    if (vert >= nA + nB) return;

    double norm = Atgt[A] + Atgt[B];
    if (vert == 0) atomicAdd(energyOut, 2.0 * (a / norm) * (a / norm));
    double w = 4.0 * a / (norm * norm);

    // Resolve which polygon this vertex belongs to, and its local ring.
    int base, n, local, edgeBase;
    if (vert < nA) { base = baseA; n = nA; local = vert; edgeBase = 0; }
    else           { base = baseB; n = nB; local = vert - nA; edgeBase = nA; }

    const double* mom = momBuf + (size_t)pairFlat * stride;
    int prev = (local - 1 + n) % n;

    // rot_k = (y_{k+1} - y_k, -(x_{k+1} - x_k)) for edge k of this polygon's ring.
    double2 pk = pos[base + local], pkn = pos[base + (local + 1) % n];
    double rotx = pkn.y - pk.y, roty = -(pkn.x - pk.x);
    double2 pp = pos[base + prev], ppn = pos[base + (prev + 1) % n];
    double prevRotx = ppn.y - pp.y, prevRoty = -(ppn.x - pp.x);

    double m0 = mom[2 * (edgeBase + local)], m1 = mom[2 * (edgeBase + local) + 1];
    double p1 = mom[2 * (edgeBase + prev) + 1];

    double gx = (m0 - m1) * rotx + p1 * prevRotx;
    double gy = (m0 - m1) * roty + p1 * prevRoty;
    gx += dsBuf[(size_t)pairFlat * stride + 2 * vert];
    gy += dsBuf[(size_t)pairFlat * stride + 2 * vert + 1];

    atomicAdd(&grad[2 * (base + local)],     w * gx);
    atomicAdd(&grad[2 * (base + local) + 1], w * gy);
}

// ---- C API (called from cudaOverlap.py via ctypes). Persistent buffers keep FIRE-loop malloc off
// the hot path; the first call sizes them, later calls at the same size reuse them. ----
static double2* g_pos = nullptr; static int* g_starts = nullptr; static double2* g_cent = nullptr;
static double* g_rad = nullptr; static double* g_Atgt = nullptr; static double* g_grad = nullptr;
static double* g_energy = nullptr; static double* g_aBuf = nullptr; static double* g_capBuf = nullptr;
static double* g_momBuf = nullptr; static double* g_dsBuf = nullptr;
static ImageWork* g_work = nullptr; static int* g_workCount = nullptr;
static int g_numVert = 0, g_numPoly = 0, g_stride = 0;

// Is there a USABLE device right now? Loading the shared library proves nothing -- a driver/library
// version mismatch (e.g. after an unattended driver update without a reboot) leaves every CUDA call
// failing while the .so still loads happily, and the kernels then return zeros. A zero force reads as
// a converged packing, so this must be checked rather than assumed.
extern "C" int cudaDeviceReady() {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) return 0;
    void* probe = nullptr;
    if (cudaMalloc(&probe, 16) != cudaSuccess) return 0;
    cudaFree(probe);
    cudaGetLastError();
    return 1;
}

// Most recent CUDA error, cleared. The Python wrappers call this after every driver invocation so a
// failure raises instead of quietly producing zeros.
extern "C" int cudaLastError() {
    return (int)cudaGetLastError();
}

// Returns 0 on success, or the offending vertex count when a polygon exceeds PLUMMER_MAXN.
extern "C" int plummerOverlapCuda(const double* positions, int numVert,
                                   const int* startIndices, int numPoly,
                                   const double* cent, const double* rad, const double* Atgt,
                                   double sigma, double gOn, double gOff, int periodic,
                                   double* energyOut, double* gradOut) {
    size_t numPairs = (size_t)numPoly * numPoly;
    int maxN = 0;
    for (int p = 0; p < numPoly; ++p) {
        int n = startIndices[p + 1] - startIndices[p];
        if (n > maxN) maxN = n;
    }
    if (maxN > PLUMMER_MAXN) return maxN;
    int stride = 4 * maxN;
    if (numVert != g_numVert || numPoly != g_numPoly || stride != g_stride) {
        if (g_pos) { cudaFree(g_pos); cudaFree(g_starts); cudaFree(g_cent); cudaFree(g_rad);
                     cudaFree(g_Atgt); cudaFree(g_grad); cudaFree(g_energy); cudaFree(g_aBuf);
                     cudaFree(g_capBuf); cudaFree(g_momBuf); cudaFree(g_dsBuf);
                     cudaFree(g_work); cudaFree(g_workCount); }
        cudaMalloc(&g_pos, numVert * sizeof(double2));
        cudaMalloc(&g_starts, (numPoly + 1) * sizeof(int));
        cudaMalloc(&g_cent, numPoly * sizeof(double2));
        cudaMalloc(&g_rad, numPoly * sizeof(double));
        cudaMalloc(&g_Atgt, numPoly * sizeof(double));
        cudaMalloc(&g_grad, 2 * numVert * sizeof(double));
        cudaMalloc(&g_energy, sizeof(double));
        cudaMalloc(&g_aBuf, numPairs * sizeof(double));
        cudaMalloc(&g_capBuf, numPairs * NUM_IMAGES * sizeof(double));
        cudaMalloc(&g_momBuf, numPairs * stride * sizeof(double));
        cudaMalloc(&g_dsBuf, numPairs * stride * sizeof(double));
        cudaMalloc(&g_work, numPairs * NUM_IMAGES * sizeof(ImageWork));
        cudaMalloc(&g_workCount, sizeof(int));
        g_numVert = numVert; g_numPoly = numPoly; g_stride = stride;
    }
    cudaMemcpy(g_pos, positions, numVert * sizeof(double2), cudaMemcpyHostToDevice);
    cudaMemcpy(g_starts, startIndices, (numPoly + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(g_cent, cent, numPoly * sizeof(double2), cudaMemcpyHostToDevice);
    cudaMemcpy(g_rad, rad, numPoly * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(g_Atgt, Atgt, numPoly * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemset(g_grad, 0, 2 * numVert * sizeof(double));
    cudaMemset(g_energy, 0, sizeof(double));
    cudaMemset(g_aBuf, 0, numPairs * sizeof(double));
    cudaMemset(g_capBuf, 0, numPairs * NUM_IMAGES * sizeof(double));
    cudaMemset(g_momBuf, 0, numPairs * stride * sizeof(double));
    cudaMemset(g_dsBuf, 0, numPairs * stride * sizeof(double));
    cudaMemset(g_workCount, 0, sizeof(int));

    int numImages = periodic ? NUM_IMAGES : 1;
    long nImage = (long)numPairs * numImages;
    long nVert  = (long)numPairs * 2 * maxN;
    cullKernel<<<(int)((nImage + 127) / 128), 128>>>(g_cent, g_rad, numPoly, sigma, gOn, gOff,
                                                     numImages, periodic != 0,
                                                     g_work, g_workCount);
    // One block per surviving (pair, image). The grid is sized for the worst case; blocks past the
    // actual count exit immediately on the device-side workCount, so no readback/sync is needed.
    panelKernel<<<(int)nImage, PANEL_BLOCK>>>(g_pos, g_starts, numPoly, sigma, g_work, g_workCount,
                                              stride, g_capBuf, g_momBuf);
    switchKernel<<<(int)((nImage + 127) / 128), 128>>>(g_starts, numPoly, g_work, g_workCount,
                                                       g_capBuf, stride, g_aBuf, g_dsBuf);
    vertexKernel<<<(int)((nVert + 127) / 128), 128>>>(g_pos, g_starts, g_Atgt, numPoly, g_aBuf,
                                                      g_momBuf, g_dsBuf, maxN, stride,
                                                      g_grad, g_energy);
    cudaMemcpy(energyOut, g_energy, sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(gradOut, g_grad, 2 * numVert * sizeof(double), cudaMemcpyDeviceToHost);
    return 0;
}
