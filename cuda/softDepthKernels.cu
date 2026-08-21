// Soft-depth boundary-area energy on the GPU -- the port of softDepth.packingEnergyForce.
//
//   E = sum over ordered pairs (A, B) of  int_{dA} phi(h_eps^B(x)) dl(x)
//
// PAIR INTERACTIONS ONLY. The container path is deliberately absent: softDepth confinement is broken
// in the numpy tier too (it contributes exactly zero, always -- reversing a loop's winding negates each
// ell_i, but h is their softMIN and min(-ell) = -MAX(ell)), and its fix needs a modeling decision.
//
// Parallel structure, mirroring plummerDriver.cu: a CULL kernel compacts the surviving ordered pairs
// into a dense work list, then ONE BLOCK PER WORK ITEM whose threads STRIDE over A's edges. Striding
// rather than one-thread-per-padded-slot is what lets ragged vertex counts need no padding.
//
// The block, not the thread, is the unit because the loop-side force of (43) touches EVERY vertex of B
// at EVERY quadrature node. Accumulating that in shared memory and issuing nB atomics once per block,
// instead of nB per node, is the difference between this being bandwidth-bound and not.
//
// Built into libplummer.so; called from ../cudaOverlap.py via ctypes. double/double2, sm_75.
#include "softDepth.cuh"

// UNVERIFIED(Cam)

// Largest vertices-per-polygon this build accepts. It caps SHARED MEMORY, not strides -- nA and nB are
// measured from starts[] at runtime. Exceeding it is REPORTED by the driver, never silently truncated.
// The reason that matters here is written in plummerDriver.cu: a hard-coded 12 once dropped the
// gradient on every vertex past the 24th of a pair while the ENERGY stayed correct, so it survived the
// test suite. Raise and rebuild (make -C cuda libplummer.so) if you need larger polygons.
#define SOFTDEPTH_MAXN 64
#define SOFTDEPTH_BLOCK 64

// Fixed iteration counts, so every thread runs the same trip count and the dominant cost carries ZERO
// warp divergence. Measured in numpy at N=32,n=32: 12 root steps cost 1.2e-13 in energy and 5.3e-13 in
// force against 32 steps, both below the ~3e-12 force noise floor. The peak only has to SEPARATE the
// two roots, so it gets fewer.
#define SOFTDEPTH_ROOT_STEPS 12
#define SOFTDEPTH_PEAK_STEPS 8

// Gauss-Legendre on [-1, 1]. Must match numpy leggauss(order) exactly, the way SELFREP_SX/SW do.
__device__ __constant__ double GAUSS16X[16] = {
    -9.89400934991649939e-01, -9.44575023073232600e-01, -8.65631202387831755e-01,
    -7.55404408355002999e-01, -6.17876244402643771e-01, -4.58016777657227370e-01,
    -2.81603550779258915e-01, -9.50125098376374544e-02,  9.50125098376374544e-02,
     2.81603550779258915e-01,  4.58016777657227370e-01,  6.17876244402643771e-01,
     7.55404408355002999e-01,  8.65631202387831755e-01,  9.44575023073232600e-01,
     9.89400934991649939e-01
};
__device__ __constant__ double GAUSS16W[16] = {
     2.71524594117540374e-02,  6.22535239386477063e-02,  9.51585116824925914e-02,
     1.24628971255534030e-01,  1.49595988816576764e-01,  1.69156519395002619e-01,
     1.82603415044923612e-01,  1.89450610455068585e-01,  1.89450610455068585e-01,
     1.82603415044923612e-01,  1.69156519395002619e-01,  1.49595988816576764e-01,
     1.24628971255534030e-01,  9.51585116824925914e-02,  6.22535239386477063e-02,
     2.71524594117540374e-02
};
__device__ __constant__ double GAUSS32X[32] = {
    -9.97263861849481570e-01, -9.85611511545268382e-01, -9.64762255587506390e-01,
    -9.34906075937739667e-01, -8.96321155766052202e-01, -8.49367613732569970e-01,
    -7.94483795967942386e-01, -7.32182118740289711e-01, -6.63044266930215231e-01,
    -5.87715757240762304e-01, -5.06899908932229359e-01, -4.21351276130635333e-01,
    -3.31868602282127667e-01, -2.39287362252137065e-01, -1.44471961582796488e-01,
    -4.83076656877383104e-02,  4.83076656877383104e-02,  1.44471961582796488e-01,
     2.39287362252137065e-01,  3.31868602282127667e-01,  4.21351276130635333e-01,
     5.06899908932229359e-01,  5.87715757240762304e-01,  6.63044266930215231e-01,
     7.32182118740289711e-01,  7.94483795967942386e-01,  8.49367613732569970e-01,
     8.96321155766052202e-01,  9.34906075937739667e-01,  9.64762255587506390e-01,
     9.85611511545268382e-01,  9.97263861849481570e-01
};
__device__ __constant__ double GAUSS32W[32] = {
     7.01861000946929839e-03,  1.62743947309059653e-02,  2.53920653092624266e-02,
     3.42738629130216257e-02,  4.28358980222264263e-02,  5.09980592623762441e-02,
     5.86840934785357038e-02,  6.58222227763617523e-02,  7.23457941088484491e-02,
     7.81938957870703111e-02,  8.33119242269468457e-02,  8.76520930044039082e-02,
     9.11738786957638631e-02,  9.38443990808045664e-02,  9.56387200792748332e-02,
     9.65400885147278121e-02,  9.65400885147278121e-02,  9.56387200792748332e-02,
     9.38443990808045664e-02,  9.11738786957638631e-02,  8.76520930044039082e-02,
     8.33119242269468457e-02,  7.81938957870703111e-02,  7.23457941088484491e-02,
     6.58222227763617523e-02,  5.86840934785357038e-02,  5.09980592623762441e-02,
     4.28358980222264263e-02,  3.42738629130216257e-02,  2.53920653092624266e-02,
     1.62743947309059653e-02,  7.01861000946929839e-03
};

// One ordered interaction: the boundary of `boundaryPoly` against the loop of `loopPoly`, with ONLY
// THE LOOP displaced by (sx, sy). Shifting both is a rigid translation that cancels -- the bug that
// silently switched periodicity off in the numpy tier while the cull still used the minimum image.
// The loop's covering ball travels with the work item so the per-edge cull below costs no global
// reads -- the cull kernel has these values in registers already.
struct PairWork {
    int boundaryPoly;
    int loopPoly;
    double sx, sy;
    double loopCx, loopCy, loopRadius;
};

// Per-plane geometry of the loop, staged in shared memory once per block. `vx, vy` and `invLength` are
// carried for the foot-of-perpendicular s_i of (41), which the loop-side force needs.
struct LoopPlane {
    double nx, ny, c;
    double vx, vy, invLength;
};

// ---------------------------------------------------------------------------
// Kernel 0 -- CULL. One thread per UNORDERED pair; a survivor emits BOTH ordered interactions, with
// the shift and its negation, so the pair is symmetrized exactly as the numpy loop does rather than
// relying on minImageShift being antisymmetric at a tie.
// ---------------------------------------------------------------------------
__global__ void softDepthCullKernel(const double2* centroids, const double* radii, int numPoly,
                                    const double2* pos, const int* starts, double epsilon,
                                    bool periodic, PairWork* work, int* workCount) {
    long t = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= (long)numPoly * numPoly) return;
    int A = (int)(t / numPoly), B = (int)(t % numPoly);
    if (B <= A) return;

    double sx = 0.0, sy = 0.0;
    if (periodic) {
        double2 a0 = pos[starts[A]], b0 = pos[starts[B]];
        sx = -floor(b0.x - a0.x + 0.5);
        sy = -floor(b0.y - a0.y + 0.5);
    }
    double dx = centroids[B].x + sx - centroids[A].x;
    double dy = centroids[B].y + sy - centroids[A].y;
    if (sqrt(dx * dx + dy * dy) >= radii[A] + radii[B] + epsilon) return;

    int slot = atomicAdd(workCount, 2);
    PairWork forward;
    forward.boundaryPoly = A; forward.loopPoly = B; forward.sx = sx; forward.sy = sy;
    forward.loopCx = centroids[B].x + sx; forward.loopCy = centroids[B].y + sy;
    forward.loopRadius = radii[B];
    PairWork reverse;
    reverse.boundaryPoly = B; reverse.loopPoly = A; reverse.sx = -sx; reverse.sy = -sy;
    reverse.loopCx = centroids[A].x - sx; reverse.loopCy = centroids[A].y - sy;
    reverse.loopRadius = radii[A];
    work[slot] = forward;
    work[slot + 1] = reverse;
}

// ---------------------------------------------------------------------------
// Kernel 1 -- FUSED per-edge. One BLOCK per work item; threads stride over the boundary polygon's
// edges. Each thread finds its edge's contact interval, walks the envelope, and runs Gauss on each
// segment -- the panels are CONSUMED AS PRODUCED and never reach memory, which is what removes the
// ragged panel count that the numpy version has to materialize.
// ---------------------------------------------------------------------------
__global__ void softDepthKernel(const double2* pos, const int* starts, const PairWork* work,
                                const int* workCount, double epsilon, double stiffness, int order,
                                double* force, double* energyOut) {
    int w = blockIdx.x;
    if (w >= *workCount) return;
    PairWork job = work[w];

    int baseA = starts[job.boundaryPoly], nA = starts[job.boundaryPoly + 1] - baseA;
    int baseB = starts[job.loopPoly],     nB = starts[job.loopPoly + 1] - baseB;

    __shared__ softDepth::Plane planes[SOFTDEPTH_MAXN];
    __shared__ LoopPlane detail[SOFTDEPTH_MAXN];
    __shared__ double loopForce[2 * SOFTDEPTH_MAXN];

    // ORIENTATION, matching softDepth.loopFrame. n_i = J t_i is outward only for a CCW loop; given a
    // clockwise one every normal points inward and the model inverts -- measured on the CPU, a
    // clockwise unit box put h at -0.5139 in its own CENTRE, turning confinement into an attractive
    // well that collapsed five squares onto a point. Recomputed redundantly by every thread because nB
    // is small and a shared-memory reduction plus barrier costs more than the loop.
    double twiceArea = 0.0;
    for (int i = 0; i < nB; ++i) {
        double2 a = pos[baseB + i], b = pos[baseB + (i + 1) % nB];
        twiceArea += a.x * b.y - b.x * a.y;
    }
    double winding = (twiceArea < 0.0) ? -1.0 : 1.0;

    // Stage the loop's half-planes: outward normal n_i = J t_i with J = [[0,1],[-1,0]] (eq 1), and
    // offset c_i = n_i . v_i (eq 2), matching softDepth.loopFrame exactly.
    for (int i = threadIdx.x; i < nB; i += blockDim.x) {
        double2 v0 = pos[baseB + i], v1 = pos[baseB + (i + 1) % nB];
        double vx = v0.x + job.sx, vy = v0.y + job.sy;
        double ex = v1.x - v0.x, ey = v1.y - v0.y;
        double length = sqrt(ex * ex + ey * ey);
        double tx = ex / length, ty = ey / length;
        double nx = winding * ty, ny = -winding * tx;
        planes[i].nx = nx; planes[i].ny = ny; planes[i].c = nx * vx + ny * vy;
        detail[i].nx = nx; detail[i].ny = ny; detail[i].c = planes[i].c;
        detail[i].vx = vx; detail[i].vy = vy; detail[i].invLength = 1.0 / length;
        loopForce[2 * i] = 0.0;
        loopForce[2 * i + 1] = 0.0;
    }
    __syncthreads();

    const double* gaussX = (order == 32) ? GAUSS32X : GAUSS16X;
    const double* gaussW = (order == 32) ? GAUSS32W : GAUSS16W;
    double blockEnergy = 0.0;

    for (int e = threadIdx.x; e < nA; e += blockDim.x) {
        double2 v0 = pos[baseA + e], v1 = pos[baseA + (e + 1) % nA];
        double2 dir = make_double2(v1.x - v0.x, v1.y - v0.y);
        double edgeLength = sqrt(dir.x * dir.x + dir.y * dir.y);

        // EDGE CULL against the loop's covering ball -- Cam's neighbor construction, at the polygon
        // level. EXACT, not a heuristic: h = min_i ell_i - eps log(total) with total >= 1, so h > 0
        // forces min_i ell_i > 0, i.e. the point lies strictly INSIDE the loop and therefore within
        // radius of its centroid. A contacting point is also within |e|/2 of the edge midpoint, so
        //     |mid - centroid| >= |e|/2 + radius   =>   no point of this edge can contact
        // drops nothing. Without it a non-contacting edge still pays ~11 depth evaluations before the
        // hPeak test can reject it, and most edges of A are nowhere near B.
        //
        // Deliberately NOT routed through candidateEdgePairs: edge-EDGE candidacy is not conservative
        // here, because an edge lying wholly inside the loop has h > 0 along its whole length while
        // crossing none of the loop's edges. Soft depth is edge-versus-INTERIOR.
        double midX = 0.5 * (v0.x + v1.x) - job.loopCx;
        double midY = 0.5 * (v0.y + v1.y) - job.loopCy;
        double reach = 0.5 * edgeLength + job.loopRadius;
        if (midX * midX + midY * midY >= reach * reach) continue;

        double hLo, slopeLo, hHi, slopeHi;
        softDepth::depthAlongEdge(planes, nB, v0, dir, 0.0, epsilon, &hLo, &slopeLo);
        softDepth::depthAlongEdge(planes, nB, v0, dir, 1.0, epsilon, &hHi, &slopeHi);

        double tPeak = softDepth::peakOf(planes, nB, v0, dir, epsilon, slopeLo, slopeHi,
                                         SOFTDEPTH_PEAK_STEPS);
        double hPeak, slopePeak;
        softDepth::depthAlongEdge(planes, nB, v0, dir, tPeak, epsilon, &hPeak, &slopePeak);
        if (hPeak <= 0.0) continue;

        // h is concave, so {h >= 0} on the edge is ONE interval, bracketed by the peak.
        double t0 = (hLo > 0.0) ? 0.0
                  : softDepth::bracketedRoot(planes, nB, v0, dir, epsilon, 0.0, tPeak, false,
                                             SOFTDEPTH_ROOT_STEPS);
        double t1 = (hHi > 0.0) ? 1.0
                  : softDepth::bracketedRoot(planes, nB, v0, dir, epsilon, tPeak, 1.0, true,
                                             SOFTDEPTH_ROOT_STEPS);
        if (t1 <= t0) continue;

        double toFirst = 0.0, toFirstY = 0.0, toSecond = 0.0, toSecondY = 0.0, edgeEnergy = 0.0;
        double invEps = 1.0 / epsilon;

        // Walk the envelope, consuming each segment as it is produced. At most nB segments; the guard
        // is belt-and-braces since nextEnvelopeCut only ever returns a parameter strictly ahead.
        double segStart = t0;
        for (int seg = 0; seg <= nB && segStart < t1; ++seg) {
            double segEnd = softDepth::nextEnvelopeCut(planes, nB, v0, dir, segStart, t1);
            double half = 0.5 * (segEnd - segStart), mid = 0.5 * (segEnd + segStart);

            for (int q = 0; q < order; ++q) {
                double t = mid + half * gaussX[q];
                double weight = edgeLength * half * gaussW[q];
                double2 x = make_double2(v0.x + t * dir.x, v0.y + t * dir.y);

                double h, nbarX, nbarY;
                softDepth::depthAndNormal(planes, nB, x, epsilon, &h, &nbarX, &nbarY);
                double density, first;
                softDepth::contactLaw(h, stiffness, &density, &first);
                if (density == 0.0 && first == 0.0) continue;

                edgeEnergy += weight * density;

                // Node force -dE/dx = phi'(h) nbar, split barycentrically onto the edge's endpoints
                // because the node rides on the edge. The split preserves torque exactly.
                double scale = weight * first;
                toFirst  += (1.0 - t) * scale * nbarX;  toFirstY  += (1.0 - t) * scale * nbarY;
                toSecond +=        t  * scale * nbarX;  toSecondY +=        t  * scale * nbarY;

                // Loop-side force (43): plane i hands (1 - s_i) of its load to vertex i and s_i to
                // vertex i+1. Recomputing the softmin weights costs one more pass over nB; storing
                // them would cost nB registers per thread.
                double lowest = 1.0e300;
                for (int i = 0; i < nB; ++i) {
                    double ell = planes[i].c - (planes[i].nx * x.x + planes[i].ny * x.y);
                    if (ell < lowest) lowest = ell;
                }
                double total = 0.0;
                for (int i = 0; i < nB; ++i) {
                    double ell = planes[i].c - (planes[i].nx * x.x + planes[i].ny * x.y);
                    total += exp(-(ell - lowest) * invEps);
                }
                double norm = -scale / total;
                for (int i = 0; i < nB; ++i) {
                    double ell = planes[i].c - (planes[i].nx * x.x + planes[i].ny * x.y);
                    double share = exp(-(ell - lowest) * invEps) * norm;
                    // s_i = t_i . (x - v_i) / |e_i|, with t_i = (-n_y, n_x) recovered from n_i = J t_i.
                    double foot = ((-detail[i].ny) * (x.x - detail[i].vx)
                                   + detail[i].nx * (x.y - detail[i].vy)) * detail[i].invLength;
                    int next = (i + 1) % nB;
                    atomicAdd(&loopForce[2 * i],        share * (1.0 - foot) * detail[i].nx);
                    atomicAdd(&loopForce[2 * i + 1],    share * (1.0 - foot) * detail[i].ny);
                    atomicAdd(&loopForce[2 * next],     share * foot * detail[i].nx);
                    atomicAdd(&loopForce[2 * next + 1], share * foot * detail[i].ny);
                }
            }
            segStart = segEnd;
        }

        if (edgeEnergy == 0.0 && toFirst == 0.0 && toSecond == 0.0) continue;
        blockEnergy += edgeEnergy;

        // The MEASURE moves too: dl = |e| dt, so the edge carries a tangential force from d|e|/dv.
        // Equal and opposite along its own edge, hence torque-free on its own.
        double tangential = edgeEnergy / (edgeLength * edgeLength);
        int first0 = baseA + e, second0 = baseA + (e + 1) % nA;
        atomicAdd(&force[2 * first0],      toFirst  + tangential * dir.x);
        atomicAdd(&force[2 * first0 + 1],  toFirstY + tangential * dir.y);
        atomicAdd(&force[2 * second0],     toSecond - tangential * dir.x);
        atomicAdd(&force[2 * second0 + 1], toSecondY - tangential * dir.y);
    }

    __syncthreads();
    for (int i = threadIdx.x; i < nB; i += blockDim.x) {
        atomicAdd(&force[2 * (baseB + i)],     loopForce[2 * i]);
        atomicAdd(&force[2 * (baseB + i) + 1], loopForce[2 * i + 1]);
    }
    if (blockEnergy != 0.0) atomicAdd(energyOut, blockEnergy);
}

// ---- C API. Persistent buffers, sized on the first call and reused. ----
static double2* g_sdPos = nullptr; static int* g_sdStarts = nullptr;
static double2* g_sdCent = nullptr; static double* g_sdRadii = nullptr;
static PairWork* g_sdWork = nullptr; static int* g_sdWorkCount = nullptr;
static double* g_sdForce = nullptr; static double* g_sdEnergy = nullptr;
static int g_sdNumVert = 0, g_sdNumPoly = 0;

// Centroids and covering radii come from the HOST, computed by energies.polygonCentroidsRadii, so the
// cull selects bit-identically to the numpy path and a disagreement can only be the arithmetic.
//
// Returns 0 on success, or the offending vertex count when a polygon exceeds SOFTDEPTH_MAXN. Reporting
// rather than truncating is the whole point -- see the note on the define.
extern "C" int softDepthCuda(const double* positions, const int* starts, int numPoly, int numVert,
                             const double* centroids, const double* radii, int container,
                             double epsilon, double stiffness, int order, int periodic,
                             double* energyOut, double* forceOut) {
    *energyOut = 0.0;
    for (int i = 0; i < 2 * numVert; ++i) forceOut[i] = 0.0;

    int stop = (container < 0) ? numPoly : container;
    int maxN = 0;
    for (int p = 0; p < stop; ++p) {
        int n = starts[p + 1] - starts[p];
        if (n > maxN) maxN = n;
    }
    if (maxN > SOFTDEPTH_MAXN) return maxN;
    if (stop < 2) return 0;
    if (order != 16 && order != 32) return -order;

    int maxWork = stop * (stop - 1);
    if (numVert != g_sdNumVert || numPoly != g_sdNumPoly) {
        if (g_sdPos) {
            cudaFree(g_sdPos); cudaFree(g_sdStarts); cudaFree(g_sdCent); cudaFree(g_sdRadii);
            cudaFree(g_sdWork); cudaFree(g_sdWorkCount); cudaFree(g_sdForce); cudaFree(g_sdEnergy);
        }
        cudaMalloc(&g_sdPos, numVert * sizeof(double2));
        cudaMalloc(&g_sdStarts, (numPoly + 1) * sizeof(int));
        cudaMalloc(&g_sdCent, numPoly * sizeof(double2));
        cudaMalloc(&g_sdRadii, numPoly * sizeof(double));
        cudaMalloc(&g_sdWork, (size_t)(numPoly * (numPoly > 1 ? numPoly - 1 : 1)) * sizeof(PairWork));
        cudaMalloc(&g_sdWorkCount, sizeof(int));
        cudaMalloc(&g_sdForce, 2 * numVert * sizeof(double));
        cudaMalloc(&g_sdEnergy, sizeof(double));
        g_sdNumVert = numVert; g_sdNumPoly = numPoly;
    }
    cudaMemcpy(g_sdPos, positions, numVert * sizeof(double2), cudaMemcpyHostToDevice);
    cudaMemcpy(g_sdStarts, starts, (numPoly + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(g_sdCent, centroids, numPoly * sizeof(double2), cudaMemcpyHostToDevice);
    cudaMemcpy(g_sdRadii, radii, numPoly * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemset(g_sdWorkCount, 0, sizeof(int));
    cudaMemset(g_sdForce, 0, 2 * numVert * sizeof(double));
    cudaMemset(g_sdEnergy, 0, sizeof(double));

    int cullThreads = 128;
    long cullSlots = (long)stop * stop;
    int cullBlocks = (int)((cullSlots + cullThreads - 1) / cullThreads);
    softDepthCullKernel<<<cullBlocks, cullThreads>>>(g_sdCent, g_sdRadii, stop, g_sdPos, g_sdStarts,
                                                     epsilon, periodic != 0, g_sdWork, g_sdWorkCount);

    // One block per work item. Launching maxWork blocks and letting the surplus exit on the workCount
    // test costs nothing measurable and saves a device-to-host round trip inside the force evaluation,
    // which is the traffic Cam asked to keep off the wire.
    softDepthKernel<<<maxWork, SOFTDEPTH_BLOCK>>>(g_sdPos, g_sdStarts, g_sdWork, g_sdWorkCount,
                                                  epsilon, stiffness, order, g_sdForce, g_sdEnergy);

    cudaMemcpy(energyOut, g_sdEnergy, sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(forceOut, g_sdForce, 2 * numVert * sizeof(double), cudaMemcpyDeviceToHost);
    return 0;
}
