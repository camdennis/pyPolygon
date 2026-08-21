// Neighbor broad phase on the GPU -- the device port of neighbors.candidateEdgePairs.
//
// Produces the candidate EDGE PAIRS whose bounding balls overlap: one ball per edge, centered at its
// midpoint with radius half its own length, and two edges are candidates when
//
//     |mid_i - mid_j|  <  |e_i|/2 + |e_j|/2 + skin
//
// which is EXACT at skin = 0 -- two segments crossing at p have their midpoints within their own
// half-lengths of p, so no crossing can fail the test. The skin is margin for Verlet reuse.
//
// ALL-PAIRS, NO TREE, DELIBERATELY. One thread per edge i, scanning j > i. The rebuild is rare (the
// skin buys tens of force evaluations per build), so an O(V^2) pass with a tiny constant beats an
// O(V log V) build with a large one until V reaches the thousands. An LBVH is the upgrade path and
// slots in behind this same C API; writing it now would be speculative.
//
// THE POLYGON-LEVEL PREFILTER IS APPLIED HERE TOO, and it has to be.
//
// It culls whole polygon pairs on |c_A - c_B| >= R_A + R_B + skin. I first argued the device could
// skip it because an edge ball sits inside its polygon's covering ball, so |mid - c| + h <= R. THAT IS
// FALSE. What the parallelogram law gives for an edge whose endpoints are within R of the centroid is
//
//     d^2 + h^2 <= R^2      (d = |mid - c|, h = half the edge length)
//
// and hence d + h <= sqrt(2) R, not R -- at d = h = R/sqrt(2) the sum is 1.41 R. So the polygon cull
// CAN reject a pair whose edge balls overlap, and a device pass without it finds strictly more
// candidates (measured 33 against 29, 45 against 34).
//
// Neither filter is wrong -- both are necessary conditions for a crossing, so either one alone still
// contains every crossing, and the resulting INTERSECTION sets agree. But the host list is tighter,
// and two broad phases that disagree cannot be checked against each other. So the device applies the
// same two levels, and the candidate lists match exactly.
//
// Built into libplummer.so; called from ../cudaOverlap.py via ctypes.
#include <math.h>

#define NEIGHBOR_BLOCK 128

__device__ __forceinline__ double2 operator-(double2 a, double2 b) {
    return make_double2(a.x - b.x, a.y - b.y);
}
__device__ __forceinline__ double2 operator+(double2 a, double2 b) {
    return make_double2(a.x + b.x, a.y + b.y);
}
__device__ __forceinline__ double2 operator*(double s, double2 a) {
    return make_double2(s * a.x, s * a.y);
}

// The minimum-image lattice translation for the unit square, matching packing.minImageShift.
__device__ __forceinline__ double2 minImageShiftNb(double2 d, bool periodic) {
    if (!periodic) return make_double2(0.0, 0.0);
    return make_double2(-rint(d.x), -rint(d.y));
}

// One shift per polygon PAIR, keyed off their first vertices -- the same single-image assumption
// energies.updateIntersections and neighbors.candidateEdgePairs both make.
__device__ __forceinline__ double2 pairShiftNb(const double2* pos, const int* starts,
                                               int sa, int sb, bool periodic) {
    return minImageShiftNb(pos[starts[sb]] - pos[starts[sa]], periodic);
}

// One thread per edge i; scans j > i. Emits the pair with the LOWER-SHAPE edge first, matching the
// host, which builds its blocks with polygon A < B.
__global__ void neighborPairKernel(const double2* pos, const int* starts, const int* nextIdx,
                                   const int* shapeId, const double2* centroid, const double* radius,
                                   int numVert, double skin, bool periodic,
                                   int* pairI, int* pairJ, int* counter, int maxPairs) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numVert) return;
    int si = shapeId[i];
    double2 a0 = pos[i], a1 = pos[nextIdx[i]];
    double2 midI = 0.5 * (a0 + a1);
    double2 ei = a1 - a0;
    double hI = 0.5 * sqrt(ei.x * ei.x + ei.y * ei.y);

    for (int j = i + 1; j < numVert; ++j) {
        int sj = shapeId[j];
        if (sj == si) continue;                    // self-intersection is the self-repulsion term's job
        int lo = si < sj ? si : sj, hi = si < sj ? sj : si;
        double2 shift = pairShiftNb(pos, starts, lo, hi, periodic);
        // Coarse level, exactly as on the host: two polygons whose covering balls do not overlap
        // cannot touch anywhere, so none of their edges can.
        double2 cd = (centroid[hi] + shift) - centroid[lo];
        double creach = radius[lo] + radius[hi] + skin;
        if (cd.x * cd.x + cd.y * cd.y >= creach * creach) continue;
        // The shift carries the HIGHER-indexed polygon into the lower one's image, so apply it to
        // whichever of the two is on that polygon.
        double2 b0 = pos[j], b1 = pos[nextIdx[j]];
        double2 midJ = 0.5 * (b0 + b1);
        double2 ej = b1 - b0;
        double hJ = 0.5 * sqrt(ej.x * ej.x + ej.y * ej.y);
        double2 d = (si < sj) ? (midJ + shift) - midI : midJ - (midI + shift);
        double reach = hI + hJ + skin;
        if (d.x * d.x + d.y * d.y >= reach * reach) continue;
        int slot = atomicAdd(counter, 1);
        if (slot < maxPairs) {
            // Lower shape first, so the emitted orientation matches the host's A < B convention.
            pairI[slot] = (si < sj) ? i : j;
            pairJ[slot] = (si < sj) ? j : i;
        }
    }
}

// ---- C API. Returns the number of pairs FOUND, which may exceed maxPairs -- the caller grows the
// buffer and calls again, the same contract the intersection buffer uses. ----
static double2* g_nbPos = nullptr;
static int *g_nbStarts = nullptr, *g_nbNext = nullptr, *g_nbShape = nullptr;
static int *g_nbPairI = nullptr, *g_nbPairJ = nullptr, *g_nbCounter = nullptr;
static double2* g_nbCent = nullptr; static double* g_nbRad = nullptr;
static int g_nbNumVert = 0, g_nbNumPoly = 0, g_nbMaxPairs = 0;

extern "C" int neighborPairsCuda(const double* positions, int numVert,
                                 const int* startIndices, int numPoly,
                                 const int* nextIdx, const int* shapeId,
                                 const double* centroids, const double* radii,
                                 double skin, int periodic, int maxPairs,
                                 int* pairIOut, int* pairJOut) {
    if (numVert < 2 || numPoly < 2) return 0;
    if (numVert != g_nbNumVert || numPoly != g_nbNumPoly) {
        if (g_nbPos) { cudaFree(g_nbPos); cudaFree(g_nbStarts); cudaFree(g_nbNext);
                       cudaFree(g_nbShape); cudaFree(g_nbCent); cudaFree(g_nbRad);
                       cudaFree(g_nbCounter); }
        cudaMalloc(&g_nbPos, numVert * sizeof(double2));
        cudaMalloc(&g_nbStarts, (numPoly + 1) * sizeof(int));
        cudaMalloc(&g_nbNext, numVert * sizeof(int));
        cudaMalloc(&g_nbShape, numVert * sizeof(int));
        cudaMalloc(&g_nbCent, numPoly * sizeof(double2));
        cudaMalloc(&g_nbRad, numPoly * sizeof(double));
        cudaMalloc(&g_nbCounter, sizeof(int));
        g_nbNumVert = numVert; g_nbNumPoly = numPoly;
    }
    if (maxPairs > g_nbMaxPairs) {
        if (g_nbPairI) { cudaFree(g_nbPairI); cudaFree(g_nbPairJ); }
        cudaMalloc(&g_nbPairI, maxPairs * sizeof(int));
        cudaMalloc(&g_nbPairJ, maxPairs * sizeof(int));
        g_nbMaxPairs = maxPairs;
    }
    cudaMemcpy(g_nbPos, positions, numVert * sizeof(double2), cudaMemcpyHostToDevice);
    cudaMemcpy(g_nbStarts, startIndices, (numPoly + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(g_nbNext, nextIdx, numVert * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(g_nbShape, shapeId, numVert * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(g_nbCent, centroids, numPoly * sizeof(double2), cudaMemcpyHostToDevice);
    cudaMemcpy(g_nbRad, radii, numPoly * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemset(g_nbCounter, 0, sizeof(int));

    neighborPairKernel<<<(numVert + NEIGHBOR_BLOCK - 1) / NEIGHBOR_BLOCK, NEIGHBOR_BLOCK>>>(
        g_nbPos, g_nbStarts, g_nbNext, g_nbShape, g_nbCent, g_nbRad, numVert, skin, periodic != 0,
        g_nbPairI, g_nbPairJ, g_nbCounter, maxPairs);

    int found = 0;
    cudaMemcpy(&found, g_nbCounter, sizeof(int), cudaMemcpyDeviceToHost);
    int copied = found < maxPairs ? found : maxPairs;
    if (copied > 0) {
        cudaMemcpy(pairIOut, g_nbPairI, copied * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(pairJOut, g_nbPairJ, copied * sizeof(int), cudaMemcpyDeviceToHost);
    }
    return found;
}
