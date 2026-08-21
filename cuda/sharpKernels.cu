// Sharp overlap -- separated-sum kernels + host driver (Track B). See sharpKernels.cuh.
// Math follows ../energies.py (verified); structure mirrors STABLE (cell-binned intersection finding,
// packed list, sort, binary-search followers, exterior-over-intersections + interior-over-vertices).
// double/double2, sm_75.
#include "sharp.cuh"
#include "sharpKernels.cuh"

#include <cub/cub.cuh>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>

using namespace sharp;

// Minimum-image lattice translation for the unit-square box: the shift that carries ``d`` to its
// nearest periodic image, i.e. packing.minImageShift. Zero in free space (periodic = false).
__device__ __forceinline__ double2 minImageShiftDev(double2 d, bool periodic) {
    if (!periodic) return make_double2(0.0, 0.0);
    return make_double2(-floor(d.x + 0.5), -floor(d.y + 0.5));
}

// The single shift that carries polygon ``sb`` into ``sa``'s image, keyed off their FIRST vertices --
// exactly energies.updateIntersections' ``minImageShift(r[starts[polyB]] - r[starts[polyA]], box)``.
// One image per pair, not a 9-image sweep: that is the single-image assumption the model is built on
// (and why build._warnLargePhi exists).
__device__ __forceinline__ double2 pairShift(const double2* pos, const int* starts,
                                             int sa, int sb, bool periodic) {
    return minImageShiftDev(pos[starts[sb]] - pos[starts[sa]], periodic);
}

// ---- device helper: test one edge pair; if it crosses (different shapes) append the intersection.
// Canonicalizes A = lower shape so orientation matches energies.updateIntersections regardless of
// the caller's argument order; si = the leave polygon (CCW overlap boundary). The higher-indexed
// polygon is min-image shifted into the lower's image before the crossing test.
__device__ __forceinline__ void tryCrossing(const double2* pos, const int* starts, const int* shapeId,
                                            int e1, int e2, uint64_t* keys, double2* tu,
                                            int* counter, int maxInter, bool periodic) {
    int s1 = shapeId[e1], s2 = shapeId[e2];
    if (s1 == s2) return;
    int ga = (s1 < s2) ? e1 : e2, gb = (s1 < s2) ? e2 : e1;
    int sa = shapeId[ga], sb = shapeId[gb];
    int baseA = starts[sa], nEa = starts[sa + 1] - baseA, la = ga - baseA;
    int baseB = starts[sb], nEb = starts[sb + 1] - baseB, lb = gb - baseB;
    // Edge VECTORS are shift-invariant, so only the base point moves.
    double2 shift = pairShift(pos, starts, sa, sb, periodic);
    double2 p0 = pos[ga], dA = pos[baseA + (la + 1) % nEa] - p0;
    double2 q0 = pos[gb] + shift, dB = pos[baseB + (lb + 1) % nEb] - pos[gb];
    double sA, sB;
    if (!segCross(p0, dA, q0, dB, &sA, &sB)) return;
    double cr = dB.x * dA.y - dB.y * dA.x;   // eB x eA
    int si, sj, edgeI, edgeJ; double sI, sJ;
    if (cr > 0.0) { si = sa; edgeI = la; sI = sA; sj = sb; edgeJ = lb; sJ = sB; }
    else          { si = sb; edgeI = lb; sI = sB; sj = sa; edgeJ = la; sJ = sA; }
    int slot = atomicAdd(counter, 1);
    if (slot < maxInter) { keys[slot] = packKey(si, sj, edgeI, edgeJ); tu[slot] = make_double2(sI, sJ); }
}

// ---- Stage 1: cell-accelerated intersection finding ------------------------------------------
// Each edge is binned by its MIDPOINT into a uniform grid (cellSize = max edge length, so any two
// crossing edges land in adjacent cells). Counting-sort edges into cells, then each edge tests only
// its 3x3 cell neighborhood (id2 > id1 dedups). Same intersection set as all-pairs.
//
// PERIODIC MODE: the midpoint is wrapped into the unit cell and the neighborhood search wraps
// modulo (nx, ny), so cell 0 and cell nx-1 are adjacent. That is what lets a pair straddling the
// boundary be found at all -- binning by absolute coordinate silently misses every such pair.
__global__ void computeCellKernel(const double2* pos, const int* starts, const int* shapeId,
                                  int numVert, double xmin, double ymin, double cellSize,
                                  int nx, int ny, bool periodic, int* cellId) {
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= numVert) return;
    int s = shapeId[e], base = starts[s], nE = starts[s + 1] - base, le = e - base;
    double2 mid = 0.5 * (pos[e] + pos[base + (le + 1) % nE]);
    if (periodic) { mid.x -= floor(mid.x); mid.y -= floor(mid.y); }
    int cx = (int)floor((mid.x - xmin) / cellSize);
    int cy = (int)floor((mid.y - ymin) / cellSize);
    if (periodic) {
        cx = ((cx % nx) + nx) % nx;
        cy = ((cy % ny) + ny) % ny;
    } else {
        if (cx < 0) cx = 0; else if (cx >= nx) cx = nx - 1;
        if (cy < 0) cy = 0; else if (cy >= ny) cy = ny - 1;
    }
    cellId[e] = cy * nx + cx;
}
__global__ void countCellKernel(const int* cellId, int numVert, int* cellCount) {
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e < numVert) atomicAdd(&cellCount[cellId[e]], 1);
}
__global__ void scatterCellKernel(const int* cellId, int numVert, const int* cellStart,
                                  int* cellFill, int* sortedEdge) {
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= numVert) return;
    int c = cellId[e];
    sortedEdge[cellStart[c] + atomicAdd(&cellFill[c], 1)] = e;
}
__global__ void findIntersectionsCellKernel(const double2* pos, const int* starts, const int* shapeId,
                                            int numVert, const int* cellId, const int* cellStart,
                                            const int* sortedEdge, int nx, int ny, bool periodic,
                                            uint64_t* keys, double2* tu, int* counter, int maxInter) {
    int id1 = blockIdx.x * blockDim.x + threadIdx.x;
    if (id1 >= numVert) return;
    int c = cellId[id1], cx = c % nx, cy = c / nx;
    for (int dy = -1; dy <= 1; ++dy) {
        int ny2 = cy + dy;
        if (periodic) {
            if (ny == 1 && dy != 0) continue;          // a single row: visiting it 3x would duplicate
            if (ny == 2 && dy == 1) continue;          // dy=-1 and dy=+1 reach the same row
            ny2 = ((ny2 % ny) + ny) % ny;
        } else if (ny2 < 0 || ny2 >= ny) continue;
        for (int dx = -1; dx <= 1; ++dx) {
            int nx2 = cx + dx;
            if (periodic) {
                if (nx == 1 && dx != 0) continue;
                if (nx == 2 && dx == 1) continue;
                nx2 = ((nx2 % nx) + nx) % nx;
            } else if (nx2 < 0 || nx2 >= nx) continue;
            int nc = ny2 * nx + nx2;
            for (int k = cellStart[nc]; k < cellStart[nc + 1]; ++k) {
                int id2 = sortedEdge[k];
                if (id2 > id1)
                    tryCrossing(pos, starts, shapeId, id1, id2, keys, tu, counter, maxInter, periodic);
            }
        }
    }
}

// ---- Stage 3: followers (over intersections) -------------------------------------------------
// For alpha, find the next-forward partner beta on the shared leave-polygon si: beta has the swapped
// pair (si_beta = sj, sj_beta = si) and minimal forward distance (cArrive - cLeave) mod nE > 0, where
// cLeave = edgeI + sI (alpha), cArrive = edgeJ_beta + sJ_beta (beta's arrival on si). energies.updateFollowers.
__global__ void followersKernel(const uint64_t* keys, const double2* tu, int M,
                                const int* starts, int* fol) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= M) return;
    uint64_t k = keys[idx];
    int si = keySi(k), sj = keySj(k), edgeI = keyEdgeI(k);
    int nE = starts[si + 1] - starts[si];
    double cLeave = edgeI + tu[idx].x;
    uint64_t lb = ((uint64_t)sj << 48) | ((uint64_t)si << 32);
    uint64_t ub = ((uint64_t)sj << 48) | ((uint64_t)(si + 1) << 32);
    int lo = 0, hi = M;                     // lower_bound of lb
    while (lo < hi) { int m = (lo + hi) >> 1; if (keys[m] < lb) lo = m + 1; else hi = m; }
    int best = -1; double bestD = 1e300;
    for (int be = lo; be < M && keys[be] < ub; ++be) {
        double cArrive = keyEdgeJ(keys[be]) + tu[be].y;
        double d = fmod(cArrive - cLeave, (double)nE);
        if (d < 0.0) d += nE;
        if (d > 0.0 && d < bestD) { bestD = d; best = be; }
    }
    fol[idx] = best;
}

// ---- Stage 4: shape ranges -------------------------------------------------------------------
__global__ void initRangesKernel(int* shapeStart, int* shapeEnd, int numPoly) {
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s < numPoly) { shapeStart[s] = INT_MAX; shapeEnd[s] = -1; }
}
__global__ void shapeRangesKernel(const uint64_t* keys, int M, int* shapeStart, int* shapeEnd) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= M) return;
    int si = keySi(keys[idx]);
    atomicMin(&shapeStart[si], idx);
    atomicMax(&shapeEnd[si], idx);
}

// ---- Stage 6: exterior kernel (U_ex), parallel OVER INTERSECTIONS -----------------------------
// The contact law is the NORMALIZED SQUARE  U = 2 k sum_{A<B} (a_AB / norm_AB)^2, matching the
// mollified tier, so the sharp result is genuinely its sigma -> 0 limit. A single total area is not
// enough for that -- the normalizer is PER PAIR -- so the geometry kernels run TWICE:
//
//   pass 1 (pairWeight == nullptr): accumulate a_AB into pairArea, deposit no gradient;
//   pass 2 (pairWeight != nullptr): deposit dU/dv = sum 4 k (a_AB / norm_AB^2) da_AB/dv, no area.
//
// Storing a gradient per pair instead would need numPoly^2 * 2 * numVert doubles; running the cheap
// geometry a second time is far less costly than that.
__device__ __forceinline__ int pairSlot(int si, int sj, int numPoly) {
    int lo = si < sj ? si : sj, hi = si < sj ? sj : si;
    return lo * numPoly + hi;
}

__global__ void exteriorKernel(const uint64_t* keys, const double2* tu, const int* fol, int M,
                               const double2* pos, const int* starts, bool periodic, int numPoly,
                               double* pairArea, const double* pairWeight,
                               double* grad) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= M) return;
    int be = fol[idx]; if (be < 0) return;
    uint64_t k = keys[idx];
    int si = keySi(k), sj = keySj(k), edgeI = keyEdgeI(k);
    double sI = tu[idx].x;
    int edgeL = keyEdgeJ(keys[be]); double sL = tu[be].y;
    int base = starts[si], nE = starts[si + 1] - base;
    int viA = base + edgeI, viAn = base + (edgeI + 1) % nE;
    int vlA = base + edgeL, vlAn = base + (edgeL + 1) % nE;
    // R is the first vertex of the LOWER-indexed polygon of the pair, and si is shifted into R's
    // image -- matching updateOverlapArea. The gradient needs no shift (eq 4.3 sees only edge
    // vectors and the s parameters), so depositGrad keeps the raw positions.
    int lower = si < sj ? si : sj;
    double2 R = pos[starts[lower]];
    double2 shift = minImageShiftDev(pos[starts[si]] - R, periodic);
    double2 qI = pos[viA] + shift + sI * (pos[viAn] - pos[viA]);
    double2 qL = pos[vlA] + shift + sL * (pos[vlAn] - pos[vlA]);
    int slot = pairSlot(si, sj, numPoly);
    double w = (pairWeight != nullptr) ? pairWeight[slot] : 0.0;
    if (edgeI == edgeL && sI <= sL) {
        if (pairWeight == nullptr) atomicAdd(&pairArea[slot], hDevice(qI, qL, R));
        else if (w != 0.0) depositGrad(grad, pos[viA], pos[viAn], viA, viAn, sI, sL, w);
    } else {
        if (pairWeight == nullptr) {
            atomicAdd(&pairArea[slot],
                      hDevice(qI, pos[viAn] + shift, R) + hDevice(pos[vlA] + shift, qL, R));
        } else if (w != 0.0) {
            depositGrad(grad, pos[viA], pos[viAn], viA, viAn, sI, 1.0, w);
            depositGrad(grad, pos[vlA], pos[vlAn], vlA, vlAn, 0.0, sL, w);
        }
    }
}

// ---- Pair energy + chain-rule weight, parallel over the pair table ----------------------------
// norm_AB = targetArea[A] + targetArea[B], as in energies._assembleOverlap. Container pairs are
// skipped: the wall is penalised by the area OUTSIDE it, with its own normalizer, so it is handled
// by the container term rather than as an ordinary contact.
__global__ void pairEnergyKernel(const double* pairArea, const double* targetArea, int numPoly,
                                 int containerIndex, double kOverlap,
                                 double* pairWeight, double* energyOut) {
    int slot = blockIdx.x * blockDim.x + threadIdx.x;
    if (slot >= numPoly * numPoly) return;
    int A = slot / numPoly, B = slot % numPoly;
    pairWeight[slot] = 0.0;
    if (A >= B) return;
    if (containerIndex >= 0 && (A == containerIndex || B == containerIndex)) return;
    double a = pairArea[slot];
    if (a == 0.0) return;
    double norm = targetArea[A] + targetArea[B];
    pairWeight[slot] = 4.0 * kOverlap * a / (norm * norm);
    atomicAdd(energyOut, 2.0 * kOverlap * (a / norm) * (a / norm));
}

// ---- Stage 7: interior kernel (U_int), parallel OVER VERTICES ---------------------------------
__global__ void interiorKernel(const uint64_t* keys, const double2* tu, const int* fol,
                               const int* shapeStart, const int* shapeEnd, const double2* pos,
                               const int* starts, const int* shapeId, int numVert, bool periodic,
                               int numPoly, double* pairArea, const double* pairWeight,
                               double* grad) {
    int m = blockIdx.x * blockDim.x + threadIdx.x;
    if (m >= numVert) return;
    int rho = shapeId[m];
    int lo = shapeStart[rho], hi = shapeEnd[rho];
    if (lo > hi) return;                     // rho has no intersections
    int base = starts[rho], nE = starts[rho + 1] - base, localM = m - base;
    int gNextM = base + (localM + 1) % nE;
    for (int al = lo; al <= hi; ++al) {
        int be = fol[al]; if (be < 0) continue;
        uint64_t k = keys[al];
        int sj = keySj(k), edgeI = keyEdgeI(k);
        double sI = tu[al].x;
        int edgeL = keyEdgeJ(keys[be]); double sL = tu[be].y;
        int span;
        if (edgeI == edgeL) span = (sI <= sL) ? 0 : nE;
        else span = ((edgeL - edgeI) % nE + nE) % nE;
        int rel = ((localM - edgeI) % nE + nE) % nE;
        if (rel > 0 && rel < span) {
            int slot = pairSlot(rho, sj, numPoly);
            if (pairWeight == nullptr) {
                double2 R = pos[starts[rho < sj ? rho : sj]];
                double2 shift = minImageShiftDev(pos[starts[rho]] - R, periodic);
                atomicAdd(&pairArea[slot], hDevice(pos[m] + shift, pos[gNextM] + shift, R));
            } else if (pairWeight[slot] != 0.0) {
                depositGrad(grad, pos[m], pos[gNextM], m, gNextM, 0.0, 1.0, pairWeight[slot]);
            }
        }
    }
}

// ---- host driver -----------------------------------------------------------------------------
void sharpOverlap(const double* positions, const int* starts, int numPoly, int numVert,
                  const double* targetArea, int containerIndex, double kOverlap,
                  double* area, double* grad, int* outNumInter, bool periodic) {
    std::vector<int> shapeId(numVert);
    for (int p = 0; p < numPoly; ++p)
        for (int v = starts[p]; v < starts[p + 1]; ++v) shapeId[v] = p;

    // Grid; cellSize = max edge length so crossing edges land in adjacent cells. Free space: over the
    // packing's bounding box. Periodic: over the unit cell, with wrap-around neighborhoods.
    double xmin = 1e300, ymin = 1e300, xmax = -1e300, ymax = -1e300, maxEdge = 0.0;
    for (int v = 0; v < numVert; ++v) {
        double x = positions[2 * v], y = positions[2 * v + 1];
        xmin = fmin(xmin, x); xmax = fmax(xmax, x); ymin = fmin(ymin, y); ymax = fmax(ymax, y);
    }
    for (int p = 0; p < numPoly; ++p) {
        int base = starts[p], nE = starts[p + 1] - base;
        for (int le = 0; le < nE; ++le) {
            int v = base + le, vn = base + (le + 1) % nE;
            double dx = positions[2 * vn] - positions[2 * v], dy = positions[2 * vn + 1] - positions[2 * v + 1];
            maxEdge = fmax(maxEdge, sqrt(dx * dx + dy * dy));
        }
    }
    double cellSize = maxEdge > 0.0 ? maxEdge * (1.0 + 1e-9) : 1.0;
    int nx, ny;
    if (periodic) {
        xmin = 0.0; ymin = 0.0;
        nx = (int)floor(1.0 / cellSize);
        ny = nx;
        if (nx < 1) nx = 1;
        cellSize = 1.0 / nx;             // exact tiling of the unit cell, so the wrap is seamless
    } else {
        nx = (int)((xmax - xmin) / cellSize) + 1;
        ny = (int)((ymax - ymin) / cellSize) + 1;
        if (nx < 1) nx = 1; if (ny < 1) ny = 1;
    }
    int numCells = nx * ny;

    const int maxInter = numVert * numVert;
    double2 *dPos, *dTu, *dTu2; int *dStarts, *dShapeId, *dCounter, *dFol, *dSStart, *dSEnd;
    int *dCellId, *dCellCount, *dCellStart, *dCellFill, *dSortedEdge;
    uint64_t *dKeys, *dKeys2; double *dGrad, *dArea;
    cudaMalloc(&dPos, numVert * sizeof(double2));
    cudaMalloc(&dStarts, (numPoly + 1) * sizeof(int));
    cudaMalloc(&dShapeId, numVert * sizeof(int));
    cudaMalloc(&dKeys, maxInter * sizeof(uint64_t));
    cudaMalloc(&dKeys2, maxInter * sizeof(uint64_t));
    cudaMalloc(&dTu, maxInter * sizeof(double2));
    cudaMalloc(&dTu2, maxInter * sizeof(double2));
    cudaMalloc(&dCounter, sizeof(int));
    cudaMalloc(&dFol, maxInter * sizeof(int));
    cudaMalloc(&dSStart, numPoly * sizeof(int));
    cudaMalloc(&dSEnd, numPoly * sizeof(int));
    cudaMalloc(&dCellId, numVert * sizeof(int));
    cudaMalloc(&dCellCount, (numCells + 1) * sizeof(int));
    cudaMalloc(&dCellStart, (numCells + 1) * sizeof(int));
    cudaMalloc(&dCellFill, numCells * sizeof(int));
    cudaMalloc(&dSortedEdge, numVert * sizeof(int));
    cudaMalloc(&dGrad, 2 * numVert * sizeof(double));
    cudaMalloc(&dArea, sizeof(double));
    // The pair table is dense numPoly^2. That is the simplest correct home for a per-pair normalizer,
    // and at the sizes this reference runs it is small (1000 polygons = 8 MB); a hash would only pay
    // off well past where the O(numVert^2) intersection buffer above already dominates.
    double *dPairArea, *dPairWeight, *dTargetArea;
    cudaMalloc(&dPairArea, (size_t)numPoly * numPoly * sizeof(double));
    cudaMalloc(&dPairWeight, (size_t)numPoly * numPoly * sizeof(double));
    cudaMalloc(&dTargetArea, numPoly * sizeof(double));
    cudaMemset(dPairArea, 0, (size_t)numPoly * numPoly * sizeof(double));
    cudaMemset(dPairWeight, 0, (size_t)numPoly * numPoly * sizeof(double));
    cudaMemcpy(dTargetArea, targetArea, numPoly * sizeof(double), cudaMemcpyHostToDevice);

    cudaMemcpy(dPos, positions, numVert * sizeof(double2), cudaMemcpyHostToDevice);
    cudaMemcpy(dStarts, starts, (numPoly + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dShapeId, shapeId.data(), numVert * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(dCounter, 0, sizeof(int));
    cudaMemset(dGrad, 0, 2 * numVert * sizeof(double));
    cudaMemset(dArea, 0, sizeof(double));
    cudaMemset(dCellCount, 0, (numCells + 1) * sizeof(int));
    cudaMemset(dCellFill, 0, numCells * sizeof(int));

    int gV = (numVert + 255) / 256;
    computeCellKernel<<<gV, 256>>>(dPos, dStarts, dShapeId, numVert, xmin, ymin, cellSize, nx, ny,
                                   periodic, dCellId);
    countCellKernel<<<gV, 256>>>(dCellId, numVert, dCellCount);
    {   // cellStart = exclusive prefix sum of cellCount (length numCells+1; last entry = total edges)
        void* dTmp = nullptr; size_t nb = 0;
        cub::DeviceScan::ExclusiveSum(dTmp, nb, dCellCount, dCellStart, numCells + 1);
        cudaMalloc(&dTmp, nb);
        cub::DeviceScan::ExclusiveSum(dTmp, nb, dCellCount, dCellStart, numCells + 1);
        cudaFree(dTmp);
    }
    scatterCellKernel<<<gV, 256>>>(dCellId, numVert, dCellStart, dCellFill, dSortedEdge);
    findIntersectionsCellKernel<<<gV, 256>>>(dPos, dStarts, dShapeId, numVert, dCellId, dCellStart,
                                             dSortedEdge, nx, ny, periodic, dKeys, dTu, dCounter,
                                             maxInter);
    int M = 0;
    cudaMemcpy(&M, dCounter, sizeof(int), cudaMemcpyDeviceToHost);
    if (outNumInter) *outNumInter = M;

    for (int i = 0; i < 2 * numVert; ++i) grad[i] = 0.0;
    *area = 0.0;
    if (M > 0) {
        if (M > maxInter) { fprintf(stderr, "intersection overflow: %d > %d\n", M, maxInter); M = maxInter; }
        // CUB radix sort of (keys, tu) so same-(si,sj) runs are contiguous.
        void* dTemp = nullptr; size_t tempBytes = 0;
        cub::DeviceRadixSort::SortPairs(dTemp, tempBytes, dKeys, dKeys2, dTu, dTu2, M);
        cudaMalloc(&dTemp, tempBytes);
        cub::DeviceRadixSort::SortPairs(dTemp, tempBytes, dKeys, dKeys2, dTu, dTu2, M);
        cudaFree(dTemp);

        int gM = (M + 255) / 256;
        initRangesKernel<<<(numPoly + 255) / 256, 256>>>(dSStart, dSEnd, numPoly);
        shapeRangesKernel<<<gM, 256>>>(dKeys2, M, dSStart, dSEnd);
        followersKernel<<<gM, 256>>>(dKeys2, dTu2, M, dStarts, dFol);
        // Pass 1: per-pair areas only. Pass 2: gradients weighted by each pair's dU/da.
        exteriorKernel<<<gM, 256>>>(dKeys2, dTu2, dFol, M, dPos, dStarts, periodic, numPoly,
                                    dPairArea, nullptr, dGrad);
        interiorKernel<<<gV, 256>>>(dKeys2, dTu2, dFol, dSStart, dSEnd, dPos, dStarts, dShapeId,
                                    numVert, periodic, numPoly, dPairArea, nullptr, dGrad);
        pairEnergyKernel<<<(numPoly * numPoly + 255) / 256, 256>>>(
            dPairArea, dTargetArea, numPoly, containerIndex, kOverlap, dPairWeight, dArea);
        exteriorKernel<<<gM, 256>>>(dKeys2, dTu2, dFol, M, dPos, dStarts, periodic, numPoly,
                                    dPairArea, dPairWeight, dGrad);
        interiorKernel<<<gV, 256>>>(dKeys2, dTu2, dFol, dSStart, dSEnd, dPos, dStarts, dShapeId,
                                    numVert, periodic, numPoly, dPairArea, dPairWeight, dGrad);
        cudaMemcpy(area, dArea, sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(grad, dGrad, 2 * numVert * sizeof(double), cudaMemcpyDeviceToHost);
    }

    cudaFree(dPos); cudaFree(dStarts); cudaFree(dShapeId); cudaFree(dKeys); cudaFree(dKeys2);
    cudaFree(dTu); cudaFree(dTu2); cudaFree(dCounter); cudaFree(dFol); cudaFree(dSStart);
    cudaFree(dSEnd); cudaFree(dCellId); cudaFree(dCellCount); cudaFree(dCellStart); cudaFree(dCellFill);
    cudaFree(dSortedEdge); cudaFree(dGrad); cudaFree(dArea);
    cudaFree(dPairArea); cudaFree(dPairWeight); cudaFree(dTargetArea);
}
