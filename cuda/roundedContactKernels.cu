// Whole-packing exact-arc contact on the GPU. Host entry behind ../cudaOverlap.py.
//
// THE WORK IS SPLIT WHERE THE SCALING CHANGES. A thread owns one (ordered pair, piece of A): it walks
// that piece's spans, subdivides them at every feature switch, and scatters each sub-stretch's
// contribution into the two bodies' gradient arrays with atomics. That is the only decomposition in
// which the integral work stops scaling with the degree-of-freedom count -- the whole reason the
// gradient was derived in the BODY arrays rather than the backbone.
//
// THE CORNER MAP STAYS ON THE HOST, deliberately. Converting dE/d(body) to dE/d(loop, rho) is
// O(bodies), not O(pairs): 3n cheap evaluations per body, once per force evaluation. Moving it here
// would buy a constant on a term that is already negligible and would add a second place for the value
// and its derivative to drift apart.
//
// THE CONTAINER ARRIVES ALREADY REVERSED. The wall is the EXTERIOR region, and which side "inside"
// means is decided by winding; the host does that flip in packingBodies so the kernel never has to
// know about it. Getting it backwards is SILENT -- the energy stays smooth and its gradient stays
// self-consistent, and only an independent area check catches it.
//
// UNVERIFIED(Cam)
#include "roundedContact.cuh"

namespace {

using namespace roundedContact;

// Bounding disc of a body, ARCS INCLUDED. The farthest point of an arc is not one of its endpoints,
// so a hull over the stored points would under-estimate and the cull would drop pairs that touch.
__device__ __forceinline__ void bodyReach(const Body* body, double2* centroid, double* reach) {
    double2 middle = make_double2(0.0, 0.0);
    const int n = body->count;
    for (int k = 0; k < n; ++k) {
        middle.x += body->tail[k].x + body->head[k].x;
        middle.y += body->tail[k].y + body->head[k].y;
    }
    middle.x /= (2.0 * n); middle.y /= (2.0 * n);
    double worst = 0.0;
    for (int k = 0; k < n; ++k) {
        double2 d = sub2(body->tail[k], middle);
        worst = fmax(worst, sqrt(dot2(d, d)));
        d = sub2(body->head[k], middle);
        worst = fmax(worst, sqrt(dot2(d, d)));
        d = sub2(body->center[k], middle);
        worst = fmax(worst, sqrt(dot2(d, d)) + body->radius[k]);
    }
    *centroid = middle; *reach = worst;
}

__global__ void buildBodies(const double* positions, const int* startIndices, const double* rho,
                            int bodyCount, Body* bodies) {
    const int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= bodyCount) return;
    const int start = startIndices[b];
    const int count = startIndices[b + 1] - start;
    double2 loop[ROUNDED_MAXN];
    double radii[ROUNDED_MAXN];
    for (int k = 0; k < count && k < ROUNDED_MAXN; ++k) {
        loop[k] = make_double2(positions[2 * (start + k)], positions[2 * (start + k) + 1]);
        radii[k] = rho[start + k];
    }
    bodyFromBackbone(loop, radii, count, &bodies[b]);
}

// One thread per (ordered pair, piece of A). Pairs are enumerated as a dense bodyCount^2 grid with the
// diagonal skipped -- BOTH directions are wanted, because the law is a sum over ordered pairs.
__global__ void pairContact(const Body* bodies, int bodyCount, int containerIndex,
                            double stiffness, double wallStiffness,
                            double* energy, double* gradient, int stride) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int pieces = 2 * ROUNDED_MAXN;
    const int pair = index / pieces;
    const int piece = index % pieces;
    if (pair >= bodyCount * bodyCount) return;
    const int a = pair / bodyCount, b = pair % bodyCount;
    if (a == b) return;
    // The exterior is never culled against itself, and never paired with itself.
    if (containerIndex >= 0 && a == containerIndex && b == containerIndex) return;

    const Body* A = &bodies[a];
    const Body* B = &bodies[b];
    if (piece >= 2 * A->count) return;
    if (pieceLength(A, piece) <= 1e-15) return;

    // Broad phase. The EXTERIOR is never culled: every body is inside its bounding disc, so a reach
    // test would drop exactly the pairs the wall exists to catch.
    const bool isWall = containerIndex >= 0 && (a == containerIndex || b == containerIndex);
    if (!isWall) {
        double2 centreA, centreB;
        double reachA, reachB;
        bodyReach(A, &centreA, &reachA);
        bodyReach(B, &centreB, &reachB);
        const double2 gap = sub2(centreB, centreA);
        if (sqrt(dot2(gap, gap)) > reachA + reachB) return;
    }

    // Symmetrized: half of each ordered direction, as contactEnergy does.
    const double pairStiffness = 0.5 * stiffness * (isWall ? wallStiffness : 1.0);
    const double scale = pairStiffness / 3.0;
    double* gradA = gradient + (size_t) a * stride;
    double* gradB = gradient + (size_t) b * stride;

    double total = 0.0;
    double low = 0.0;
    while (low < 1.0) {
        const double high = nextCrossing(A, B, piece, low, 1e-9);
        if (high - low > 1e-14 && spanInside(A, B, piece, low, high)) {
            double from = low;
            while (from < high) {
                const double to = nextSwitch(A, B, piece, from, high, 1e-7);
                if (to - from > 1e-14) {
                    int kind, feature;
                    substretchWinner(A, B, piece, from, to, &kind, &feature);
                    total += substretchGradient(A, B, piece, from, to, kind, feature,
                                                scale, gradA, gradB);
                }
                if (to <= from) break;
                from = to;
            }
        }
        if (high <= low) break;
        low = high;
    }
    if (total != 0.0) atomicAdd(energy, total);
}

}  // namespace

// (energy, dE/d(body arrays)) for a whole packing. `gradient` is bodyCount blocks of
// 8 * ROUNDED_MAXN doubles, laid out as the host's BodyGradient.flat(): centre, radius, sweep, tail,
// head. Returns 0 on success, or the CUDA error code.
extern "C" int roundedContactCuda(const double* positions, int vertexCount,
                                  const int* startIndices, int bodyCount,
                                  const double* rho, int containerIndex,
                                  double stiffness, double wallStiffness,
                                  double* energyOut, double* gradientOut) {
    const int stride = 8 * ROUNDED_MAXN;
    double *dPositions = nullptr, *dRho = nullptr, *dEnergy = nullptr, *dGradient = nullptr;
    int* dStart = nullptr;
    roundedContact::Body* dBodies = nullptr;
    cudaError_t status = cudaSuccess;

    #define ROUNDED_TRY(call) do { status = (call); if (status != cudaSuccess) goto cleanup; } while (0)
    ROUNDED_TRY(cudaMalloc(&dPositions, (size_t) 2 * vertexCount * sizeof(double)));
    ROUNDED_TRY(cudaMalloc(&dRho, (size_t) vertexCount * sizeof(double)));
    ROUNDED_TRY(cudaMalloc(&dStart, (size_t) (bodyCount + 1) * sizeof(int)));
    ROUNDED_TRY(cudaMalloc(&dBodies, (size_t) bodyCount * sizeof(roundedContact::Body)));
    ROUNDED_TRY(cudaMalloc(&dEnergy, sizeof(double)));
    ROUNDED_TRY(cudaMalloc(&dGradient, (size_t) bodyCount * stride * sizeof(double)));

    ROUNDED_TRY(cudaMemcpy(dPositions, positions, (size_t) 2 * vertexCount * sizeof(double),
                           cudaMemcpyHostToDevice));
    ROUNDED_TRY(cudaMemcpy(dRho, rho, (size_t) vertexCount * sizeof(double),
                           cudaMemcpyHostToDevice));
    ROUNDED_TRY(cudaMemcpy(dStart, startIndices, (size_t) (bodyCount + 1) * sizeof(int),
                           cudaMemcpyHostToDevice));
    ROUNDED_TRY(cudaMemset(dEnergy, 0, sizeof(double)));
    ROUNDED_TRY(cudaMemset(dGradient, 0, (size_t) bodyCount * stride * sizeof(double)));

    buildBodies<<<(bodyCount + 63) / 64, 64>>>(dPositions, dStart, dRho, bodyCount, dBodies);
    ROUNDED_TRY(cudaGetLastError());

    {
        const long long threads = (long long) bodyCount * bodyCount * 2 * ROUNDED_MAXN;
        const int block = 64;
        const long long blocks = (threads + block - 1) / block;
        pairContact<<<(int) blocks, block>>>(dBodies, bodyCount, containerIndex,
                                             stiffness, wallStiffness, dEnergy, dGradient, stride);
        ROUNDED_TRY(cudaGetLastError());
    }
    ROUNDED_TRY(cudaDeviceSynchronize());

    ROUNDED_TRY(cudaMemcpy(energyOut, dEnergy, sizeof(double), cudaMemcpyDeviceToHost));
    ROUNDED_TRY(cudaMemcpy(gradientOut, dGradient, (size_t) bodyCount * stride * sizeof(double),
                           cudaMemcpyDeviceToHost));
    #undef ROUNDED_TRY

cleanup:
    cudaFree(dPositions); cudaFree(dRho); cudaFree(dStart);
    cudaFree(dBodies); cudaFree(dEnergy); cudaFree(dGradient);
    return (int) status;
}

namespace {

// TWO PHASES, BECAUSE THE AREA LAW IS A FUNCTION OF THE AREA. U = 2k (a/norm)^2, so dU/d(body) needs
// dU/da = 4k a/norm^2 -- which is not known until every span of the pair has been walked. Phase one
// measures the areas, the host turns them into weights (which is where the container's normalizer and
// stiffness live, so that convention exists in exactly one place), phase two scatters.
//
// Index layout for both: (pair, side, piece), side 0 walking dA inside B and side 1 walking dB inside A.
__device__ __forceinline__ bool unpackWork(int index, int bodyCount, int* a, int* b,
                                           int* side, int* piece) {
    const int pieces = 2 * ROUNDED_MAXN;
    *piece = index % pieces;
    const int rest = index / pieces;
    *side = rest % 2;
    const int pair = rest / 2;
    if (pair >= bodyCount * bodyCount) return false;
    *a = pair / bodyCount;
    *b = pair % bodyCount;
    return *a < *b;                 // UNORDERED pairs: the area of A and B is counted once
}

__global__ void pairAreas(const Body* bodies, int bodyCount, int containerIndex, double* areas) {
    int a, b, side, piece;
    if (!unpackWork(blockIdx.x * blockDim.x + threadIdx.x, bodyCount, &a, &b, &side, &piece)) return;
    const Body* first = side == 0 ? &bodies[a] : &bodies[b];
    const Body* second = side == 0 ? &bodies[b] : &bodies[a];
    if (piece >= 2 * first->count || pieceLength(first, piece) <= 1e-15) return;

    const bool isWall = containerIndex >= 0 && (a == containerIndex || b == containerIndex);
    if (!isWall) {
        double2 centreA, centreB;
        double reachA, reachB;
        bodyReach(first, &centreA, &reachA);
        bodyReach(second, &centreB, &reachB);
        const double2 gap = sub2(centreB, centreA);
        if (sqrt(dot2(gap, gap)) > reachA + reachB) return;
    }

    double total = 0.0;
    double low = 0.0;
    while (low < 1.0) {
        const double high = nextCrossing(first, second, piece, low, 1e-9);
        if (high - low > 1e-14 && spanInside(first, second, piece, low, high))
            total += 0.5 * greenIntegral(first, piece, low, high);
        if (high <= low) break;
        low = high;
    }
    if (total != 0.0) atomicAdd(areas + (size_t) a * bodyCount + b, total);
}

__global__ void pairAreaGradients(const Body* bodies, int bodyCount, int containerIndex,
                                  const double* weights, double* gradient, int stride) {
    int a, b, side, piece;
    if (!unpackWork(blockIdx.x * blockDim.x + threadIdx.x, bodyCount, &a, &b, &side, &piece)) return;
    const double weight = weights[(size_t) a * bodyCount + b];
    if (weight == 0.0) return;
    const Body* first = side == 0 ? &bodies[a] : &bodies[b];
    const Body* second = side == 0 ? &bodies[b] : &bodies[a];
    if (piece >= 2 * first->count || pieceLength(first, piece) <= 1e-15) return;
    double* target = gradient + (size_t) (side == 0 ? a : b) * stride;

    double low = 0.0;
    while (low < 1.0) {
        const double high = nextCrossing(first, second, piece, low, 1e-9);
        if (high - low > 1e-14 && spanInside(first, second, piece, low, high))
            spanAreaGradient(first, piece, low, high, weight, target);
        if (high <= low) break;
        low = high;
    }
}

}  // namespace

// Phase one: per-pair overlap AREA, as a bodyCount x bodyCount upper-triangular matrix.
extern "C" int roundedAreaCuda(const double* positions, int vertexCount,
                               const int* startIndices, int bodyCount,
                               const double* rho, int containerIndex, double* areasOut) {
    double *dPositions = nullptr, *dRho = nullptr, *dAreas = nullptr;
    int* dStart = nullptr;
    roundedContact::Body* dBodies = nullptr;
    cudaError_t status = cudaSuccess;
    #define ROUNDED_TRY(call) do { status = (call); if (status != cudaSuccess) goto cleanup; } while (0)
    ROUNDED_TRY(cudaMalloc(&dPositions, (size_t) 2 * vertexCount * sizeof(double)));
    ROUNDED_TRY(cudaMalloc(&dRho, (size_t) vertexCount * sizeof(double)));
    ROUNDED_TRY(cudaMalloc(&dStart, (size_t) (bodyCount + 1) * sizeof(int)));
    ROUNDED_TRY(cudaMalloc(&dBodies, (size_t) bodyCount * sizeof(roundedContact::Body)));
    ROUNDED_TRY(cudaMalloc(&dAreas, (size_t) bodyCount * bodyCount * sizeof(double)));
    ROUNDED_TRY(cudaMemcpy(dPositions, positions, (size_t) 2 * vertexCount * sizeof(double),
                           cudaMemcpyHostToDevice));
    ROUNDED_TRY(cudaMemcpy(dRho, rho, (size_t) vertexCount * sizeof(double), cudaMemcpyHostToDevice));
    ROUNDED_TRY(cudaMemcpy(dStart, startIndices, (size_t) (bodyCount + 1) * sizeof(int),
                           cudaMemcpyHostToDevice));
    ROUNDED_TRY(cudaMemset(dAreas, 0, (size_t) bodyCount * bodyCount * sizeof(double)));
    buildBodies<<<(bodyCount + 63) / 64, 64>>>(dPositions, dStart, dRho, bodyCount, dBodies);
    ROUNDED_TRY(cudaGetLastError());
    {
        const long long threads = (long long) bodyCount * bodyCount * 2 * 2 * ROUNDED_MAXN;
        pairAreas<<<(int) ((threads + 63) / 64), 64>>>(dBodies, bodyCount, containerIndex, dAreas);
        ROUNDED_TRY(cudaGetLastError());
    }
    ROUNDED_TRY(cudaDeviceSynchronize());
    ROUNDED_TRY(cudaMemcpy(areasOut, dAreas, (size_t) bodyCount * bodyCount * sizeof(double),
                           cudaMemcpyDeviceToHost));
    #undef ROUNDED_TRY
cleanup:
    cudaFree(dPositions); cudaFree(dRho); cudaFree(dStart); cudaFree(dBodies); cudaFree(dAreas);
    return (int) status;
}

// Phase two: scatter `weights[a][b] * d(area_ab)/d(body arrays)` into per-body gradient blocks.
extern "C" int roundedAreaGradientCuda(const double* positions, int vertexCount,
                                       const int* startIndices, int bodyCount,
                                       const double* rho, int containerIndex,
                                       const double* weights, double* gradientOut) {
    const int stride = 8 * ROUNDED_MAXN;
    double *dPositions = nullptr, *dRho = nullptr, *dWeights = nullptr, *dGradient = nullptr;
    int* dStart = nullptr;
    roundedContact::Body* dBodies = nullptr;
    cudaError_t status = cudaSuccess;
    #define ROUNDED_TRY(call) do { status = (call); if (status != cudaSuccess) goto cleanup; } while (0)
    ROUNDED_TRY(cudaMalloc(&dPositions, (size_t) 2 * vertexCount * sizeof(double)));
    ROUNDED_TRY(cudaMalloc(&dRho, (size_t) vertexCount * sizeof(double)));
    ROUNDED_TRY(cudaMalloc(&dStart, (size_t) (bodyCount + 1) * sizeof(int)));
    ROUNDED_TRY(cudaMalloc(&dBodies, (size_t) bodyCount * sizeof(roundedContact::Body)));
    ROUNDED_TRY(cudaMalloc(&dWeights, (size_t) bodyCount * bodyCount * sizeof(double)));
    ROUNDED_TRY(cudaMalloc(&dGradient, (size_t) bodyCount * stride * sizeof(double)));
    ROUNDED_TRY(cudaMemcpy(dPositions, positions, (size_t) 2 * vertexCount * sizeof(double),
                           cudaMemcpyHostToDevice));
    ROUNDED_TRY(cudaMemcpy(dRho, rho, (size_t) vertexCount * sizeof(double), cudaMemcpyHostToDevice));
    ROUNDED_TRY(cudaMemcpy(dStart, startIndices, (size_t) (bodyCount + 1) * sizeof(int),
                           cudaMemcpyHostToDevice));
    ROUNDED_TRY(cudaMemcpy(dWeights, weights, (size_t) bodyCount * bodyCount * sizeof(double),
                           cudaMemcpyHostToDevice));
    ROUNDED_TRY(cudaMemset(dGradient, 0, (size_t) bodyCount * stride * sizeof(double)));
    buildBodies<<<(bodyCount + 63) / 64, 64>>>(dPositions, dStart, dRho, bodyCount, dBodies);
    ROUNDED_TRY(cudaGetLastError());
    {
        const long long threads = (long long) bodyCount * bodyCount * 2 * 2 * ROUNDED_MAXN;
        pairAreaGradients<<<(int) ((threads + 63) / 64), 64>>>(dBodies, bodyCount, containerIndex,
                                                              dWeights, dGradient, stride);
        ROUNDED_TRY(cudaGetLastError());
    }
    ROUNDED_TRY(cudaDeviceSynchronize());
    ROUNDED_TRY(cudaMemcpy(gradientOut, dGradient, (size_t) bodyCount * stride * sizeof(double),
                           cudaMemcpyDeviceToHost));
    #undef ROUNDED_TRY
cleanup:
    cudaFree(dPositions); cudaFree(dRho); cudaFree(dStart);
    cudaFree(dBodies); cudaFree(dWeights); cudaFree(dGradient);
    return (int) status;
}
