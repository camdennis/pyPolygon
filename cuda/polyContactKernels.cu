// Polygon-polygon contact on the GPU -- the port of polyContact.pairGradient, assembled over a packing.
//
// Parallel structure: a CULL kernel compacts surviving ordered body pairs into a dense work list, then
// ONE BLOCK PER WORK ITEM whose threads STRIDE over the edges of A. B is staged in shared memory with
// its periodic shift already folded in, so the inner loops never touch global memory.
//
// WHY THIS LAW SUITS A GPU BETTER THAN softDepth DID. There the depth was a softmin over ALL of B's
// half-planes, so every quadrature node's gradient touched every vertex of B -- nB atomics per node,
// which forced a shared-memory accumulator and a block-per-pair decomposition just to survive. Here the
// nearest feature is a SINGLE edge or vertex, so a sub-stretch's gradient touches at most two vertices
// of A and two of B. Atomics are cheap and no accumulator is needed.
//
// NOTHING IS STORED PER THREAD. Sorted crossings and envelope breakpoints are both walked by repeated
// minimum-search rather than materialized, so a thread needs no local arrays and never spills.
//
// Built into libplummer.so; called from ../cudaOverlap.py via ctypes. double/double2, sm_75.
#include "polyContact.cuh"

// UNVERIFIED(Cam)

// Largest vertices-per-polygon this build accepts. Caps SHARED MEMORY, not strides -- vertex counts are
// read from starts[] at runtime. Exceeding it is REPORTED by the driver, never silently truncated: a
// hard-coded cap in this codebase once dropped gradients past the 24th vertex of a pair while the
// ENERGY stayed correct, so the failure survived the whole test suite (see cuda/plummerDriver.cu).
#define POLYCONTACT_MAXN 64
#define POLYCONTACT_BLOCK 64

struct ContactPair {
    int boundaryBody;
    int loopBody;
    double sx, sy;
};

// ---------------------------------------------------------------------------
// Kernel 0 -- CULL. One thread per unordered body pair; a survivor emits BOTH ordered interactions,
// since the law is E = 1/2 sum over ORDERED pairs and each direction integrates a different boundary.
// ---------------------------------------------------------------------------
__global__ void contactCullKernel(const double2* centroids, const double* radii, int bodyCount,
                                  double boxSize, bool periodic, ContactPair* work, int* workCount) {
    long t = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= (long)bodyCount * bodyCount) return;
    int first = (int)(t / bodyCount), second = (int)(t % bodyCount);
    if (second <= first) return;

    double sx = 0.0, sy = 0.0;
    double dx = centroids[second].x - centroids[first].x;
    double dy = centroids[second].y - centroids[first].y;
    if (periodic) {
        sx = -boxSize * nearbyint(dx / boxSize);
        sy = -boxSize * nearbyint(dy / boxSize);
    }
    if (sqrt((dx + sx) * (dx + sx) + (dy + sy) * (dy + sy)) >= radii[first] + radii[second]) return;

    int slot = atomicAdd(workCount, 2);
    ContactPair forward;
    forward.boundaryBody = first; forward.loopBody = second; forward.sx = sx; forward.sy = sy;
    ContactPair reverse;
    reverse.boundaryBody = second; reverse.loopBody = first; reverse.sx = -sx; reverse.sy = -sy;
    work[slot] = forward;
    work[slot + 1] = reverse;
}

// The smallest crossing of edge (base, dir) with dB strictly after `at`, clamped to 1.
__device__ __forceinline__ double nextCrossing(const polyContact::LoopFrame* frames, int count,
                                               double2 base, double2 dir, double at) {
    double best = 1.0;
    for (int j = 0; j < count; ++j) {
        double2 a = frames[j].base;
        double fx = frames[j].tangent.x * frames[j].length;
        double fy = frames[j].tangent.y * frames[j].length;
        double determinant = dir.x * fy - dir.y * fx;
        if (fabs(determinant) < 1e-14) continue;
        double rx = a.x - base.x, ry = a.y - base.y;
        double alongA = (rx * fy - ry * fx) / determinant;
        double alongB = (rx * dir.y - ry * dir.x) / determinant;
        if (alongA > 1e-12 && alongA < 1.0 - 1e-12 && alongB >= -1e-12 && alongB <= 1.0 + 1e-12)
            if (alongA > at + 1e-12 && alongA < best) best = alongA;
    }
    return best;
}

// ---------------------------------------------------------------------------
// Kernel 1 -- one BLOCK per ordered pair; threads stride over the boundary body's edges. Each thread
// walks its edge's crossings, keeps the stretches whose midpoint is inside B, walks each stretch's
// nearest-feature envelope, and accumulates energy and gradient in closed form.
// ---------------------------------------------------------------------------
__global__ void contactKernel(const double2* positions, const int* starts, const ContactPair* work,
                              const int* workCount, double stiffness,
                              int exterior, double wallStiffness,
                              double* gradient, double* energyOut) {
    int item = blockIdx.x;
    if (item >= *workCount) return;
    ContactPair job = work[item];

    // THE WALL MAY BE STIFFER THAN THE BODIES, and this is the whole of what that costs on device.
    // Both energy and gradient are linear in the stiffness, so a per-pair multiplier is exact -- there
    // is no second code path, just a different k for work items that touch the exterior body.
    //
    // It is needed because body contact and wall penetration are ALTERNATIVES: a stressed packing
    // relieves itself through whichever is softer, and escaping lowers the confinement for everyone
    // while overlapping a neighbour relieves nothing globally. At equal stiffness the wall loses
    // outright -- measured, a packing held at its target excess carried 100.00% of its contact energy
    // in wall penetration, 1.83e-19 between bodies, and a pair overlap of EXACTLY zero.
    if (exterior >= 0 && (job.boundaryBody == exterior || job.loopBody == exterior))
        stiffness *= wallStiffness;

    int baseA = starts[job.boundaryBody], countA = starts[job.boundaryBody + 1] - baseA;
    int baseB = starts[job.loopBody], countB = starts[job.loopBody + 1] - baseB;

    // B's edge frames staged ONCE, with the periodic shift folded in. Recomputing them inside
    // candidateQuadratic (~2 x 2M calls per breakpoint, each a normalize and a sqrt) was 37 ms of a
    // 39 ms evaluation at N=32, n=32.
    __shared__ polyContact::LoopFrame frameB[POLYCONTACT_MAXN];
    for (int j = threadIdx.x; j < countB; j += blockDim.x) {
        double2 a = positions[baseB + j];
        double2 b = positions[baseB + (j + 1) % countB];
        double ex = b.x - a.x, ey = b.y - a.y;
        double len = sqrt(ex * ex + ey * ey);
        frameB[j].base = make_double2(a.x + job.sx, a.y + job.sy);
        frameB[j].tangent = make_double2(ex / len, ey / len);
        frameB[j].normal = make_double2(ey / len, -ex / len);
        frameB[j].length = len;
    }
    __syncthreads();

    double blockEnergy = 0.0;

    for (int edge = threadIdx.x; edge < countA; edge += blockDim.x) {
        double2 base = positions[baseA + edge];
        double2 head = positions[baseA + (edge + 1) % countA];
        double2 dir = make_double2(head.x - base.x, head.y - base.y);
        double length = sqrt(dir.x * dir.x + dir.y * dir.y);
        double2 tangent = make_double2(dir.x / length, dir.y / length);
        int firstSlot = baseA + edge, secondSlot = baseA + (edge + 1) % countA;

        double at = 0.0;
        for (int guard = 0; guard <= 2 * countB + 2 && at < 1.0 - 1e-15; ++guard) {
            double stretchEnd = nextCrossing(frameB, countB, base, dir, at);
            double middle = 0.5 * (at + stretchEnd);
            double2 probe = make_double2(base.x + middle * dir.x, base.y + middle * dir.y);
            int kind, index;
            double distance;
            bool inside;
            polyContact::nearestFeature(probe, frameB, countB, &kind, &index, &distance, &inside);
            if (!inside) { at = stretchEnd; continue; }

            // --- a SPAN: walk its nearest-feature envelope --------------------------------------
            double spanIntegral = 0.0;
            double walk = at;
            for (int step = 0; step <= 4 * countB + 4 && walk < stretchEnd - 1e-15; ++step) {
                double pieceEnd = polyContact::nextBreakpoint(frameB, countB, base, dir,
                                                              walk, stretchEnd);
                if (pieceEnd - walk < 1e-14) { walk = pieceEnd; continue; }
                double low = walk, high = pieceEnd;
                double centre = 0.5 * (low + high);
                double2 at2 = make_double2(base.x + centre * dir.x, base.y + centre * dir.y);
                // Re-identified at the MIDPOINT, never taken from the envelope walk: the walk
                // identifies its winner at t + 1e-13, which does not separate two candidates that
                // cross shallowly (measured on the host: reported E3 where the truth was E1,
                // inflating that pair's energy eightfold).
                polyContact::nearestFeature(at2, frameB, countB, &kind, &index, &distance, &inside);

                double m0 = high - low;
                double m1 = (high * high - low * low) / 2.0;
                double m2 = (high * high * high - low * low * low) / 3.0;
                double m3 = (high * high * high * high - low * low * low * low) / 4.0;

                if (kind == 0) {
                    double2 vertexBase = frameB[index].base;
                    double2 edgeTangent = frameB[index].tangent;
                    double2 normal = frameB[index].normal;
                    double edgeLength = frameB[index].length;
                    double offX = vertexBase.x - base.x, offY = vertexBase.y - base.y;
                    double alpha = normal.x * offX + normal.y * offY;
                    double slope = normal.x * dir.x + normal.y * dir.y;
                    // POLYNOMIAL in the moments, never the 1/m antiderivative: m vanishes exactly for
                    // face-parallel contact, the dominant configuration, and cancels near it.
                    spanIntegral += stiffness * (alpha * alpha * alpha * m0
                                                 - 3.0 * alpha * alpha * slope * m1
                                                 + 3.0 * alpha * slope * slope * m2
                                                 - slope * slope * slope * m3) / 3.0;
                    double first = stiffness * (alpha * alpha * m0 - 2.0 * alpha * slope * m1
                                                + slope * slope * m2);
                    double second = stiffness * (alpha * alpha * m1 - 2.0 * alpha * slope * m2
                                                 + slope * slope * m3);
                    atomicAdd(&gradient[2 * firstSlot],      -length * (first - second) * normal.x);
                    atomicAdd(&gradient[2 * firstSlot + 1],  -length * (first - second) * normal.y);
                    atomicAdd(&gradient[2 * secondSlot],     -length * second * normal.x);
                    atomicAdd(&gradient[2 * secondSlot + 1], -length * second * normal.y);

                    double footStart = (edgeTangent.x * (base.x - vertexBase.x)
                                        + edgeTangent.y * (base.y - vertexBase.y)) / edgeLength;
                    double footRate = (edgeTangent.x * dir.x + edgeTangent.y * dir.y) / edgeLength;
                    double shared = footStart * first + footRate * second;
                    int slotB = baseB + index, slotBnext = baseB + (index + 1) % countB;
                    atomicAdd(&gradient[2 * slotB],          length * (first - shared) * normal.x);
                    atomicAdd(&gradient[2 * slotB + 1],      length * (first - shared) * normal.y);
                    atomicAdd(&gradient[2 * slotBnext],      length * shared * normal.x);
                    atomicAdd(&gradient[2 * slotBnext + 1],  length * shared * normal.y);
                } else {
                    double2 v = frameB[index].base;
                    double along = tangent.x * (v.x - base.x) + tangent.y * (v.y - base.y);
                    double foot = along / length;
                    // The FULL perpendicular vector, undivided -- using its direction gives ~1e-3
                    // relative errors that conservation does not catch.
                    double perpX = (base.x - v.x) + along * tangent.x;
                    double perpY = (base.y - v.y) + along * tangent.y;
                    double radius = fmax(sqrt(perpX * perpX + perpY * perpY), 1e-14);
                    double wLow = length * (low - foot), wHigh = length * (high - foot);
                    double sLow = sqrt(wLow * wLow + radius * radius);
                    double sHigh = sqrt(wHigh * wHigh + radius * radius);
                    // The /3 matches the edge branch. Omitting it triples every vertex-nearest
                    // sub-stretch, and a convex-only test suite cannot see that: those stretches come
                    // only from REFLEX vertices of the loop body.
                    spanIntegral += stiffness * (polyContact::vertexAntiderivative(wHigh, radius)
                                                 - polyContact::vertexAntiderivative(wLow, radius))
                                    / (3.0 * length);
                    double dJ0 = ((wHigh * sHigh + radius * radius * asinh(wHigh / radius))
                                  - (wLow * sLow + radius * radius * asinh(wLow / radius))) / 2.0;
                    double dJ1 = (sHigh * sHigh * sHigh - sLow * sLow * sLow) / 3.0;
                    double dJ2 = (wHigh * sHigh * sHigh * sHigh / 4.0
                                  - radius * radius * wHigh * sHigh / 8.0
                                  - radius * radius * radius * radius * asinh(wHigh / radius) / 8.0)
                               - (wLow * sLow * sLow * sLow / 4.0
                                  - radius * radius * wLow * sLow / 8.0
                                  - radius * radius * radius * radius * asinh(wLow / radius) / 8.0);
                    double v0x = stiffness * (tangent.x * dJ1 + perpX * dJ0) / length;
                    double v0y = stiffness * (tangent.y * dJ1 + perpY * dJ0) / length;
                    double v1x = stiffness * (foot * (tangent.x * dJ1 + perpX * dJ0)
                                              + (tangent.x * dJ2 + perpX * dJ1) / length) / length;
                    double v1y = stiffness * (foot * (tangent.y * dJ1 + perpY * dJ0)
                                              + (tangent.y * dJ2 + perpY * dJ1) / length) / length;
                    atomicAdd(&gradient[2 * firstSlot],      length * (v0x - v1x));
                    atomicAdd(&gradient[2 * firstSlot + 1],  length * (v0y - v1y));
                    atomicAdd(&gradient[2 * secondSlot],     length * v1x);
                    atomicAdd(&gradient[2 * secondSlot + 1], length * v1y);
                    int slotB = baseB + index;
                    atomicAdd(&gradient[2 * slotB],     -length * v0x);
                    atomicAdd(&gradient[2 * slotB + 1], -length * v0y);
                }
                walk = pieceEnd;
            }

            // MEASURE term, per span: dL/dv times that span's whole integral.
            atomicAdd(&gradient[2 * secondSlot],      spanIntegral * tangent.x);
            atomicAdd(&gradient[2 * secondSlot + 1],  spanIntegral * tangent.y);
            atomicAdd(&gradient[2 * firstSlot],      -spanIntegral * tangent.x);
            atomicAdd(&gradient[2 * firstSlot + 1],  -spanIntegral * tangent.y);
            blockEnergy += length * spanIntegral;
            at = stretchEnd;
        }
    }
    if (blockEnergy != 0.0) atomicAdd(energyOut, blockEnergy);
}

// ---- C API. Persistent buffers, sized on the first call and reused. ----
static double2* g_pcPos = nullptr; static int* g_pcStarts = nullptr;
static double2* g_pcCent = nullptr; static double* g_pcRadii = nullptr;
static ContactPair* g_pcWork = nullptr; static int* g_pcWorkCount = nullptr;
static double* g_pcGrad = nullptr; static double* g_pcEnergy = nullptr;
static int g_pcVerts = 0, g_pcBodies = 0;

// Returns 0 on success, or the offending vertex count when a body exceeds POLYCONTACT_MAXN.
extern "C" int polyContactCuda(const double* positions, const int* starts, int bodyCount, int vertexCount,
                               const double* centroids, const double* radii,
                               double stiffness, int exterior, double wallStiffness,
                               double boxSize, int periodic,
                               double* energyOut, double* gradientOut) {
    *energyOut = 0.0;
    for (int i = 0; i < 2 * vertexCount; ++i) gradientOut[i] = 0.0;

    int maxCount = 0;
    for (int b = 0; b < bodyCount; ++b) {
        int n = starts[b + 1] - starts[b];
        if (n > maxCount) maxCount = n;
    }
    if (maxCount > POLYCONTACT_MAXN) return maxCount;
    if (bodyCount < 2) return 0;

    int maxWork = bodyCount * (bodyCount - 1);
    if (vertexCount != g_pcVerts || bodyCount != g_pcBodies) {
        if (g_pcPos) {
            cudaFree(g_pcPos); cudaFree(g_pcStarts); cudaFree(g_pcCent); cudaFree(g_pcRadii);
            cudaFree(g_pcWork); cudaFree(g_pcWorkCount); cudaFree(g_pcGrad); cudaFree(g_pcEnergy);
        }
        cudaMalloc(&g_pcPos, vertexCount * sizeof(double2));
        cudaMalloc(&g_pcStarts, (bodyCount + 1) * sizeof(int));
        cudaMalloc(&g_pcCent, bodyCount * sizeof(double2));
        cudaMalloc(&g_pcRadii, bodyCount * sizeof(double));
        cudaMalloc(&g_pcWork, (size_t)maxWork * sizeof(ContactPair));
        cudaMalloc(&g_pcWorkCount, sizeof(int));
        cudaMalloc(&g_pcGrad, 2 * vertexCount * sizeof(double));
        cudaMalloc(&g_pcEnergy, sizeof(double));
        g_pcVerts = vertexCount; g_pcBodies = bodyCount;
    }
    cudaMemcpy(g_pcPos, positions, vertexCount * sizeof(double2), cudaMemcpyHostToDevice);
    cudaMemcpy(g_pcStarts, starts, (bodyCount + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(g_pcCent, centroids, bodyCount * sizeof(double2), cudaMemcpyHostToDevice);
    cudaMemcpy(g_pcRadii, radii, bodyCount * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemset(g_pcWorkCount, 0, sizeof(int));
    cudaMemset(g_pcGrad, 0, 2 * vertexCount * sizeof(double));
    cudaMemset(g_pcEnergy, 0, sizeof(double));

    int cullThreads = 128;
    long slots = (long)bodyCount * bodyCount;
    contactCullKernel<<<(int)((slots + cullThreads - 1) / cullThreads), cullThreads>>>(
        g_pcCent, g_pcRadii, bodyCount, boxSize, periodic != 0, g_pcWork, g_pcWorkCount);
    contactKernel<<<maxWork, POLYCONTACT_BLOCK>>>(
        g_pcPos, g_pcStarts, g_pcWork, g_pcWorkCount, stiffness, exterior, wallStiffness,
        g_pcGrad, g_pcEnergy);

    cudaMemcpy(energyOut, g_pcEnergy, sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(gradientOut, g_pcGrad, 2 * vertexCount * sizeof(double), cudaMemcpyDeviceToHost);

    // THE LAW CARRIES A FACTOR 1/2: E = 1/2 sum over ORDERED pairs. The cull emits both orders and the
    // kernel integrates each, so the raw sum is E_AB + E_BA and has to be halved. Omitting this gave
    // EXACTLY twice the host's energy and gradient -- a clean factor of two, which is easy to mistake
    // for a stiffness convention rather than a missing term.
    *energyOut *= 0.5;
    for (int i = 0; i < 2 * vertexCount; ++i) gradientOut[i] *= 0.5;
    return 0;
}
