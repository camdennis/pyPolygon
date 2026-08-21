// Device math for the exact-arc contact law, checked against the numpy reference.
//
// Stages one and two of the CUDA port: the ROOT SOLVER, the CORNER FRAME, the DISTANCE FIELD, and
// the SPAN WALK. Each is
// compared against values printed by ../roundedContact.py for the same inputs, because the whole point
// of the port is that it computes the same thing -- a kernel that agrees with itself proves nothing.
//
// UNVERIFIED(Cam)
#include <stdio.h>
#include <stdlib.h>
#include "roundedContact.cuh"

using namespace roundedContact;

// One backbone square and its radii, matching the case the host prints.
__constant__ double2 kLoop[4];
__constant__ double  kRho[4];

__global__ void frameKernel(double2* center, double* radius, double* sweep,
                            double2* tail, double2* head) {
    Body body;
    bodyFromBackbone(kLoop, kRho, 4, &body);
    for (int k = 0; k < 4; ++k) {
        center[k] = body.center[k];
        radius[k] = body.radius[k];
        sweep[k]  = body.sweep[k];
        tail[k]   = body.tail[k];
        head[k]   = body.head[k];
    }
}

__global__ void distanceKernel(const double2* points, int count, double* out, int* kind, int* feature) {
    Body body;
    bodyFromBackbone(kLoop, kRho, 4, &body);
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    // The SIGNED distance is what the contact law integrates, so that is what is compared.
    out[i] = signedDistance(&body, points[i], &kind[i], &feature[i]);
}

__global__ void pieceKernel(const double* parameters, int count, double2* out) {
    Body body;
    bodyFromBackbone(kLoop, kRho, 4, &body);
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    out[i] = evaluatePiece(&body, i % 8, parameters[i]);
}

// Spans of dA inside B, walked exactly as a kernel would. Two bodies, both from constant memory.
__constant__ double2 kLoopB[4];
__constant__ double  kRhoB[4];

// Sub-stretches: the span walk subdivided again at every feature switch, winner re-identified at each
// midpoint. This is the full `substretches` pipeline as a kernel would run it.
__global__ void substretchKernel(int* pieceOut, double* lowOut, double* highOut,
                                 int* kindOut, int* featureOut, int* countOut, int limit) {
    Body A, B;
    bodyFromBackbone(kLoop, kRho, 4, &A);
    bodyFromBackbone(kLoopB, kRhoB, 4, &B);
    int written = 0;
    for (int p = 0; p < 2 * A.count; ++p) {
        if (pieceLength(&A, p) <= 1e-15) continue;
        double low = 0.0;
        while (low < 1.0) {
            const double high = nextCrossing(&A, &B, p, low, 1e-9);
            if (high - low > 1e-14 && spanInside(&A, &B, p, low, high)) {
                double a = low;
                while (a < high) {
                    const double b = nextSwitch(&A, &B, p, a, high, 1e-7);
                    if (b - a > 1e-14 && written < limit) {
                        int kind, feature;
                        substretchWinner(&A, &B, p, a, b, &kind, &feature);
                        pieceOut[written] = p; lowOut[written] = a; highOut[written] = b;
                        kindOut[written] = kind; featureOut[written] = feature; ++written;
                    }
                    if (b <= a) break;
                    a = b;
                }
            }
            if (high <= low) break;
            low = high;
        }
    }
    *countOut = written;
}

// The whole ordered-pair pipeline: spans, sub-stretches, energy and dE/d(body arrays).
__global__ void pairKernel(double* energyOut, double* gradA, double* gradB, double scale) {
    Body A, B;
    bodyFromBackbone(kLoop, kRho, 4, &A);
    bodyFromBackbone(kLoopB, kRhoB, 4, &B);
    double total = 0.0;
    for (int p = 0; p < 2 * A.count; ++p) {
        if (pieceLength(&A, p) <= 1e-15) continue;
        double low = 0.0;
        while (low < 1.0) {
            const double high = nextCrossing(&A, &B, p, low, 1e-9);
            if (high - low > 1e-14 && spanInside(&A, &B, p, low, high)) {
                double a = low;
                while (a < high) {
                    const double b = nextSwitch(&A, &B, p, a, high, 1e-7);
                    if (b - a > 1e-14) {
                        int kind, feature;
                        substretchWinner(&A, &B, p, a, b, &kind, &feature);
                        total += substretchGradient(&A, &B, p, a, b, kind, feature,
                                                    scale, gradA, gradB);
                    }
                    if (b <= a) break;
                    a = b;
                }
            }
            if (high <= low) break;
            low = high;
        }
    }
    *energyOut = total;
}

__global__ void spanKernel(int* pieceOut, double* lowOut, double* highOut, int* countOut, int limit) {
    Body A, B;
    bodyFromBackbone(kLoop, kRho, 4, &A);
    bodyFromBackbone(kLoopB, kRhoB, 4, &B);
    int written = 0;
    for (int p = 0; p < 2 * A.count; ++p) {
        if (pieceLength(&A, p) <= 1e-15) continue;
        double low = 0.0;
        while (low < 1.0) {
            const double high = nextCrossing(&A, &B, p, low, 1e-9);
            if (high - low > 1e-14 && spanInside(&A, &B, p, low, high) && written < limit) {
                pieceOut[written] = p; lowOut[written] = low; highOut[written] = high; ++written;
            }
            if (high <= low) break;
            low = high;
        }
    }
    *countOut = written;
}

__global__ void rootKernel(const double* coefficients, int count, double* out, int* found) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    double roots[4];
    const int m = quarticRoots(coefficients + 5 * i, 4, roots);
    found[i] = m;
    for (int k = 0; k < 4; ++k) out[4 * i + k] = (k < m) ? roots[k] : 0.0;
}

__global__ void trigKernel(const double* coefficients, int count, double* out, int* found) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const double* e = coefficients + 5 * i;
    double roots[5];
    const int m = solveTrig(e[0], e[1], e[2], e[3], e[4], roots);
    found[i] = m;
    for (int k = 0; k < 5; ++k) out[5 * i + k] = (k < m) ? roots[k] : 0.0;
}

#define CHECK(call) do { cudaError_t status = (call); if (status != cudaSuccess) { \
    printf("CUDA error %d at line %d: %s\n", (int) status, __LINE__, cudaGetErrorString(status)); \
    return 1; } } while (0)

int main(int argc, char** argv) {
    // The square and radii the host prints for. Kept identical on both sides on purpose.
    double2 loop[4] = { {0.0, 0.0}, {1.0, 0.0}, {1.0, 1.0}, {0.0, 1.0} };
    double  rho[4]  = { 0.30, 0.15, 0.42, 0.22 };
    CHECK(cudaMemcpyToSymbol(kLoop, loop, sizeof(loop)));
    CHECK(cudaMemcpyToSymbol(kRho, rho, sizeof(rho)));

    double2 *dCenter, *dTail, *dHead;
    double *dRadius, *dSweep;
    CHECK(cudaMalloc(&dCenter, 4 * sizeof(double2)));
    CHECK(cudaMalloc(&dTail,   4 * sizeof(double2)));
    CHECK(cudaMalloc(&dHead,   4 * sizeof(double2)));
    CHECK(cudaMalloc(&dRadius, 4 * sizeof(double)));
    CHECK(cudaMalloc(&dSweep,  4 * sizeof(double)));
    frameKernel<<<1, 1>>>(dCenter, dRadius, dSweep, dTail, dHead);
    CHECK(cudaDeviceSynchronize());

    double2 center[4], tail[4], head[4];
    double radius[4], sweep[4];
    CHECK(cudaMemcpy(center, dCenter, sizeof(center), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(tail,   dTail,   sizeof(tail),   cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(head,   dHead,   sizeof(head),   cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(radius, dRadius, sizeof(radius), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(sweep,  dSweep,  sizeof(sweep),  cudaMemcpyDeviceToHost));

    printf("FRAME\n");
    for (int k = 0; k < 4; ++k)
        printf("%d %.17g %.17g %.17g %.17g %.17g %.17g %.17g %.17g\n", k,
               center[k].x, center[k].y, radius[k], sweep[k],
               tail[k].x, tail[k].y, head[k].x, head[k].y);

    // Distance field on a grid covering the square and its surroundings.
    const int side = 11;
    const int total = side * side;
    double2 host[121];
    for (int i = 0; i < side; ++i)
        for (int j = 0; j < side; ++j)
            host[i * side + j] = make_double2(-0.25 + 1.5 * j / (side - 1.0),
                                              -0.25 + 1.5 * i / (side - 1.0));
    double2* dPoints;
    double* dDistance;
    int *dKind, *dFeature;
    CHECK(cudaMalloc(&dPoints, total * sizeof(double2)));
    CHECK(cudaMalloc(&dDistance, total * sizeof(double)));
    CHECK(cudaMalloc(&dKind, total * sizeof(int)));
    CHECK(cudaMalloc(&dFeature, total * sizeof(int)));
    CHECK(cudaMemcpy(dPoints, host, total * sizeof(double2), cudaMemcpyHostToDevice));
    distanceKernel<<<(total + 63) / 64, 64>>>(dPoints, total, dDistance, dKind, dFeature);
    CHECK(cudaDeviceSynchronize());
    double distance[121];
    int kind[121], feature[121];
    CHECK(cudaMemcpy(distance, dDistance, sizeof(distance), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(kind, dKind, sizeof(kind), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(feature, dFeature, sizeof(feature), cudaMemcpyDeviceToHost));
    printf("DISTANCE\n");
    for (int i = 0; i < total; ++i)
        printf("%.17g %.17g %.17g %d %d\n", host[i].x, host[i].y, distance[i], kind[i], feature[i]);

    // Pieces, one parameter each, cycling through all eight.
    double parameters[24];
    for (int i = 0; i < 24; ++i) parameters[i] = (i / 8) * 0.5 + 0.17;
    double* dParameters;
    double2* dPoint;
    CHECK(cudaMalloc(&dParameters, sizeof(parameters)));
    CHECK(cudaMalloc(&dPoint, 24 * sizeof(double2)));
    CHECK(cudaMemcpy(dParameters, parameters, sizeof(parameters), cudaMemcpyHostToDevice));
    pieceKernel<<<1, 32>>>(dParameters, 24, dPoint);
    CHECK(cudaDeviceSynchronize());
    double2 point[24];
    CHECK(cudaMemcpy(point, dPoint, sizeof(point), cudaMemcpyDeviceToHost));
    printf("PIECE\n");
    for (int i = 0; i < 24; ++i)
        printf("%d %.17g %.17g %.17g\n", i % 8, parameters[i], point[i].x, point[i].y);

    // Root solving, on coefficient sets read from stdin so the host chooses the hard cases.
    int cases = 0;
    static double coefficients[5 * 4096];
    while (cases < 4096 && scanf("%lf %lf %lf %lf %lf",
                                 &coefficients[5 * cases + 0], &coefficients[5 * cases + 1],
                                 &coefficients[5 * cases + 2], &coefficients[5 * cases + 3],
                                 &coefficients[5 * cases + 4]) == 5) ++cases;
    if (cases > 0) {
        double* dCoefficients;
        double* dRoots;
        int* dFound;
        CHECK(cudaMalloc(&dCoefficients, 5 * cases * sizeof(double)));
        CHECK(cudaMalloc(&dRoots, 5 * cases * sizeof(double)));
        CHECK(cudaMalloc(&dFound, cases * sizeof(int)));
        CHECK(cudaMemcpy(dCoefficients, coefficients, 5 * cases * sizeof(double),
                         cudaMemcpyHostToDevice));
        rootKernel<<<(cases + 63) / 64, 64>>>(dCoefficients, cases, dRoots, dFound);
        CHECK(cudaDeviceSynchronize());
        static double roots[5 * 4096];
        static int found[4096];
        CHECK(cudaMemcpy(roots, dRoots, 4 * cases * sizeof(double), cudaMemcpyDeviceToHost));
        CHECK(cudaMemcpy(found, dFound, cases * sizeof(int), cudaMemcpyDeviceToHost));
        printf("QUARTIC\n");
        for (int i = 0; i < cases; ++i) {
            printf("%d", found[i]);
            for (int k = 0; k < found[i]; ++k) printf(" %.17g", roots[4 * i + k]);
            printf("\n");
        }
        trigKernel<<<(cases + 63) / 64, 64>>>(dCoefficients, cases, dRoots, dFound);
        CHECK(cudaDeviceSynchronize());
        CHECK(cudaMemcpy(roots, dRoots, 5 * cases * sizeof(double), cudaMemcpyDeviceToHost));
        CHECK(cudaMemcpy(found, dFound, cases * sizeof(int), cudaMemcpyDeviceToHost));
        printf("TRIG\n");
        for (int i = 0; i < cases; ++i) {
            printf("%d", found[i]);
            for (int k = 0; k < found[i]; ++k) printf(" %.17g", roots[5 * i + k]);
            printf("\n");
        }
    }
    // Spans against a second body, supplied on the command line as 8 loop coordinates + 4 radii.
    if (argc >= 13) {
        double2 loopB[4];
        double rhoB[4];
        for (int k = 0; k < 4; ++k)
            loopB[k] = make_double2(atof(argv[1 + 2 * k]), atof(argv[2 + 2 * k]));
        for (int k = 0; k < 4; ++k) rhoB[k] = atof(argv[9 + k]);
        CHECK(cudaMemcpyToSymbol(kLoopB, loopB, sizeof(loopB)));
        CHECK(cudaMemcpyToSymbol(kRhoB, rhoB, sizeof(rhoB)));
        const int limit = 512;
        int *dPiece, *dCount;
        double *dLow, *dHigh;
        CHECK(cudaMalloc(&dPiece, limit * sizeof(int)));
        CHECK(cudaMalloc(&dLow, limit * sizeof(double)));
        CHECK(cudaMalloc(&dHigh, limit * sizeof(double)));
        CHECK(cudaMalloc(&dCount, sizeof(int)));
        spanKernel<<<1, 1>>>(dPiece, dLow, dHigh, dCount, limit);
        CHECK(cudaDeviceSynchronize());
        int spanCount = 0;
        CHECK(cudaMemcpy(&spanCount, dCount, sizeof(int), cudaMemcpyDeviceToHost));
        static int piece[512];
        static double low[512], high[512];
        CHECK(cudaMemcpy(piece, dPiece, spanCount * sizeof(int), cudaMemcpyDeviceToHost));
        CHECK(cudaMemcpy(low, dLow, spanCount * sizeof(double), cudaMemcpyDeviceToHost));
        CHECK(cudaMemcpy(high, dHigh, spanCount * sizeof(double), cudaMemcpyDeviceToHost));
        printf("SPAN\n");
        for (int i = 0; i < spanCount; ++i)
            printf("%d %.17g %.17g\n", piece[i], low[i], high[i]);

        int *dKind2, *dFeature2;
        CHECK(cudaMalloc(&dKind2, limit * sizeof(int)));
        CHECK(cudaMalloc(&dFeature2, limit * sizeof(int)));
        substretchKernel<<<1, 1>>>(dPiece, dLow, dHigh, dKind2, dFeature2, dCount, limit);
        CHECK(cudaDeviceSynchronize());
        int subCount = 0;
        CHECK(cudaMemcpy(&subCount, dCount, sizeof(int), cudaMemcpyDeviceToHost));
        static int subKind[512], subFeature[512];
        CHECK(cudaMemcpy(piece, dPiece, subCount * sizeof(int), cudaMemcpyDeviceToHost));
        CHECK(cudaMemcpy(low, dLow, subCount * sizeof(double), cudaMemcpyDeviceToHost));
        CHECK(cudaMemcpy(high, dHigh, subCount * sizeof(double), cudaMemcpyDeviceToHost));
        CHECK(cudaMemcpy(subKind, dKind2, subCount * sizeof(int), cudaMemcpyDeviceToHost));
        CHECK(cudaMemcpy(subFeature, dFeature2, subCount * sizeof(int), cudaMemcpyDeviceToHost));
        printf("SUBSTRETCH\n");
        for (int i = 0; i < subCount; ++i)
            printf("%d %.17g %.17g %d %d\n", piece[i], low[i], high[i], subKind[i], subFeature[i]);
    }
    if (argc >= 13) {
        const int flat = 8 * 4;
        double *dEnergy, *dGradA, *dGradB;
        CHECK(cudaMalloc(&dEnergy, sizeof(double)));
        CHECK(cudaMalloc(&dGradA, flat * sizeof(double)));
        CHECK(cudaMalloc(&dGradB, flat * sizeof(double)));
        CHECK(cudaMemset(dGradA, 0, flat * sizeof(double)));
        CHECK(cudaMemset(dGradB, 0, flat * sizeof(double)));
        pairKernel<<<1, 1>>>(dEnergy, dGradA, dGradB, 1.7 / 3.0);
        CHECK(cudaDeviceSynchronize());
        double energy = 0.0, gradA[32], gradB[32];
        CHECK(cudaMemcpy(&energy, dEnergy, sizeof(double), cudaMemcpyDeviceToHost));
        CHECK(cudaMemcpy(gradA, dGradA, flat * sizeof(double), cudaMemcpyDeviceToHost));
        CHECK(cudaMemcpy(gradB, dGradB, flat * sizeof(double), cudaMemcpyDeviceToHost));
        printf("PAIR\n");
        printf("%.17g\n", energy);
        for (int i = 0; i < flat; ++i) printf("%.17g ", gradA[i]);
        printf("\n");
        for (int i = 0; i < flat; ++i) printf("%.17g ", gradB[i]);
        printf("\n");
    }
    printf("DONE\n");
    (void) argc; (void) argv;
    return 0;
}
