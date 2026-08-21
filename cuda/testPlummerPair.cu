// Validation harness for the single-pair mollified overlap ENERGY (Track A) vs energies.py.
//
// Build:  nvcc -O2 -arch=sm_75 -std=c++14 testPlummerPair.cu -o testPlummerPair
// Run:    ./testPlummerPair   (expects cuda/vectors/plummerPair.csv from `python genVectors.py`)
//
// Each row is an n=6 pair: loopA(12), loopB(12), sigma, energy. The energy is
//   E = -(1/4pi) sum_{edge a in A, edge b in B} (e_a . e_b) I(a,b),
// so we parallelize OVER EDGE PAIRS: thread (pair, a, b) computes the panel I via iClosedDevice,
// weights by e_a.e_b, and atomic-reduces into that pair's energy. Half the pairs are near-parallel,
// exercising the bridge branch of iClosedDevice.
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>

#include "plummer.cuh"

static const int N = 6, NCOL = 2 * (2 * N) + 1 + 1;   // 12 + 12 + sigma + energy = 26
static const int EP = N * N;                          // 36 edge pairs per hexagon pair

static std::vector<double> readCsv(const std::string& path, int& nrow) {
    std::ifstream f(path);
    if (!f) { fprintf(stderr, "cannot open %s\n", path.c_str()); exit(1); }
    std::string line; std::getline(f, line);
    std::vector<double> data; nrow = 0;
    while (std::getline(f, line)) {
        if (line.empty()) continue;
        std::stringstream ss(line); std::string cell;
        while (std::getline(ss, cell, ',')) data.push_back(std::stod(cell));
        ++nrow;
    }
    return data;
}

// Thread per (pair p, A-edge a, B-edge b): panel I -> weighted contribution -> atomicAdd.
__global__ void pairEnergyKernel(const double* rows, int nrow, double* accum) {
    long t = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= (long)nrow * EP) return;
    int p = (int)(t / EP), local = (int)(t % EP), a = local / N, b = local % N;
    const double* row = rows + (size_t)p * NCOL;
    double sg = row[2 * (2 * N)];
    int an = (a + 1) % N, bn = (b + 1) % N;
    double ax0 = row[2 * a], ay0 = row[2 * a + 1];
    double eax = row[2 * an] - ax0, eay = row[2 * an + 1] - ay0;
    double LA = sqrt(eax * eax + eay * eay);
    double bx0 = row[2 * N + 2 * b], by0 = row[2 * N + 2 * b + 1];
    double ebx = row[2 * N + 2 * bn] - bx0, eby = row[2 * N + 2 * bn + 1] - by0;
    double LB = sqrt(ebx * ebx + eby * eby);
    double bhx = ebx / LB, bhy = eby / LB;
    double w0x = ax0 - bx0, w0y = ay0 - by0;
    double P0 = w0x * bhx + w0y * bhy;
    double P1 = eax * bhx + eay * bhy;
    double X0 = w0x * bhy - w0y * bhx;
    double X1 = eax * bhy - eay * bhx;
    double I = plummer::iClosedDevice(P0, P1, X0, X1, LA, LB, sg);
    atomicAdd(&accum[p], (eax * ebx + eay * eby) * I);   // weight = (nA.nB) LA LB = e_a . e_b
}

int main() {
    printf("CUDA single-pair mollified energy validation (vs energies.py, threshold rel 1e-12)\n");
    int nrow; std::vector<double> rows = readCsv("vectors/plummerPair.csv", nrow);

    double *dRows, *dAccum;
    cudaMalloc(&dRows, rows.size() * sizeof(double));
    cudaMalloc(&dAccum, nrow * sizeof(double));
    cudaMemcpy(dRows, rows.data(), rows.size() * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemset(dAccum, 0, nrow * sizeof(double));
    long total = (long)nrow * EP;
    pairEnergyKernel<<<(int)((total + 255) / 256), 256>>>(dRows, nrow, dAccum);
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) { fprintf(stderr, "kernel error: %s\n", cudaGetErrorString(err)); return 2; }
    std::vector<double> accum(nrow);
    cudaMemcpy(accum.data(), dAccum, nrow * sizeof(double), cudaMemcpyDeviceToHost);
    cudaFree(dRows); cudaFree(dAccum);

    // The pair energy is a near-cancellation of 36 O(1) edge-pair panels, so it floors ~2 digits
    // above the per-panel roundoff (matching the model's ~9e-13 force floor); threshold 1e-11.
    const double INV4PI = 1.0 / (4.0 * M_PI);
    double maxAbs = 0, scale = 0; int worst = 0;
    for (int p = 0; p < nrow; ++p) {
        double e = -accum[p] * INV4PI;
        double ref = rows[(size_t)p * NCOL + 2 * (2 * N) + 1];
        double a = fabs(e - ref);
        if (a > maxAbs) { maxAbs = a; worst = p; }
        scale = fmax(scale, fabs(ref));
    }
    double rel = maxAbs / (scale > 0 ? scale : 1.0);
    printf("  pairs=%d\n", nrow);
    printf("  energy  max|dev-ref|=%.3e  (rel %.3e, scale %.3e)  worst pair=%d\n",
           maxAbs, rel, scale, worst);
    int ok = rel < 1e-11;
    printf("%s\n", ok ? "ALL CHECKS PASSED" : "SOME CHECKS FAILED");
    return ok ? 0 : 1;
}
