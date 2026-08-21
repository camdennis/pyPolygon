// Validation harness: run each ported device function over the vectors exported
// by genVectors.py and report max abs/rel error vs the energies.py reference.
//
// Build:  make            (or: nvcc -O2 -arch=sm_75 testPlummer.cu -o testPlummer)
// Run:    ./testPlummer   (expects cuda/vectors/*.csv from `python genVectors.py`)
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>

#include "plummer.cuh"

// Read a CSV with a header line; return all numeric rows flattened, plus the column count.
static std::vector<double> readCsv(const std::string& path, int& ncol) {
    std::ifstream f(path);
    if (!f) { fprintf(stderr, "cannot open %s\n", path.c_str()); exit(1); }
    std::string line;
    std::getline(f, line);   // header
    std::vector<double> data;
    ncol = 0;
    while (std::getline(f, line)) {
        if (line.empty()) continue;
        std::stringstream ss(line);
        std::string cell;
        int c = 0;
        while (std::getline(ss, cell, ',')) { data.push_back(std::stod(cell)); ++c; }
        ncol = c;
    }
    return data;
}

// ---- one kernel per device function under test ----
__global__ void cl2Kernel(const double* x, double* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = plummer::cl2Device(x[i]);
}

// One kernel macro per 4-input device function (args a,b,c,d -> out).
#define K4(NAME, FN) __global__ void NAME(const double* a, const double* b, const double* c, \
    const double* d, double* out, int n) { int i = blockIdx.x*blockDim.x+threadIdx.x; \
    if (i < n) out[i] = plummer::FN(a[i], b[i], c[i], d[i]); }
K4(tCoreRealKernel, tCoreRealDevice)
K4(m1Kernel, m1Device)
K4(m2Kernel, m2Device)
K4(m1PrimeKernel, m1PrimeDevice)
K4(lam0Kernel, lam0Device)
K4(lam1Kernel, lam1Device)
typedef void (*Launch4)(const double*, const double*, const double*, const double*, double*, int);

// Compare device output of a unary fn against the expected column; print a verdict.
static int checkUnary(const std::string& name,
                      void (*launch)(const double*, double*, int)) {
    int ncol;
    std::vector<double> rows = readCsv("vectors/" + name + ".csv", ncol);   // [in..., expected]
    int n = (int)rows.size() / ncol;
    std::vector<double> in(n), expected(n);
    for (int i = 0; i < n; ++i) { in[i] = rows[i * ncol + 0]; expected[i] = rows[i * ncol + ncol - 1]; }

    double *dIn, *dOut;
    cudaMalloc(&dIn, n * sizeof(double));
    cudaMalloc(&dOut, n * sizeof(double));
    cudaMemcpy(dIn, in.data(), n * sizeof(double), cudaMemcpyHostToDevice);
    launch(dIn, dOut, n);
    cudaDeviceSynchronize();
    std::vector<double> out(n);
    cudaMemcpy(out.data(), dOut, n * sizeof(double), cudaMemcpyDeviceToHost);
    cudaFree(dIn); cudaFree(dOut);

    double maxAbs = 0.0, maxRel = 0.0, scale = 0.0;
    int worst = 0;
    for (int i = 0; i < n; ++i) {
        double a = fabs(out[i] - expected[i]);
        if (a > maxAbs) { maxAbs = a; worst = i; }
        scale = fmax(scale, fabs(expected[i]));
    }
    maxRel = maxAbs / (scale > 0 ? scale : 1.0);
    const char* verdict = (maxRel < 1e-13) ? "PASS" : "FAIL";
    printf("  %-10s  n=%-5d  max|dev-ref|=%.3e  (rel %.3e, scale %.3e)  worst x=%.6g  -> %s\n",
           name.c_str(), n, maxAbs, maxRel, scale, in[worst], verdict);
    return (maxRel < 1e-13) ? 0 : 1;
}

static void launchCl2(const double* x, double* out, int n) {
    cl2Kernel<<<(n + 255) / 256, 256>>>(x, out, n);
}

// Generic 4-input check: read <name>.csv (4 input cols + expected), run the kernel, report max err.
static int check4(const std::string& name, Launch4 launch) {
    int ncol;
    std::vector<double> rows = readCsv("vectors/" + name + ".csv", ncol);
    int n = (int)rows.size() / ncol;
    std::vector<double> col[4], expected(n);
    for (int c = 0; c < 4; ++c) col[c].resize(n);
    for (int i = 0; i < n; ++i) {
        for (int c = 0; c < 4; ++c) col[c][i] = rows[i * ncol + c];
        expected[i] = rows[i * ncol + ncol - 1];
    }
    double *d[4], *dOut;
    for (int c = 0; c < 4; ++c) { cudaMalloc(&d[c], n * sizeof(double));
        cudaMemcpy(d[c], col[c].data(), n * sizeof(double), cudaMemcpyHostToDevice); }
    cudaMalloc(&dOut, n * sizeof(double));
    launch(d[0], d[1], d[2], d[3], dOut, n);
    cudaDeviceSynchronize();
    std::vector<double> out(n);
    cudaMemcpy(out.data(), dOut, n * sizeof(double), cudaMemcpyDeviceToHost);
    for (int c = 0; c < 4; ++c) cudaFree(d[c]);
    cudaFree(dOut);

    double maxAbs = 0, scale = 0; int worst = 0;
    for (int i = 0; i < n; ++i) {
        double a = fabs(out[i] - expected[i]);
        if (a > maxAbs) { maxAbs = a; worst = i; }
        scale = fmax(scale, fabs(expected[i]));
    }
    double rel = maxAbs / (scale > 0 ? scale : 1.0);
    const char* verdict = (rel < 1e-13) ? "PASS" : "FAIL";
    printf("  %-10s  n=%-5d  max|dev-ref|=%.3e  (rel %.3e, scale %.3e)  worst[%.4g,%.4g,%.4g,%.4g]  -> %s\n",
           name.c_str(), n, maxAbs, rel, scale, col[0][worst], col[1][worst], col[2][worst],
           col[3][worst], verdict);
    return (rel < 1e-13) ? 0 : 1;
}

// launch wrappers (grid over n)
#define L4(NAME, KER) static void NAME(const double* a, const double* b, const double* c, \
    const double* d, double* out, int n) { KER<<<(n + 255) / 256, 256>>>(a, b, c, d, out, n); }
L4(lTCore, tCoreRealKernel) L4(lM1, m1Kernel) L4(lM2, m2Kernel)
L4(lM1p, m1PrimeKernel) L4(lLam0, lam0Kernel) L4(lLam1, lam1Kernel)

int main() {
    printf("CUDA Plummer math-library validation (vs energies.py, threshold rel 1e-13)\n");
    int fails = 0;
    fails += checkUnary("cl2", launchCl2);
    fails += check4("tCoreReal", lTCore);
    fails += check4("m1", lM1);
    fails += check4("m2", lM2);
    fails += check4("m1Prime", lM1p);
    fails += check4("lam0", lLam0);
    fails += check4("lam1", lLam1);
    printf("%s\n", fails ? "SOME CHECKS FAILED" : "ALL CHECKS PASSED");
    return fails;
}
