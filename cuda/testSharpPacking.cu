// Validation harness for the whole-packing separated-sum sharp overlap (Track B) vs energies.py.
//
// Build:  nvcc -O2 -arch=sm_75 -std=c++14 testSharpPacking.cu sharpKernels.cu -o testSharpPacking
// Run:    ./testSharpPacking   (expects cuda/vectors/sharpPacking.csv from `python genVectors.py`)
//
// Each row is an N=6, n=6 packing: positions(72), area(1), gradient(72). The driver computes total
// area + full vertex gradient via the separated pipeline; we report max error vs the reference.
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>

#include "sharpKernels.cuh"

static const int NPOLY = 6, NVERT = 36, NCOL = 2 * NVERT + 1 + 2 * NVERT + 1;   // 72+1+72+nInter = 146

static std::vector<double> readCsv(const std::string& path, int& nrow) {
    std::ifstream f(path);
    if (!f) { fprintf(stderr, "cannot open %s\n", path.c_str()); exit(1); }
    std::string line; std::getline(f, line);   // header
    std::vector<double> data; nrow = 0;
    while (std::getline(f, line)) {
        if (line.empty()) continue;
        std::stringstream ss(line); std::string cell;
        while (std::getline(ss, cell, ',')) data.push_back(std::stod(cell));
        ++nrow;
    }
    return data;
}

int main() {
    printf("CUDA whole-packing sharp overlap validation (separated sum, vs energies.py, rel 1e-12)\n");
    int nrow; std::vector<double> rows = readCsv("vectors/sharpPacking.csv", nrow);
    int starts[NPOLY + 1];
    for (int p = 0; p <= NPOLY; ++p) starts[p] = 6 * p;

    double maxAreaErr = 0, areaScale = 0, maxGradErr = 0, gradScale = 0;
    int worst = 0, countMismatches = 0;
    for (int r = 0; r < nrow; ++r) {
        const double* row = rows.data() + (size_t)r * NCOL;
        const double* positions = row;
        double refArea = row[2 * NVERT];
        const double* refGrad = row + 2 * NVERT + 1;
        int refNInter = (int)llround(row[NCOL - 1]);

        // STALE REFERENCE VECTORS. sharpOverlap now returns the normalized-squared ENERGY
        // U = 2 k sum (a_AB / norm_AB)^2, not the raw overlap area these vectors were generated
        // against, so the area/gradient comparisons below no longer mean what they did. The live
        // check is CPU-vs-GPU from Python (tests/cudaTermsCheck.py); regenerate with genVectors.py
        // before trusting this one again. Unit targets make norm_AB = 2, i.e. U = (1/2) sum a^2.
        double unitTargets[NPOLY];
        for (int p = 0; p < NPOLY; ++p) unitTargets[p] = 1.0;
        double area, grad[2 * NVERT]; int nInter = 0;
        sharpOverlap(positions, starts, NPOLY, NVERT, unitTargets, -1, 1.0, &area, grad, &nInter);

        if (nInter != refNInter) ++countMismatches;   // cell-candidate set vs all-pairs (energies)
        double ae = fabs(area - refArea);
        if (ae > maxAreaErr) { maxAreaErr = ae; worst = r; }
        areaScale = fmax(areaScale, fabs(refArea));
        for (int i = 0; i < 2 * NVERT; ++i) {
            maxGradErr = fmax(maxGradErr, fabs(grad[i] - refGrad[i]));
            gradScale = fmax(gradScale, fabs(refGrad[i]));
        }
    }
    double areaRel = maxAreaErr / (areaScale > 0 ? areaScale : 1);
    double gradRel = maxGradErr / (gradScale > 0 ? gradScale : 1);
    printf("  packings=%d\n", nrow);
    printf("  intersection count (cells vs all-pairs): %d mismatches\n", countMismatches);
    printf("  area  max|dev-ref|=%.3e  (rel %.3e, scale %.3e)  worst=%d\n",
           maxAreaErr, areaRel, areaScale, worst);
    printf("  grad  max|dev-ref|=%.3e  (rel %.3e, scale %.3e)\n", maxGradErr, gradRel, gradScale);
    int ok = (areaRel < 1e-12) && (gradRel < 1e-12) && (countMismatches == 0);
    printf("%s\n", ok ? "ALL CHECKS PASSED" : "SOME CHECKS FAILED");
    return ok ? 0 : 1;
}
