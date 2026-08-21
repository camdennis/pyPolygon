// Mollified membership field Psi(y) on a grid -- the GPU port of energies.plummerMeasure summed over
// every polygon and its 8 periodic images (what Model.draw(indicatorColorMap = ...) renders).
//
//   Psi_B(y) = sum over B's edges of  (w . n) L * (2/sq)(atan((2a+b)/sq) - atan(b/sq)) / 2pi
//   w = v0 - y,  a = |e|^2,  b = 2 w.e,  c = |w|^2 + sigma^2,  sq = sqrt(4ac - b^2)
//
// This is a VISUALIZATION cost, not a physics cost, but it dominated interactive use: at N=32, n=10
// and the default 160x160 grid it is 288 polygon-images x 25600 points x 10 edges = 74M edge
// evaluations, which numpy took ~10.5 s to do (42 s at 320x320) against 33 ms for a plain draw.
//
// One thread per GRID POINT, looping polygons / images / edges. Perfectly parallel, no atomics: each
// thread owns its own output pixel.
//
// SINGLE PRECISION on purpose. The inner loop is two atan() calls per edge, and this GPU (GTX 1650,
// sm_75) runs fp64 at 1/32 the fp32 rate, so double precision is a ~30x tax on a quantity that only
// ever reaches imshow. Measured agreement with the fp64 numpy reference is ~1e-6 absolute on a field
// spanning 0..4 -- far below anything a colormap can show. The PHYSICS path is untouched: this
// kernel is used only by Model.draw, while energies.plummerMeasure (fp64) still backs
// _pairHessianOneSided. Positions arrive as double and are narrowed here rather than on the host, so
// the caller's contract is unchanged.
#include <math.h>

#define MEASURE_BLOCK 128

__global__ void measureKernel(const double2* pos, const int* starts, int numPoly,
                              const double* gridX, const double* gridY, int numPoints,
                              double sigma, int numImages, double* field) {
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= numPoints) return;
    float yx = (float)gridX[p], yy = (float)gridY[p];
    float sigma2 = (float)(sigma * sigma);
    float total = 0.0f;

    for (int s = 0; s < numPoly; ++s) {
        int base = starts[s], n = starts[s + 1] - base;
        for (int k = 0; k < n; ++k) {
            double2 vd = pos[base + k], vnd = pos[base + (k + 1) % n];
            float vx = (float)vd.x, vy = (float)vd.y;
            float ex = (float)(vnd.x - vd.x), ey = (float)(vnd.y - vd.y);
            float length = sqrtf(ex * ex + ey * ey);
            if (length <= 0.0f) continue;
            // Outward unit normal of a CCW loop: (tau.y, -tau.x) -- matches energies._edges.
            float nx = ey / length, ny = -ex / length;
            float a = ex * ex + ey * ey;
            // numImages is 9 for a periodic box and 1 in free space, where the tiled copies would be
            // fictitious -- and where they would also make the field read 1 everywhere, since the
            // images of a box-sized shape tile the plane.
            for (int img = 0; img < numImages; ++img) {
                float wx = vx + (numImages == 1 ? 0.0f : (float)(img / 3 - 1)) - yx;
                float wy = vy + (numImages == 1 ? 0.0f : (float)(img % 3 - 1)) - yy;
                float b = 2.0f * (wx * ex + wy * ey);
                float c = wx * wx + wy * wy + sigma2;
                float sq = sqrtf(4.0f * a * c - b * b);
                float j = (2.0f / sq) * (atanf((2.0f * a + b) / sq) - atanf(b / sq));
                total += (wx * nx + wy * ny) * length * j;
            }
        }
    }
    field[p] = (double)total / (2.0 * M_PI);
}

// ---- C API. Buffers are sized per call: the grid changes with the requested resolution, and this
// runs once per frame rather than once per force evaluation, so allocation is not on a hot path. ----
extern "C" void plummerMeasureGridCuda(const double* positions, int numVert,
                                       const int* startIndices, int numPoly,
                                       const double* gridX, const double* gridY, int numPoints,
                                       double sigma, int periodic, double* fieldOut) {
    double2* dPos = nullptr; int* dStarts = nullptr;
    double *dGx = nullptr, *dGy = nullptr, *dField = nullptr;
    cudaMalloc(&dPos, numVert * sizeof(double2));
    cudaMalloc(&dStarts, (numPoly + 1) * sizeof(int));
    cudaMalloc(&dGx, numPoints * sizeof(double));
    cudaMalloc(&dGy, numPoints * sizeof(double));
    cudaMalloc(&dField, numPoints * sizeof(double));

    cudaMemcpy(dPos, positions, numVert * sizeof(double2), cudaMemcpyHostToDevice);
    cudaMemcpy(dStarts, startIndices, (numPoly + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dGx, gridX, numPoints * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dGy, gridY, numPoints * sizeof(double), cudaMemcpyHostToDevice);

    int blocks = (numPoints + MEASURE_BLOCK - 1) / MEASURE_BLOCK;
    measureKernel<<<blocks, MEASURE_BLOCK>>>(dPos, dStarts, numPoly, dGx, dGy, numPoints,
                                             sigma, periodic ? 9 : 1, dField);
    cudaMemcpy(fieldOut, dField, numPoints * sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(dPos); cudaFree(dStarts); cudaFree(dGx); cudaFree(dGy); cudaFree(dField);
}
