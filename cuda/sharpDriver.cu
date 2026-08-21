// extern "C" entry point for the sharp overlap, so ctypes can reach the C++ host driver in
// sharpKernels.cu (which is name-mangled and carries default arguments).
//
// Returns the overlap AREA and its vertex gradient dA/dv; the physical force is -kOverlap * grad,
// applied on the Python side (energies.sharpOverlapEnergyForce contract).
#include "sharpKernels.cuh"

extern "C" void sharpOverlapCuda(const double* positions, const int* starts, int numPoly, int numVert,
                                 const double* targetArea, int containerIndex, double kOverlap,
                                 int periodic, double* area, double* grad, int* outNumInter) {
    sharpOverlap(positions, starts, numPoly, numVert, targetArea, containerIndex, kOverlap,
                 area, grad, outNumInter, periodic != 0);
}
