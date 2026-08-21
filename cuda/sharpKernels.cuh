// Sharp overlap -- separated-sum host driver (Track B).
//
// Runs the full whole-packing sharp overlap as the separated sum designed in energies.py and
// mirrored by STABLE: build a packed intersection list -> CUB sort -> binary-search followers ->
// shape ranges -> one kernel OVER INTERSECTIONS (U_ex) + one kernel OVER VERTICES (U_int).
//
// Host-array interface for validation; a device-pointer version follows at engine integration.
#pragma once

// Compute the total sharp overlap area and its vertex gradient dA/dv (energies.overlapGradient;
// physical force = -grad). Host arrays in/out:
//   positions : 2*numVert doubles, interleaved [x0,y0,x1,y1,...]
//   starts    : numPoly+1 ints (CSR polygon offsets)
//   area      : 1 double out
//   grad      : 2*numVert doubles out
//   outNumInter (optional) : number of edge-edge intersections found (for candidate-set validation)
void sharpOverlap(const double* positions, const int* starts, int numPoly, int numVert,
                  const double* targetArea, int containerIndex, double kOverlap,
                  double* area, double* grad, int* outNumInter = nullptr, bool periodic = false);
