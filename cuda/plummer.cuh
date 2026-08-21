// Mollified Plummer overlap -- device math library (Track A).
//
// Faithful CUDA C++ port of the fully-analytic tier in ../energies.py, validated
// device-vs-Python to ~1e-13 by testPlummer.cu (vectors from genVectors.py). All
// double precision, to match the CPU reference; sm_75.
//
// Dependency order (each builds on the ones above):
//   cl2Device         Clausen Cl2(x)                    <- energies._cl2
//   tCoreRealDevice   the single transcendental T=-2J   <- energies._tCoreReal
//   [next] master primitives, single-pair energy
#pragma once
#include <math.h>

namespace plummer {

// Cl2 series coefficients |B_2k| / (2k (2k+1)!) (energies._CL2COEFFS; notes/arcsinhClausen.nb).
// 20 terms give ~1e-15 on the reduced range [0, pi]. C[k] multiplies t^(2k+3) in the series tail.
__device__ __constant__ double CL2COEFFS[20] = {
    1.388888888888888889e-2, 6.944444444444444444e-5, 7.873519778281683044e-7,
    1.148221634332745444e-8, 1.897886998897099907e-10, 3.387301370953521272e-12,
    6.372636443183180397e-14, 1.246205991295067230e-15, 2.510544460899954551e-17,
    5.178258806090623507e-19, 1.088735736830084884e-20, 2.325744114302087224e-22,
    5.035195213147389561e-24, 1.102649929438121533e-25, 2.438658550900734474e-27,
    5.440142678856252316e-29, 1.222834013121735212e-30, 2.767263468967950584e-32,
    6.300090591832013949e-34, 1.442086838841847521e-35};

// Cl2(x) = -int_0^x ln|2 sin(t/2)| dt. Reduce the argument to (-pi, pi] by 2pi-periodicity
// (matching numpy's non-negative mod exactly) and oddness, then sum t - t ln t + sum c_k t^(2k+1).
__device__ __forceinline__ double cl2Device(double x) {
    const double PI = 3.14159265358979323846264338328;
    const double TWO_PI = 6.28318530717958647692528676656;
    double t = x - TWO_PI * floor(x / TWO_PI);   // [0, 2pi), matches np.mod(x, 2pi)
    if (t > PI) t -= TWO_PI;                       // (-pi, pi]
    double s = (t < 0.0) ? -1.0 : 1.0;
    t = fabs(t);
    if (t == 0.0) return 0.0;
    double t2 = t * t;
    double p = 0.0;
    for (int k = 19; k >= 0; --k) p = p * t2 + CL2COEFFS[k];   // Horner in t^2
    double val = t - t * log(t) + p * t * t2;
    return s * val;
}

// The single transcendental T = -2 J_arcsinh, via the real Clausen form (energies._tCoreReal;
// mollifiedDerivation.tex sec 5). m = al, nu = be/sg, w = sqrt(1+m^2+nu^2), y = (xi+sqrt(xi^2+sg^2))/sg;
// the two upper-half poles give Im G(eta_+/-) as Bloch-Wigner D -> Clausen; T = Im G(y).
__device__ __forceinline__ double tCoreRealDevice(double xi, double al, double be, double sg) {
    double m = al, nu = be / sg;
    double w = sqrt(1.0 + m * m + nu * nu);
    double psi = atan2(1.0, m);
    double L = log(w - nu) - 0.5 * log(1.0 + m * m);
    double y = (xi + sqrt(xi * xi + sg * sg)) / sg;
    double phiP = atan2(y, w - nu - m * y);
    double phiM = atan2(y, w + nu + m * y);
    double c2psi = cl2Device(2.0 * psi);                         // shared by both Bloch-Wigner terms
    double DP = 0.5 * (cl2Device(2.0 * psi + 2.0 * phiP) - c2psi - cl2Device(2.0 * phiP));
    double DM = 0.5 * (c2psi - cl2Device(2.0 * phiM) - cl2Device(2.0 * psi - 2.0 * phiM));
    return (DP + L * phiP) - (DM - L * phiM);
}

// ---- elementary master primitives (energies.py secs 6.3); q(s)=a s^2+b s+c, 4ac-b^2>0 always ----
__device__ __forceinline__ double lam0Device(double s, double a, double b, double c) {   // int ln q ds
    double q = a * s * s + b * s + c, sD = sqrt(4.0 * a * c - b * b);
    return (s + b / (2.0 * a)) * log(q) - 2.0 * s + (sD / a) * atan((2.0 * a * s + b) / sD);
}
__device__ __forceinline__ double lam1Device(double s, double a, double b, double c) {   // int s ln q ds
    double q = a * s * s + b * s + c, sD = sqrt(4.0 * a * c - b * b);
    return (s * s * 0.5 - (b * b - 2.0 * a * c) / (4.0 * a * a)) * log(q) - s * s * 0.5
         + b * s / (2.0 * a) - (b * sD / (2.0 * a * a)) * atan((2.0 * a * s + b) / sD);
}
__device__ __forceinline__ double q2Device(double xi, double al, double be, double sg) {
    return (1.0 + al * al) * xi * xi + 2.0 * al * be * xi + be * be + sg * sg;
}
__device__ __forceinline__ double xi0Device(double xi, double al, double be, double sg) {   // int dxi / Q2
    double A = 1.0 + al * al, delta = sqrt(be * be + A * sg * sg);
    return atan((A * xi + al * be) / delta) / delta;
}
__device__ __forceinline__ double rPrimDevice(double xi, double al, double be, double sg) {  // int xi(al sg^2-be xi)/Q2
    double A = 1.0 + al * al, x0 = xi0Device(xi, al, be, sg);
    double x1 = log(q2Device(xi, al, be, sg)) / (2.0 * A) - (al * be / A) * x0;
    return -be * xi / A + (al * (sg * sg * A + 2.0 * be * be) / A) * x1 + (be * (be * be + sg * sg) / A) * x0;
}
__device__ __forceinline__ double thetaDevice(double xi, double al, double be, double sg) {
    return atan((al * xi + be) / sqrt(xi * xi + sg * sg));
}
__device__ __forceinline__ double m1Device(double xi, double al, double be, double sg) {  // int xi/sqrt Theta (elementary)
    double A = 1.0 + al * al, delta = sqrt(be * be + A * sg * sg);
    return sqrt(xi * xi + sg * sg) * thetaDevice(xi, al, be, sg) + (be / (2.0 * A)) * log(q2Device(xi, al, be, sg))
         - (al * delta / A) * atan((A * xi + al * be) / delta);
}
__device__ __forceinline__ double vPlusDevice(double xi, double sg) {
    return (xi * sqrt(xi * xi + sg * sg) + sg * sg * asinh(xi / sg)) * 0.5;
}
__device__ __forceinline__ double vMinusDevice(double xi, double sg) {
    return (xi * sqrt(xi * xi + sg * sg) - sg * sg * asinh(xi / sg)) * 0.5;
}
// Master arctan-integral M[V] = V Theta - rho/2 + kappa (sg^2/2) T. v=vPlus,kappa=-1 -> M2 (energy
// panel); v=vMinus,kappa=+1 -> M1' (gradient W1). Both share the single transcendental T=tCoreReal.
__device__ __forceinline__ double m2Device(double xi, double al, double be, double sg) {
    return vPlusDevice(xi, sg) * thetaDevice(xi, al, be, sg) - rPrimDevice(xi, al, be, sg) * 0.5
         - (sg * sg * 0.5) * tCoreRealDevice(xi, al, be, sg);
}
__device__ __forceinline__ double m1PrimeDevice(double xi, double al, double be, double sg) {
    return vMinusDevice(xi, sg) * thetaDevice(xi, al, be, sg) - rPrimDevice(xi, al, be, sg) * 0.5
         + (sg * sg * 0.5) * tCoreRealDevice(xi, al, be, sg);
}

// ---- single-pair energy panel (energies._iClosedVec + _ceBridge) -----------------------------
// Near-parallel bridge: 2x24 Gauss on [0,1] per half of the peak-split interval, triggered per edge
// pair when |X1| <= _NEARPAR * LA, where the 1/X1 closed form loses precision (energies BNODES/BWTS).
#define PLUMMER_NEARPAR 1e-2
__device__ __constant__ double BNODES[24] = {
    2.40639000148934468e-03, 1.26357220143452631e-02, 3.08627239986336011e-02,
    5.67922364977995198e-02, 8.99990070130485265e-02, 1.29937904210722821e-01,
    1.75953174031512227e-01, 2.27289264305580219e-01, 2.83103246186977464e-01,
    3.42478660151918302e-01, 4.04440566263191859e-01, 4.67971553568697185e-01,
    5.32028446431302759e-01, 5.95559433736808197e-01, 6.57521339848081698e-01,
    7.16896753813022536e-01, 7.72710735694419837e-01, 8.24046825968487773e-01,
    8.70062095789277179e-01, 9.10000992986951474e-01, 9.43207763502200480e-01,
    9.69137276001366343e-01, 9.87364277985654737e-01, 9.97593609998510655e-01};
__device__ __constant__ double BWTS[24] = {
    6.17061489999354545e-03, 1.42656943144668716e-02, 2.21387194087097755e-02,
    2.96492924577183709e-02, 3.66732407055402054e-02, 4.30950807659766441e-02,
    4.88093260520570324e-02, 5.37221350579828033e-02, 5.77528340268628065e-02,
    6.08352364639017096e-02, 6.29187281734141513e-02, 6.39690976733761074e-02,
    6.39690976733761074e-02, 6.29187281734141513e-02, 6.08352364639017096e-02,
    5.77528340268628065e-02, 5.37221350579828033e-02, 4.88093260520570324e-02,
    4.30950807659766441e-02, 3.66732407055402054e-02, 2.96492924577183709e-02,
    2.21387194087097755e-02, 1.42656943144668716e-02, 6.17061489999354545e-03};

// fa(u) = u arctan(u/h) - (h/2) ln(u^2 + h^2)   (the parallel-branch elementary primitive)
__device__ __forceinline__ double faDevice(double u, double h) {
    return u * atan(u / h) - 0.5 * h * log(u * u + h * h);
}

// Near-parallel-stable value of the arctan panel term: int_0^1 sqrt(xi^2+sg^2) arctan((U0+P1 u)/...) du,
// xi = X0 + u X1. Peak-split at u* = -U0/P1, 2x24 Gauss (energies._ceBridge).
__device__ __forceinline__ double ceBridgeDevice(double P1, double X0, double X1, double U0, double sg) {
    double P1s = (fabs(P1) < 1e-300) ? 1.0 : P1;
    double ustar = fmax(0.0, fmin(1.0, -U0 / P1s));
    double lo[2] = {0.0, ustar}, hi[2] = {ustar, 1.0};
    double tot = 0.0;
    for (int s = 0; s < 2; ++s) {
        double a = lo[s], span = hi[s] - a, acc = 0.0;
        for (int k = 0; k < 24; ++k) {
            double u = a + span * BNODES[k];
            double r = X0 + u * X1;
            double R = sqrt(r * r + sg * sg);
            acc += BWTS[k] * (R * atan((U0 + P1 * u) / R));
        }
        tot += span * acc;
    }
    return tot;
}

// Panel integral I = int int ln Q ds dt for ONE edge pair; general / near-parallel bridge / exactly-
// parallel branch selected per pair (energies._iClosedVec). LA = |e_A|, LB = |e_B|.
__device__ __forceinline__ double iClosedDevice(double P0, double P1, double X0, double X1,
                                                double LA, double LB, double sg) {
    const double tol = 1e-12;
    bool par = fabs(X1) <= tol * LA;
    bool near = !par && (fabs(X1) <= PLUMMER_NEARPAR * LA);
    double X1s = par ? 1.0 : X1;
    double h = sqrt(X0 * X0 + sg * sg);
    double tot = 0.0;
    for (int e = 0; e <= 1; ++e) {
        double eps = (e == 0) ? 1.0 : -1.0;
        double U0 = P0 - e * LB;
        double aq = LA * LA, bq = 2.0 * (U0 * P1 + X0 * X1), cq = U0 * U0 + X0 * X0 + sg * sg;
        double Ae = U0 * (lam0Device(1.0, aq, bq, cq) - lam0Device(0.0, aq, bq, cq))
                  + P1 * (lam1Device(1.0, aq, bq, cq) - lam1Device(0.0, aq, bq, cq));
        double Be = U0 + P1 * 0.5;
        double ce;
        if (par) {
            double P1s = (fabs(P1) < 1e-300) ? 1.0 : P1;
            ce = (h / P1s) * (faDevice(U0 + P1, h) - faDevice(U0, h));
        } else if (near) {
            ce = ceBridgeDevice(P1, X0, X1, U0, sg);
        } else {
            double al = P1 / X1s, be = U0 - al * X0;
            ce = (m2Device(X0 + X1, al, be, sg) - m2Device(X0, al, be, sg)) / X1s;
        }
        tot += eps * (Ae - 2.0 * Be + 2.0 * ce);
    }
    return tot / LB;
}

// ---- single-pair gradient moments (energies._wClosedVec + _wBridge) ---------------------------
// fua(u) = ((u^2+h^2)/2) arctan(u/h) - h u/2  (parallel-branch primitive for the W1 moment)
__device__ __forceinline__ double fuaDevice(double u, double h) {
    return 0.5 * (u * u + h * h) * atan(u / h) - 0.5 * h * u;
}

// Near-parallel-stable W0, W1 moment cores: wb0 = int g du, wb1 = int u g du, g = (xi/R) arctan(...),
// xi = X0 + u X1, R = sqrt(xi^2+sg^2). Peak-split 2x24 Gauss (energies._wBridge).
__device__ __forceinline__ void wBridgeDevice(double P1, double X0, double X1, double U0, double sg,
                                              double* wb0, double* wb1) {
    double P1s = (fabs(P1) < 1e-300) ? 1.0 : P1;
    double ustar = fmax(0.0, fmin(1.0, -U0 / P1s));
    double lo[2] = {0.0, ustar}, hi[2] = {ustar, 1.0};
    double s0 = 0.0, s1 = 0.0;
    for (int s = 0; s < 2; ++s) {
        double a = lo[s], span = hi[s] - a, a0 = 0.0, a1 = 0.0;
        for (int k = 0; k < 24; ++k) {
            double u = a + span * BNODES[k];
            double r = X0 + u * X1;
            double R = sqrt(r * r + sg * sg);
            double g = (r / R) * atan((U0 + P1 * u) / R);
            a0 += BWTS[k] * g;
            a1 += BWTS[k] * (u * g);
        }
        s0 += span * a0; s1 += span * a1;
    }
    *wb0 = s0; *wb1 = s1;
}

// Gradient moments W0, W1 for ONE edge pair (energies._wClosedVec); general / bridge / parallel branch.
__device__ __forceinline__ void wClosedDevice(double P0, double P1, double X0, double X1,
                                              double LA, double LB, double sg,
                                              double* W0out, double* W1out) {
    const double tol = 1e-12, INV2PI = 0.15915494309189533577;
    bool par = fabs(X1) <= tol * LA;
    bool near = !par && (fabs(X1) <= PLUMMER_NEARPAR * LA);
    double X1s = par ? 1.0 : X1;
    double h = sqrt(X0 * X0 + sg * sg);
    double W0 = 0.0, W1 = 0.0;
    for (int e = 0; e <= 1; ++e) {
        double eps = (e == 0) ? 1.0 : -1.0;
        double U0 = P0 - e * LB;
        double w0v, w1v;
        if (par) {
            double P1s = (fabs(P1) < 1e-300) ? 1.0 : P1;
            double dfa = faDevice(U0 + P1, h) - faDevice(U0, h);
            double d0 = dfa / P1s;
            double d1 = ((fuaDevice(U0 + P1, h) - fuaDevice(U0, h)) / P1s - U0 * dfa / P1s) / P1s;
            double c = X0 * INV2PI / h;
            w0v = -eps * c * d0;
            w1v = -eps * c * d1;
        } else if (near) {
            double wb0, wb1;
            wBridgeDevice(P1, X0, X1, U0, sg, &wb0, &wb1);
            w0v = -eps * INV2PI * wb0;
            w1v = -eps * INV2PI * wb1;
        } else {
            double al = P1 / X1s, be = U0 - al * X0;
            double dM1 = m1Device(X0 + X1, al, be, sg) - m1Device(X0, al, be, sg);
            double dM1p = m1PrimeDevice(X0 + X1, al, be, sg) - m1PrimeDevice(X0, al, be, sg);
            w0v = -eps * INV2PI * dM1 / X1s;
            w1v = -eps * INV2PI * (dM1p - X0 * dM1) / (X1s * X1s);
        }
        W0 += w0v; W1 += w1v;
    }
    *W0out = W0; *W1out = W1;
}

// FUSED panel I + gradient moments W0,W1 for ONE edge pair, sharing the transcendental tCoreReal (and
// theta, rPrim) between the energy's M2 and the gradient's M1' -- energies.py evaluates them twice on
// the same frame; here it is once. Equivalent to iClosedDevice + wClosedDevice for the same arguments.
__device__ __forceinline__ void fusedPanelMomentA(double P0, double P1, double X0, double X1,
                                                  double LA, double LB, double sg,
                                                  double* Iout, double* W0out, double* W1out) {
    const double tol = 1e-12, INV2PI = 0.15915494309189533577;
    bool par = fabs(X1) <= tol * LA;
    bool near = !par && (fabs(X1) <= PLUMMER_NEARPAR * LA);
    double X1s = par ? 1.0 : X1;
    double h = sqrt(X0 * X0 + sg * sg);
    double I = 0.0, W0 = 0.0, W1 = 0.0;
    for (int e = 0; e <= 1; ++e) {
        double eps = (e == 0) ? 1.0 : -1.0;
        double U0 = P0 - e * LB;
        double aq = LA * LA, bq = 2.0 * (U0 * P1 + X0 * X1), cq = U0 * U0 + X0 * X0 + sg * sg;
        double Ae = U0 * (lam0Device(1.0, aq, bq, cq) - lam0Device(0.0, aq, bq, cq))
                  + P1 * (lam1Device(1.0, aq, bq, cq) - lam1Device(0.0, aq, bq, cq));
        double Be = U0 + P1 * 0.5;
        double ce, w0v, w1v;
        if (par) {
            double P1s = (fabs(P1) < 1e-300) ? 1.0 : P1;
            double dfa = faDevice(U0 + P1, h) - faDevice(U0, h);
            ce = (h / P1s) * dfa;
            double d0 = dfa / P1s;
            double d1 = ((fuaDevice(U0 + P1, h) - fuaDevice(U0, h)) / P1s - U0 * dfa / P1s) / P1s;
            double c = X0 * INV2PI / h;
            w0v = -eps * c * d0; w1v = -eps * c * d1;
        } else if (near) {
            ce = ceBridgeDevice(P1, X0, X1, U0, sg);
            double wb0, wb1; wBridgeDevice(P1, X0, X1, U0, sg, &wb0, &wb1);
            w0v = -eps * INV2PI * wb0; w1v = -eps * INV2PI * wb1;
        } else {
            double al = P1 / X1s, be = U0 - al * X0, s2 = sg * sg * 0.5;
            double xh = X0 + X1;
            double thi = thetaDevice(xh, al, be, sg), rpi = rPrimDevice(xh, al, be, sg), Ti = tCoreRealDevice(xh, al, be, sg);
            double tlo = thetaDevice(X0, al, be, sg), rlo = rPrimDevice(X0, al, be, sg), Tlo = tCoreRealDevice(X0, al, be, sg);
            double m2hi = vPlusDevice(xh, sg) * thi - rpi * 0.5 - s2 * Ti;
            double m2lo = vPlusDevice(X0, sg) * tlo - rlo * 0.5 - s2 * Tlo;
            ce = (m2hi - m2lo) / X1s;
            double dM1p = (vMinusDevice(xh, sg) * thi - rpi * 0.5 + s2 * Ti)
                        - (vMinusDevice(X0, sg) * tlo - rlo * 0.5 + s2 * Tlo);
            double dM1 = m1Device(xh, al, be, sg) - m1Device(X0, al, be, sg);
            w0v = -eps * INV2PI * dM1 / X1s;
            w1v = -eps * INV2PI * (dM1p - X0 * dM1) / (X1s * X1s);
        }
        I += eps * (Ae - 2.0 * Be + 2.0 * ce);
        W0 += w0v; W1 += w1v;
    }
    *Iout = I / LB; *W0out = W0; *W1out = W1;
}

}  // namespace plummer
