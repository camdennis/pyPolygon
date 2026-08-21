// Exact-arc contact, device math. CUDA C++ port of ../roundedContact.py.
//
//   E = 1/2 sum over ordered pairs (P,Q) of  int_{dP cap Q} (k/3) d_Q(x)^3 dl(x)
//
// with the boundary made of n circular ARCS alternating with n straight SEGMENTS, and d_Q the exact
// distance to that boundary. No chording anywhere.
//
// CONVENTIONS ARE FIXED AND MUST NOT BE RE-DERIVED -- the host reference lost four sign errors to
// re-deriving the normal, and this file inherits its choices verbatim:
//
//     CCW backbone, signed area > 0
//     corner k:          arc of radius rho_k about z_k, running a^-_k -> a^+_k, sweep psi_k > 0 convex
//     segment k:         a^+_k -> a^-_{k+1}, stored as tail[k] -> head[k], so head[k] = a^-_{k+1}
//     arc start vector:  w0_k = head[k-1] - z_k     (NEVER an absolute angle: atan2 is not needed and
//                                                    the host avoids it so the same code differentiates)
//     inward normal of a to-segment:  n = (e_y, -e_x)/|e|,  d = n . (tail_B - x) > 0 inside
//     to-arc distance:                d = rho - |x - z| > 0 inside
//
// SIGNED DISTANCES, NOT SQUARED. Inside Q every distance is positive, so d_segment is LINEAR along a
// segment and d_arc = rho - sqrt(quadratic). That is what drops the switch equations from degree 4/8 to
// the shapes solved here, and it is why the whole switch solver is one quartic.
//
// THE BOUNDARY IS C1 AND HAS NO VERTICES. Each arc meets its two incident segments tangentially by
// construction, so there is no vertex feature, the nearest-feature partition has only two cell kinds,
// and the elliptic "arc versus vertex" integral never arises.
//
// NOTHING VARIABLE-LENGTH IS STORED PER THREAD, following ../cuda/polyContact.cuh: breakpoints are
// walked by repeated minimum-search over the candidates rather than collected and sorted, so a thread
// needs no local arrays and never spills. Cost is O(M^2) per piece, which at M <= 2*ROUNDED_MAXN is a
// few thousand flops and far cheaper than the spill.
//
// EVERY BUG FOUND IN THE HOST REFERENCE WAS A TANGENCY, and they will appear here too, because this
// geometry is BUILT from tangencies -- each corner circle is tangent to both its edges. The rules that
// came out of four of them, all of which this file follows:
//
//   * test tangency on the ABSOLUTE discriminant and BEFORE rejecting negatives;
//   * keep a root by whether it SOLVES the equation, never by whether the solver called it real --
//     a double root's near-conjugate pair has a tiny residual and is a genuine switch;
//   * merge breakpoints agreeing to about sqrt(machine epsilon);
//   * a SPURIOUS breakpoint costs nothing (it subdivides an interval whose winner does not change);
//     a MISSED one mislabels a sub-stretch. Always err toward more.
#pragma once
#include <math.h>

// UNVERIFIED(Cam)
namespace roundedContact {

// Corners per body. The depth kernel in polyContact.cuh caps at 64 VERTICES; here the cap is on
// CORNERS, and each carries one arc and one segment, so the feature count is 2*ROUNDED_MAXN.
#define ROUNDED_MAXN 16

struct Body {
    double2 center[ROUNDED_MAXN];   // z_k, the corner circle's centre
    double  radius[ROUNDED_MAXN];   // rho_k
    double  sweep [ROUNDED_MAXN];   // psi_k, SIGNED: positive at a convex corner
    double  start [ROUNDED_MAXN];   // absolute angle of a^-_k about z_k. THE ONLY atan2 in the file,
                                    // and it is read only by the arc's WEDGE TEST -- never by the
                                    // integrals, which rotate w0 instead so the host can differentiate
    double2 tail  [ROUNDED_MAXN];   // a^+_k
    double2 head  [ROUNDED_MAXN];   // a^-_{k+1}
    int     count;
};

__device__ __forceinline__ double2 sub2(double2 a, double2 b) {
    return make_double2(a.x - b.x, a.y - b.y);
}
__device__ __forceinline__ double dot2(double2 a, double2 b) { return a.x * b.x + a.y * b.y; }
__device__ __forceinline__ double2 turn2(double2 a) { return make_double2(-a.y, a.x); }

// ---------------------------------------------------------------------------------------------
// Polynomial roots, closed form. numpy.roots is a companion-matrix eigensolve and cannot come here.
// ---------------------------------------------------------------------------------------------

// Real roots of the monic cubic t^3 + a2 t^2 + a1 t + a0. Returns the count; fills out[0..2].
//
// The three-real-root branch uses the TRIGONOMETRIC form, not Cardano: there Cardano needs the cube
// root of a complex number and loses most of its digits.
__device__ __forceinline__ int cubicRoots(double a2, double a1, double a0, double* out,
                                          bool candidates = false) {
    const double shift = a2 / 3.0;
    const double p = a1 - a2 * a2 / 3.0;
    const double q = 2.0 * a2 * a2 * a2 / 27.0 - a2 * a1 / 3.0 + a0;
    if (fabs(p) < 1e-300) { out[0] = cbrt(-q) - shift; return 1; }
    const double discriminant = -4.0 * p * p * p - 27.0 * q * q;
    if (discriminant > 0.0) {
        const double magnitude = 2.0 * sqrt(-p / 3.0);
        double argument = 3.0 * q / (p * magnitude);
        argument = fmin(1.0, fmax(-1.0, argument));
        const double angle = acos(argument) / 3.0;
        const double third = 2.0 * M_PI / 3.0;
        out[0] = magnitude * cos(angle) - shift;
        out[1] = magnitude * cos(angle - third) - shift;
        out[2] = magnitude * cos(angle - 2.0 * third) - shift;
        return 3;
    }
    const double inner = fmax(q * q / 4.0 + p * p * p / 27.0, 0.0);
    const double root = sqrt(inner);
    const double single = cbrt(-0.5 * q + root) + cbrt(-0.5 * q - root);
    out[0] = single - shift;
    if (!candidates) return 1;
    // The depressed roots sum to zero, so the conjugate pair sits at -single/2. Its real part is a
    // legitimate CANDIDATE -- see the tangency rule in the file header.
    out[1] = -0.5 * single - shift;
    return 2;
}

// Real roots of a x^2 + b x + c, PLUS the real part of a complex pair. Returns the count.
__device__ __forceinline__ int quadraticCandidates(double a, double b, double c, double* out) {
    if (fabs(a) < 1e-300) {
        if (fabs(b) < 1e-300) return 0;
        out[0] = -c / b;
        return 1;
    }
    const double discriminant = b * b - 4.0 * a * c;
    if (discriminant < 0.0) { out[0] = -b / (2.0 * a); return 1; }
    const double root = sqrt(discriminant);
    out[0] = (-b - root) / (2.0 * a);
    out[1] = (-b + root) / (2.0 * a);
    return 2;
}

// Horner value and derivative of the degree-`degree` polynomial in c.
__device__ __forceinline__ void horner(const double* c, int degree, double x,
                                       double* value, double* slope) {
    double v = c[0], s = 0.0;
    for (int i = 1; i <= degree; ++i) { s = s * x + v; v = v * x + c[i]; }
    *value = v; *slope = s;
}

// Real roots (and complex-pair real parts) of a polynomial of degree at most four. Returns the count;
// `out` must hold at least 4.
//
// Ferrari in the factored form y^4 + p y^2 + q y + r = (y^2 + a y + b)(y^2 - a y + c). Matching gives
// a^2 = u with u the resolvent root of u^3 + 2p u^2 + (p^2 - 4r) u - q^2 = 0, whose constant term is
// -q^2 <= 0, so a real u >= 0 always exists; the LARGEST is taken because a larger a divides q/a more
// safely.
//
// POLISHING IS MONOTONE OR ABSENT. Plain Newton is DESTRUCTIVE at a double root, and double roots are
// the normal case here: measured on the host, an unguarded polish moved a switch at -0.25287 to
// -0.33893 and one at 2.35222 to -82.94, losing both and the sub-stretches that ended there. A step is
// accepted only when it reduces |p(x)|, which can sharpen a simple root and can never wreck a
// degenerate one.
__device__ inline int quarticRoots(const double* coefficients, int degree, double* out) {
    double c[5];
    double scale = 0.0;
    for (int i = 0; i <= degree; ++i) scale = fmax(scale, fabs(coefficients[i]));
    if (scale <= 0.0) return 0;
    for (int i = 0; i <= degree; ++i) c[i] = coefficients[i] / scale;

    int start = 0;
    while (start < degree && fabs(c[start]) < 1e-14) ++start;
    const int n = degree - start;
    const double* a = c + start;
    if (n <= 0) return 0;

    int found = 0;
    if (n == 1) {
        out[0] = -a[1] / a[0];
        found = 1;
    } else if (n == 2) {
        found = quadraticCandidates(a[0], a[1], a[2], out);
    } else if (n == 3) {
        found = cubicRoots(a[1] / a[0], a[2] / a[0], a[3] / a[0], out, true);
    } else {
        const double b = a[1] / a[0], cc = a[2] / a[0], d = a[3] / a[0], e = a[4] / a[0];
        const double shift = b / 4.0;
        const double p = cc - 6.0 * shift * shift;
        const double q = d - 2.0 * cc * shift + 8.0 * shift * shift * shift;
        const double r = e - d * shift + cc * shift * shift - 3.0 * shift * shift * shift * shift;
        if (fabs(q) < 1e-14) {
            // Biquadratic: y^2 solves a quadratic, and Ferrari would divide by a -> 0.
            double squares[2];
            const int m = quadraticCandidates(1.0, p, r, squares);
            for (int i = 0; i < m; ++i) {
                const double root = sqrt(fmax(squares[i], 0.0));
                out[found++] = root - shift;
                out[found++] = -root - shift;
            }
        } else {
            double resolvent[3];
            const int m = cubicRoots(2.0 * p, p * p - 4.0 * r, -q * q, resolvent, false);
            double u = -1.0;
            for (int i = 0; i < m; ++i) if (resolvent[i] > u) u = resolvent[i];
            if (u <= 0.0) return 0;
            const double alpha = sqrt(u);
            const double beta  = 0.5 * (p + u - q / alpha);
            const double gamma = 0.5 * (p + u + q / alpha);
            double part[2];
            int k = quadraticCandidates(1.0, alpha, beta, part);
            for (int i = 0; i < k; ++i) out[found++] = part[i] - shift;
            k = quadraticCandidates(1.0, -alpha, gamma, part);
            for (int i = 0; i < k; ++i) out[found++] = part[i] - shift;
        }
    }

    for (int i = 0; i < found; ++i) {
        double x = out[i];
        if (!isfinite(x)) continue;
        double value, slope;
        horner(a, n, x, &value, &slope);
        double best = fabs(value);
        for (int step = 0; step < 2; ++step) {
            horner(a, n, x, &value, &slope);
            if (slope == 0.0 || fabs(slope) < 1e-13 * fmax(fabs(value), 1e-300)) break;
            const double candidate = x - value / slope;
            if (!isfinite(candidate)) break;
            double trial, ignored;
            horner(a, n, candidate, &trial, &ignored);
            if (fabs(trial) >= best) break;
            x = candidate; best = fabs(trial);
        }
        out[i] = x;
    }
    return found;
}

// Real solutions in (-pi, pi] of  e0 + e1c cos p + e1s sin p + e2c cos 2p + e2s sin 2p = 0.
//
// Every feature-switch equation in this module reduces to this one shape. u = tan(p/2) turns it into a
// quartic; the substitution cannot represent p = pi, so that value is ALWAYS offered as an extra
// candidate. Returns the count; `out` must hold at least 5.
__device__ inline int solveTrig(double e0, double e1c, double e1s, double e2c, double e2s,
                                double* out) {
    double quartic[5];
    quartic[0] = e0 - e1c + e2c;
    quartic[1] = 2.0 * e1s - 4.0 * e2s;
    quartic[2] = 2.0 * e0 - 6.0 * e2c;
    quartic[3] = 2.0 * e1s + 4.0 * e2s;
    quartic[4] = e0 + e1c + e2c;

    int found = 0;
    double biggest = 0.0;
    for (int i = 0; i < 5; ++i) biggest = fmax(biggest, fabs(quartic[i]));
    if (biggest > 1e-11) {
        double roots[4];
        const int m = quarticRoots(quartic, 4, roots);
        const double scale = fmax(fabs(e0) + fabs(e1c) + fabs(e1s) + fabs(e2c) + fabs(e2s), 1e-300);
        for (int i = 0; i < m; ++i) {
            const double angle = 2.0 * atan(roots[i]);
            // Kept by whether it SOLVES the equation, never by whether the solver called it real.
            const double residual = fabs(e0 + e1c * cos(angle) + e1s * sin(angle)
                                         + e2c * cos(2.0 * angle) + e2s * sin(2.0 * angle));
            if (residual <= 1e-6 * scale) out[found++] = angle;
        }
    }
    out[found++] = M_PI;
    return found;
}

// ---------------------------------------------------------------------------------------------
// The body: corner frames from a backbone loop.
// ---------------------------------------------------------------------------------------------

// One corner's frame from (previous, vertex, next, rho). Cam's notation, notes/roundedDefinitions.tex:
//   t = rho / tan(theta/2), z = v + (rho / sin(theta/2)) bHat, psi = pi - theta,
//   a^- = v - t eInHat, a^+ = v + t eOutHat.
__device__ inline void cornerFrame(double2 previous, double2 vertex, double2 next, double rho,
                                   double2* center, double2* aMinus, double2* aPlus, double* sweep) {
    double2 incoming = sub2(vertex, previous);
    double2 outgoing = sub2(next, vertex);
    const double inLength = sqrt(dot2(incoming, incoming));
    const double outLength = sqrt(dot2(outgoing, outgoing));
    const double2 inHat = make_double2(incoming.x / inLength, incoming.y / inLength);
    const double2 outHat = make_double2(outgoing.x / outLength, outgoing.y / outLength);

    // Interior angle at the vertex, between the reversed incoming edge and the outgoing one.
    const double cosine = fmin(1.0, fmax(-1.0, -dot2(inHat, outHat)));
    const double theta = acos(cosine);
    const double half = 0.5 * theta;
    const double tangent = rho / fmax(tan(half), 1e-300);

    // Bisector, pointing INTO the polygon: the normalized sum of the two edge directions away from v.
    double2 bisector = make_double2(outHat.x - inHat.x, outHat.y - inHat.y);
    const double bisectorLength = sqrt(dot2(bisector, bisector));
    if (bisectorLength > 1e-300) {
        bisector.x /= bisectorLength; bisector.y /= bisectorLength;
    }
    const double offset = rho / fmax(sin(half), 1e-300);
    *center = make_double2(vertex.x + offset * bisector.x, vertex.y + offset * bisector.y);
    *aMinus = make_double2(vertex.x - tangent * inHat.x, vertex.y - tangent * inHat.y);
    *aPlus  = make_double2(vertex.x + tangent * outHat.x, vertex.y + tangent * outHat.y);
    // Convexity is the sign of the cross product of the two edges; the sweep carries it.
    const double cross = incoming.x * outgoing.y - incoming.y * outgoing.x;
    *sweep = (cross >= 0.0 ? 1.0 : -1.0) * (M_PI - theta);
}

// Fill a Body from one CCW backbone loop and its per-vertex radii.
__device__ inline void bodyFromBackbone(const double2* loop, const double* rho, int count,
                                        Body* body) {
    body->count = count;
    for (int k = 0; k < count; ++k) {
        const int previous = (k + count - 1) % count;
        const int next = (k + 1) % count;
        double2 center, aMinus, aPlus;
        double sweep;
        cornerFrame(loop[previous], loop[k], loop[next], rho[k], &center, &aMinus, &aPlus, &sweep);
        const bool degenerate = rho[k] <= 0.0;
        body->center[k] = degenerate ? loop[k] : center;
        body->radius[k] = rho[k];
        body->sweep[k]  = degenerate ? 0.0 : sweep;
        const double2 toStart = sub2(aMinus, center);
        body->start[k]  = degenerate ? 0.0 : atan2(toStart.y, toStart.x);
        body->tail[k]   = aPlus;
        // head[k] IS a^-_{k+1}, so corner k's own a^- belongs to segment k-1. Writing it here keeps
        // a^- computed exactly once, and matches the host's `head = aMinus[nextIndex]`.
        body->head[previous] = aMinus;
        (void) next;
    }
}

// ---------------------------------------------------------------------------------------------
// The distance field.
// ---------------------------------------------------------------------------------------------

// UNSIGNED distance from x to segment k, with the foot CLAMPED to the segment, plus `side`, the
// outward-normal component (positive outside).
//
// NOT THE DISTANCE TO THE SUPPORTING LINE. The signed line form d = A - M t is what the SWITCH
// equations use, and conflating the two is a real error: a far segment's line distance can undercut
// the true nearest feature and win the reduce. `side` is only meaningful where this segment is the
// nearest feature, which is the only place the caller reads it.
__device__ __forceinline__ double segmentDistance(const Body* body, int k, double2 x, double* side) {
    const double2 tail = body->tail[k];
    const double2 edge = sub2(body->head[k], tail);
    const double length2 = fmax(dot2(edge, edge), 1e-300);
    const double2 delta = sub2(x, tail);
    double t = dot2(delta, edge) / length2;
    t = fmin(1.0, fmax(0.0, t));
    const double2 offset = make_double2(delta.x - t * edge.x, delta.y - t * edge.y);
    const double length = sqrt(length2);
    // OUTWARD normal of a CCW loop is (tangent.y, -tangent.x). Do not re-derive it.
    *side = (delta.x * edge.y - delta.y * edge.x) / length;
    return sqrt(dot2(offset, offset));
}

// UNSIGNED distance from x to arc k, or INFINITY outside its angular wedge; `insideCircle` says which
// side of the corner circle x is on.
//
// INFINITY OUTSIDE THE WEDGE IS EXACT, NOT AN APPROXIMATION. Outside it the nearest point of the arc
// is an ENDPOINT, and every endpoint is also an endpoint of a tangent segment, so the segment branch
// already covers it. That is also what keeps the elliptic arc-versus-vertex case from ever arising.
__device__ __forceinline__ double arcDistance(const Body* body, int k, double2 x,
                                              bool* insideCircle) {
    const double2 delta = sub2(x, body->center[k]);
    const double reach = sqrt(dot2(delta, delta));
    *insideCircle = reach < body->radius[k];
    const double sweep = body->sweep[k];
    if (body->radius[k] <= 0.0 || fabs(sweep) < 1e-300) return 1e300;
    const double angle = atan2(delta.y, delta.x);
    double turned = (angle - body->start[k]) * (sweep >= 0.0 ? 1.0 : -1.0);
    turned = fmod(turned, 2.0 * M_PI);
    if (turned < 0.0) turned += 2.0 * M_PI;
    if (turned > fabs(sweep)) return 1e300;
    return fabs(reach - body->radius[k]);
}

// Nearest feature of `body` to x: UNSIGNED distance, with `inside` carrying the sign separately.
// `kind` is 0 for a segment and 1 for an arc.
//
// MEMBERSHIP COMES FROM THE NEAREST FEATURE, no ray cast, and the rule is simple because the boundary
// is C1 with no vertices: nearest is a segment -> inside iff on its inner side; nearest is an arc ->
// inside iff within that corner circle, which is right because the circle is pushed in from INSIDE so
// its centre is interior.
__device__ inline double nearestFeature(const Body* body, double2 x, bool* inside,
                                        int* kind, int* feature) {
    double bestSegment = 1e300, bestArc = 1e300, bestSide = 0.0;
    int segmentIndex = 0, arcIndex = 0;
    bool bestInsideCircle = false;
    for (int k = 0; k < body->count; ++k) {
        double side;
        const double d = segmentDistance(body, k, x, &side);
        if (d < bestSegment) { bestSegment = d; segmentIndex = k; bestSide = side; }
    }
    for (int k = 0; k < body->count; ++k) {
        bool insideCircle;
        const double d = arcDistance(body, k, x, &insideCircle);
        if (d < bestArc) { bestArc = d; arcIndex = k; bestInsideCircle = insideCircle; }
    }
    if (bestArc < bestSegment) {
        *inside = bestInsideCircle; *kind = 1; *feature = arcIndex;
        return bestArc;
    }
    *inside = bestSide < 0.0; *kind = 0; *feature = segmentIndex;
    return bestSegment;
}

// Distance to the boundary, POSITIVE INSIDE -- the sign convention the contact law integrates.
__device__ __forceinline__ double signedDistance(const Body* body, double2 x,
                                                 int* kind, int* feature) {
    bool inside;
    const double distance = nearestFeature(body, x, &inside, kind, feature);
    return inside ? distance : -distance;
}

// A point on piece `p` of the body at parameter t in [0,1]. Pieces 0..n-1 are arcs, n..2n-1 segments.
// The arc is parametrized by ROTATING w0 = head[k-1] - z, never by an absolute angle.
__device__ __forceinline__ double2 evaluatePiece(const Body* body, int p, double t) {
    const int n = body->count;
    if (p >= n) {
        const int k = p - n;
        const double2 tail = body->tail[k];
        const double2 edge = sub2(body->head[k], tail);
        return make_double2(tail.x + t * edge.x, tail.y + t * edge.y);
    }
    const double2 center = body->center[p];
    const double2 start = sub2(body->head[(p + n - 1) % n], center);
    const double2 turned = turn2(start);
    const double angle = body->sweep[p] * t;
    const double c = cos(angle), s = sin(angle);
    return make_double2(center.x + c * start.x + s * turned.x,
                        center.y + c * start.y + s * turned.y);
}

// ---------------------------------------------------------------------------------------------
// Crossings of dA with dB, and the spans of dA that lie inside B.
// ---------------------------------------------------------------------------------------------

// Where a point already ON arc k's circle sits along it, as a fraction of the sweep; a negative return
// means OFF the arc. Only the angular window is resolved -- a circle-circle or line-circle solve
// leaves exactly that undetermined.
__device__ __forceinline__ double arcParameter(const Body* body, int k, double2 point,
                                               double tolerance) {
    const double sweep = body->sweep[k];
    if (body->radius[k] <= 0.0 || fabs(sweep) <= 0.0) return -1.0;
    const double2 delta = sub2(point, body->center[k]);
    const double angle = atan2(delta.y, delta.x);
    double fraction = (angle - body->start[k]) * (sweep >= 0.0 ? 1.0 : -1.0);
    fraction = fmod(fraction, 2.0 * M_PI);
    if (fraction < 0.0) fraction += 2.0 * M_PI;
    fraction /= fabs(sweep);
    if (fraction > 1.0 + tolerance) {
        // A crossing exactly at an endpoint belongs to the arc; the wrap puts it just under the full
        // turn rather than just over zero.
        if (fraction > (2.0 * M_PI / fabs(sweep)) - tolerance) return 0.0;
        return -1.0;
    }
    return fmin(1.0, fmax(0.0, fraction));
}

// Parameters in [0,1] along tail->head where it meets the circle. Returns the count (0, 1 or 2).
//
// THE TANGENCY TEST COMES BEFORE THE SIGN TEST, and is relative to the terms that cancelled. Near a
// double root sqrt(discriminant) amplifies roundoff to about sqrt(machine epsilon): measured on the
// host, a tangency at t = 0.125 came back as 0.12499998097 and 0.12500001903, and keeping both left a
// 3.8e-08 span whose midpoint sat exactly on the tangency, where the nearest feature is ambiguous.
__device__ __forceinline__ int lineCircle(double2 tail, double2 head, double2 center, double radius,
                                          double* out) {
    const double2 vector = sub2(head, tail);
    const double2 offset = sub2(tail, center);
    const double a = dot2(vector, vector);
    if (a <= 1e-300) return 0;
    const double b = 2.0 * dot2(offset, vector);
    const double c = dot2(offset, offset) - radius * radius;
    const double discriminant = b * b - 4.0 * a * c;
    double candidate[2];
    int found;
    if (fabs(discriminant) <= 1e-13 * fmax(b * b + 4.0 * fabs(a * c), 1e-300)) {
        candidate[0] = -b / (2.0 * a);
        found = 1;
    } else if (discriminant < 0.0) {
        return 0;
    } else {
        const double root = sqrt(discriminant);
        candidate[0] = (-b - root) / (2.0 * a);
        candidate[1] = (-b + root) / (2.0 * a);
        found = 2;
    }
    int kept = 0;
    for (int i = 0; i < found; ++i)
        if (candidate[i] >= -1e-12 && candidate[i] <= 1.0 + 1e-12) out[kept++] = candidate[i];
    return kept;
}

// Intersection points of two circles. Returns the count (0, 1 or 2); tangent circles give ONE.
__device__ __forceinline__ int circleCircle(double2 centerA, double radiusA,
                                            double2 centerB, double radiusB, double2* out) {
    const double2 delta = sub2(centerB, centerA);
    const double separation = sqrt(dot2(delta, delta));
    const double slack = 1e-9 * fmax(radiusA + radiusB, 1e-300);
    if (separation <= 1e-300 || separation > radiusA + radiusB + slack ||
        separation < fabs(radiusA - radiusB) - slack) return 0;
    const double along = (separation * separation + radiusA * radiusA - radiusB * radiusB)
                         / (2.0 * separation);
    const double height2 = radiusA * radiusA - along * along;
    const double2 base = make_double2(centerA.x + along * delta.x / separation,
                                      centerA.y + along * delta.y / separation);
    if (fabs(height2) <= 1e-13 * fmax(radiusA * radiusA + along * along, 1e-300) || height2 < 0.0) {
        out[0] = base;
        return 1;
    }
    const double height = sqrt(height2);
    const double2 perpendicular = make_double2(-delta.y / separation, delta.x / separation);
    out[0] = make_double2(base.x + height * perpendicular.x, base.y + height * perpendicular.y);
    out[1] = make_double2(base.x - height * perpendicular.x, base.y - height * perpendicular.y);
    return 2;
}

// The next crossing of piece `p` of A with dB strictly beyond `after`, or 1.0 if there is none.
//
// EVERY PAIR IS TESTED, never a prefiltered subset -- polyContact records that pruning cost five
// genuine switches across 107 spans and moved the energy 0.7%. A spurious crossing is harmless (it
// subdivides an interval whose state does not change); a missed one is not.
//
// SEARCHED RATHER THAN COLLECTED. Returning the minimum above `after` reproduces the host's
// sort-then-dedup exactly -- both keep the FIRST of a cluster within `tolerance` -- while needing no
// per-thread array, which is the whole reason this file can run one thread per piece.
__device__ inline double nextCrossing(const Body* A, const Body* B, int p, double after,
                                      double tolerance) {
    const int n = A->count;
    double best = 1.0;
    const double floorValue = after + tolerance;
    double hits[2];
    double2 points[2];

    if (p < n) {
        if (A->radius[p] <= 0.0 || fabs(A->sweep[p]) <= 0.0) return 1.0;
        for (int j = 0; j < B->count; ++j) {
            if (B->radius[j] <= 0.0 || fabs(B->sweep[j]) <= 0.0) continue;
            const int m = circleCircle(A->center[p], A->radius[p],
                                       B->center[j], B->radius[j], points);
            for (int i = 0; i < m; ++i) {
                const double mine = arcParameter(A, p, points[i], 1e-9);
                const double theirs = arcParameter(B, j, points[i], 1e-9);
                if (mine >= 0.0 && theirs >= 0.0 && mine > floorValue && mine < best) best = mine;
            }
        }
        for (int j = 0; j < B->count; ++j) {
            const int m = lineCircle(B->tail[j], B->head[j], A->center[p], A->radius[p], hits);
            for (int i = 0; i < m; ++i) {
                const double2 edge = sub2(B->head[j], B->tail[j]);
                const double2 point = make_double2(B->tail[j].x + hits[i] * edge.x,
                                                   B->tail[j].y + hits[i] * edge.y);
                const double mine = arcParameter(A, p, point, 1e-9);
                if (mine >= 0.0 && mine > floorValue && mine < best) best = mine;
            }
        }
        return best;
    }

    const int k = p - n;
    const double2 tail = A->tail[k];
    const double2 vector = sub2(A->head[k], tail);
    for (int j = 0; j < B->count; ++j) {
        if (B->radius[j] <= 0.0 || fabs(B->sweep[j]) <= 0.0) continue;
        const int m = lineCircle(tail, A->head[k], B->center[j], B->radius[j], hits);
        for (int i = 0; i < m; ++i) {
            const double2 point = make_double2(tail.x + hits[i] * vector.x,
                                               tail.y + hits[i] * vector.y);
            if (arcParameter(B, j, point, 1e-9) >= 0.0 && hits[i] > floorValue && hits[i] < best)
                best = hits[i];
        }
    }
    for (int j = 0; j < B->count; ++j) {
        const double2 other = sub2(B->head[j], B->tail[j]);
        const double denominator = vector.x * other.y - vector.y * other.x;
        if (fabs(denominator) < 1e-300) continue;
        const double2 gap = sub2(B->tail[j], tail);
        const double t = (gap.x * other.y - gap.y * other.x) / denominator;
        const double u = (gap.x * vector.y - gap.y * vector.x) / denominator;
        if (t >= -1e-12 && t <= 1.0 + 1e-12 && u >= -1e-12 && u <= 1.0 + 1e-12 &&
            t > floorValue && t < best) best = t;
    }
    return best;
}

// Arc length of a piece. Zero for a degenerate corner or a vanished straight run.
__device__ __forceinline__ double pieceLength(const Body* body, int p) {
    const int n = body->count;
    if (p < n) return body->radius[p] * fabs(body->sweep[p]);
    const double2 edge = sub2(body->head[p - n], body->tail[p - n]);
    return sqrt(dot2(edge, edge));
}

// ---------------------------------------------------------------------------------------------
// Feature switches: where the nearest feature of B changes along a piece of A.
//
// SIGNED DISTANCES ARE WHAT KEEP THE DEGREES DOWN. Inside B every distance is positive, so along a
// SEGMENT of A a to-segment reads d = A - M t (linear) and a to-arc reads d = rho - sqrt(quadratic).
// The pairwise equations are then linear, or quadratic after ONE squaring -- arc-versus-arc included,
// because both quadratics share the leading coefficient |v|^2 and their difference is linear. Along an
// ARC of A the same distances become first-harmonic trigonometric polynomials and every equation lands
// on solveTrig's single shape.
//
// NOTHING IS STAGED. Coefficients are recomputed inside the pair loop rather than held for all 2n
// features: at ROUNDED_MAXN = 16 that array would be ~1.5 KB per thread and would spill, and each set
// costs about ten flops to rebuild.
// ---------------------------------------------------------------------------------------------

// Real roots of a x^2 + b x + c, with a TANGENCY returned as one root. Distinct from
// quadraticCandidates: this one rejects a genuinely negative discriminant, and is what the segment
// switch equations want.
__device__ __forceinline__ int realQuadratic(double a, double b, double c, double* out) {
    if (fabs(a) < 1e-300) {
        if (fabs(b) < 1e-300) return 0;
        out[0] = -c / b;
        return 1;
    }
    const double discriminant = b * b - 4.0 * a * c;
    // Tangency is tested on the ABSOLUTE discriminant and BEFORE negatives are rejected. Rejecting
    // first loses exactly the roots that matter: measured on the host, a real switch whose
    // discriminant landed at -2.2e-16 was discarded that way.
    if (fabs(discriminant) <= 1e-13 * fmax(b * b + 4.0 * fabs(a * c), 1e-300)) {
        out[0] = -b / (2.0 * a);
        return 1;
    }
    if (discriminant < 0.0) return 0;
    const double root = sqrt(discriminant);
    out[0] = (-b - root) / (2.0 * a);
    out[1] = (-b + root) / (2.0 * a);
    return 2;
}

// Distance from B's segment j to the point tail + t*vector: d = value - slope * t, positive inside.
__device__ __forceinline__ void segmentAlongSegment(const Body* B, int j, double2 tail,
                                                    double2 vector, double* value, double* slope) {
    const double2 edge = sub2(B->head[j], B->tail[j]);
    const double length = fmax(sqrt(dot2(edge, edge)), 1e-300);
    const double2 normal = make_double2(edge.y / length, -edge.x / length);
    *value = dot2(normal, sub2(B->tail[j], tail));
    *slope = dot2(normal, vector);
}

// Distance from B's arc j along the same ray: d = rho - sqrt(qa t^2 + qb t + qc).
__device__ __forceinline__ void arcAlongSegment(const Body* B, int j, double2 tail, double2 vector,
                                                double* rho, double* qa, double* qb, double* qc) {
    const double2 offset = sub2(tail, B->center[j]);
    *rho = B->radius[j];
    *qa = dot2(vector, vector);
    *qb = 2.0 * dot2(offset, vector);
    *qc = dot2(offset, offset);
}

// Distance from B's segment j along a circle of `radius` about `center`:
// d = value + xTerm cos p + yTerm sin p.
__device__ __forceinline__ void segmentAlongArc(const Body* B, int j, double2 center, double radius,
                                                double* value, double* xTerm, double* yTerm) {
    const double2 edge = sub2(B->head[j], B->tail[j]);
    const double length = fmax(sqrt(dot2(edge, edge)), 1e-300);
    const double2 normal = make_double2(edge.y / length, -edge.x / length);
    *value = dot2(normal, sub2(B->tail[j], center));
    *xTerm = -radius * normal.x;
    *yTerm = -radius * normal.y;
}

// Distance from B's arc j along the same circle: d = rho - sqrt(p + cx cos + cy sin).
__device__ __forceinline__ void arcAlongArc(const Body* B, int j, double2 center, double radius,
                                            double* rho, double* p, double* cx, double* cy) {
    const double2 toCenter = sub2(center, B->center[j]);
    *rho = B->radius[j];
    *p = dot2(toCenter, toCenter) + radius * radius;
    *cx = 2.0 * radius * toCenter.x;
    *cy = 2.0 * radius * toCenter.y;
}

// Coefficients of (g0 + gx cos + gy sin)^2 - (p + cx cos + cy sin) in solveTrig's basis. The squaring
// is where the second harmonic appears.
__device__ __forceinline__ void squareFirstHarmonic(double g0, double gx, double gy,
                                                    double p, double cx, double cy, double* e) {
    e[0] = g0 * g0 + 0.5 * (gx * gx + gy * gy) - p;
    e[1] = 2.0 * g0 * gx - cx;
    e[2] = 2.0 * g0 * gy - cy;
    e[3] = 0.5 * (gx * gx - gy * gy);
    e[4] = gx * gy;
}

// Smallest feature switch strictly beyond `after` and below `limit` along a SEGMENT of A, or `limit`.
__device__ inline double nextSwitchAlongSegment(const Body* B, double2 tail, double2 vector,
                                                double after, double limit, double tolerance) {
    double best = limit;
    const double floorValue = after + tolerance;
    const int count = B->count;
    double roots[2];

    // segment versus segment: linear.
    for (int i = 0; i < count; ++i) {
        double valueI, slopeI;
        segmentAlongSegment(B, i, tail, vector, &valueI, &slopeI);
        for (int j = i + 1; j < count; ++j) {
            double valueJ, slopeJ;
            segmentAlongSegment(B, j, tail, vector, &valueJ, &slopeJ);
            const double slope = slopeI - slopeJ;
            if (fabs(slope) <= 1e-300) continue;
            const double t = (valueI - valueJ) / slope;
            if (t > floorValue && t < best) best = t;
        }
    }
    // segment versus arc: one squaring.
    for (int i = 0; i < count; ++i) {
        double valueI, slopeI;
        segmentAlongSegment(B, i, tail, vector, &valueI, &slopeI);
        for (int j = 0; j < count; ++j) {
            if (B->radius[j] <= 0.0) continue;
            double rho, qa, qb, qc;
            arcAlongSegment(B, j, tail, vector, &rho, &qa, &qb, &qc);
            const double g0 = rho - valueI, g1 = slopeI;
            const int m = realQuadratic(qa - g1 * g1, qb - 2.0 * g0 * g1, qc - g0 * g0, roots);
            for (int r = 0; r < m; ++r)
                if (roots[r] > floorValue && roots[r] < best) best = roots[r];
        }
    }
    // arc versus arc: the two quadratics share |v|^2, so their difference is LINEAR and one squaring
    // is enough. This is why signed distances were used.
    for (int i = 0; i < count; ++i) {
        if (B->radius[i] <= 0.0) continue;
        double rhoI, qaI, qbI, qcI;
        arcAlongSegment(B, i, tail, vector, &rhoI, &qaI, &qbI, &qcI);
        for (int j = i + 1; j < count; ++j) {
            if (B->radius[j] <= 0.0) continue;
            double rhoJ, qaJ, qbJ, qcJ;
            arcAlongSegment(B, j, tail, vector, &rhoJ, &qaJ, &qbJ, &qcJ);
            const double delta = rhoI - rhoJ;
            const double db = qbI - qbJ, dc = qcI - qcJ;
            if (fabs(delta) < 1e-14) {
                if (fabs(db) > 1e-300) {
                    const double t = -dc / db;
                    if (t > floorValue && t < best) best = t;
                }
                continue;
            }
            const double half = 1.0 / (2.0 * delta);
            const double h1 = db * half, h0 = (dc - delta * delta) * half;
            const int m = realQuadratic(qaJ - h1 * h1, qbJ - 2.0 * h0 * h1, qcJ - h0 * h0, roots);
            for (int r = 0; r < m; ++r)
                if (roots[r] > floorValue && roots[r] < best) best = roots[r];
        }
    }
    return best;
}

// Smallest feature switch strictly beyond `after` and below `limit` along ARC `p` of A, or `limit`.
//
// EVERY ANGLE IS OFFERED IN THREE WRAPPINGS. solveTrig returns values in (-pi, pi]; the arc's own
// parameter runs start -> start + sweep, which can straddle the branch cut, so a switch just outside
// the principal branch would otherwise be lost.
__device__ inline double nextSwitchAlongArc(const Body* A, const Body* B, int p,
                                            double after, double limit, double tolerance) {
    const double2 center = A->center[p];
    const double radius = A->radius[p];
    const double start = A->start[p], sweep = A->sweep[p];
    if (radius <= 0.0 || fabs(sweep) <= 0.0) return limit;

    double best = limit;
    const double floorValue = after + tolerance;
    const int count = B->count;
    double e[5], angles[5];

    #define ROUNDED_OFFER(count_, source_)                                                       \
        for (int r = 0; r < (count_); ++r) {                                                     \
            for (int w = -1; w <= 1; ++w) {                                                      \
                const double t = ((source_)[r] + w * 2.0 * M_PI - start) / sweep;                \
                if (t > floorValue && t < best) best = t;                                        \
            }                                                                                    \
        }

    for (int i = 0; i < count; ++i) {
        double valueI, xI, yI;
        segmentAlongArc(B, i, center, radius, &valueI, &xI, &yI);
        for (int j = i + 1; j < count; ++j) {
            double valueJ, xJ, yJ;
            segmentAlongArc(B, j, center, radius, &valueJ, &xJ, &yJ);
            const int m = solveTrig(valueI - valueJ, xI - xJ, yI - yJ, 0.0, 0.0, angles);
            ROUNDED_OFFER(m, angles)
        }
    }
    for (int i = 0; i < count; ++i) {
        double valueI, xI, yI;
        segmentAlongArc(B, i, center, radius, &valueI, &xI, &yI);
        for (int j = 0; j < count; ++j) {
            if (B->radius[j] <= 0.0) continue;
            double rho, pp, cx, cy;
            arcAlongArc(B, j, center, radius, &rho, &pp, &cx, &cy);
            squareFirstHarmonic(rho - valueI, -xI, -yI, pp, cx, cy, e);
            const int m = solveTrig(e[0], e[1], e[2], e[3], e[4], angles);
            ROUNDED_OFFER(m, angles)
        }
    }
    for (int i = 0; i < count; ++i) {
        if (B->radius[i] <= 0.0) continue;
        double rhoI, pI, cxI, cyI;
        arcAlongArc(B, i, center, radius, &rhoI, &pI, &cxI, &cyI);
        for (int j = i + 1; j < count; ++j) {
            if (B->radius[j] <= 0.0) continue;
            double rhoJ, pJ, cxJ, cyJ;
            arcAlongArc(B, j, center, radius, &rhoJ, &pJ, &cxJ, &cyJ);
            const double delta = rhoI - rhoJ;
            const double dP = pI - pJ, dX = cxI - cxJ, dY = cyI - cyJ;
            int m;
            if (fabs(delta) < 1e-14) {
                m = solveTrig(dP, dX, dY, 0.0, 0.0, angles);
            } else {
                const double half = 1.0 / (2.0 * delta);
                squareFirstHarmonic((dP - delta * delta) * half, dX * half, dY * half,
                                    pJ, cxJ, cyJ, e);
                m = solveTrig(e[0], e[1], e[2], e[3], e[4], angles);
            }
            ROUNDED_OFFER(m, angles)
        }
    }
    #undef ROUNDED_OFFER
    return best;
}

// The next sub-stretch boundary after `after`, within the span ending at `limit`.
//
// THE SEARCH STOPS `tolerance` SHORT OF THE SPAN END, matching the host's `cuts < high - tolerance`.
// Without that a switch landing just inside the end opens a sliver narrower than the tolerance, and
// while a spurious breakpoint is harmless to the ENERGY -- the integrand is continuous, so the halves
// sum to the whole -- it is NOT harmless to the frozen-partition GRADIENT: adjacent sub-stretches
// cancel each other's boundary terms only where the two features really do read the same distance, so
// a sliver at an approximate switch leaves a residual with nothing to cancel against. Measured, that
// alone moved the gradient 3.7e-07 relative while the energy stayed at 1e-15.
__device__ __forceinline__ double nextSwitch(const Body* A, const Body* B, int p,
                                             double after, double limit, double tolerance) {
    const int n = A->count;
    const double ceiling = limit - tolerance;
    double found;
    if (p < n) {
        found = nextSwitchAlongArc(A, B, p, after, ceiling, tolerance);
    } else {
        const int k = p - n;
        found = nextSwitchAlongSegment(B, A->tail[k], sub2(A->head[k], A->tail[k]),
                                       after, ceiling, tolerance);
    }
    return (found >= ceiling) ? limit : found;
}

// The winning feature of B at the MIDPOINT of a sub-stretch. Never taken from the walk: trusting a
// marcher's winner list inflated the host's energy EIGHTFOLD and surfaced only as an FD failure.
__device__ __forceinline__ void substretchWinner(const Body* A, const Body* B, int p,
                                                 double low, double high, int* kind, int* feature) {
    const double2 midpoint = evaluatePiece(A, p, 0.5 * (low + high));
    bool inside;
    nearestFeature(B, midpoint, &inside, kind, feature);
}

// Is the midpoint of [low, high] on piece `p` of A inside B?
//
// MEMBERSHIP IS READ AT THE MIDPOINT, never at a crossing: a state read exactly at a transition is not
// reliably on either side of it.
__device__ __forceinline__ bool spanInside(const Body* A, const Body* B, int p,
                                           double low, double high) {
    const double2 midpoint = evaluatePiece(A, p, 0.5 * (low + high));
    bool inside;
    int kind, feature;
    nearestFeature(B, midpoint, &inside, &kind, &feature);
    return inside;
}


// ---------------------------------------------------------------------------------------------
// The four integrals, each returning its VALUE and its PARTIALS in one pass.
//
// Which are elementary, worked out before the design was committed to:
//
//     along   | to      | integrand                    | status
//     --------|---------|------------------------------|-------------------------------------
//     segment | segment | (alpha - m t)^3              | elementary
//     segment | arc     | (r - sqrt(w^2 + h^2))^3      | ELEMENTARY in ARC LENGTH -- see below
//     arc     | segment | (a + b cos + c sin)^3        | elementary, polynomial in cos and sin
//     arc     | arc     | (rho - sqrt(P + C cos))^3    | ELLIPTIC, so Gauss-Legendre
//
// See ../notes/roundedContactGradient.tex for the derivation of every partial here.
// ---------------------------------------------------------------------------------------------

// Gauss-Legendre order 24, generated by numpy.polynomial.legendre.leggauss and MATCHING the
// host's default exactly -- the arc-along/arc-to integrand is elliptic, so this is the one
// branch that is quadrature rather than closed form. It converges geometrically on a
// sub-stretch (measured 8.9e-07 at order 2, 2.8e-16 by order 8, flat at 4e-19 past 12), so 24
// is far inside the noise and the order is a convergence knob rather than a tolerance.
#define ROUNDED_QUADRATURE 24
__constant__ double kNodes[ROUNDED_QUADRATURE] = {
    -0.99518721999702131, -0.97472855597130947, -0.9382745520027328, -0.88641552700440096, -0.82000198597390295, -0.74012419157855436,
    -0.64809365193697555, -0.54542147138883956, -0.43379350762604513, -0.3150426796961634, -0.19111886747361631, -0.06405689286260563,
    0.06405689286260563, 0.19111886747361631, 0.3150426796961634, 0.43379350762604513, 0.54542147138883956, 0.64809365193697555,
    0.74012419157855436, 0.82000198597390295, 0.88641552700440096, 0.9382745520027328, 0.97472855597130947, 0.99518721999702131
};
__constant__ double kWeights[ROUNDED_QUADRATURE] = {
    0.012341229799987091, 0.028531388628933743, 0.044277438817419551, 0.059298584915436742, 0.073346481411080411, 0.086190161531953288,
    0.097618652104114065, 0.10744427011596561, 0.11550566805372561, 0.12167047292780342, 0.1258374563468283, 0.12793819534675221,
    0.12793819534675221, 0.1258374563468283, 0.12167047292780342, 0.11550566805372561, 0.10744427011596561, 0.097618652104114065,
    0.086190161531953288, 0.073346481411080411, 0.059298584915436742, 0.044277438817419551, 0.028531388628933743, 0.012341229799987091
};

// int (a - m t)^3 dt over [lo, hi], with d/da and d/dm.
//
// EXPANDED, NOT -((a - m hi)^4 - (a - m lo)^4)/(4m). That closed form divides by m, which is ZERO
// whenever the two segments are PARALLEL -- routine between axis-aligned squares.
__device__ __forceinline__ void lineIntegral(double a, double m, double lo, double hi,
                                             double* value, double* dA, double* dM) {
    const double d1 = hi - lo;
    const double d2 = hi * hi - lo * lo;
    const double d3 = hi * hi * hi - lo * lo * lo;
    const double d4 = hi * hi * hi * hi - lo * lo * lo * lo;
    *value = a * a * a * d1 - 1.5 * a * a * m * d2 + a * m * m * d3 - 0.25 * m * m * m * d4;
    *dA = 3.0 * a * a * d1 - 3.0 * a * m * d2 + m * m * d3;
    *dM = -1.5 * a * a * d2 + 2.0 * a * m * d3 - 0.75 * m * m * d4;
}

// int (r - sqrt(w^2 + h^2))^3 dw over [lowW, highW], with partials in r, h and both endpoints.
//
// THE SEGMENT-ALONG, ARC-TO CASE IN ITS NATURAL COORDINATE. Written as a general quadratic in the
// segment's own parameter, the partials pick up q_a^(13/2) denominators. Rewriting in arc length --
// w measured from the foot of the perpendicular, h the perpendicular distance -- absorbs the speed
// into the measure and leaves partials made of the SAME five antiderivatives the value needs.
//
// h -> 0 (the arc centre on the chord's line) is removable: h asinh(w/h) tends to zero, so d/dh does.
__device__ __forceinline__ void arcToIntegral(double r, double h, double lowW, double highW,
                                              double* value, double* dR, double* dH,
                                              double* dLow, double* dHigh) {
    const double h2 = h * h;
    double arc[2], zero[2], first[2], second[2], third[2], root[2];
    const double w[2] = { lowW, highW };
    for (int i = 0; i < 2; ++i) {
        root[i] = sqrt(w[i] * w[i] + h2);
        arc[i] = (h > 0.0) ? asinh(w[i] / h) : 0.0;
        zero[i] = w[i];
        first[i] = 0.5 * w[i] * root[i] + 0.5 * h2 * arc[i];
        second[i] = w[i] * w[i] * w[i] / 3.0 + h2 * w[i];
        third[i] = w[i] * (2.0 * w[i] * w[i] + 5.0 * h2) * root[i] / 8.0
                   + 0.375 * h2 * h2 * arc[i];
    }
    const double dArc = arc[1] - arc[0], d0 = zero[1] - zero[0];
    const double d1 = first[1] - first[0], d2 = second[1] - second[0], d3 = third[1] - third[0];
    *value = r * r * r * d0 - 3.0 * r * r * d1 + 3.0 * r * d2 - d3;
    *dR = 3.0 * r * r * d0 - 6.0 * r * d1 + 3.0 * d2;
    *dH = -3.0 * h * (r * r * dArc - 2.0 * r * d0 + d1);
    *dLow = -(r - root[0]) * (r - root[0]) * (r - root[0]);
    *dHigh = (r - root[1]) * (r - root[1]) * (r - root[1]);
}

// Antiderivative of (a + b cos psi + c sin psi)^3 at psi, with partials in a, b, c and psi.
//
// NO PHASE ANGLE ANYWHERE. The single-amplitude form a + C cos(psi - gamma) is tidier but needs
// atan2, which is not analytic; the expanded form costs a few terms and stays differentiable, which
// is what lets the host complex-step the same expression this transcribes.
//
// dF/dpsi is the integrand itself, by the fundamental theorem -- and it is the cheapest available
// check on the other three.
__device__ __forceinline__ void harmonicIntegral(double a, double b, double c, double psi,
                                                 double* value, double* dA, double* dB,
                                                 double* dC, double* dPsi) {
    const double cosine = cos(psi), sine = sin(psi);
    const double doubleCos = cos(2.0 * psi), doubleSin = sin(2.0 * psi);
    const double linear = b * sine - c * cosine;
    const double square = (b * b + c * c) * psi / 2.0 + (b * b - c * c) * doubleSin / 4.0
                          - b * c * doubleCos / 2.0;
    const double cube = b * b * b * (sine - sine * sine * sine / 3.0) - b * b * c * cosine * cosine * cosine
                        + b * c * c * sine * sine * sine + c * c * c * (-cosine + cosine * cosine * cosine / 3.0);
    *value = a * a * a * psi + 3.0 * a * a * linear + 3.0 * a * square + cube;
    *dA = 3.0 * a * a * psi + 6.0 * a * linear - 1.5 * b * c * doubleCos
          + 1.5 * psi * (b * b + c * c) + 0.75 * (b * b - c * c) * doubleSin;
    *dB = 3.0 * a * a * sine + 1.5 * a * (2.0 * b * psi + b * doubleSin - c * doubleCos)
          + b * b * (cosine * cosine + 2.0) * sine - 2.0 * b * c * cosine * cosine * cosine
          + c * c * sine * sine * sine;
    *dC = -3.0 * a * a * cosine - 1.5 * a * (b * doubleCos - 2.0 * c * psi + c * doubleSin)
          - b * b * cosine * cosine * cosine + 2.0 * b * c * sine * sine * sine
          + c * c * (cosine * cosine - 3.0) * cosine;
    const double inner = a + b * cosine + c * sine;
    *dPsi = inner * inner * inner;
}

// ---------------------------------------------------------------------------------------------
// One sub-stretch's contribution to the energy and to dE/d(body arrays).
//
// THE GRADIENT IS TAKEN IN THE BODY ARRAYS, not in the backbone: centre, radius, sweep, tail, head.
// That is what makes the cost independent of the degree-of-freedom count -- one thread per
// sub-stretch, a fixed amount of arithmetic, an atomic scatter -- and it leaves the corner map's own
// derivative to a separate per-body pass that runs once per force evaluation rather than once per pair.
//
// THE PARTITION IS FROZEN, and that is exact rather than an approximation. Differentiating the true
// energy would add a Leibniz term wherever a breakpoint moves, and every one is zero: at a span
// endpoint d_B = 0, and at an interior feature switch d_B is continuous so the two adjacent
// sub-stretches cancel. So the argmin and the root solving never have to be differentiated.
//
// Flat gradient layout, matching the host's BodyGradient.flat():
//     [0, 2n)   centre      [2n, 3n)  radius     [3n, 4n)  sweep
//     [4n, 6n)  tail        [6n, 8n)  head
// ---------------------------------------------------------------------------------------------

__device__ __forceinline__ void scatter2(double* gradient, int base, double2 value) {
    atomicAdd(gradient + base + 0, value.x);
    atomicAdd(gradient + base + 1, value.y);
}

// d/d(edge) of (g . n) for the inward unit normal n = K e / |e|, K e = (e_y, -e_x).
//
// THE TWO TERMS CARRY DIFFERENT POWERS OF |e|. Factoring a single 1/|e| out of both is the obvious
// slip and is nearly invisible: measured on the host it left one component 1.6% wrong and the other
// completely wrong, so a spot check on the wrong component reads as a pass.
__device__ __forceinline__ double2 normalPullback(double2 g, double2 edge, double2 normal,
                                                  double length) {
    const double along = dot2(g, normal);
    return make_double2((-g.y - along * edge.x / length) / length,
                        ( g.x - along * edge.y / length) / length);
}

__device__ inline double substretchGradient(const Body* A, const Body* B, int p,
                                            double lo, double hi, int kind, int feature,
                                            double scale, double* gradA, double* gradB) {
    if (hi - lo <= 0.0) return 0.0;
    const int n = A->count;
    const int f = feature;

    // B's edge frame, needed by both to-segment cases.
    const double2 edge = sub2(B->head[f], B->tail[f]);
    const double length = fmax(sqrt(dot2(edge, edge)), 1e-300);
    const double2 normal = make_double2(edge.y / length, -edge.x / length);

    if (p >= n) {
        const int j = p - n;
        const double2 tail = A->tail[j];
        const double2 vector = sub2(A->head[j], tail);
        const double speed = sqrt(dot2(vector, vector));
        if (speed <= 0.0) return 0.0;
        const double2 unit = make_double2(vector.x / speed, vector.y / speed);

        if (kind == 0) {
            const double2 offset = sub2(B->tail[f], tail);
            const double a = dot2(normal, offset);
            const double m = dot2(normal, vector);
            double value, dA, dM;
            lineIntegral(a, m, lo, hi, &value, &dA, &dM);
            const double weight = scale * speed;
            scatter2(gradA, 4 * ROUNDED_MAXN + 2 * j,
                     make_double2(-unit.x * scale * value - normal.x * weight * (dA + dM),
                                  -unit.y * scale * value - normal.y * weight * (dA + dM)));
            scatter2(gradA, 6 * ROUNDED_MAXN + 2 * j,
                     make_double2(unit.x * scale * value + normal.x * weight * dM,
                                  unit.y * scale * value + normal.y * weight * dM));
            const double2 toNormal = make_double2(weight * (dA * offset.x + dM * vector.x),
                                                  weight * (dA * offset.y + dM * vector.y));
            const double2 conjugate = normalPullback(toNormal, edge, normal, length);
            scatter2(gradB, 6 * ROUNDED_MAXN + 2 * f, conjugate);
            scatter2(gradB, 4 * ROUNDED_MAXN + 2 * f,
                     make_double2(-conjugate.x + normal.x * weight * dA,
                                  -conjugate.y + normal.y * weight * dA));
            return scale * speed * value;
        }

        // Segment along, arc to -- in ARC LENGTH, which is what keeps the partials elementary.
        const double2 delta = sub2(tail, B->center[f]);
        const double radius = B->radius[f];
        const double along = dot2(delta, unit);
        const double2 perpendicular = make_double2(delta.x - along * unit.x,
                                                   delta.y - along * unit.y);
        const double height = sqrt(dot2(perpendicular, perpendicular));
        double value, dRadius, dHeight, dLow, dHigh;
        arcToIntegral(radius, height, speed * lo + along, speed * hi + along,
                      &value, &dRadius, &dHeight, &dLow, &dHigh);
        const double shift = dLow + dHigh;
        const double stretch = lo * dLow + hi * dHigh;
        double2 direction = make_double2(0.0, 0.0);
        if (height > 0.0) {
            direction = make_double2(perpendicular.x / height, perpendicular.y / height);
        } else {
            dHeight = 0.0;   // h asinh(w/h) -> 0, so the whole partial does
        }
        const double2 toDelta = make_double2(scale * (shift * unit.x + dHeight * direction.x),
                                             scale * (shift * unit.y + dHeight * direction.y));
        const double2 toVector = make_double2(
            scale * (stretch * unit.x + (shift * perpendicular.x - along * dHeight * direction.x) / speed),
            scale * (stretch * unit.y + (shift * perpendicular.y - along * dHeight * direction.y) / speed));
        scatter2(gradA, 4 * ROUNDED_MAXN + 2 * j, make_double2(toDelta.x - toVector.x, toDelta.y - toVector.y));
        scatter2(gradA, 6 * ROUNDED_MAXN + 2 * j, toVector);
        scatter2(gradB, 2 * f, make_double2(-toDelta.x, -toDelta.y));
        atomicAdd(gradB + 2 * ROUNDED_MAXN + f, scale * dRadius);
        return scale * value;
    }

    // Arc along.
    const double radius = A->radius[p];
    const double sweep = A->sweep[p];
    if (radius <= 0.0 || sweep == 0.0) return 0.0;
    const double2 center = A->center[p];
    const int previous = (p + n - 1) % n;
    const double2 startVector = sub2(A->head[previous], center);
    const double2 turned = turn2(startVector);
    const double orientation = (sweep >= 0.0) ? 1.0 : -1.0;

    if (kind == 0) {
        const double2 offset = sub2(B->tail[f], center);
        const double a = dot2(normal, offset);
        const double b = -dot2(normal, startVector);
        const double c = -dot2(normal, turned);
        double valueHi, dAHi, dBHi, dCHi, dPsiHi;
        double valueLo, dALo, dBLo, dCLo, dPsiLo;
        harmonicIntegral(a, b, c, sweep * hi, &valueHi, &dAHi, &dBHi, &dCHi, &dPsiHi);
        harmonicIntegral(a, b, c, sweep * lo, &valueLo, &dALo, &dBLo, &dCLo, &dPsiLo);
        const double value = valueHi - valueLo;
        const double dA = dAHi - dALo, dB = dBHi - dBLo, dC = dCHi - dCLo;
        const double weight = scale * radius * orientation;
        atomicAdd(gradA + 2 * ROUNDED_MAXN + p, scale * orientation * value);
        atomicAdd(gradA + 3 * ROUNDED_MAXN + p, weight * (hi * dPsiHi - lo * dPsiLo));
        const double2 turnedNormal = turn2(normal);
        const double2 toStart = make_double2(weight * (-dB * normal.x + dC * turnedNormal.x),
                                             weight * (-dB * normal.y + dC * turnedNormal.y));
        scatter2(gradA, 6 * ROUNDED_MAXN + 2 * previous, toStart);
        scatter2(gradA, 2 * p, make_double2(-weight * dA * normal.x - toStart.x,
                                            -weight * dA * normal.y - toStart.y));
        const double2 toNormal = make_double2(
            weight * (dA * offset.x - dB * startVector.x - dC * turned.x),
            weight * (dA * offset.y - dB * startVector.y - dC * turned.y));
        const double2 conjugate = normalPullback(toNormal, edge, normal, length);
        scatter2(gradB, 6 * ROUNDED_MAXN + 2 * f, conjugate);
        scatter2(gradB, 4 * ROUNDED_MAXN + 2 * f,
                 make_double2(-conjugate.x + normal.x * weight * dA,
                              -conjugate.y + normal.y * weight * dA));
        return weight * value;
    }

    // Arc along, arc to: elliptic, so QUADRATURE -- and the integrand is differentiated POINTWISE at
    // the nodes, with the same weights carrying the derivative. Exact for the same reason the value is.
    const double2 gap = sub2(center, B->center[f]);
    const double alongStart = dot2(gap, startVector);
    const double alongTurned = dot2(gap, turned);
    const double measure = scale * radius * fabs(sweep) * 0.5 * (hi - lo);
    double cubed = 0.0, totalInner = 0.0, cosineSum = 0.0, sineSum = 0.0;
    double toDepthSum = 0.0, sweepSum = 0.0;
    for (int q = 0; q < ROUNDED_QUADRATURE; ++q) {
        const double parameter = 0.5 * (lo + hi) + 0.5 * (hi - lo) * kNodes[q];
        const double psi = sweep * parameter;
        const double cosine = cos(psi), sine = sin(psi);
        const double inner = dot2(gap, gap) + radius * radius
                             + 2.0 * (cosine * alongStart + sine * alongTurned);
        const double root = sqrt(fmax(inner, 0.0));
        const double depth = B->radius[f] - root;
        cubed += kWeights[q] * depth * depth * depth;
        const double toDepth = measure * kWeights[q] * 3.0 * depth * depth;
        const double toInner = -0.5 * toDepth / fmax(root, 1e-300);
        toDepthSum += toDepth;
        totalInner += toInner;
        cosineSum += toInner * cosine;
        sineSum += toInner * sine;
        sweepSum += toInner * parameter * 2.0 * (cosine * alongTurned - sine * alongStart);
    }
    const double2 turnedGap = turn2(gap);
    const double2 toGap = make_double2(
        2.0 * (totalInner * gap.x + cosineSum * startVector.x + sineSum * turned.x),
        2.0 * (totalInner * gap.y + cosineSum * startVector.y + sineSum * turned.y));
    const double2 toStart = make_double2(2.0 * (cosineSum * gap.x - sineSum * turnedGap.x),
                                         2.0 * (cosineSum * gap.y - sineSum * turnedGap.y));
    atomicAdd(gradB + 2 * ROUNDED_MAXN + f, toDepthSum);
    scatter2(gradB, 2 * f, make_double2(-toGap.x, -toGap.y));
    scatter2(gradA, 6 * ROUNDED_MAXN + 2 * previous, toStart);
    scatter2(gradA, 2 * p, make_double2(toGap.x - toStart.x, toGap.y - toStart.y));
    atomicAdd(gradA + 2 * ROUNDED_MAXN + p,
              2.0 * radius * totalInner + scale * fabs(sweep) * 0.5 * (hi - lo) * cubed);
    atomicAdd(gradA + 3 * ROUNDED_MAXN + p,
              sweepSum + scale * radius * orientation * 0.5 * (hi - lo) * cubed);
    return measure * cubed;
}

// ---------------------------------------------------------------------------------------------
// The overlap AREA and its shape derivative.
//
// NO CLIPPER IS NEEDED. The boundary of A and B is exactly the pieces of dA inside B plus the pieces
// of dB inside A, each kept in its original direction -- which is what the span walk already produces.
// Green's theorem adds them with no loop ordering, because int (x dy - y dx) is additive over them
// however they are sorted.
//
// THE TWO TIERS NEED DIFFERENT DERIVATIVE ARGUMENTS, and this is the one thing here that cannot be
// guessed. The ENERGY's integrand vanishes at a crossing, so its moving breakpoints drop out and the
// frozen partition is exact. Green's integrand does NOT vanish there, and freezing is measured 40-60%
// wrong. The shape derivative has no boundary terms at all:
//
//     d|A n B|/dp = int_{dA n B} (v_A . n_A) ds + int_{dB n A} (v_B . n_B) ds,   v = dx/dp
//
// so the spans may still be frozen -- for a different reason, and through a different formula.
// ---------------------------------------------------------------------------------------------

// int (x dy - y dx) along piece `p` over [lo, hi]: TWICE the swept area.
//
// Segment: with x = tail + t v the t terms cancel identically. Arc: with x = z + w and w = Rot(psi) w0,
// dx = J w dpsi and the integrand collapses to (z . w + r^2) dpsi because |w| = r.
__device__ __forceinline__ double greenIntegral(const Body* body, int p, double lo, double hi) {
    const int n = body->count;
    if (p >= n) {
        const int k = p - n;
        const double2 tail = body->tail[k];
        const double2 vector = sub2(body->head[k], tail);
        return (tail.x * vector.y - tail.y * vector.x) * (hi - lo);
    }
    const double2 center = body->center[p];
    const double radius = body->radius[p], sweep = body->sweep[p];
    const double2 start = sub2(body->head[(p + n - 1) % n], center);
    const double2 turned = turn2(start);
    const double psiLo = sweep * lo, psiHi = sweep * hi;
    return radius * radius * (psiHi - psiLo)
           + dot2(center, start) * (sin(psiHi) - sin(psiLo))
           - dot2(center, turned) * (cos(psiHi) - cos(psiLo));
}

// Scatter `weight * d(area)/d(body arrays)` for one span. NO QUADRATURE ANYWHERE -- written out, both
// pieces integrate in closed form and three of the five arrays drop out:
//
//   * x contains neither radius nor sweep (an arc point is z + cos psi w0 + sin psi J w0), so their
//     velocities are zero and only centre, tail and head can move the area;
//   * along an arc the TANGENTIAL velocity has no normal component -- offset . (-sin psi w0 +
//     cos psi J w0) = 0 identically -- which is why d/dsweep vanishes rather than merely being small;
//   * (cos psi I + sin psi J)^T offset = Rot(-psi) offset = w0, collapsing the whole w0 term to
//     |sweep| (hi - lo) (dw0 . w0) with no trigonometry left in it.
//
// So this is EXACTER than the order-12 Gauss-Legendre the host reference originally used, as well as
// cheaper: that integrand is trigonometric and the quadrature was accurate but never exact.
__device__ __forceinline__ void spanAreaGradient(const Body* body, int p, double lo, double hi,
                                                 double weight, double* gradient) {
    const int n = body->count;
    if (hi - lo <= 0.0) return;
    if (p >= n) {
        const int k = p - n;
        const double2 vector = sub2(body->head[k], body->tail[k]);
        const double2 scaledNormal = make_double2(vector.y, -vector.x);
        const double second = 0.5 * (hi * hi - lo * lo);
        scatter2(gradient, 4 * ROUNDED_MAXN + 2 * k,
                 make_double2(weight * scaledNormal.x * ((hi - lo) - second),
                              weight * scaledNormal.y * ((hi - lo) - second)));
        scatter2(gradient, 6 * ROUNDED_MAXN + 2 * k,
                 make_double2(weight * scaledNormal.x * second,
                              weight * scaledNormal.y * second));
        return;
    }
    const double radius = body->radius[p], sweep = body->sweep[p];
    if (radius <= 0.0 || sweep == 0.0) return;
    const double2 center = body->center[p];
    const int previous = (p + n - 1) % n;
    const double2 start = sub2(body->head[previous], center);
    const double2 turned = turn2(start);
    const double psiLo = sweep * lo, psiHi = sweep * hi;
    const double2 moment = make_double2(
        (sin(psiHi) - sin(psiLo)) / sweep * start.x + (cos(psiLo) - cos(psiHi)) / sweep * turned.x,
        (sin(psiHi) - sin(psiLo)) / sweep * start.y + (cos(psiLo) - cos(psiHi)) / sweep * turned.y);
    const double magnitude = fabs(sweep);
    const double2 toStart = make_double2(magnitude * (hi - lo) * start.x,
                                         magnitude * (hi - lo) * start.y);
    scatter2(gradient, 2 * p, make_double2(weight * (magnitude * moment.x - toStart.x),
                                           weight * (magnitude * moment.y - toStart.y)));
    scatter2(gradient, 6 * ROUNDED_MAXN + 2 * previous,
             make_double2(weight * toStart.x, weight * toStart.y));
}

}  // namespace roundedContact
