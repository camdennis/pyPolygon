"""Take a contact graph to arbitrary precision, then recognise the side length as an algebraic number.

THE MINIMIZER CANNOT GIVE YOU FIFTEEN DIGITS AND DOES NOT HAVE TO. Its force noise floor is ~3e-12 and
its answer comes from a bisection on overlap area, so the side is good to eight or nine digits. That is
already far more than enough to decide WHO TOUCHES WHOM, and once the contact graph is known the
numerics are thrown away: the contacts are a square system of polynomials, Newton on it converges
quadratically, and the precision is whatever mpmath is asked for.

PRECISION IS THE PART THAT BITES. Recovering a minimal polynomial of degree d with coefficients of
height H needs roughly ``d log10(H)`` digits plus guard, so a degree-18 answer with three-digit
coefficients already wants sixty. Fifteen is nowhere near enough for anything but the easy cases; the
default here is 300.

THE CONTACT SET IS A HYPOTHESIS AND NEWTON IS THE TEST. Include a pair that is not really touching and
the Jacobian goes singular or the residual stalls; both are clean failures. Getting a converged
solution that then violates a non-overlap inequality is the third failure, and ``verify`` is what
catches it.

# UNVERIFIED(Cam)
"""

import mpmath as mp
import numpy as np

import contacts as ct

_CORNERS = [(mp.mpf(1) / 2, mp.mpf(1) / 2), (-mp.mpf(1) / 2, mp.mpf(1) / 2),
            (-mp.mpf(1) / 2, -mp.mpf(1) / 2), (mp.mpf(1) / 2, -mp.mpf(1) / 2)]
_NORMALS = [(mp.mpf(1), mp.mpf(0)), (mp.mpf(0), mp.mpf(1)),
            (-mp.mpf(1), mp.mpf(0)), (mp.mpf(0), -mp.mpf(1))]


# UNVERIFIED(Cam)
def _slot(order, i):
    """Where square ``i``'s four variables start in the reduced state."""
    return 4 * order[i]


# UNVERIFIED(Cam)
def residualAndJacobian(state, equations, order, count):
    """``(F, J)`` for the contact system, both analytic.

    Rows are the contact equations followed by one ``a^2 + b^2 = 1`` per square. Columns are
    ``x, y, a, b`` per square and then ``s``. Every derivative below is exact -- no differencing --
    which is what lets Newton reach three hundred digits instead of stalling at the step size."""
    unknowns = 4 * count + 1
    rows = len(equations) + count
    F = mp.matrix(rows, 1)
    J = mp.matrix(rows, unknowns)
    # INDEXED EXPLICITLY, NEVER WITH -1. An mpmath matrix returns 0.0 for a negative index instead of
    # wrapping or raising, so `state[-1]` silently reads ZERO rather than s: measured, that made every
    # wall-at-s equation carry a residual of exactly s, and Newton sat at a fixed point with the box
    # collapsed to nothing while every square stayed put. Nothing warns.
    s = state[unknowns - 1]

    for row, (kind, i, k, j, edge) in enumerate(equations):
        ui, vi = _CORNERS[k]
        pi = _slot(order, i)
        xi, yi, ai, bi = state[pi], state[pi + 1], state[pi + 2], state[pi + 3]
        # The vertex, and how it moves with i's own variables.
        vx = xi + ai * ui - bi * vi
        vy = yi + bi * ui + ai * vi

        if kind == "wall":
            if edge in (0, 2):
                F[row] = vx - (s if edge == 2 else 0)
                J[row, pi] = 1
                J[row, pi + 2] = ui
                J[row, pi + 3] = -vi
            else:
                F[row] = vy - (s if edge == 3 else 0)
                J[row, pi + 1] = 1
                J[row, pi + 2] = vi
                J[row, pi + 3] = ui
            if edge in (2, 3):
                J[row, unknowns - 1] = -1
            continue

        em, en = _NORMALS[edge]
        pj = _slot(order, j)
        xj, yj, aj, bj = state[pj], state[pj + 1], state[pj + 2], state[pj + 3]
        nx = aj * em - bj * en                      # n_j^m, the outward normal of j's edge
        ny = bj * em + aj * en
        wx, wy = vx - xj, vy - yj
        F[row] = nx * wx + ny * wy - mp.mpf(1) / 2

        J[row, pi] = nx
        J[row, pi + 1] = ny
        # dv/da_i = (u, v) and dv/db_i = (-v, u), so these contract the normal with each.
        J[row, pi + 2] = nx * ui + ny * vi
        J[row, pi + 3] = -nx * vi + ny * ui
        J[row, pj] = -nx
        J[row, pj + 1] = -ny
        # dn/da_j = (em, en) and dn/db_j = (-en, em).
        J[row, pj + 2] = em * wx + en * wy
        J[row, pj + 3] = -en * wx + em * wy

    for index in range(count):
        row = len(equations) + index
        p = 4 * index
        F[row] = state[p + 2] ** 2 + state[p + 3] ** 2 - 1
        J[row, p + 2] = 2 * state[p + 2]
        J[row, p + 3] = 2 * state[p + 3]
    return F, J


# UNVERIFIED(Cam)
def _leastSquares(J, F):
    """``J x = F`` in least squares, with a fallback for an awkward column ORDER.

    ``mp.qr_solve`` is Householder WITHOUT COLUMN PIVOTING, so it breaks down when a sub-column comes
    out exactly zero -- which a contact Jacobian produces easily, every row touching only two squares
    and being zero everywhere else. THAT IS NOT A RANK PROBLEM, THOUGH THE MESSAGE IT RAISES SAYS IT
    IS. Measured: the same n = 5 contact graph, the same singular values and an extended condition
    number of 8 solved from ``data/packingKnown5.npz`` and raised "matrix is numerically singular"
    from ``data/packing5.npz`` -- the two differ only in which square got which index. The normal
    equations square the condition number, which is free at the precision this runs at, so they are
    the fallback rather than the default."""
    try:
        return mp.qr_solve(J, F)[0]
    except (ValueError, ZeroDivisionError):
        # BOTH EXCEPTION TYPES, because mpmath is not consistent about which it uses for a singular
        # system: householder raises ValueError('matrix is numerically singular') while the LU path
        # raises ZeroDivisionError, sometimes with that same text and sometimes -- from a bare mpf
        # divide -- with NO MESSAGE AT ALL. Catching only ValueError let the fallback's own failure
        # escape, which is what surfaced as "refinement refused ()" with an empty parenthesis.
        transpose = J.T
        return mp.lu_solve(transpose * J, transpose * F)


# UNVERIFIED(Cam)
def newton(state, equations, order, count, steps = 60, digits = 300,
           null = None, anchor = None):
    """Gauss-Newton to ``digits`` places. Returns ``(state, residual)``.

    LEAST SQUARES RATHER THAN A SQUARE SOLVE, because a real optimum is usually HYPERSTATIC: symmetry
    makes some contacts dependent, so the system is overdetermined and consistent. ``qr_solve`` handles
    both cases, and the returned residual is the test of whether the contact set was right -- a wrong
    one leaves it stuck near where it started instead of falling to 1e-300."""
    mp.mp.dps = digits
    z = mp.matrix([mp.mpf(v) for v in state])
    start = mp.matrix(z) if anchor is None else mp.matrix(anchor)

    # FIRST-ORDER FLEXIBLE DIRECTIONS ARE PINNED, NOT REGULARIZED. A contact graph can be first-order
    # flexible and second-order rigid: in the proved n = 5 optimum every corner square's vertex lands on
    # the MIDPOINT of a face of the tilted middle square, where w is parallel to n, so
    # d(n.w)/dtheta = (Jn).w vanishes identically and that square's rotation is a null direction of J.
    #
    # A RIDGE DOES NOT FIX THIS AND MAKES IT WORSE. The step along a direction with singular value sigma
    # scales as sigma/(sigma^2 + lambda), which for a nearly-null sigma is an AMPLIFICATION rather than
    # a damping: measured, lambda = 1e-80 against sigma = 1e-9 drove s to zero on the first step.
    # Instead each null direction gets an explicit equation holding it at its input value, which makes
    # the system full rank, keeps Newton quadratic on everything else, and -- the point -- says out loud
    # which coordinates were INHERITED from the relaxed packing rather than solved for.
    if null is None:
        # PINNED BY RETRY, NOT BY A THRESHOLD. How many directions are genuinely null is not knowable
        # from a double-precision singular value at a point this far from the solution: on a loosely
        # jammed packing the near-null directions grade smoothly and any fixed cut either misses some
        # (leaving the extended system singular, which qr_solve refuses) or pins real freedoms. So the
        # smallest directions are pinned one at a time until the solve actually works.
        _, J0 = residualAndJacobian(z, equations, order, count)
        dense = np.array([[float(J0[r, c]) for c in range(J0.cols)] for r in range(J0.rows)])
        _, singular, rows = np.linalg.svd(dense)
        columns = dense.shape[1]
        # THE CRITERION IS CONDITIONING, NOT RANK. Asking whether a singular value is "zero" needs a
        # threshold, and any fixed one is wrong here: the n = 5 optimum's genuine null direction sits at
        # 8.6e-10 against a largest of 2, which a 1e-11 rank test calls full rank -- so nothing gets
        # pinned and the solve fails anyway. Pinning until the system is merely WELL CONDITIONED asks
        # the question that actually matters, and a condition number of 1e9 is harmless when the solve
        # itself runs at fifty digits or more.
        # BOUNDED, because needing many pinned directions is a verdict rather than a hurdle: each one
        # is a coordinate the contacts fail to determine, and past a handful the contact set is simply
        # wrong. The bound also keeps this cheap -- every trial is an SVD, and sweeping all 74 columns
        # of a 26-square system took minutes to reach a conclusion already visible in the first few.
        limit = min(columns, 12)
        null = []
        for extra in range(limit + 1):
            candidate = [rows[len(rows) - 1 - k] for k in range(extra)]
            probe = np.vstack([dense] + [d[None, :] for d in candidate]) if candidate else dense
            values = np.linalg.svd(probe, compute_uv = False)
            if len(values) >= columns and values[columns - 1] > values[0] * 1e-9:
                null = candidate
                break
        else:
            raise ValueError(
                f"this contact set leaves more than {limit} directions undetermined "
                f"(Jacobian {dense.shape[0]} by {columns}). That is a verdict, not a numerical "
                f"problem: the contacts are not pinning the packing, which happens when the set "
                f"includes pairs that are not really touching, or when the packing has many rattlers. "
                f"Check contacts.toleranceLadder -- a real contact set holds a plateau with no "
                f"rattlers across decades.")

    residual = mp.inf
    for _ in range(steps):
        F, J = residualAndJacobian(z, equations, order, count)
        residual = max(abs(F[r]) for r in range(F.rows))
        if residual < mp.mpf(10) ** (-(digits - 20)):
            break
        if null:
            extended = mp.matrix(J.rows + len(null), J.cols)
            target = mp.matrix(J.rows + len(null), 1)
            for r in range(J.rows):
                target[r] = F[r]
                for c in range(J.cols):
                    extended[r, c] = J[r, c]
            for index, direction in enumerate(null):
                row = J.rows + index
                offset = mp.mpf(0)
                for c in range(J.cols):
                    extended[row, c] = mp.mpf(float(direction[c]))
                    offset += mp.mpf(float(direction[c])) * (z[c] - start[c])
                target[row] = offset
            J, F = extended, target
        step = _leastSquares(J, F)
        for r in range(len(z)):
            z[r] -= step[r]
    return z, residual, null


# UNVERIFIED(Cam)
def jacobianRank(state, equations, order, count, tolerance = 1e-9):
    """``(rank, columns)`` of the contact Jacobian, in double precision.

    WORTH REPORTING SEPARATELY FROM THE EQUATION COUNT. A hyperstatic verdict says there are more
    equations than unknowns; the rank says whether they actually PIN everything. A deficiency of one
    means a first-order flexible direction -- the packing may still be rigid at second order, as the
    n = 5 optimum is -- and Newton will not determine that coordinate, so it has to be inherited from
    the input rather than trusted as solved."""
    _, J = residualAndJacobian(state, equations, order, count)
    dense = np.array([[float(J[r, c]) for c in range(J.cols)] for r in range(J.rows)])
    return int(np.linalg.matrix_rank(dense, tol = tolerance)), int(J.cols)


# UNVERIFIED(Cam)
def stationary(state, equations, order, count, null, digits = 300, window = None):
    """Slide along a first-order flexible direction until ``s`` is STATIONARY, and return the offset.

    WITHOUT THIS, A SECOND-ORDER RIGID PACKING CAPS AT THE SQUARE OF THE INPUT ERROR. Pinning the
    flexible coordinate at its relaxed value leaves it wrong by ~1e-09; the constraint manifold is
    curved, so ``s`` along it goes as ``s* + c (t - t*)^2`` and the side comes out wrong by ~1e-18 no
    matter how many Newton steps are taken. Measured on the proved n = 5 optimum: residual 4e-121 and
    still only twenty correct digits.

    THE CONDITION IS LOCAL OPTIMALITY, WHICH IS THE POINT OF THE WHOLE EXERCISE. A flexible direction
    that changed ``s`` at first order would mean the packing is not extremal at all; that it does not
    means ``s`` has a critical point along it, and finding that point is what pins the coordinate. So
    this is not a numerical patch -- it is the second-order half of the statement "this contact graph
    has a locally maximal packing"."""
    if not null:
        return state, mp.mpf(0)
    if len(null) > 1:
        # MORE THAN ONE FLEXIBLE DIRECTION IS A SURFACE, NOT A CURVE, so making s stationary over it is
        # a genuine optimization rather than a scalar search. Rather than refuse outright, the
        # directions stay pinned at their input values and the caller is told: the side is then only as
        # good as the SQUARE of the input error, which is the honest limit and still worth having.
        return state, mp.mpf("nan")
    direction = [mp.mpf(float(v)) for v in null[0]]
    unknowns = 4 * count + 1

    def sideAt(offset):
        moved = mp.matrix(state)
        for c in range(unknowns):
            moved[c] += offset * direction[c]
        solved, _, _ = newton(moved, equations, order, count, digits = digits, null = null,
                              anchor = moved)
        return solved[unknowns - 1]

    # A PARABOLIC ITERATION, NOT A ROOT-FIND ON THE DERIVATIVE. Near the critical point s(u) is
    # quadratic, so three evaluations locate the vertex EXACTLY in one step and the iteration then just
    # cleans up the quartic terms. Handing the numerical derivative to findroot does not work: ds/du is
    # already ~1e-19 at the starting point, which findroot reads as a root and accepts, leaving the
    # coordinate exactly where it began.
    if window is None:
        window = mp.mpf(10) ** -7          # comfortably wider than any relaxed packing's error
    best = mp.mpf(0)
    floor = mp.mpf(10) ** (-(digits // 3))
    for _ in range(8):
        lower, middle, upper = sideAt(best - window), sideAt(best), sideAt(best + window)
        curvature = upper - 2 * middle + lower
        if curvature == 0:
            break
        step = -window * (upper - lower) / (2 * curvature)
        best += step
        if abs(step) < mp.mpf(10) ** (-(digits // 2)):
            break
        window = max(abs(step), floor)

    moved = mp.matrix(state)
    for c in range(unknowns):
        moved[c] += best * direction[c]
    solved, residual, _ = newton(moved, equations, order, count, digits = digits, null = null,
                                 anchor = moved)
    return solved, residual


# UNVERIFIED(Cam)
def verify(state, count, tolerance = None):
    """``(worstOverlap, worstOutside)`` for the refined packing: the inequalities Newton never saw.

    THE EQUATIONS SAY CERTAIN PAIRS TOUCH; THEY DO NOT SAY THE REST STAY APART. A contact set with one
    spurious member can converge beautifully to a configuration where two other squares interpenetrate,
    so this is not optional. Separation is measured by the SEPARATING-AXIS margin, which for two convex
    squares is exact: positive means disjoint."""
    if tolerance is None:
        tolerance = mp.mpf(10) ** (-(mp.mp.dps // 2))
    s = state[4 * count]          # never state[-1]: mpmath returns 0.0 for negative indices
    worstOverlap = -mp.inf
    for i in range(count):
        for j in range(i + 1, count):
            margin = -mp.inf
            for owner, other in ((i, j), (j, i)):
                a, b = state[4 * owner + 2], state[4 * owner + 3]
                for em, en in _NORMALS:
                    nx, ny = a * em - b * en, b * em + a * en
                    # Support of `other` along -n, against the face of `owner` at 1/2.
                    lowest = mp.inf
                    for uk, vk in _CORNERS:
                        ax, ay = state[4 * other + 2], state[4 * other + 3]
                        px = state[4 * other] + ax * uk - ay * vk - state[4 * owner]
                        py = state[4 * other + 1] + ay * uk + ax * vk - state[4 * owner + 1]
                        lowest = min(lowest, nx * px + ny * py)
                    margin = max(margin, lowest - mp.mpf(1) / 2)
            worstOverlap = max(worstOverlap, -margin)
    worstOutside = -mp.inf
    for i in range(count):
        a, b = state[4 * i + 2], state[4 * i + 3]
        for uk, vk in _CORNERS:
            px = state[4 * i] + a * uk - b * vk
            py = state[4 * i + 1] + b * uk + a * vk
            worstOutside = max(worstOutside, -px, -py, px - s, py - s)
    return worstOverlap, worstOutside


# UNVERIFIED(Cam)
def identifySide(value, maxDegree = 24, constants = ("sqrt(2)", "sqrt(3)", "sqrt(5)")):
    """``(expression, minimalPolynomial)`` for the side length, or ``(None, None)``.

    Two passes, cheap first. ``identify`` recognises the shapes that actually occur in the record table
    -- ``k + m sqrt(2)/2`` and relatives -- and ``pslq`` then searches for a minimal polynomial of any
    degree up to ``maxDegree``.

    A FAILURE HERE IS INFORMATIVE RATHER THAN A DEAD END. The clean closed forms in ``records.py``
    belong to packings whose tilted squares sit at exactly 45 degrees; a packing whose tilt is some
    other algebraic angle has a high-degree minimal polynomial and will not be recognised by eye. If
    nothing lands, try the TILTS instead -- ``tan(theta)`` is often lower degree than the side, and once
    the angles are known the rest of the system frequently collapses."""
    expression = mp.identify(value, constants)
    for degree in range(2, maxDegree + 1):
        relation = mp.pslq([value ** k for k in range(degree + 1)],
                           maxcoeff = 10 ** 12, maxsteps = 10 ** 5)
        if relation and any(relation):
            return expression, list(relation)
    return expression, None


# UNVERIFIED(Cam)
def refine(packing, tolerance = None, digits = 300):
    """The whole pipeline: audit the contact graph, Newton it, verify it, name the side.

    Returns a dict; ``report`` renders it."""
    state, startingSide = ct.unitState(packing)
    count = (len(state) - 1) // 4
    if tolerance is None:
        # CHOSEN FROM THE GAP SPECTRUM RATHER THAN ASSUMED. A fixed 1e-6 is right for a packing whose
        # contacts are machine-tight and makes every square a rattler for one that ended on a
        # bisection; see contacts.gaps.
        tolerance = ct.suggestTolerance(state, count)
        if tolerance is None:
            spectrum = ct.gaps(state, count)
            raise ValueError(
                "no tolerance separates the contacts: the gaps grade smoothly rather than clustering, "
                "so nothing distinguishes a contact from a near-miss. That means the packing is not "
                f"jammed. Smallest gaps are {np.array2string(spectrum[:8], precision = 3)}. "
                "Inspect contacts.toleranceLadder before choosing one by hand.")
    verdict = ct.audit(state, count, tolerance)
    free = list(verdict["free"])
    if not free:
        raise ValueError(
            f"every one of {count} squares is a rattler, so there is no system to solve. The packing "
            f"is not jammed: read it at the JAMMING point instead. A bisection down to zero overlap "
            f"ends at the largest VALID size, where the depth law has no force left and the final "
            f"relaxation disengages every contact but the marginal one -- measured on an n = 11 run, "
            f"one gap at 1e-08 and the next at 2e-04. Save the state from the overlapping side of the "
            f"bisection, where every contact is still engaged.")
    order = {body: index for index, body in enumerate(free)}
    reduced = np.concatenate([np.concatenate([state[4 * b:4 * b + 4] for b in free]), [state[-1]]])
    refined, residual, null = newton(reduced, verdict["equations"], order, len(free),
                                     digits = digits)
    stationaryDone = False
    if null:
        moved, movedResidual = stationary(refined, verdict["equations"], order, len(free), null,
                                          digits = digits)
        if not mp.isnan(movedResidual):
            refined, residual, stationaryDone = moved, movedResidual, True
    overlap, outside = verify(refined, len(free))
    side = refined[4 * len(free)]
    expression, polynomial = identifySide(side)
    rank, unknowns = jacobianRank(refined, verdict["equations"], order, len(free))
    return {"audit": verdict, "tolerance": tolerance, "startingSide": startingSide, "side": side, "residual": residual,
            "state": refined, "order": order, "worstOverlap": overlap, "worstOutside": outside,
            "expression": expression, "polynomial": polynomial, "count": count,
            "rank": rank, "columns": unknowns, "pinned": len(null),
            "stationary": stationaryDone}


# UNVERIFIED(Cam)
def report(result, places = 30):
    """Print the verdict, the digits and the certificate in one place."""
    audit = result["audit"]
    print(f"contacts     {len(audit['contacts'])} found, {len(audit['equations'])} as equations, "
          f"{len(audit['corners'])} corner touches (not equations)")
    print(f"tolerance    {result['tolerance']:.2e} (square-side units)")
    print(f"rattlers     {list(audit['rattlers'])}")
    print(f"system       {len(audit['equations'])} equations against {audit['unknowns']} unknowns "
          f"-- {audit['verdict'].upper()}")
    if result["pinned"] == 0:
        note = "fully determined at first order"
    elif result["stationary"]:
        note = "made STATIONARY in s (second-order rigid)"
    else:
        note = ("PINNED at their input values -- more than one, so s is only as good as the SQUARE "
                "of the input error")
    print(f"jacobian     rank {result['rank']} of {result['columns']} columns; "
          f"{result['pinned']} flexible direction(s), {note}")
    print(f"residual     {mp.nstr(result['residual'], 5)}")
    print(f"side         {mp.nstr(result['side'], places)}")
    print(f"             started from {result['startingSide']:.12f}")
    # AT AN EXACT CONTACT BOTH OF THESE ARE EXACTLY ZERO, so they arrive at the precision floor with
    # whatever sign the rounding hands them -- measured, +9.8e-91 at 120 digits. Reading "must be <= 0"
    # literally then calls every correct answer invalid. The test is whether they clear the FLOOR.
    floor = mp.mpf(10) ** (-(mp.mp.dps // 2))
    breach = max(result["worstOverlap"], result["worstOutside"])
    print(f"valid        worst overlap {mp.nstr(result['worstOverlap'], 5)}, "
          f"worst outside {mp.nstr(result['worstOutside'], 5)}"
          f"   -- {'OK, at the precision floor' if breach <= floor else 'VIOLATED'}")
    print(f"expression   {result['expression']}")
    print(f"minimal poly {result['polynomial']}")


# UNVERIFIED(Cam)
def search(packing, levels = None, digits = 80, quiet = False):
    """Try every tolerance and let CONVERGENCE decide which contact set is right.

    THE CONTACT SET IS A HYPOTHESIS AND NEWTON IS THE TEST -- so when the gaps do not separate cleanly,
    the answer is not to pick a number more carefully but to try them all and read the three outcomes:

      * the residual stalls               -> the set includes a pair that is not really touching
      * it converges but an inequality is positive -> likewise, and the packing it found is invalid
      * it converges AND verifies         -> that contact set is consistent

    A LOOSE INPUT IS NOT A PROBLEM, WHICH IS WHY THIS WORKS. Newton only needs the combinatorics; the
    coordinates are a starting guess. A packing read after its final decompression has every contact
    opened by the decompression amount -- measured, 1e-04 to 1e-03 of a square side on a 26-square run
    -- and Newton simply closes them in a few extra iterations.

    Several tolerances usually agree on the same side length. That agreement is the real evidence: a
    contact set that survives a decade of tolerance and reproduces the same algebraic number is a much
    stronger claim than any single run."""
    state, startingSide = ct.unitState(packing)
    count = (len(state) - 1) // 4
    if levels is None:
        levels = [10.0 ** k for k in range(-9, -1)]
    rows = []
    for level in levels:
        try:
            verdict = ct.audit(state, count, tolerance = level)
            if not len(verdict["free"]):
                rows.append((level, "all rattlers", None, None, None))
                continue
            result = refine(packing, tolerance = level, digits = digits)
            valid = max(result["worstOverlap"], result["worstOutside"]) <= mp.mpf(10) ** (-(digits // 3))
            rows.append((level, verdict["verdict"], result["residual"], result["side"], valid))
        except Exception as exc:
            rows.append((level, f"failed: {type(exc).__name__}", None, None, None))
    if not quiet:
        print(f"{'tolerance':>10} {'verdict':>14} {'residual':>12} {'valid':>6}  side")
        for level, verdict, residual, side, valid in rows:
            residualText = "-" if residual is None else mp.nstr(residual, 3)
            sideText = "-" if side is None else mp.nstr(side, 20)
            validText = "-" if valid is None else ("yes" if valid else "NO")
            print(f"{level:10.0e} {verdict:>14} {residualText:>12} {validText:>6}  {sideText}")
    return rows
