"""Verification for the QR factorization behind SHAKE and the projector.

``ShapeConstraints`` used a rank-revealing thin SVD for both of the things it needs -- an orthonormal
basis of the constraint-NORMAL space, and the minimum-norm Newton correction. It now uses a QR of
``J^T`` when the blocks are full rank, because the SVD was 93% of SHAKE's cost and SHAKE was ~50% of a
FIRE step:

    J^T = Q R    ->    normalBasis = Q^T,    J^+ C = Q R^-T C

the second because ``J J^T = R^T R`` for full row rank. Note it never FORMS ``J J^T``: ``R`` carries
``J``'s condition number, not its square, so this is not the normal-equations shortcut the module
docstring rules out.

The bar is that nothing observable changes. QR and SVD give DIFFERENT bases of the same subspace --
signs and rotations differ -- so the tests compare the projector and the step, never the vectors.

  1. the projectors agree: V^T V from QR equals V^T V from the SVD;
  2. the SHAKE step agrees, so the retraction follows the same path;
  3. the retraction lands on the same residual in the same iteration count, at the drift a FIRE step
     actually leaves;
  4. rank-deficient blocks FALL BACK rather than returning a wrong factorization -- triangles under
     area+edge, perimeter+edge, and the zero-padded blocks of a ragged packing;
  5. forward substitution matches a general solve on the same triangular system.

Run: python tests/shakeFactorCheck.py
"""

# UNVERIFIED(Cam)

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import Model
from constraints import ShapeConstraints, _forwardSubstitute


def buildPacking(n = 16, N = 8, seed = 5, phi = 0.5):
    model = Model(N = N, n = n, seed = seed)
    model.generateEquilateralPolygons(phi = phi,
                                      kappa = float(np.sqrt(4.0 * n * np.tan(np.pi / n))))
    model.syncTargetPerimeters()
    model.syncTargetAreas()
    return model


def svdBasis(constraints, J):
    _, _, Vh, keep = constraints._decompose(J)
    return np.where(keep[:, :, None], Vh, 0.0)


def checkProjectorsAgree():
    """1. The two factorizations give the same projector, which is all any consumer uses."""
    worst = 0.0
    for n in (4, 8, 16, 32):
        model = buildPacking(n = n)
        constraints = ShapeConstraints(model.packing, area = True, edge = True)
        J = constraints.jacobian(model.packing)
        factored = constraints._qrFactor(J)
        assert factored is not None, f"n={n}: expected full rank but the QR guard refused"
        fromQr = np.swapaxes(factored[0], 1, 2)
        fromSvd = svdBasis(constraints, J)
        projectorQr = np.einsum("pmd,pme->pde", fromQr, fromQr)
        projectorSvd = np.einsum("pmd,pme->pde", fromSvd, fromSvd)
        error = float(np.abs(projectorQr - projectorSvd).max())
        worst = max(worst, error)
        print(f"  1. n={n:2d}   max |P_qr - P_svd| = {error:.3e}")
        assert error < 1e-12, f"n={n}: projectors differ by {error:.3e}"


def checkStepsAgree():
    """2. The Newton correction itself agrees, not just the subspace."""
    for n in (4, 16, 32):
        model = buildPacking(n = n)
        packing = model.packing
        constraints = ShapeConstraints(packing, area = True, edge = True)
        rng = np.random.default_rng(2)
        packing.positions += 1e-4 * rng.standard_normal(packing.positions.size)
        J = constraints.jacobian(packing)
        C = constraints.residual(packing)

        Q, R = constraints._qrFactor(J)
        stepQr = np.einsum("pdj,pj->pd", Q, _forwardSubstitute(R, C))

        U, S, Vh, keep = constraints._decompose(J)
        y = np.where(keep, np.einsum("pij,pi->pj", U, C) / np.where(keep, S, 1.0), 0.0)
        stepSvd = np.einsum("pjd,pj->pd", Vh, y)

        scale = float(np.abs(stepSvd).max())
        error = float(np.abs(stepQr - stepSvd).max())
        print(f"  2. n={n:2d}   max |step_qr - step_svd| = {error:.3e}   (steps ~ {scale:.3e})")
        assert error < 1e-10 * max(scale, 1.0), f"n={n}: Newton steps differ by {error:.3e}"


def checkRetractionMatches():
    """3. Same residual, same iteration count, at the drift a FIRE step leaves."""
    for n, N in ((16, 32), (32, 32)):
        results = {}
        for label in ("svd", "qr"):
            model = buildPacking(n = n, N = N)
            packing = model.packing
            constraints = ShapeConstraints(packing, area = True, edge = True)
            if label == "svd":
                constraints._qrFactor = staticmethod(lambda J: None)
            constraints.projectPositions(packing)
            rng = np.random.default_rng(1)
            packing.positions += 1e-5 * rng.standard_normal(packing.positions.size)
            results[label] = constraints.projectPositions(packing)
        (iterSvd, worstSvd), (iterQr, worstQr) = results["svd"], results["qr"]
        print(f"  3. n={n:2d} N={N}   svd {iterSvd} iters -> {worstSvd:.3e}   "
              f"qr {iterQr} iters -> {worstQr:.3e}")
        assert iterQr == iterSvd, f"n={n}: {iterQr} iterations against the SVD's {iterSvd}"
        assert abs(worstQr - worstSvd) <= 1e-15 + 0.05 * worstSvd, \
            f"n={n}: residual {worstQr:.3e} against the SVD's {worstSvd:.3e}"


def checkRankDeficientFallsBack():
    """4. The guard REFUSES rather than returning a wrong factorization."""
    # A triangle's area is determined by its three edges, so area+edge is redundant by construction.
    model = buildPacking(n = 3, N = 8)
    constraints = ShapeConstraints(model.packing, area = True, edge = True)
    J = constraints.jacobian(model.packing)
    refused = constraints._qrFactor(J) is None
    _, _, _, keep = constraints._decompose(J)
    print(f"  4. triangle area+edge   rank kept {int(keep.sum())}/{keep.size}   "
          f"qr refused: {refused}")
    assert not keep.all(), "the triangle case is supposed to be rank deficient"
    assert refused, "the QR guard accepted a rank-deficient block"

    # perimeter is a function of the edge rows.
    model = buildPacking(n = 8, N = 6)
    constraints = ShapeConstraints(model.packing, area = True, perimeter = True, edge = True)
    J = constraints.jacobian(model.packing)
    refused = constraints._qrFactor(J) is None
    print(f"  4. perimeter+edge       redundant: {constraints.redundancyReason() is not None}   "
          f"qr refused: {refused}")
    assert refused, "the QR guard accepted a redundant perimeter+edge set"

    # A RAGGED packing pads its blocks with all-zero rows, which are exactly rank deficient.
    model = buildPacking(n = 6, N = 5)
    model.addShape(np.array([[0.0, 0.0], [0.6, 0.0], [0.3, 0.5]]))
    constraints = ShapeConstraints(model.packing, area = True, edge = True)
    J = constraints.jacobian(model.packing)
    refused = constraints._qrFactor(J) is None
    print(f"  4. ragged (6-gons + triangle)   ragged={constraints.ragged}   qr refused: {refused}")
    assert refused, "the QR guard accepted zero-padded ragged blocks"


def checkForwardSubstitution():
    """5. Substitution matches a general solve on the same triangular system."""
    model = buildPacking(n = 32, N = 32)
    constraints = ShapeConstraints(model.packing, area = True, edge = True)
    J = constraints.jacobian(model.packing)
    C = constraints.residual(model.packing)
    Q, R = constraints._qrFactor(J)
    mine = _forwardSubstitute(R, C)
    reference = np.linalg.solve(np.swapaxes(R, 1, 2), C[:, :, None])[:, :, 0]
    error = float(np.abs(mine - reference).max())
    print(f"  5. forward substitution vs linalg.solve   max |diff| = {error:.3e}")
    assert error < 1e-20, f"forward substitution differs by {error:.3e}"


def main():
    print("SHAKE factorization: QR against the SVD it replaced")
    checkProjectorsAgree()
    checkStepsAgree()
    checkRetractionMatches()
    checkRankDeficientFallsBack()
    checkForwardSubstitution()
    print("all checks passed")


if __name__ == "__main__":
    main()
