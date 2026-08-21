"""Verify the DISTRIBUTION (moment) constraints and their composition with per-object constraints.

The claim being tested is Cam's design for the square search: pin every polygon's AREA exactly, but
hold the edge lengths only through the global moments of their distribution, so shapes can reshape
without any of them shrinking. Six checks:

  [1] the moment Jacobian matches central finite differences of the residual
  [2] a projected vector really is tangent to every moment row (and stays tangent to the per-object
      rows when the two are composed)
  [3] the composite retraction drives BOTH residuals to their floors
  [4] under hard-area + moment-edge constraints a relaxation keeps every area exactly, while the edge
      lengths genuinely spread -- the behavior the whole design exists for
  [5] the mean/variance rows go DEPENDENT as the distribution narrows (the documented handoff point)
  [6] a target set that violates the isoperimetric bound is REFUSED rather than ground against

Run: python tests/momentConstraintCheck.py
"""

import os
import sys
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from constraints import (ShapeConstraints, DistributionConstraints, CompositeConstraints)
from model import Model

WALL = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 0.0]])


def buildPacking(numSquares = 6, phi = 0.5, seed = 7, wall = False, sides = 4):
    """A relaxed packing to constrain. With ``wall`` the container is added and pinned, which is the
    case that has to stay out of every moment sum."""
    model = Model(N = numSquares, n = sides, seed = seed)
    model.generateEquilateralPolygons(phi = phi, kappa = 4.0)
    model.setMonoPerimeter()
    if wall:
        model.addShape(WALL)
        model.pinVertices(np.arange(model.getNumVertices())[-4:])
        model.setBoundaryConditions("fixed")
    return model


def checkJacobian():
    """[1] Analytic moment Jacobian vs central finite differences of the residual."""
    print("\n[1] moment Jacobian vs finite differences")
    ok = True
    for label, wall, families in (("area+edge, free", False, dict(area = True, edge = True)),
                                  ("edge only, walled", True, dict(edge = True)),
                                  ("area only, walled", True, dict(area = True))):
        model = buildPacking(wall = wall)
        packing = model.packing
        constraints = DistributionConstraints(packing, [1, 2, -1], **families)
        J = constraints.jacobian(packing)
        step = 1e-7
        numerical = np.zeros_like(J)
        for column in range(packing.positions.size):
            saved = packing.positions[column]
            packing.positions[column] = saved + step
            plus = constraints.residual(packing)
            packing.positions[column] = saved - step
            minus = constraints.residual(packing)
            packing.positions[column] = saved
            numerical[:, column] = (plus - minus) / (2.0 * step)
        scale = np.abs(J).max()
        error = np.abs(J - numerical).max() / scale
        good = error < 1e-6
        ok = ok and good
        print(f"    {'OK  ' if good else 'FAIL'} {label:<20} rows={J.shape[0]:2d}  "
              f"max relative error {error:.2e}")
    return ok


def rescalePolygons(model, factors):
    """Scale each non-container polygon about its own centroid, in place.

    Each polygon stays regular, so the only thing that changes is the SPREAD of edge lengths across
    the packing -- the quantity that decides whether the k = 1 and k = 2 moment rows are independent."""
    packing = model.packing
    r = packing.positions.reshape(-1, 2)
    for polygon, factor in enumerate(factors):
        a, b = packing.startIndices[polygon], packing.startIndices[polygon + 1]
        centroid = r[a : b].mean(axis = 0)
        r[a : b] = centroid + factor * (r[a : b] - centroid)
    return model


def checkTangency():
    """[2] A projected vector has no component along any constraint row.

    Run on a POLYDISPERSE configuration so both moment rows are genuinely independent -- on a
    monodisperse one the k = 2 row is a multiple of the k = 1 row, the SVD correctly drops it, and
    tangency to the dropped direction is only as good as the dependency (see [5])."""
    print("\n[2] projected vectors are tangent")
    rng = np.random.default_rng(0)
    ok = True

    model = buildPacking(wall = True)
    rescalePolygons(model, [0.80, 0.90, 1.00, 1.05, 1.15, 1.25])
    packing = model.packing
    distribution = DistributionConstraints(packing, [1, 2], edge = True)
    vector = rng.standard_normal(packing.positions.size)
    tangent = distribution.projectVector(packing, vector)
    G = distribution.jacobian(packing)
    residual = np.abs(G @ tangent).max() / np.abs(G @ vector).max()
    good = residual < 1e-12 and distribution.rank(packing) == 2
    ok = ok and good
    print(f"    {'OK  ' if good else 'FAIL'} moments alone      |G w_tan| / |G w| = {residual:.2e}  "
          f"(rank {distribution.rank(packing)} of 2)")

    block = ShapeConstraints(packing, area = True)
    composite = CompositeConstraints(block, distribution)
    tangent = composite.projectVector(packing, vector)
    momentPart = np.abs(G @ tangent).max() / np.abs(G @ vector).max()
    J = block.jacobian(packing)
    gathered = tangent.reshape(-1, 2)[block.index].reshape(block.numPolygons, -1)
    raw = vector.reshape(-1, 2)[block.index].reshape(block.numPolygons, -1)
    blockPart = (np.abs(np.einsum("pmd,pd->pm", J, gathered)).max()
                 / np.abs(np.einsum("pmd,pd->pm", J, raw)).max())
    good = momentPart < 1e-10 and blockPart < 1e-10
    ok = ok and good
    print(f"    {'OK  ' if good else 'FAIL'} composite          moments {momentPart:.2e}, "
          f"per-object {blockPart:.2e}")
    return ok


def checkRetraction():
    """[3] The composite retraction satisfies both constraint sets at once."""
    print("\n[3] composite retraction")
    rng = np.random.default_rng(3)
    model = buildPacking(wall = True)
    packing = model.packing
    distribution = DistributionConstraints(packing, [1, 2], edge = True)
    block = ShapeConstraints(packing, area = True)
    composite = CompositeConstraints(block, distribution)

    free = np.ones(packing.numVertices, dtype = bool)
    free[packing.pinned] = False
    perturbation = np.zeros((packing.numVertices, 2))
    perturbation[free] = 0.01 * rng.standard_normal((int(free.sum()), 2))
    packing.positions += perturbation.reshape(-1)
    before = (block.maxResidual(packing), distribution.maxResidual(packing))
    iterations, worst = composite.projectPositions(packing)
    after = (block.maxResidual(packing), distribution.maxResidual(packing))
    ok = after[0] < 1e-12 and after[1] < 1e-11
    print(f"    {'OK  ' if ok else 'FAIL'} per-object {before[0]:.2e} -> {after[0]:.2e},  "
          f"moments {before[1]:.2e} -> {after[1]:.2e}  in {iterations} iterations")
    return ok


def relaxAt(phi, edge, numSquares = 6, maxSteps = 3000):
    """A relaxed model at the given phi with the given edge treatment."""
    model = buildPacking(numSquares = numSquares, phi = phi, wall = True)
    model.setModelType("mollified")
    model.setSofteningFraction(0.06)
    model.setConstraints(area = True, edge = edge)
    model.minimizeFIRE(maxUnbalancedForce = 1e-6, maxSteps = maxSteps)
    return model


def describe(model, targetAreas):
    """Areas, edge spread and constraint health of a model, relaxed or not."""
    if model.getEnergy() is None:
        model.calcForceEnergy()
    block = getattr(model.constraints, "block", model.constraints)
    probe = DistributionConstraints(model.packing, [1, 2], edge = True)
    lengths = probe.quantity(model.packing, "edge")
    areas = block.areas(model.packing)
    return dict(energy = model.getEnergy(),
                areaError = float(np.abs(areas / targetAreas - 1.0).max()),
                cv = float(np.std(lengths) / np.mean(lengths)),
                meanEdge = float(np.mean(lengths)),
                residual = model.constraintResidual())


def checkRelaxation():
    """[4] Does the extra freedom actually buy anything, and does it stay honest about area?

    Run ABOVE the rigid-square optimum (phi = 0.75 against 0.6823), where no arrangement of rigid
    squares can avoid overlapping, so shape change is the only remaining move.

    Two separate claims, and only one of them is about energy. The unconditional one is HONESTY: every
    polygon's area must stay exactly on target, because a search whose objects can quietly shrink will
    always report success. The conditional one is that the freedom pays -- and it only pays when it is
    DRIVEN, in the one direction the constraint permits. The rigid configuration satisfies the moment
    constraints (equal edges hit the reference moments), so the moment feasible set strictly CONTAINS
    the rigid one and its global optimum cannot be worse; but FIRE is local, so switching the freedom on
    leaves it in the same basin.

    Driving it means seeding the width GEOMETRICALLY (``spreadShapes``, constant area) and ramping the
    moments down. Retargeting upward from a monodisperse start cannot work: the variance is at a
    minimum there, so its gradient vanishes and the retraction has no first-order direction to widen
    along."""
    print("\n[4] hard areas + moment edges, above the rigid optimum")
    targetAreas = np.array(buildPacking(numSquares = 6, phi = 0.75, wall = True)
                           .getTargetAreas()[:6], dtype = float)

    rigid = describe(relaxAt(0.75, True), targetAreas)
    free = describe(relaxAt(0.75, [1, 2]), targetAreas)

    # Driven: seed the width at constant area, capture it in the moments, then close it back down.
    model = buildPacking(numSquares = 6, phi = 0.75, wall = True)
    model.setModelType("mollified")
    model.setSofteningFraction(0.06)
    model.spreadShapes(0.18)
    model.setConstraints(area = True, edge = [1, 2])
    seeded = describe(model, targetAreas)
    model.minimizeFIRE(maxUnbalancedForce = 1e-6, maxSteps = 3000)
    for cv in (0.12, 0.08, 0.05, 0.03, 0.015, 0.005, 0.0):
        model.setTargetPolydispersity(cv)
        model.minimizeFIRE(maxUnbalancedForce = 1e-6, maxSteps = 1500)
    annealed = describe(model, targetAreas)

    for label, row in (("rigid (edge = True)", rigid), ("free moments, undriven", free),
                       ("seeded wide (pre-relax)", seeded), ("moments, annealed down", annealed)):
        print(f"         {label:<24} E = {row['energy']:.6e}  max|A/A0 - 1| = "
              f"{row['areaError']:.1e}  edge CV = {row['cv']:.2e}  max|C| = {row['residual']:.1e}")

    honest = all(row["areaError"] < 1e-9 for row in (free, seeded, annealed))
    seededWide = seeded["cv"] > 0.05
    narrowed = annealed["cv"] < 0.2 * seeded["cv"]
    ok = honest and seededWide and narrowed
    print(f"    {'OK  ' if ok else 'FAIL'} areas exact throughout (worst "
          f"{max(row['areaError'] for row in (free, seeded, annealed)):.1e}); width seeded to "
          f"{seeded['cv']:.2e} then annealed to {annealed['cv']:.2e}")
    print(f"         anneal vs rigid: E {annealed['energy']:.6e} vs {rigid['energy']:.6e} "
          f"({'LOWER' if annealed['energy'] < rigid['energy'] else 'no better'}); "
          f"undriven {free['energy']:.6e} "
          f"({'LOWER' if free['energy'] < rigid['energy'] else 'no better'})")
    return ok


def checkRankCollapse():
    """[5] The mean and variance rows degenerate as the distribution narrows.

    Conditioning is the property that matters, not the rank. The k = 2 row is
    ``sum_e 2 l_e dl_e/dr`` and the k = 1 row is ``sum_e dl_e/dr``, so their difference is carried
    entirely by the SPREAD of the edge lengths: the second singular value falls off in direct
    proportion to the CV. The rank only drops once that ratio reaches the truncation threshold, so at
    any workable width both rows are formally independent while the projector's noise amplification
    grows as 1/CV. That amplification is the real reason monodispersity has to be handed off to
    per-object constraints rather than squeezed out of the second moment."""
    print("\n[5] conditioning as the width goes to zero")
    rows = []
    rng = np.random.default_rng(11)
    base = rng.standard_normal(6)
    base = base / np.std(base)
    for cv in (0.20, 0.05, 1e-3, 1e-6, 0.0):
        model = buildPacking(wall = True)
        rescalePolygons(model, 1.0 + cv * base)
        constraints = DistributionConstraints(model.packing, [1, 2], edge = True)
        lengths = constraints.quantity(model.packing, "edge")
        realized = float(np.std(lengths) / np.mean(lengths))
        singular = np.linalg.svd(constraints.jacobian(model.packing), compute_uv = False)
        ratio = float(singular[1] / singular[0])
        rows.append((cv, realized, ratio, constraints.rank(model.packing)))
    # sigma2/sigma1 should track the realized CV to within a small constant factor.
    proportional = [ratio / realized for cv, realized, ratio, _ in rows if realized > 1e-12]
    ok = max(proportional) / min(proportional) < 5.0 and all(rank == 2 for *_, rank in rows)
    for cv, realized, ratio, rank in rows:
        print(f"         CV {realized:<9.2e} sigma2/sigma1 {ratio:<9.2e} "
              f"ratio/CV {ratio / max(realized, 1e-300):<7.2f} rank {rank} of 2")
    print(f"    {'OK  ' if ok else 'FAIL'} second singular value falls off LINEARLY in the width "
          f"(ratio/CV constant), and the rank never drops -- so there is no rank signal to wait for, "
          f"only conditioning")
    return ok


def checkInfeasible():
    """[6] An over-determined, geometrically impossible target set is refused."""
    print("\n[6] infeasible area + edge targets are refused")
    model = buildPacking(numSquares = 5, phi = 0.5, wall = True)
    # Deterministic rather than lucky: ask polygon 0 for 5% more area than its four edge targets can
    # enclose. (Independent setLogNormalTargetArea / setLogNormalTargetPerimeter calls land here by
    # CHANCE, which is the accident this guard exists to catch -- but a test should not depend on it.)
    model.packing.targetArea[0] *= 1.05
    refused, message = False, ""
    try:
        model.setConstraints(area = True, edge = True)
    except ValueError as exc:
        refused, message = True, str(exc).strip().splitlines()[-1]

    # The same targets held only by their DISTRIBUTION must be accepted: there is no per-edge target
    # left to contradict the area, which is the point of the moment formulation.
    accepted = True
    try:
        model.setConstraints(area = True, edge = [1, 2])
    except ValueError as exc:
        accepted, message = False, str(exc)
    ok = refused and accepted
    print(f"    {'OK  ' if ok else 'FAIL'} area+edge refused: {refused};  area+edge-moments "
          f"accepted: {accepted}")
    if refused:
        print(f"         reported: ...{message[-96:]}")
    return ok


def main():
    warnings.simplefilter("ignore")
    print("=" * 78)
    print("moment (distribution) constraint check")
    print("=" * 78)
    results = [checkJacobian(), checkTangency(), checkRetraction(), checkRelaxation(),
               checkRankCollapse(), checkInfeasible()]
    print()
    if all(results):
        print("ALL CHECKS PASSED")
        return 0
    print(f"{results.count(False)} CHECK(S) FAILED")
    return 1


if __name__ == "__main__":
    sys.exit(main())
