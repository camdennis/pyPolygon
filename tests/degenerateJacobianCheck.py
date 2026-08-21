"""Where does a collapsing polygon first turn a constraint Jacobian non-finite?

A packing run reported ``1 of 1 constraint rows are NOT FINITE`` on the AREA moment row at the start
of a cascade stage, alongside ``worstFlat 1.00000`` -- every selected vertex already exactly collinear
before the flattening ramp had done any work. Those two readings together say a polygon had collapsed,
but the warning fired on the moment row, which is the one part of the composite that does not divide
by anything geometric.

This walks a polygon into two different degeneracies and reports, at each step, which layer is the
first to go non-finite:

    area -> 0 with edges intact   (squash onto a line: what "everything is flat" looks like)
    the whole polygon -> a point  (uniform shrink: area and edges together)

and separately whether ``_qrFactor`` accepts the block. That last column is the point of the file. QR
propagates a NaN silently and every comparison against a NaN is False, so the rank test
``diagonal <= tol`` REPORTS FULL RANK on an all-NaN block; the poisoned basis then leaves as if it
were healthy and the first visible symptom is a moment row failing several projections later.

Free space, pure numpy -- no contacts, no container, no GPU.

    python tests/degenerateJacobianCheck.py
"""

# UNVERIFIED(Cam)

import os
import sys
import warnings

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(_ROOT))
sys.path.insert(0, _ROOT)

import pyPolygon as pp
import constraints as cs

_KAPPA = 4.0


def model(N = 3, n = 16, seed = 42):
    m = pp.Model(N = N, n = n, seed = seed)
    m.generatePolygons(phi = 0.2, kappa = _KAPPA)
    m.setConstraints(equilateral = _KAPPA, edge = False, area = [1])
    return m


def squash(m, polygon, factor, uniform = False):
    """Collapse one polygon about its centroid: onto a LINE, or to a point."""
    r = m.packing.positions.reshape(-1, 2)
    a, b = int(m.packing.startIndices[polygon]), int(m.packing.startIndices[polygon + 1])
    centre = r[a : b].mean(axis = 0)
    scale = np.array([factor, factor]) if uniform else np.array([1.0, factor])
    r[a : b] = centre + scale * (r[a : b] - centre)


def probe(m, polygon):
    """One row of the sweep: the geometry, then each layer's finiteness."""
    block = m.constraints.block
    moments = m.constraints.distribution
    packing = m.packing

    area = float(np.abs(block.areas(packing))[polygon])
    lengths = np.where(block.valid, block.edgeLengths(packing), np.inf)
    edge = float(lengths[polygon].min())

    blockJ = block.jacobian(packing)
    momentJ = np.atleast_2d(moments.jacobian(packing))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        basis = m.constraints.normalBasis(packing)
    return {
        "area": area,
        "edge": edge,
        "blockJ": bool(np.all(np.isfinite(blockJ))),
        "blockMax": float(np.abs(blockJ[np.isfinite(blockJ)]).max(initial = 0.0)),
        "qr": block._qrFactor(blockJ) is not None,
        "momentJ": bool(np.all(np.isfinite(momentJ))),
        "basis": bool(np.all(np.isfinite(basis["block"]))),
        "extra": bool(np.all(np.isfinite(basis["extra"]))),
    }


def sweep(name, uniform):
    print(f"\n{name}")
    print(f"  {'factor':>9}  {'|area|':>10}  {'minEdge':>10}  {'max|blockJ|':>12}  "
          f"blockJ  qrOk  momentJ  basis  extra")
    firstBad = None
    for factor in (1.0, 1e-2, 1e-4, 1e-6, 1e-8, 1e-10, 1e-12, 1e-14, 1e-16, 0.0):
        m = model()
        squash(m, 0, factor, uniform = uniform)
        try:
            row = probe(m, 0)
        except FloatingPointError as error:
            print(f"  {factor:9.1e}  raised FloatingPointError: {str(error).splitlines()[0]}")
            firstBad = firstBad or ("raised", factor)
            continue
        flags = (row["blockJ"], row["qr"], row["momentJ"], row["basis"], row["extra"])
        print(f"  {factor:9.1e}  {row['area']:10.3e}  {row['edge']:10.3e}  "
              f"{row['blockMax']:12.3e}  "
              f"{str(row['blockJ']):>6}  {str(row['qr']):>4}  {str(row['momentJ']):>7}  "
              f"{str(row['basis']):>5}  {str(row['extra']):>5}")
        if firstBad is None and not all(flags[i] for i in (0, 2, 3, 4)):
            firstBad = ("non-finite", factor)
    return firstBad


def checkPoisonedInput():
    """The two things that are NOT floored: a non-finite POSITION and a zeroed stored TARGET.

    This is the scenario that matches the reported failure. Before the guard, a non-finite block
    Jacobian reached QR, was laundered into a healthy-looking basis, and surfaced as a warning about
    the innocent moment row; the block itself said nothing. After it, ``_decompose`` raises and names
    the polygon."""
    ok = True
    for name, poison in (("a NaN POSITION", "position"), ("a ZEROED targetArea", "target")):
        m = model()
        if poison == "position":
            m.packing.positions[3] = np.nan
        else:
            m.setConstraints(equilateral = None, edge = True, area = True)
            m.packing.targetArea[0] = 0.0
        block = getattr(m.constraints, "block", None) or m.constraints
        blockJ = block.jacobian(m.packing)
        polluted = not np.all(np.isfinite(blockJ))
        laundered = block._qrFactor(blockJ) is not None
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                m.constraints.normalBasis(m.packing)
            raised = None
        except FloatingPointError as error:
            raised = str(error)
        good = polluted and not laundered and raised is not None \
            and "POSITIONS:" in (raised or "")
        ok = ok and good
        print(f"\n  {name}: block Jacobian non-finite {polluted}, laundered by QR {laundered}")
        if raised is None:
            print("    NO ERROR RAISED -- the poison flows downstream")
        else:
            print(f"    {raised.splitlines()[0]}")
            for line in raised.splitlines():
                if line.startswith("    ") and ":" in line and "NOT" not in line:
                    print(f"    {line.strip()}")
        print(f"    {'ok' if good else 'FAIL'}")
    return ok


def checkQrLaundersNaN():
    """The defect on its own terms: does _qrFactor accept an all-NaN block?"""
    J = np.full((1, 4, 8), np.nan)
    accepted = cs.ShapeConstraints._qrFactor(J) is not None
    print(f"\n_qrFactor on an all-NaN block: {'ACCEPTS (laundered)' if accepted else 'returns None'}")
    diagonal = np.abs(np.diagonal(np.linalg.qr(np.swapaxes(J, 1, 2))[1], axis1 = 1, axis2 = 2))
    print(f"  its rank test reads {np.any(diagonal <= 1e-12)} -- every comparison to NaN is False")
    return not accepted


def main():
    print("degenerate geometry: which layer breaks first")
    flat = sweep("squash onto a LINE (area -> 0, edges intact)", uniform = False)
    point = sweep("shrink to a POINT (area and edges -> 0)", uniform = True)
    print("\nthe two unfloored inputs")
    poisoned = checkPoisonedInput()
    guarded = checkQrLaundersNaN()
    print(f"\nfirst failure, line squash: {flat}")
    print(f"first failure, point shrink: {point}")
    print(f"degenerate geometry stays finite: {'PASS' if flat is None and point is None else 'FAIL'}")
    print(f"poisoned input is caught and named: {'PASS' if poisoned else 'FAIL'}")
    print(f"_qrFactor guards NaN: {'PASS' if guarded else 'FAIL'}")


if __name__ == "__main__":
    main()
