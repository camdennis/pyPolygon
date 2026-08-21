# UNVERIFIED(Cam)
"""Read the live state left behind by a FloatingPointError, WITHOUT re-running.

The exception aborts the minimizer mid-loop and rolls nothing back, so the corrupted arrays are still
sitting in the model. That makes the post-mortem free -- which matters here, because reproducing the
fault costs an hour and a kernel restart drops the evidence on the floor.

Paste into the notebook after the traceback:

    import sys; sys.path.insert(0, "tests")
    import inspectPoisonedState
    inspectPoisonedState.report(packing)
"""

import numpy as np


def report(model):
    """Name the poisoned input on a model that has just raised."""
    inner = getattr(model, "packing", model)
    positions = np.asarray(inner.positions, dtype = float).reshape(-1, 2)
    finite = np.isfinite(positions).all(axis = 1)
    print(f"POSITIONS   {int((~finite).sum())} of {positions.shape[0]} vertices non-finite")
    if finite.any():
        print(f"            max|finite| {np.abs(positions[finite]).max():.6e}")
    if not finite.all():
        print(f"            first bad vertex index {int(np.flatnonzero(~finite)[0])}")

    for label, getter in (("targetArea", "getTargetAreas"),
                          ("targetPerimeter", "getTargetPerimeters"),
                          ("area", "getAreas"),
                          ("edgeLength", "getEdgeLengths")):
        try:
            values = np.asarray(getattr(model, getter)(), dtype = float)
        except Exception as error:
            print(f"{label:11s} <{type(error).__name__}: {error}>")
            continue
        flat = values[np.isfinite(values)]
        bad = int((~np.isfinite(values)).sum())
        nonPositive = int((flat <= 0.0).sum())
        note = "clean" if not (bad or nonPositive) else \
            f"{bad} non-finite, {nonPositive} <= 0"
        span = f"min {flat.min():.6e}  max {flat.max():.6e}" if flat.size else "all non-finite"
        print(f"{label:11s} {note:28s} {span}")

    # The rows divide by these, so a zero here is the fault even when every geometric quantity is
    # healthy. Reported as a RATIO because that is the quantity the constraint actually forms.
    try:
        areas = np.asarray(model.getAreas(), dtype = float)
        targets = np.asarray(model.getTargetAreas(), dtype = float)
        with np.errstate(divide = "ignore", invalid = "ignore"):
            ratio = areas / targets
        print(f"area/target  min {np.nanmin(ratio):.6e}  max {np.nanmax(ratio):.6e}  "
              f"{int((~np.isfinite(ratio)).sum())} non-finite")
    except Exception as error:
        print(f"area/target  <{type(error).__name__}: {error}>")
