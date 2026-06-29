"""Area distributions for setting up a packing of N polygons.

Areas are drawn so they sum to ``phi * cellArea`` (phi = packing fraction; cellArea
is the periodic cell area, 1 by the area-1 calibration). Two distributions:

  logNormal  -- lognormal areas with log-std ``sigma`` (continuous polydispersity).
  biDisperse -- two sizes; ``sizeRatio`` is the LENGTH ratio (so the area ratio is
                ``sizeRatio**2``), with ``fractionLarge`` of the polygons large.
  mono       -- all equal (each polygon gets phi/N).
"""

import numpy as np

def asRng(rng):
    """Coerce None / int seed / Generator into a numpy Generator."""
    if rng is None or isinstance(rng, (int, np.integer)):
        return np.random.default_rng(rng)
    return rng

def sampleAreas(N, kind = "logNormal", phi = 1.0, cellArea = 1.0, rng = None,
                sigma = 0.2, sizeRatio = 1.4, fractionLarge = 0.5):
    """Return N polygon areas (np.ndarray) that sum to ``phi * cellArea``.

    logNormal:  raw_i ~ Lognormal(mean=0, sigma), then rescaled to the target sum.
    biDisperse: ``fractionLarge`` of the polygons get area ``sizeRatio**2``, the rest
                area 1, shuffled, then rescaled to the target sum.
    """
    rng = asRng(rng)
    if kind == "logNormal":
        raw = rng.lognormal(mean = 0.0, sigma = sigma, size = N)
    elif kind == "biDisperse":
        nLarge = int(round(fractionLarge * N))
        raw = np.ones(N)
        raw[:nLarge] = sizeRatio ** 2
    elif kind == "mono":
        raw = np.ones(N)
    else:
        raise ValueError(f"unknown area distribution: {kind!r}")
    return raw / raw.sum() * (phi * cellArea)
