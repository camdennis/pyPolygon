# UNVERIFIED(Cam)
"""Does skipping the protocol's decompression bisection change the answer?

``runPacking.cornerCut`` defaults to ``bisect = False``, on the argument that ``squeeze`` subsumes the
bisection: the bisection searches for the size at which the first pair collides WITH THE ARRANGEMENT
FROZEN, while the squeeze minimizes the box side over every center, angle and ``s`` at once. That is
an argument, not a measurement, and the default was flipped on it.

THE CONTROL IS THE POINT. CUDA is not bit-reproducible -- about 3e-12 over 120 steps -- so two runs of
the SAME configuration do not agree exactly, and a difference between the two paths means nothing until
that floor is known. So each seed is run three times: the default path twice, and the bisection path
once. If |default - default| is the same size as |default - bisect|, the flag does not matter.
"""
import os
import sys
import time
import warnings

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root not in sys.path:
    sys.path.insert(0, root)

import records
import runPacking as rp

COUNT = 11
SEEDS = (2055282738, 3018100038)
STEPS = 100
DIGITS = 60


# UNVERIFIED(Cam)
def run(count, seed, bisect, steps = STEPS):
    """``(side, rattlers, seconds)`` for one full protocol plus refinement."""
    started = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = rp.cornerCut(count, seed, scheduleSteps = steps, bisect = bisect)
        state, rattlers, _, result = rp.analyze(rp.asPacking(model, count), count, DIGITS)
        _, side = rp.unitCorners(state, count, result, DIGITS)
    return float(side), len(rattlers), time.time() - started


if __name__ == "__main__":
    known = records.bestKnownSide(COUNT)
    print(f"n = {COUNT}, best known {known:.8f}, {STEPS} rungs\n")
    for seed in SEEDS:
        first, rattlersA, timeA = run(COUNT, seed, False)
        print(f"  seed {seed}  default  {first:.12f}  rattlers {rattlersA}  {timeA:.0f}s",
              flush = True)
        second, rattlersB, timeB = run(COUNT, seed, False)
        print(f"  seed {seed}  default  {second:.12f}  rattlers {rattlersB}  {timeB:.0f}s",
              flush = True)
        third, rattlersC, timeC = run(COUNT, seed, True)
        print(f"  seed {seed}  bisect   {third:.12f}  rattlers {rattlersC}  {timeC:.0f}s",
              flush = True)
        print(f"    reproducibility floor |default - default| = {abs(first - second):.3e}")
        print(f"    effect of the flag    |default - bisect|   = {abs(first - third):.3e}")
        print(f"    bisection cost        {timeC - 0.5 * (timeA + timeB):+.0f}s\n", flush = True)
