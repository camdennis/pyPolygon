"""Published packing records, for comparing a run against something outside this code.

A search that only reports its own number cannot be right or wrong. These are the best known values
from the literature, so a run has something to be measured against -- and for the small n where the
optimum is PROVED, being wrong is a real possibility rather than a matter of taste.

Currently one table: ``squaresInSquares``, the smallest known square containing n unit squares. Each
record carries whether the value is proved or merely the best known, which is the difference between
"this run failed" and "this run did not beat the record".

    import records
    records.bestKnownSide(5)     # 2.70710678..., proved
    records.record(5)            # the whole row, with its exact expression
    records.bestKnownSide(12)    # None -- the source does not list it, and nothing is inferred

NOTHING IS EXTRAPOLATED. The source lists only some n, and a plausible-looking rule for the rest
(``ceil(sqrt(n))``, say) would be a guess wearing the source's authority. Absent n return None.
"""

import json
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_PATH = os.path.join(_HERE, "data", "squaresInSquares.json")
_cache = None


# UNVERIFIED(Cam)
def squaresInSquares():
    """The whole table as loaded, including its provenance fields. Read once and cached."""
    global _cache
    if _cache is None:
        with open(_PATH) as handle:
            _cache = json.load(handle)
        _cache["byCount"] = {int(row["n"]): row for row in _cache["records"]}
    return _cache


# UNVERIFIED(Cam)
def record(n):
    """The full row for ``n`` unit squares, or None when the source does not list it.

    Keys: ``n``, ``side``, ``exact`` (symbolic form or None), ``proved``, and ``note`` on the one row
    whose transcription is internally inconsistent."""
    return squaresInSquares()["byCount"].get(int(n))


# UNVERIFIED(Cam)
def bestKnownSide(n):
    """Best known container side for ``n`` unit squares, or None when the source does not list it."""
    row = record(n)
    return None if row is None else float(row["side"])


# UNVERIFIED(Cam)
def maximumDensity(n):
    """Largest packing fraction ``n`` equal squares can reach: ``n / s(n)^2``. None when unlisted.

    The bound the whole search is measured against, kept here so it has ONE definition. Asking for more
    than this is not a hard optimization, it is impossible -- and the failure does not look like
    impossibility from inside a run: the constraint retraction simply stops converging, which reads as
    a budget problem and invites a larger iteration count that cannot ever help. A cascade burned an
    hour on exactly that, with area targets summing to phi 0.971 against a ceiling of 0.500."""
    side = bestKnownSide(n)
    return None if side is None else float(n) / (side * side)


# UNVERIFIED(Cam)
def isProved(n):
    """True when ``n``'s value is a PROVED optimum, False when it is only the best known, None when
    the source does not list it.

    Worth branching on. Against a proved optimum a run that lands above it has definitely failed to
    find the arrangement; against a record it may simply have found a different one."""
    row = record(n)
    return None if row is None else bool(row["proved"])


# UNVERIFIED(Cam)
def describe(n, achieved = None):
    """One line comparing an achieved side against the published value, for printing after a run.

    Says what is known rather than implying more: the status word distinguishes a proved optimum from
    a record, and an unlisted n says so instead of quietly comparing against nothing."""
    row = record(n)
    if row is None:
        return (f"n = {n}: not listed in {squaresInSquares()['source']}, so there is nothing to "
                f"compare against.")
    status = "PROVED optimum" if row["proved"] else "best known (not proved)"
    exact = f" = {row['exact']}" if row.get("exact") else ""
    text = f"n = {n}: {status} is side {row['side']:.8f}{exact}"
    if achieved is not None:
        gap = 100.0 * (float(achieved) / row["side"] - 1.0)
        text += f";  this run reached {float(achieved):.6f}, {gap:+.3f}%"
        if row["proved"] and gap < -1e-9:
            text += "  <-- BELOW A PROVED OPTIMUM, so the run is wrong, not brilliant"
    if row.get("note"):
        text += f"\n    NOTE: {row['note']}"
    return text
