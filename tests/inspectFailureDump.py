"""Dissect a state dump written by a minimizer guard, offline -- no GPU, no rerun.

The stage-2 NaN resisted two measured hypotheses (an exactly collinear vertex and a fully collapsed
polygon; `tests/flatVertexEnergyCheck.py` shows the depth tier is finite through BOTH). Each of those
cost a guess, and reproducing the fault costs an hour of cascade. This reads the configuration the
guard saved at the raise, so every further hypothesis is tested against the real geometry for free.

    python tests/inspectFailureDump.py data/failure-force-<stamp>.npz

Reports the geometry the tier actually saw: per-polygon area and edge extremes, the smallest distance
between any two vertices, how far anything sits outside the unit cell, and which quantities are already
non-finite -- ranked, so the smallest/largest outliers name themselves.
"""

# UNVERIFIED(Cam)

import os
import sys

import numpy as np


def polygons(positions, startIndices = None):
    """Split a flat position array into polygons along the CSR boundaries.

    Driven by `startIndices` and never by an assumed uniform n: a cascade is exactly where vertex
    counts stop being uniform, and a reader that assumes one n merges every polygon into a single ring
    and then reports an area for a shape that does not exist."""
    points = positions.reshape(-1, 2)
    if startIndices is None:
        return [points]
    edges = np.asarray(startIndices).reshape(-1)
    return [points[edges[k]:edges[k + 1]] for k in range(len(edges) - 1)]


def signedArea(polygon):
    return 0.5 * float(np.cross(polygon, np.roll(polygon, -1, axis = 0)).sum())


def report(path):
    data = np.load(path)
    print(f"dump: {path}")
    print(f"  arrays: {', '.join(sorted(data.files))}\n")

    positions = data["positions"].astype(float)
    points = positions.reshape(-1, 2)
    finite = np.isfinite(points).all(axis = 1)
    print(f"vertices {points.shape[0]}   non-finite {int((~finite).sum())}")
    # Losing every position must NOT end the report. The TARGETS survive independently of the
    # coordinates, and they are what say whether the run was asked for something impossible -- which is
    # exactly the case where the positions are the part that got destroyed. Bailing here hid the
    # infeasible area targets behind "nothing further can be measured".
    if finite.any():
        geometry(points, finite, data)
    else:
        print("  every position is non-finite -- geometry cannot be measured, but the TARGETS below")
        print("  survive, and they are the more informative half when the coordinates are gone")
    targets(data)


def geometry(points, finite, data):
    inside = points[finite]
    print(f"  extent   x [{inside[:, 0].min():.6f}, {inside[:, 0].max():.6f}]"
          f"   y [{inside[:, 1].min():.6f}, {inside[:, 1].max():.6f}]")
    # Only meaningful with a FIXED container. Under a periodic box, coordinates outside [0, 1) are
    # ordinary unwrapped images and mean nothing -- so this is reported, never flagged as a fault.
    outside = np.maximum(np.maximum(-inside, inside - 1.0).max(), 0.0)
    print(f"  farthest outside [0, 1): {outside:.6e}"
          + ("   (a fault under a fixed container; ordinary unwrapping under a periodic box)"
             if outside > 1e-9 else ""))

    positions = data["positions"].astype(float)
    starts = data["startIndices"] if "startIndices" in data.files else None
    shapes = polygons(positions, starts)
    print(f"\npolygons {len(shapes)}"
          + ("   (CSR boundaries read from the dump)" if starts is not None
             else "   (no startIndices in dump -- treated as ONE ring; every area below is"
                  " meaningless)"))
    print(f"  {'idx':>4s}  {'n':>4s}  {'signed area':>14s}  {'min edge':>12s}  {'max edge':>12s}")
    for index, polygon in enumerate(shapes):
        if not np.isfinite(polygon).all():
            print(f"  {index:4d}  {polygon.shape[0]:4d}  {'non-finite':>14s}")
            continue
        edges = np.linalg.norm(np.roll(polygon, -1, axis = 0) - polygon, axis = 1)
        print(f"  {index:4d}  {polygon.shape[0]:4d}  {signedArea(polygon):14.6e}  "
              f"{edges.min():12.4e}  {edges.max():12.4e}")

    # Coincident vertices are the degeneracy neither earlier test covered: a zero-length edge or two
    # bodies touching at a point makes a direction undefined, where a merely small polygon does not.
    gaps = np.linalg.norm(inside[:, None, :] - inside[None, :, :], axis = 2)
    np.fill_diagonal(gaps, np.inf)
    a, b = np.unravel_index(np.argmin(gaps), gaps.shape)
    print(f"\nclosest vertex pair: {gaps[a, b]:.6e}  (indices {a}, {b})")


def targets(data):
    # The container is the LAST shape and carries a NEGATIVE signed area (it is wound clockwise, so
    # its interior is the exterior region). That sign is the marker: it separates the wall from the
    # polygons without needing containerIndex, which is not stored in the dump.
    if "targetArea" in data.files:
        areas = np.asarray(data["targetArea"], dtype = float).reshape(-1)
        wall = areas < 0.0
        if wall.any() and (~wall).any():
            held = float(abs(areas[wall].sum()))
            asked = float(areas[~wall].sum())
            print(f"\nAREA TARGETS   {int((~wall).sum())} polygons ask for {asked:.6f} "
                  f"inside a container of {held:.6f}")
            print(f"  implied packing fraction: {asked / held:.6f}")
            try:
                sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                import records
                ceiling = records.maximumDensity(int((~wall).sum()))
            except Exception:
                ceiling = None
            if ceiling is not None:
                verdict = "INFEASIBLE" if asked / held > ceiling else "feasible"
                print(f"  published ceiling for that many squares: {ceiling:.6f}   -> {verdict}")
                if verdict == "INFEASIBLE":
                    print("  the targets cannot be met by any arrangement, so the retraction cannot")
                    print("  converge and no iteration count will help -- fix the DENSITY, not the")
                    print("  minimizer")
    print()
    for name in ("targetArea", "targetPerimeter", "targetEdgeLength", "targetDiagonal",
                 "failingForce", "force", "velocities"):
        if name not in data.files:
            continue
        values = np.asarray(data[name], dtype = float).reshape(-1)
        good = values[np.isfinite(values)]
        bad = int((~np.isfinite(values)).sum())
        span = f"min {good.min():.6e}  max {good.max():.6e}" if good.size else "all non-finite"
        flag = ""
        # The container's target area is negative BY DESIGN (clockwise winding), so only a
        # non-container entry at or below zero is a fault. Flagging the wall trains the reader to
        # ignore the line that would matter.
        if good.size and name.startswith("target"):
            suspect = good[:-1] if name == "targetArea" else good
            if suspect.size and suspect.min() <= 0.0:
                flag = "   <-- a POLYGON target has reached zero or gone negative"
        print(f"{name:18s} {bad:4d} non-finite   {span}{flag}")


def main():
    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        folder = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
        dumps = sorted(f for f in os.listdir(folder) if f.startswith("failure-")) \
            if os.path.isdir(folder) else []
        if not dumps:
            print("no dump given and none found in data/; pass a path")
            return
        path = os.path.join(folder, dumps[-1])
        print(f"(no path given; using the most recent dump)\n")
    report(path)


if __name__ == "__main__":
    main()
