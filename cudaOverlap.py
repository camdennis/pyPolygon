"""ctypes binding to the CUDA whole-packing mollified overlap driver (cuda/plummerDriver.cu).

Drop-in replacement for energies.plummerOverlapExact: same (energy, gradient) contract, computed on
the GPU. Requires the shared library built by ``make -C cuda libplummer.so`` (or the Makefile's
``libplummer`` target). ``isAvailable()`` reports whether the library loaded, so the model can fall
back to the Python tier when the GPU/build is absent.
"""
import ctypes
import os
import warnings

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_SO = os.path.join(_HERE, "cuda", "libplummer.so")

_lib = None
try:
    _lib = ctypes.CDLL(_SO)
    _dp = ctypes.POINTER(ctypes.c_double)
    _ip = ctypes.POINTER(ctypes.c_int)
    _lib.plummerOverlapCuda.restype = ctypes.c_int
    _lib.plummerOverlapCuda.argtypes = [
        _dp, ctypes.c_int, _ip, ctypes.c_int, _dp, _dp, _dp,
        ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_int, _dp, _dp]
    _lib.selfRepulsionCuda.restype = None
    _lib.selfRepulsionCuda.argtypes = [
        _dp, ctypes.c_int, _ip, ctypes.c_int, ctypes.c_double, ctypes.c_double, _dp, _dp]
    _lib.springsCuda.restype = None
    _lib.springsCuda.argtypes = [
        _dp, ctypes.c_int, _ip, _ip, _ip, ctypes.c_int, _dp, _dp,
        ctypes.c_double, ctypes.c_double, _dp, _dp]
    _lib.sharpOverlapCuda.restype = None
    _lib.sharpOverlapCuda.argtypes = [
        _dp, _ip, ctypes.c_int, ctypes.c_int, _dp, ctypes.c_int, ctypes.c_double,
        ctypes.c_int, _dp, _dp, _ip]
    _lib.plummerMeasureGridCuda.restype = None
    _lib.plummerMeasureGridCuda.argtypes = [
        _dp, ctypes.c_int, _ip, ctypes.c_int, _dp, _dp, ctypes.c_int, ctypes.c_double,
        ctypes.c_int, _dp]
    _lib.containerEnergyForceCuda.restype = ctypes.c_int
    _lib.containerEnergyForceCuda.argtypes = [
        _dp, ctypes.c_int, _ip, ctypes.c_int, ctypes.c_int, _ip, _ip, _ip, _dp,
        ctypes.c_double, ctypes.c_double, ctypes.c_double, _dp, _dp]
    _lib.neighborPairsCuda.restype = ctypes.c_int
    _lib.neighborPairsCuda.argtypes = [
        _dp, ctypes.c_int, _ip, ctypes.c_int, _ip, _ip, _dp, _dp,
        ctypes.c_double, ctypes.c_int, ctypes.c_int, _ip, _ip]
    _lib.softDepthCuda.restype = ctypes.c_int
    _lib.softDepthCuda.argtypes = [
        _dp, _ip, ctypes.c_int, ctypes.c_int, _dp, _dp, ctypes.c_int,
        ctypes.c_double, ctypes.c_double, ctypes.c_int, ctypes.c_int, _dp, _dp]
    _lib.polyContactCuda.restype = ctypes.c_int
    _lib.polyContactCuda.argtypes = [
        _dp, _ip, ctypes.c_int, ctypes.c_int, _dp, _dp,
        ctypes.c_double, ctypes.c_int, ctypes.c_double,
        ctypes.c_double, ctypes.c_int, _dp, _dp]
    _lib.roundedContactCuda.restype = ctypes.c_int
    _lib.roundedContactCuda.argtypes = [
        _dp, ctypes.c_int, _ip, ctypes.c_int, _dp, ctypes.c_int,
        ctypes.c_double, ctypes.c_double, _dp, _dp]
    _lib.roundedAreaCuda.restype = ctypes.c_int
    _lib.roundedAreaCuda.argtypes = [
        _dp, ctypes.c_int, _ip, ctypes.c_int, _dp, ctypes.c_int, _dp]
    _lib.roundedAreaGradientCuda.restype = ctypes.c_int
    _lib.roundedAreaGradientCuda.argtypes = [
        _dp, ctypes.c_int, _ip, ctypes.c_int, _dp, ctypes.c_int, _dp, _dp]
    _lib.cudaDeviceReady.restype = ctypes.c_int
    _lib.cudaDeviceReady.argtypes = []
    _lib.cudaLastError.restype = ctypes.c_int
    _lib.cudaLastError.argtypes = []
except OSError as exc:
    _lib = None
    _LOAD_ERROR_MESSAGE = str(exc)
except AttributeError as exc:
    _lib = None
    _LOAD_ERROR_MESSAGE = f"missing symbol: {exc}"
else:
    _LOAD_ERROR_MESSAGE = None


_deviceReady = None
_LOAD_ERROR = _LOAD_ERROR_MESSAGE


def _warnFallback(reason, remedy):
    """Announce, once, that the numpy tier is being used instead of the GPU.

    Worth being loud about: the numpy overlap is 150-650x slower, so a silent fallback looks like the
    machine got mysteriously slow rather than like a configuration problem. ``once`` is set explicitly
    rather than relying on the warnings registry, since this is polled from inside the force loop."""
    warnings.warn(
        f"\n*** pyPolygon is falling back to the PYTHON (numpy) tier -- {reason} ***\n"
        f"    The GPU overlap is 150-650x faster; expect minimization to be very slow.\n"
        f"    {remedy}", stacklevel = 3)


def isAvailable():
    """True only if the library loaded AND a usable GPU responds right now. Warns ONCE, loudly, on
    either failure, so a run never quietly drops onto the slow path.

    The load alone proves nothing: a driver/library version mismatch (an unattended driver update
    without a reboot is the usual cause) leaves every CUDA call failing while the .so still loads,
    and the kernels then hand back zeros. A zero force looks exactly like a converged packing, so the
    device is probed and the answer cached rather than re-tested on every force evaluation.

    THE CACHE IS STICKY, AND THAT CAN STRAND A LONG-LIVED PROCESS. A single failed probe holds for the
    life of the interpreter, so a TRANSIENT failure -- the shared library being rebuilt underneath a
    running kernel, or another process briefly exhausting the device -- silently demotes a Jupyter
    session to the numpy tier until it is restarted, at 150-650x the cost. Call
    ``resetAvailability()`` to re-probe once the cause is cleared, instead of restarting the kernel."""
    global _deviceReady
    if _deviceReady is not None:
        return _deviceReady
    if _lib is None:
        _deviceReady = False
        _warnFallback(
            f"the CUDA library could not be loaded ({_LOAD_ERROR})",
            "Build it with 'make -C cuda libplummer.so'.")
        return False
    try:
        _deviceReady = bool(_lib.cudaDeviceReady())
    except AttributeError:
        _deviceReady = False
    if not _deviceReady:
        _warnFallback(
            "the library loaded but no usable GPU responded",
            "A driver/library version mismatch after a driver update is the usual cause -- compare "
            "'cat /proc/driver/nvidia/version' with the installed libcuda; a reboot normally clears "
            "it. If the device was only BUSY or the library was rebuilt under this process, the "
            "failure is transient: call cudaOverlap.resetAvailability() to probe again.")
    return _deviceReady


def resetAvailability():
    """Forget a cached probe result and re-test the device. Returns the new answer.

    For recovering a session that was demoted to the numpy tier by a transient failure -- the usual one
    being ``libplummer.so`` rebuilt while this process had it mapped. Without this the only cure is
    restarting the interpreter, which for a notebook means losing the packing being worked on."""
    global _deviceReady
    _deviceReady = None
    return isAvailable()


def _checkError(what):
    """Raise if the last CUDA call failed, so a failure can never masquerade as a zero result."""
    if _lib is None:
        return
    try:
        code = int(_lib.cudaLastError())
    except AttributeError:
        return
    if code != 0:
        raise RuntimeError(f"CUDA error {code} during {what}; results are not trustworthy")


def _c(a, dtype):
    a = np.ascontiguousarray(a, dtype = dtype)
    return a, a.ctypes.data_as(ctypes.POINTER(ctypes.c_double if dtype == np.float64 else ctypes.c_int))


def plummerOverlapCuda(packing, sigma, targetArea, targetPerimeter, gOn = 2.0, gOff = 3.0,
                       numActive = None):
    """Whole-packing mollified overlap energy + vertex gradient dU/dv on the GPU (matches
    energies.plummerOverlapExact). ``targetArea`` / ``targetPerimeter`` are per-polygon arrays (the
    normalizer and covering radii). Returns ``(energy, grad)`` with grad shape (numVertices, 2)."""
    if _lib is None:
        raise RuntimeError("libplummer.so not loaded; build it (make -C cuda libplummer.so)")
    r = packing.positions.reshape(-1, 2)
    starts = np.asarray(packing.startIndices, dtype = np.int32)
    numPoly = int(starts.size - 1) if numActive is None else int(numActive)
    starts = np.ascontiguousarray(starts[:numPoly + 1])
    numVert = int(r.shape[0])
    from energies import polygonCentroidsRadii
    cent, rad = polygonCentroidsRadii(packing)
    cent = np.ascontiguousarray(cent[:numPoly])
    rad = np.ascontiguousarray(rad[:numPoly], dtype = np.float64)
    Atgt = np.ascontiguousarray(np.asarray(targetArea, dtype = np.float64)[:numPoly])

    pos, posP = _c(r.ravel(), np.float64)
    startsC, startsP = _c(starts, np.int32)
    centC, centP = _c(cent.ravel(), np.float64)
    radC, radP = _c(rad, np.float64)
    AtgtC, AtgtP = _c(Atgt, np.float64)
    energy = np.zeros(1, dtype = np.float64)
    grad = np.zeros(2 * numVert, dtype = np.float64)
    status = _lib.plummerOverlapCuda(posP, numVert, startsP, numPoly, centP, radP, AtgtP,
                                     float(sigma), float(gOn), float(gOff),
                            1 if packing.box is not None else 0,
                            energy.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                            grad.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))
    if status != 0:
        raise ValueError(
            f"a polygon has {status} vertices, above the CUDA driver's PLUMMER_MAXN limit. Raise "
            f"PLUMMER_MAXN in cuda/plummerDriver.cu and rebuild with 'make -C cuda libplummer.so'. "
            f"(The limit caps buffer memory, not accuracy -- it is reported rather than truncated "
            f"because silently dropping the gradient is far worse than failing.)")
    _checkError("plummerOverlapCuda")
    return float(energy[0]), grad.reshape(-1, 2)


def selfRepulsionCuda(packing, kSelf, delta):
    """Intra-polygon self-repulsion energy + force on the GPU (matches
    energies.selfRepulsionEnergyForce). Returns ``(energy, force)`` with force a flat (2N,) array.

    The non-adjacent edge-pair list is the same topology index energies._selfRepEdgePairs caches on
    the packing, interleaved to (K, 4) for the device."""
    from energies import _selfRepEdgePairs
    if _lib is None:
        raise RuntimeError("libplummer.so not loaded; build it (make -C cuda libplummer.so)")
    numVert = int(packing.positions.size // 2)
    iA0, iA1, iB0, iB1 = _selfRepEdgePairs(packing)
    force = np.zeros(2 * numVert, dtype = np.float64)
    if iA0.size == 0:
        return 0.0, force
    pairs = np.ascontiguousarray(np.stack([iA0, iA1, iB0, iB1], axis = 1).ravel(), dtype = np.int32)
    pos, posP = _c(packing.positions.ravel(), np.float64)
    energy = np.zeros(1, dtype = np.float64)
    dp = ctypes.POINTER(ctypes.c_double)
    _lib.selfRepulsionCuda(posP, numVert, pairs.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
                           int(iA0.size), float(kSelf), float(delta),
                           energy.ctypes.data_as(dp), force.ctypes.data_as(dp))
    _checkError("selfRepulsionCuda")
    return float(energy[0]), force


def sharpOverlapCuda(packing, kOverlap = 1.0):
    """Sharp (unmollified) overlap energy + force on the GPU, matching
    energies.sharpOverlapEnergyForce. Returns ``(energy, force)`` with force shape (numVertices, 2).

    The NORMALIZED-SQUARED contact law ``U = 2 k sum (a_AB / norm_AB)^2`` with
    ``norm_AB = targetArea[A] + targetArea[B]``, the same functional as the mollified tier -- so the
    device returns the energy directly rather than an area, and ``targetArea`` has to be uploaded.
    Container pairs are skipped, the wall being handled by the container term with its own normalizer.

    Periodicity is taken from ``packing.box``: a square box turns on the minimum-image pair shift and
    the wrap-around cell neighborhoods; free space (box is None) runs the plain bounded grid."""
    if _lib is None:
        raise RuntimeError("libplummer.so not loaded; build it (make -C cuda libplummer.so)")
    numVert = int(packing.positions.size // 2)
    numPoly = int(packing.numPolygons)
    pos, posP = _c(packing.positions.ravel(), np.float64)
    starts = np.ascontiguousarray(packing.startIndices, dtype = np.int32)
    targets, targetsP = _c(np.asarray(packing.targetArea, dtype = np.float64).ravel(), np.float64)
    container = getattr(packing, "containerIndex", None)
    energy = np.zeros(1, dtype = np.float64)
    grad = np.zeros(2 * numVert, dtype = np.float64)
    numInter = np.zeros(1, dtype = np.int32)
    ip = ctypes.POINTER(ctypes.c_int)
    dp = ctypes.POINTER(ctypes.c_double)
    _lib.sharpOverlapCuda(posP, starts.ctypes.data_as(ip), numPoly, numVert,
                          targetsP, -1 if container is None else int(container), float(kOverlap),
                          1 if packing.box is not None else 0,
                          energy.ctypes.data_as(dp), grad.ctypes.data_as(dp),
                          numInter.ctypes.data_as(ip))
    _checkError("sharpOverlapCuda")
    return float(energy[0]), -grad.reshape(-1, 2)


def springsCuda(packing, kEdge, kArea):
    """eqSoftBody shape springs (RELATIVE form) on the GPU, matching
    softBody.eqSoftBodyEnergyForce(..., relative = True). Returns ``(energy, force)`` flat (2N,)."""
    if _lib is None:
        raise RuntimeError("libplummer.so not loaded; build it (make -C cuda libplummer.so)")
    numVert = int(packing.positions.size // 2)
    numPoly = int(packing.numPolygons)
    pos, posP = _c(packing.positions.ravel(), np.float64)
    shapeId = np.ascontiguousarray(packing.shapeId, dtype = np.int32)
    nextIdx = np.ascontiguousarray(packing.next, dtype = np.int32)
    prevIdx = np.ascontiguousarray(packing.prev, dtype = np.int32)
    l0 = np.ascontiguousarray(packing.targetEdgeLength, dtype = np.float64)
    A0 = np.ascontiguousarray(packing.targetArea, dtype = np.float64)
    energy = np.zeros(1, dtype = np.float64)
    force = np.zeros(2 * numVert, dtype = np.float64)
    ip = ctypes.POINTER(ctypes.c_int)
    dp = ctypes.POINTER(ctypes.c_double)
    _lib.springsCuda(posP, numVert, shapeId.ctypes.data_as(ip), nextIdx.ctypes.data_as(ip),
                     prevIdx.ctypes.data_as(ip), numPoly,
                     l0.ctypes.data_as(dp), A0.ctypes.data_as(dp), float(kEdge), float(kArea),
                     energy.ctypes.data_as(dp), force.ctypes.data_as(dp))
    _checkError("springsCuda")
    return float(energy[0]), force


def softDepthCuda(packing, epsilon, stiffness = 1.0, quadratureOrder = 16):
    """Soft-depth boundary-area energy + force on the GPU, matching
    softDepth.packingEnergyForce. Returns ``(energy, force)`` flat (2N,).

    PAIR INTERACTIONS ONLY -- no container, matching the kernel. The caller is responsible for not
    reaching here with a container packing that expects confinement (which is broken on both tiers).

    Centroids and covering radii are computed HOST-side by ``energies.polygonCentroidsRadii`` and
    uploaded, so the device culls exactly the pairs the numpy path culls and any disagreement in the
    result can only be the arithmetic, never the pair selection."""
    if _lib is None:
        raise RuntimeError("libplummer.so not loaded; build it (make -C cuda libplummer.so)")
    from energies import polygonCentroidsRadii
    numVert = int(packing.positions.size // 2)
    numPoly = int(packing.numPolygons)
    pos, posP = _c(packing.positions.ravel(), np.float64)
    starts = np.ascontiguousarray(packing.startIndices, dtype = np.int32)
    centroids, radii = polygonCentroidsRadii(packing)
    cent, centP = _c(np.ascontiguousarray(centroids, dtype = np.float64).ravel(), np.float64)
    rad, radP = _c(np.ascontiguousarray(radii, dtype = np.float64).ravel(), np.float64)
    container = getattr(packing, "containerIndex", None)
    energy = np.zeros(1, dtype = np.float64)
    force = np.zeros(2 * numVert, dtype = np.float64)
    ip = ctypes.POINTER(ctypes.c_int)
    dp = ctypes.POINTER(ctypes.c_double)
    status = _lib.softDepthCuda(posP, starts.ctypes.data_as(ip), numPoly, numVert, centP, radP,
                                -1 if container is None else int(container),
                                float(epsilon), float(stiffness), int(quadratureOrder),
                                1 if packing.box is not None else 0,
                                energy.ctypes.data_as(dp), force.ctypes.data_as(dp))
    if status > 0:
        raise ValueError(
            f"a polygon has {status} vertices, past this build's SOFTDEPTH_MAXN. Raise it in "
            f"cuda/softDepthKernels.cu and rebuild (make -C cuda libplummer.so). It is REPORTED "
            f"rather than truncated on purpose -- a silent cap here once dropped gradients while the "
            f"energy stayed correct.")
    if status < 0:
        raise ValueError(f"quadratureOrder {-status} is not tabulated on the device (16 or 32)")
    _checkError("softDepthCuda")
    return float(energy[0]), force


def polyContactCuda(bodies, stiffness = 1.0, wallStiffness = 1.0):
    """Depth-contact energy and gradient on the GPU, matching
    polyContactSystem.systemEnergyGradient. Returns ``(energy, gradient)`` with gradient (V, 2).

    ``bodies`` is a ``polyContactSystem.BodySet``. Centroids and circumradii are computed HOST-side and
    uploaded so the device culls exactly the pairs the host culls, leaving any disagreement to be
    arithmetic rather than pair selection.

    ``wallStiffness`` multiplies the stiffness for work items touching ``bodies.exterior``. The kernel
    applies it as a per-pair scalar, which is exact: energy and gradient are both linear in k, so there
    is no separate code path for the wall. ``bodies.exterior`` of None sends -1, which disables it."""
    if _lib is None:
        raise RuntimeError("libplummer.so not loaded; build it (make -C cuda libplummer.so)")
    import polyContactSystem
    vertexCount = len(bodies.positions)
    pos, posP = _c(np.ascontiguousarray(bodies.positions, dtype = np.float64).ravel(), np.float64)
    starts = np.ascontiguousarray(bodies.startIndices, dtype = np.int32)
    centroids = polyContactSystem.bodyCentroids(bodies)
    radii = polyContactSystem.circumradii(bodies)
    cent, centP = _c(np.ascontiguousarray(centroids, dtype = np.float64).ravel(), np.float64)
    rad, radP = _c(np.ascontiguousarray(radii, dtype = np.float64).ravel(), np.float64)
    energy = np.zeros(1, dtype = np.float64)
    gradient = np.zeros(2 * vertexCount, dtype = np.float64)
    ip = ctypes.POINTER(ctypes.c_int)
    dp = ctypes.POINTER(ctypes.c_double)
    exterior = getattr(bodies, "exterior", None)
    status = _lib.polyContactCuda(posP, starts.ctypes.data_as(ip), bodies.count, vertexCount,
                                  centP, radP, float(stiffness),
                                  -1 if exterior is None else int(exterior), float(wallStiffness),
                                  0.0 if bodies.boxSize is None else float(bodies.boxSize),
                                  0 if bodies.boxSize is None else 1,
                                  energy.ctypes.data_as(dp), gradient.ctypes.data_as(dp))
    if status > 0:
        raise ValueError(
            f"a body has {status} vertices, past this build's POLYCONTACT_MAXN. Raise it in "
            f"cuda/polyContactKernels.cu and rebuild. It is REPORTED rather than truncated on "
            f"purpose -- a silent cap here once dropped gradients while the energy stayed correct.")
    _checkError("polyContactCuda")
    return float(energy[0]), gradient.reshape(-1, 2)


def plummerMeasureGridCuda(packing, gridX, gridY, sigma, polygons = None):
    """Mollified membership field Psi at the given grid points, summed over every polygon and its 8
    periodic images -- the GPU port of the loop in ``Model.draw(indicatorColorMap = ...)``.

    ``gridX`` / ``gridY`` are flat arrays of the same length; returns Psi as a flat array. This is a
    visualization cost rather than a physics one, but at the default 160x160 grid the numpy version
    is ~10.5 s per frame against 33 ms for a plain draw, which makes it the slowest thing in an
    interactive session."""
    if _lib is None:
        raise RuntimeError("libplummer.so not loaded; build it (make -C cuda libplummer.so)")
    numVert = int(packing.positions.size // 2)
    # ``polygons`` = (first, last) restricts the sum to that CSR slice. The container is drawn as its
    # OUTSIDE rather than its inside, so it must be measured separately from the shapes.
    allStarts = np.asarray(packing.startIndices, dtype = np.int32)
    if polygons is None:
        starts = np.ascontiguousarray(allStarts)
    else:
        first, last = polygons
        starts = np.ascontiguousarray(allStarts[first:last + 1])
    numPoly = int(starts.size - 1)
    pos, posP = _c(packing.positions.ravel(), np.float64)
    gx = np.ascontiguousarray(gridX, dtype = np.float64).ravel()
    gy = np.ascontiguousarray(gridY, dtype = np.float64).ravel()
    field = np.zeros(gx.size, dtype = np.float64)
    dp = ctypes.POINTER(ctypes.c_double)
    _lib.plummerMeasureGridCuda(posP, numVert,
                                starts.ctypes.data_as(ctypes.POINTER(ctypes.c_int)), numPoly,
                                gx.ctypes.data_as(dp), gy.ctypes.data_as(dp), gx.size,
                                float(sigma), 1 if packing.box is not None else 0,
                                field.ctypes.data_as(dp))
    _checkError("plummerMeasureGridCuda")
    return field


def containerEnergyForceCuda(packing, sigma, kContainer = 1.0):
    """Fixed-boundary confinement energy + force on the GPU (matches energies.containerEnergyForce).

    Returns ``(energy, force)`` with force a flat (2N,) array. The WALL's own gradient is not
    computed -- it is pinned in every use so far, so the caller must fall back to the numpy routine
    if any wall vertex is free."""
    from energies import containerOrientationSign
    if _lib is None:
        raise RuntimeError("libplummer.so not loaded; build it (make -C cuda libplummer.so)")
    containerIndex = packing.containerIndex
    numVert = int(packing.positions.size // 2)
    numPoly = int(packing.numPolygons)
    pos, posP = _c(packing.positions.ravel(), np.float64)
    starts = np.ascontiguousarray(packing.startIndices, dtype = np.int32)
    nextIdx = np.ascontiguousarray(packing.next, dtype = np.int32)
    prevIdx = np.ascontiguousarray(packing.prev, dtype = np.int32)
    shapeId = np.ascontiguousarray(packing.shapeId, dtype = np.int32)
    Atgt = np.ascontiguousarray(packing.targetArea, dtype = np.float64)
    energy = np.zeros(1, dtype = np.float64)
    force = np.zeros(2 * numVert, dtype = np.float64)
    ip = ctypes.POINTER(ctypes.c_int); dp = ctypes.POINTER(ctypes.c_double)
    status = _lib.containerEnergyForceCuda(
        posP, numVert, starts.ctypes.data_as(ip), numPoly, int(containerIndex),
        nextIdx.ctypes.data_as(ip), prevIdx.ctypes.data_as(ip), shapeId.ctypes.data_as(ip),
        Atgt.ctypes.data_as(dp), float(sigma), float(kContainer),
        float(containerOrientationSign(packing, containerIndex)),
        energy.ctypes.data_as(dp), force.ctypes.data_as(dp))
    if status != 0:
        raise ValueError(f"containerEnergyForceCuda rejected the packing (status {status}): it needs "
                         f"at least one ordinary polygon and a wall with 3+ vertices")
    _checkError("containerEnergyForceCuda")
    return float(energy[0]), force


def neighborPairsCuda(packing, skin = 0.0, maxPairs = None):
    """Candidate edge pairs whose bounding balls overlap, found on the GPU.

    Returns ``(edgeI, edgeJ)`` global edge-index arrays, the same object
    ``neighbors.candidateEdgePairs`` returns and the same SET of pairs -- see the note in
    ``cuda/neighbors.cu`` for why the host's polygon-level prefilter can be skipped here without
    changing the answer.

    ``maxPairs`` sizes the output buffer. The kernel reports how many pairs it FOUND even when that
    exceeds the buffer, so an undersized guess is detected rather than silently truncating: the call
    retries once at the true size. That matters more than it sounds -- a truncated candidate list
    silently under-reports overlap, which is the failure mode the whole design guards against."""
    if _lib is None:
        raise RuntimeError("libplummer.so not loaded; build it (make -C cuda libplummer.so)")
    numVert = int(packing.positions.size // 2)
    numPoly = int(packing.numPolygons)
    if numVert < 2 or numPoly < 2:
        return np.zeros(0, dtype = np.int32), np.zeros(0, dtype = np.int32)
    pos, posP = _c(packing.positions.ravel(), np.float64)
    starts = np.ascontiguousarray(packing.startIndices, dtype = np.int32)
    nextIdx = np.ascontiguousarray(packing.next, dtype = np.int32)
    shapeId = np.ascontiguousarray(packing.shapeId, dtype = np.int32)
    from energies import polygonCentroidsRadii
    centroids, radii = polygonCentroidsRadii(packing)
    cent, centP = _c(np.asarray(centroids, dtype = np.float64).ravel(), np.float64)
    rad, radP = _c(np.asarray(radii, dtype = np.float64).ravel(), np.float64)
    ip = ctypes.POINTER(ctypes.c_int)
    periodic = 1 if packing.box is not None else 0
    budget = int(maxPairs) if maxPairs is not None else max(4096, 64 * numVert)

    for _ in range(2):
        pairI = np.zeros(budget, dtype = np.int32)
        pairJ = np.zeros(budget, dtype = np.int32)
        found = _lib.neighborPairsCuda(
            posP, numVert, starts.ctypes.data_as(ip), numPoly,
            nextIdx.ctypes.data_as(ip), shapeId.ctypes.data_as(ip), centP, radP,
            float(skin), periodic, budget,
            pairI.ctypes.data_as(ip), pairJ.ctypes.data_as(ip))
        _checkError("neighborPairsCuda")
        if found <= budget:
            return pairI[:found].astype(int), pairJ[:found].astype(int)
        budget = found
    raise RuntimeError("neighborPairsCuda could not size its buffer in two attempts")


# UNVERIFIED(Cam)
def roundedContactCuda(positions, startIndices, rho, containerIndex = None,
                       stiffness = 1.0, wallStiffness = 1.0, maxCorners = 16):
    """EXACT-ARC depth contact on the GPU. Returns ``(energy, gradient)`` with gradient shaped
    ``(bodies, 8 * maxCorners)`` -- ``dE/d(body arrays)``, NOT ``dE/d(loop, rho)``.

    THE CORNER MAP IS DELIBERATELY LEFT TO THE CALLER. Converting to backbone coordinates is
    O(bodies) and runs once per force evaluation, against O(pairs) for the integrals; doing it here
    would buy a constant on a negligible term and add a second place for a value and its derivative to
    drift apart. ``roundedContact.packingEnergyForceCuda`` does the conversion.

    The container must arrive ALREADY REVERSED if it was drawn counter-clockwise -- the wall is the
    exterior region, and the kernel does not re-derive winding. ``roundedContact.packingBodies``
    performs that flip.

    Gradient block layout matches ``roundedContact.BodyGradient.flat()``: centre, radius, sweep, tail,
    head, each padded out to ``maxCorners`` so every body has the same stride."""
    if _lib is None:
        raise RuntimeError("libplummer.so not loaded; build it (make -C cuda libplummer.so)")
    positions = np.ascontiguousarray(np.asarray(positions, dtype = np.float64).reshape(-1, 2))
    startIndices = np.ascontiguousarray(np.asarray(startIndices, dtype = np.int32).reshape(-1))
    rho = np.ascontiguousarray(np.asarray(rho, dtype = np.float64).reshape(-1))
    bodyCount = len(startIndices) - 1
    widest = int(np.diff(startIndices).max()) if bodyCount else 0
    if widest > maxCorners:
        raise ValueError(
            f"the exact-arc kernel is compiled for at most {maxCorners} corners per body and the "
            f"widest here has {widest}. Raise ROUNDED_MAXN in cuda/roundedContact.cuh and rebuild, or "
            f"use the numpy path. It is a COMPILE-TIME stride: exceeding it silently reads past the "
            f"end of a body rather than failing.")

    energy = np.zeros(1, dtype = np.float64)
    gradient = np.zeros(bodyCount * 8 * maxCorners, dtype = np.float64)
    status = _lib.roundedContactCuda(
        positions.ctypes.data_as(_dp), ctypes.c_int(len(positions)),
        startIndices.ctypes.data_as(_ip), ctypes.c_int(bodyCount),
        rho.ctypes.data_as(_dp),
        ctypes.c_int(-1 if containerIndex is None else int(containerIndex)),
        ctypes.c_double(float(stiffness)), ctypes.c_double(float(wallStiffness)),
        energy.ctypes.data_as(_dp), gradient.ctypes.data_as(_dp))
    if status != 0:
        raise RuntimeError(f"roundedContactCuda failed with CUDA status {status}")
    return float(energy[0]), gradient.reshape(bodyCount, 8 * maxCorners)


# UNVERIFIED(Cam)
def roundedAreaCuda(positions, startIndices, rho, containerIndex = None):
    """Per-pair EXACT overlap area, as an upper-triangular ``(bodies, bodies)`` matrix.

    Phase one of the area law. It is split from the gradient because ``U = 2k (a/norm)^2`` needs the
    pair's whole area before it can weight that pair's contribution -- and keeping the weighting on the
    host means the container's normalizer and stiffness convention exists in exactly one place."""
    if _lib is None:
        raise RuntimeError("libplummer.so not loaded; build it (make -C cuda libplummer.so)")
    positions = np.ascontiguousarray(np.asarray(positions, dtype = np.float64).reshape(-1, 2))
    startIndices = np.ascontiguousarray(np.asarray(startIndices, dtype = np.int32).reshape(-1))
    rho = np.ascontiguousarray(np.asarray(rho, dtype = np.float64).reshape(-1))
    bodyCount = len(startIndices) - 1
    areas = np.zeros(bodyCount * bodyCount, dtype = np.float64)
    status = _lib.roundedAreaCuda(
        positions.ctypes.data_as(_dp), ctypes.c_int(len(positions)),
        startIndices.ctypes.data_as(_ip), ctypes.c_int(bodyCount), rho.ctypes.data_as(_dp),
        ctypes.c_int(-1 if containerIndex is None else int(containerIndex)),
        areas.ctypes.data_as(_dp))
    if status != 0:
        raise RuntimeError(f"roundedAreaCuda failed with CUDA status {status}")
    return areas.reshape(bodyCount, bodyCount)


# UNVERIFIED(Cam)
def roundedAreaGradientCuda(positions, startIndices, rho, weights, containerIndex = None,
                            maxCorners = 16):
    """Phase two: ``sum_ab weights[a,b] * d(area_ab)/d(body arrays)``, one block per body.

    ``weights`` is ``dU/da`` per pair, upper-triangular to match ``roundedAreaCuda``."""
    if _lib is None:
        raise RuntimeError("libplummer.so not loaded; build it (make -C cuda libplummer.so)")
    positions = np.ascontiguousarray(np.asarray(positions, dtype = np.float64).reshape(-1, 2))
    startIndices = np.ascontiguousarray(np.asarray(startIndices, dtype = np.int32).reshape(-1))
    rho = np.ascontiguousarray(np.asarray(rho, dtype = np.float64).reshape(-1))
    weights = np.ascontiguousarray(np.asarray(weights, dtype = np.float64).reshape(-1))
    bodyCount = len(startIndices) - 1
    gradient = np.zeros(bodyCount * 8 * maxCorners, dtype = np.float64)
    status = _lib.roundedAreaGradientCuda(
        positions.ctypes.data_as(_dp), ctypes.c_int(len(positions)),
        startIndices.ctypes.data_as(_ip), ctypes.c_int(bodyCount), rho.ctypes.data_as(_dp),
        ctypes.c_int(-1 if containerIndex is None else int(containerIndex)),
        weights.ctypes.data_as(_dp), gradient.ctypes.data_as(_dp))
    if status != 0:
        raise RuntimeError(f"roundedAreaGradientCuda failed with CUDA status {status}")
    return gradient.reshape(bodyCount, 8 * maxCorners)
