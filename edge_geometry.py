"""Edge-based screen geometry: fit a line per edge, derive corners by intersection.

Define each screen boundary by points along its *visible* part, fit a line, and
get each corner as the intersection of two adjacent edge lines. The intersection
is computed analytically, so it is valid even when the true corner is **off the
screen** -- which is the whole point: a vertex you cannot see cannot be clicked
or tracked, but the edges it lies on usually are visible.

Pure numpy, no OpenCV, so it can be unit-tested without a video backend.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence

Point = "tuple[float, float]"
Line = "tuple[float, float, float]"

EDGE_ORDER: Final = ("top", "right", "bottom", "left")
MIN_POINTS: Final = 2
PARALLEL_EPS: Final = 1e-9


def fit_line(points: Sequence[tuple[float, float]]) -> tuple[float, float, float]:
    """Total-least-squares line ``a*x + b*y + c = 0`` (unit normal) from >=2 points.

    Handles any orientation, including vertical edges, unlike a y = mx + b fit.
    """
    pts = np.asarray(points, dtype=np.float64)
    if pts.shape[0] < MIN_POINTS:
        message = "need at least two points to fit a line"
        raise ValueError(message)
    centroid = pts.mean(axis=0)
    _u, _s, vh = np.linalg.svd(pts - centroid, full_matrices=False)
    normal = vh[-1]  # direction of least variance is the line normal
    a, b = float(normal[0]), float(normal[1])
    c = -(a * float(centroid[0]) + b * float(centroid[1]))
    return a, b, c


def intersect(
    line1: tuple[float, float, float],
    line2: tuple[float, float, float],
) -> tuple[float, float]:
    """Intersection of two lines in ``a*x + b*y + c = 0`` form (may be off-screen)."""
    a1, b1, c1 = line1
    a2, b2, c2 = line2
    det = a1 * b2 - a2 * b1
    if abs(det) < PARALLEL_EPS:
        message = "edges are parallel; no intersection"
        raise ValueError(message)
    x = (b1 * c2 - b2 * c1) / det
    y = (a2 * c1 - a1 * c2) / det
    return float(x), float(y)


def corners_from_edges(
    edges: dict[str, Sequence[tuple[float, float]]],
) -> list[tuple[float, float]]:
    """Map top/right/bottom/left edge points to corners ``[TL, TR, BR, BL]``.

    Order matches a clockwise destination quad ``[[0,0],[W,0],[W,H],[0,H]]`` for
    ``cv2.getPerspectiveTransform``.
    """
    lines = {name: fit_line(edges[name]) for name in EDGE_ORDER}
    top_left = intersect(lines["left"], lines["top"])
    top_right = intersect(lines["top"], lines["right"])
    bottom_right = intersect(lines["right"], lines["bottom"])
    bottom_left = intersect(lines["bottom"], lines["left"])
    return [top_left, top_right, bottom_right, bottom_left]
