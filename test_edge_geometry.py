"""Tests for edge_geometry: line fitting + (possibly off-screen) corner derivation."""

from __future__ import annotations

import math

from edge_geometry import corners_from_edges, fit_line, intersect


def _normalize(line):
    a, b, c = line
    norm = math.hypot(a, b) or 1.0
    sign = 1.0 if a > 0 or (a == 0 and b > 0) else -1.0
    return (a / norm * sign, b / norm * sign, c / norm * sign)


def test_fit_line_recovers_horizontal() -> None:
    # y = 10  ->  0*x + 1*y - 10 = 0
    assert _normalize(fit_line([(0, 10), (50, 10), (100, 10)])) == _normalize((0.0, 1.0, -10.0))


def test_fit_line_recovers_vertical() -> None:
    # x = 5  ->  1*x + 0*y - 5 = 0   (a y=mx+b fit cannot represent this)
    assert _normalize(fit_line([(5, 0), (5, 40), (5, 90)])) == _normalize((1.0, 0.0, -5.0))


def test_intersect_on_screen() -> None:
    x, y = intersect(fit_line([(0, 0), (100, 0)]), fit_line([(0, 0), (0, 100)]))
    assert abs(x) < 1e-6
    assert abs(y) < 1e-6


def test_intersect_off_screen_corner() -> None:
    # top edge y = -10, left edge x = -20  ->  corner at (-20, -10), outside any frame
    top = fit_line([(0, -10), (100, -10)])
    left = fit_line([(-20, 0), (-20, 50)])
    x, y = intersect(left, top)
    assert abs(x - (-20.0)) < 1e-6
    assert abs(y - (-10.0)) < 1e-6


def test_corners_from_edges_with_off_screen_vertex() -> None:
    # A trapezoid whose top-left corner sits off-screen at (-20, -10).
    edges = {
        "top": [(0, -10), (200, -10)],
        "right": [(210, 0), (210, 300)],
        "bottom": [(0, 320), (200, 320)],
        "left": [(-20, 0), (-20, 300)],
    }
    top_left, top_right, bottom_right, bottom_left = corners_from_edges(edges)
    assert (round(top_left[0]), round(top_left[1])) == (-20, -10)  # off-screen, still derived
    assert (round(top_right[0]), round(top_right[1])) == (210, -10)
    assert (round(bottom_right[0]), round(bottom_right[1])) == (210, 320)
    assert (round(bottom_left[0]), round(bottom_left[1])) == (-20, 320)
