#!/usr/bin/env python3
"""Edge-based dashcam crop.

Instead of tracking 4 corner points (see ``main.py``), track points ALONG each
screen edge, fit a line per edge, and derive the corners by intersecting the
lines. A corner that falls outside the frame is still recovered, because we only
ever click/template/match points on the *visible* part of each edge.

Headless (no GUI -- works over SSH, e.g. on the RTX PRO 6000):

    python edge_crop.py sample.mp4 --edges edges.json --out out.mp4

Interactive (needs a display): click >=2 points along each edge, in the order
top, right, bottom, left; press 'n' to advance to the next edge, ESC when done.

    python edge_crop.py sample.mp4 --pick

edges.json format:
    {"edges": {"top": [[x,y],[x,y]], "right": [...], "bottom": [...], "left": [...]}}
"""

from __future__ import annotations

import argparse
import json
import os

import cv2
import numpy as np

from edge_geometry import EDGE_ORDER, corners_from_edges, fit_line

TEMPLATE_SIZE = 35


def load_edges(path):
    with open(path, encoding="utf-8") as stream:
        edges = json.load(stream)["edges"]
    return {name: [(float(x), float(y)) for x, y in edges[name]] for name in EDGE_ORDER}


def make_template(frame, x, y, size):
    height, width = frame.shape[:2]
    x0, y0 = int(round(x)) - size, int(round(y)) - size
    x1, y1 = int(round(x)) + size, int(round(y)) + size
    if x0 < 0 or y0 < 0 or x1 > width or y1 > height:
        return None
    return frame[y0:y1, x0:x1].copy()


def track_point(frame, template, prev, size):
    if template is None:
        return prev
    height, width = frame.shape[:2]
    px, py = prev
    margin = size * 2
    x0, y0 = max(0, int(px) - margin), max(0, int(py) - margin)
    x1, y1 = min(width, int(px) + margin), min(height, int(py) + margin)
    area = frame[y0:y1, x0:x1]
    if area.shape[0] < template.shape[0] or area.shape[1] < template.shape[1]:
        return prev
    result = cv2.matchTemplate(area, template, cv2.TM_CCOEFF_NORMED)
    _min_v, _max_v, _min_l, max_l = cv2.minMaxLoc(result)
    new_x = x0 + max_l[0] + size
    new_y = y0 + max_l[1] + size
    if np.hypot(new_x - px, new_y - py) > margin:
        return prev
    return (float(new_x), float(new_y))


def low_pass(new, prev, alpha):
    return (alpha * new[0] + (1 - alpha) * prev[0], alpha * new[1] + (1 - alpha) * prev[1])


def draw_line(frame, line, width, height, color):
    a, b, c = line
    if abs(b) > abs(a):
        p1, p2 = (0, int(-c / b)), (width, int(-(a * width + c) / b))
    else:
        p1, p2 = (int(-c / a), 0), (int(-(b * height + c) / a), height)
    cv2.line(frame, p1, p2, color, 1)


def annotate(frame, edges, corners, width, height):
    for name in EDGE_ORDER:
        draw_line(frame, fit_line(edges[name]), width, height, (255, 0, 0))
        for x, y in edges[name]:
            cv2.circle(frame, (int(x), int(y)), 4, (0, 255, 0), -1)
    for cx, cy in corners:
        if 0 <= cx < width and 0 <= cy < height:
            cv2.circle(frame, (int(cx), int(cy)), 6, (0, 0, 255), -1)


def pick_edges(first_frame):
    edges = {name: [] for name in EDGE_ORDER}
    state = {"edge": 0}
    view = first_frame.copy()

    def on_mouse(event, x, y, _flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN and state["edge"] < len(EDGE_ORDER):
            edges[EDGE_ORDER[state["edge"]]].append((float(x), float(y)))
            cv2.circle(view, (x, y), 4, (0, 255, 0), -1)

    cv2.namedWindow("pick edges")
    cv2.setMouseCallback("pick edges", on_mouse)
    while True:
        label = EDGE_ORDER[state["edge"]] if state["edge"] < len(EDGE_ORDER) else "done"
        shown = view.copy()
        cv2.putText(shown, f"edge: {label} (n=next, ESC=run)", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.imshow("pick edges", shown)
        key = cv2.waitKey(20) & 0xFF
        if key == ord("n") and state["edge"] < len(EDGE_ORDER):
            state["edge"] += 1
        elif key == 27:
            break
    cv2.destroyAllWindows()
    return edges


def main():
    parser = argparse.ArgumentParser(description="Edge-based dashcam crop.")
    _ = parser.add_argument("input_video")
    _ = parser.add_argument("--edges", help="edges.json (top/right/bottom/left points)")
    _ = parser.add_argument("--pick", action="store_true", help="click points per edge (GUI)")
    _ = parser.add_argument("--out", default=None)
    _ = parser.add_argument("--perspective-out", default=None)
    _ = parser.add_argument("--corners-out", default=None)
    _ = parser.add_argument("--alpha", type=float, default=0.5, help="low-pass 0..1 (1=off)")
    _ = parser.add_argument("--template-size", type=int, default=TEMPLATE_SIZE)
    _ = parser.add_argument("--max-frames", type=int, default=0, help="0 = all frames")
    _ = parser.add_argument("--show", action="store_true", help="show windows (needs display)")
    args = parser.parse_args()

    stem = os.path.splitext(os.path.basename(args.input_video))[0]
    out_path = args.out or f"{stem}_edge_output.mp4"
    persp_path = args.perspective_out or f"{stem}_edge_perspective.mp4"
    corners_path = args.corners_out or f"{stem}_edge_corners.json"

    cap = cv2.VideoCapture(args.input_video)
    ok, first = cap.read()
    if not ok:
        raise SystemExit(f"could not read video: {args.input_video}")
    height, width = first.shape[:2]
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    if args.pick:
        edges = pick_edges(first)
        with open(f"{stem}_edges.json", "w", encoding="utf-8") as stream:
            json.dump({"edges": {k: [[x, y] for x, y in v] for k, v in edges.items()}}, stream, indent=2)
    elif args.edges:
        edges = load_edges(args.edges)
    else:
        raise SystemExit("provide --edges edges.json or --pick")
    for name in EDGE_ORDER:
        if len(edges[name]) < 2:
            raise SystemExit(f"edge '{name}' needs at least two points")

    size = args.template_size
    tracked = [
        [name, make_template(first, x, y, size), (float(x), float(y))]
        for name in EDGE_ORDER
        for (x, y) in edges[name]
    ]

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
    persp = cv2.VideoWriter(persp_path, fourcc, fps, (width, height))
    dst = np.float32([[0, 0], [width, 0], [width, height], [0, height]])

    records = []
    prev_corners = None
    frame = first
    frame_idx = 0
    while True:
        for entry in tracked:
            entry[2] = low_pass(track_point(frame, entry[1], entry[2], size), entry[2], args.alpha)
        current = {name: [] for name in EDGE_ORDER}
        for name, _template, pos in tracked:
            current[name].append(pos)
        try:
            corners = corners_from_edges(current)
            prev_corners = corners
        except ValueError:
            corners = prev_corners
        if corners is None:
            break

        records.append({"frame": frame_idx, "corners": [[round(cx, 2), round(cy, 2)] for cx, cy in corners]})
        matrix = cv2.getPerspectiveTransform(np.float32(corners), dst)
        persp.write(cv2.warpPerspective(frame, matrix, (width, height)))
        annotate(frame, current, corners, width, height)
        out.write(frame)

        if args.show:
            cv2.imshow("edge", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break

        frame_idx += 1
        if args.max_frames and frame_idx >= args.max_frames:
            break
        ok, frame = cap.read()
        if not ok:
            break

    cap.release()
    out.release()
    persp.release()
    if args.show:
        cv2.destroyAllWindows()
    with open(corners_path, "w", encoding="utf-8") as stream:
        json.dump({"video": args.input_video, "width": width, "height": height, "frames": records}, stream, indent=2)
    first_corners = records[0]["corners"] if records else None
    print(f"frames={frame_idx} out={out_path} perspective={persp_path} corners={corners_path}")
    print(f"first-frame corners (TL,TR,BR,BL)={first_corners}")


if __name__ == "__main__":
    main()
