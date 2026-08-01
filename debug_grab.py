#!/usr/bin/env python3
"""Снимки живого потока для отладки маски/контура."""

from __future__ import annotations

import argparse
import json
import ssl
import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np

import classify

CTX = ssl._create_unverified_context()
ROOT = Path(__file__).resolve().parent


def grab(url: str = "https://127.0.0.1:5050/api/raw.jpg") -> np.ndarray | None:
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req, context=CTX, timeout=6) as r:
        data = r.read()
    if not data:
        return None
    return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)


def panel(frame: np.ndarray, roi_quad, mm_per_px: float, min_area: float, out: Path) -> dict:
    h, w = frame.shape[:2]
    quad = classify.quad_to_pixels(h, w, roi_quad)
    roi, inv, rw, rh = classify.warp_roi(frame, quad)
    app = classify._appearance_mask(roi)
    gray = cv2.cvtColor(cv2.GaussianBlur(roi, (3, 3), 0), cv2.COLOR_BGR2GRAY)
    mag = classify._sobel_magnitude(gray)
    mag_vis = cv2.applyColorMap(
        np.clip(mag / max(1.0, float(np.percentile(mag, 99))) * 255, 0, 255).astype(np.uint8),
        cv2.COLORMAP_MAGMA,
    )

    tracker = classify.MotionTracker(use_sam=True)
    dets: list = []
    for _ in range(3):
        dets = tracker.detect(frame, mm_per_px, min_area, roi_quad=roi_quad)
    vis = roi.copy()
    info: dict = {"n": len(dets)}
    cnts, _ = cv2.findContours(app, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(vis, cnts, -1, (60, 200, 255), 1)
    if dets:
        d = dets[0]
        cnt_roi = cv2.perspectiveTransform(
            d.contour.astype(np.float32), np.linalg.inv(np.asarray(inv, np.float32))
        ).astype(np.int32)
        cv2.drawContours(vis, [cnt_roi], -1, (60, 255, 120), 2)
        info.update(
            size_mm=[round(x, 1) for x in d.size_mm],
            zone=classify.ZONE[d.category],
            ratio=round(d.circle_ratio, 3),
            rect_fill=round(d.rect_fill, 3),
            area=round(d.area_px, 1),
        )

    def to3(m):
        return cv2.cvtColor(m, cv2.COLOR_GRAY2BGR) if m.ndim == 2 else m

    top = np.hstack([roi, to3(app)])
    bot = np.hstack([mag_vis, vis])
    cv2.imwrite(str(out), np.vstack([top, bot]))
    return info


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--gap", type=float, default=1.2)
    ap.add_argument("--prefix", default="/tmp/dbg")
    args = ap.parse_args()

    cal = json.loads((ROOT / "calibration.json").read_text())
    mm = float(cal.get("mm_per_px", 0.58))
    quad = cal.get("roi_quad")
    for i in range(args.n):
        frame = grab()
        if frame is None:
            print(i, "no frame", flush=True)
            continue
        out = Path(f"{args.prefix}_{i}.jpg")
        info = panel(frame, quad, mm, 1400.0, out)
        print(i, out, info, flush=True)
        time.sleep(args.gap)


if __name__ == "__main__":
    main()
