#!/usr/bin/env python3
"""Оценка пропусков: объект в ROI есть (по маске), а детекции нет."""

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
BASE = "https://127.0.0.1:5050"
ROOT = Path(__file__).resolve().parent


def get(path: str) -> bytes:
    with urllib.request.urlopen(f"{BASE}{path}", context=CTX, timeout=6) as r:
        return r.read()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=120)
    ap.add_argument("--gap", type=float, default=0.25)
    ap.add_argument("--min-area", type=float, default=1200.0)
    ap.add_argument("--prefix", default="/tmp/miss")
    args = ap.parse_args()

    quad = json.loads((ROOT / "calibration.json").read_text())["roi_quad"]
    present = miss = extra = 0
    saved = 0
    for i in range(args.iters):
        try:
            frame = cv2.imdecode(np.frombuffer(get("/api/raw.jpg"), np.uint8), cv2.IMREAD_COLOR)
            st = json.loads(get("/api/status"))
        except Exception:
            time.sleep(args.gap)
            continue
        if frame is None:
            time.sleep(args.gap)
            continue
        h, w = frame.shape[:2]
        roi, _inv, rw, rh = classify.warp_roi(frame, classify.quad_to_pixels(h, w, quad))
        app = classify._appearance_mask(roi)
        dm = classify._belt_distance_map(roi)
        cnts, _ = cv2.findContours(app, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best = None
        for c in cnts:
            if cv2.contourArea(c) < args.min_area:
                continue
            mean_d, frac_d = classify._blob_contrast(roi, c, dm)
            if mean_d < 30.0 or frac_d < 0.6:
                continue
            if best is None or cv2.contourArea(c) > cv2.contourArea(best):
                best = c
        has_obj = best is not None
        det = st.get("live_det")
        present += has_obj
        if has_obj and not det:
            miss += 1
            print(f"  drop: {st.get('live_drop')}", flush=True)
            if saved < 6:
                out = f"{args.prefix}_{saved}.jpg"
                x, y, bw, bh = cv2.boundingRect(best)
                vis = roi.copy()
                cv2.drawContours(vis, [best], -1, (0, 0, 255), 2)
                cv2.imwrite(out, np.hstack([roi, cv2.cvtColor(app, cv2.COLOR_GRAY2BGR), vis]))
                print(f"MISS {out} bbox={bw}x{bh} area={int(cv2.contourArea(best))}", flush=True)
                saved += 1
        if det and not has_obj:
            extra += 1
            if saved < 8:
                out = f"{args.prefix}_extra{saved}.jpg"
                cv2.imwrite(out, np.hstack([roi, cv2.cvtColor(app, cv2.COLOR_GRAY2BGR)]))
                a, b = det["size_mm"]
                print(f"EXTRA {out} {a}x{b}mm bbox={det['bbox']}", flush=True)
                saved += 1
        time.sleep(args.gap)
    print(f"=== present={present} miss={miss} extra={extra} of {args.iters} ===", flush=True)


if __name__ == "__main__":
    main()
