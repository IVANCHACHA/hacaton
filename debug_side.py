#!/usr/bin/env python3
"""Отладка боковой камеры: тянем кадры с сервера и смотрим силуэт/высоту."""

from __future__ import annotations

import argparse
import ssl
import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np

from classify import quad_to_pixels
from side_view import SideView, draw_side_overlay

CTX = ssl.create_default_context()
CTX.check_hostname = False
CTX.verify_mode = ssl.CERT_NONE


def get(url: str) -> bytes:
    with urllib.request.urlopen(url, context=CTX, timeout=4) as r:
        return r.read()


def get_json(url: str) -> dict:
    import json

    return json.loads(get(url).decode("utf-8"))


def grab(base: str) -> np.ndarray | None:
    try:
        data = get(f"{base}/api/side.jpg")
    except Exception:
        return None
    arr = np.frombuffer(data, np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--base", default="https://127.0.0.1:5050")
    p.add_argument("--frames", type=int, default=90)
    p.add_argument("--out", default="debug_side")
    args = p.parse_args()

    st = get_json(f"{args.base}/api/status")["side"]
    quad = st["roi_quad"]
    mm = float(st["mm_per_px"])
    print(f"ROI={quad} mm/px={mm} rot={st['rot']} fps={st['fps']}")

    out = Path(args.out)
    out.mkdir(exist_ok=True)
    sv = SideView()
    hits = 0
    last_frame = None
    for i in range(args.frames):
        frame = grab(args.base)
        if frame is None:
            time.sleep(0.1)
            continue
        last_frame = frame
        m = sv.measure(frame, mm, 600.0, roi_quad=quad)
        if m is not None:
            hits += 1
            if hits <= 6:
                ov = draw_side_overlay(frame, m, quad, mm)
                cv2.imwrite(str(out / f"hit_{hits:02d}.jpg"), ov)
            print(
                f"{i:3d} H={m.height_mm:6.1f}мм L={m.length_mm:6.1f}мм "
                f"K_side={m.k_side:.3f} rf={m.rect_fill:.2f} "
                f"bbox={m.bbox_roi} area={int(m.area_px)}"
            )
        else:
            if i % 15 == 0:
                print(f"{i:3d} — {sv.drop}")
        time.sleep(0.05)

    print(f"\nизмерений: {hits}/{args.frames}")
    if last_frame is not None:
        h, w = last_frame.shape[:2]
        qpx = quad_to_pixels(h, w, quad)
        mask = sv._silhouette(
            cv2.warpPerspective(
                last_frame,
                cv2.getPerspectiveTransform(
                    qpx.astype(np.float32),
                    np.array(
                        [[0, 0], [639, 0], [639, 399], [0, 399]], dtype=np.float32
                    ),
                ),
                (640, 400),
            )
        )
        cv2.imwrite(str(out / "last_mask.jpg"), mask)
        vis = last_frame.copy()
        cv2.polylines(vis, [qpx.astype(np.int32)], True, (0, 220, 255), 2)
        cv2.imwrite(str(out / "last_roi.jpg"), vis)
        print(f"сохранил {out}/last_roi.jpg и {out}/last_mask.jpg")


if __name__ == "__main__":
    main()
