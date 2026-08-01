#!/usr/bin/env python3
"""Слежение за живой детекцией: снимки кадров с обводкой + метрики."""

from __future__ import annotations

import argparse
import json
import ssl
import time
import urllib.request
from pathlib import Path

CTX = ssl._create_unverified_context()
BASE = "https://127.0.0.1:5050"


def get_json(path: str) -> dict:
    with urllib.request.urlopen(f"{BASE}{path}", context=CTX, timeout=4) as r:
        return json.load(r)


def get_bytes(path: str) -> bytes:
    with urllib.request.urlopen(f"{BASE}{path}", context=CTX, timeout=6) as r:
        return r.read()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=300.0)
    ap.add_argument("--shots", type=int, default=14)
    ap.add_argument("--prefix", default="/tmp/live")
    args = ap.parse_args()

    t0 = time.time()
    saved = 0
    seen_moments: set[int] = set()
    last_shot = 0.0
    frames = 0
    with_det = 0
    while time.time() - t0 < args.seconds:
        try:
            st = get_json("/api/status")
        except Exception:
            time.sleep(0.4)
            continue
        det = st.get("live_det")
        frames += 1
        if det:
            with_det += 1
        for m in st.get("moments") or []:
            if m["id"] in seen_moments:
                continue
            seen_moments.add(m["id"])
            a, b = m["size_mm"]
            print(f"MOMENT #{m['id']} {m['zone']} {a}x{b} r={m['circle_ratio']}", flush=True)
        if det and saved < args.shots and time.time() - last_shot > 1.4:
            out = Path(f"{args.prefix}_{saved}.jpg")
            try:
                out.write_bytes(get_bytes("/api/overlay.jpg"))
                a, b = det["size_mm"]
                print(
                    f"SHOT {out} {a}x{b}mm r={det['circle_ratio']} "
                    f"rf={det['rect_fill']} area={det['area_px']} bbox={det['bbox']}",
                    flush=True,
                )
                saved += 1
                last_shot = time.time()
            except Exception as exc:  # noqa: BLE001
                print("shot fail", exc, flush=True)
        time.sleep(0.3)
    rate = (with_det / frames * 100.0) if frames else 0.0
    print(f"=== det in {with_det}/{frames} polls ({rate:.0f}%) ===", flush=True)


if __name__ == "__main__":
    main()
