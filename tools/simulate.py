#!/usr/bin/env python3
"""Standalone simulation runner - no ROS2 required.

Builds the STL-derived item catalog (if missing), runs the conveyor +
2-shiber kinematic simulation, writes metrics/events JSON, and renders a
top-down animation (GIF) of the belt, the 2 sensors, the 2 shibers and
the B/C/D routing outcomes.

Usage:
    python3 tools/simulate.py                  # full run: catalog + metrics + animation
    python3 tools/simulate.py --no-animation    # skip the (slower) GIF render
    python3 tools/simulate.py --duration 300    # metrics-run length, seconds
"""
from __future__ import annotations

import argparse
import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "src", "sorter_pak"))

from sorter_pak.core import stl_dataset as sd            # noqa: E402
from sorter_pak.core import conveyor_sim as csim          # noqa: E402

OUT_DIR = os.path.join(REPO_ROOT, "sim_output")
CATALOG_PATH = os.path.join(OUT_DIR, "item_catalog.json")


def ensure_catalog():
    if not os.path.exists(CATALOG_PATH):
        print("building item catalog from тз/Stl ...")
        cat = sd.build_catalog()
        sd.save_catalog(cat, CATALOG_PATH)
    return sd.load_catalog(CATALOG_PATH)


def run_metrics(catalog, duration_s: float, seed: int):
    cfg = csim.SimConfig(duration_s=duration_s, seed=seed)
    result = csim.run_simulation(catalog, cfg)
    out_path = os.path.join(OUT_DIR, "sim_metrics_run.json")
    result.save(out_path, include_frames=False)
    print(f"metrics run: {result.metrics['n_items']} items, "
          f"{result.metrics['throughput_items_per_hour']} items/h, "
          f"accuracy={result.metrics['accuracy_vs_reference']}, "
          f"unsafe_to_B={result.metrics['unsafe_items_routed_to_B']}")
    print(f"wrote {out_path}")
    return result


def render_animation(catalog, anim_duration_s: float, seed: int, fps: int = 12):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import animation
    from matplotlib.patches import Rectangle

    cfg = csim.SimConfig(duration_s=anim_duration_s, seed=seed, frame_dt=1.0 / fps)
    result = csim.run_simulation(catalog, cfg)
    frames = result.frames

    cat_color = {"B": "#2e7d32", "C": "#e65100", "D": "#6a1b9a", None: "#9e9e9e"}
    zone_positions = {"C": csim.X_SHIBER1, "D": csim.X_SHIBER2}

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.set_xlim(-0.3, csim.X_END + 1.0)
    ax.set_ylim(-1.9, 1.6)
    ax.set_aspect("equal")
    ax.set_yticks([])
    ax.set_xlabel("belt position, m")
    ax.set_title("PAK: конвейер A -> 2 сенсора -> 2 шибера -> B / roll-cage C / roll-cage D")

    ax.add_patch(Rectangle((0, -0.35), csim.X_END, 0.7, facecolor="#222222", zorder=0))
    ax.add_patch(Rectangle((0, -0.35), csim.X_FIXED_END, 0.7, facecolor="none",
                            edgecolor="#ffffff", linestyle="--", linewidth=1, zorder=1))
    ax.text(csim.X_FIXED_END / 2, 0.55, "конвейер A (фиксирован)", ha="center", fontsize=8, color="#555")

    ax.axvline(csim.X_SENSOR, color="#1565c0", linestyle=":", linewidth=1.5)
    ax.text(csim.X_SENSOR, 1.0, "сенсоры\n(top+profiler)", ha="center", fontsize=8, color="#1565c0")

    ax.text(csim.X_END + 0.15, 0, "-> B\n(осн.\nсортировщик)", fontsize=8, va="center", color="#2e7d32")

    shiber_patches = {}
    for name, x, color in [("shiber1", csim.X_SHIBER1, "#e65100"), ("shiber2", csim.X_SHIBER2, "#6a1b9a")]:
        p = Rectangle((x - 0.05, -0.45), 0.1, 0.9, facecolor=color, alpha=0.25, zorder=2)
        ax.add_patch(p)
        shiber_patches[name] = p
        ax.plot([x], [-1.1], marker="v", color=color)
    ax.text(csim.X_SHIBER1, -1.25, "шибер 1\n(C, негабарит)", ha="center", fontsize=7, color="#e65100")
    ax.text(csim.X_SHIBER2, -1.55, "шибер 2\n(D, круглые)", ha="center", fontsize=7, color="#6a1b9a")

    item_patches = {}
    counters_text = ax.text(csim.X_END + 0.15, 1.2, "", fontsize=8, va="top")

    def init():
        return []

    def update(frame_idx):
        frame = frames[frame_idx]
        seen_ids = set()
        for it in frame["items"]:
            iid = it["id"]
            seen_ids.add(iid)
            color = cat_color.get(it["cat"], cat_color[None])
            if iid not in item_patches:
                rect = Rectangle((it["x"] - 0.09, -0.18), 0.18, 0.36, facecolor=color,
                                  edgecolor="white", linewidth=0.5, zorder=3)
                ax.add_patch(rect)
                item_patches[iid] = rect
            else:
                item_patches[iid].set_x(it["x"] - 0.09)
                item_patches[iid].set_facecolor(color)
        for iid in list(item_patches):
            if iid not in seen_ids:
                item_patches[iid].remove()
                del item_patches[iid]

        shiber_patches["shiber1"].set_alpha(0.85 if frame["shiber1"] == "ACTUATING" else 0.2)
        shiber_patches["shiber2"].set_alpha(0.85 if frame["shiber2"] == "ACTUATING" else 0.2)

        n_seen = sum(1 for e in result.events if e["t"] <= frame["t"] and e["event"] == "spawn")
        n_c = sum(1 for e in result.events if e["t"] <= frame["t"] and e["event"] == "shiber_actuate"
                  and e.get("zone") == "C")
        n_d = sum(1 for e in result.events if e["t"] <= frame["t"] and e["event"] == "shiber_actuate"
                  and e.get("zone") == "D")
        n_b = sum(1 for e in result.events if e["t"] <= frame["t"] and e["event"] == "handoff_B")
        counters_text.set_text(f"t={frame['t']:5.1f}s\nB:{n_b}  C:{n_c}  D:{n_d}")
        return list(item_patches.values())

    anim = animation.FuncAnimation(fig, update, frames=len(frames), init_func=init,
                                    interval=1000 / fps, blit=False)
    out_path = os.path.join(OUT_DIR, "animation.gif")
    anim.save(out_path, writer=animation.PillowWriter(fps=fps))
    plt.close(fig)
    print(f"wrote {out_path} ({len(frames)} frames)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--duration", type=float, default=300.0, help="metrics-run length, s")
    ap.add_argument("--anim-duration", type=float, default=25.0, help="animation length, s")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--no-animation", action="store_true")
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    catalog = ensure_catalog()
    run_metrics(catalog, args.duration, args.seed)
    if not args.no_animation:
        render_animation(catalog, args.anim_duration, args.seed + 1)


if __name__ == "__main__":
    main()
