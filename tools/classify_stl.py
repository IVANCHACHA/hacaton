#!/usr/bin/env python3
"""Runs the reference (ground-truth) classifier over the organizers' STL
set and prints a table + validates the one publicly-confirmed data point
from the Q&A (Мешок.stl, K=0.817 -> category D, Q&A #113).

No ROS2 required:
    python3 tools/classify_stl.py
"""
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "src", "sorter_pak"))

from sorter_pak.core import stl_dataset as sd    # noqa: E402


def main():
    catalog = sd.build_catalog()
    print(f"{'item':28s} {'dims_mm (desc)':28s} {'category':8s} K per plane")
    print("-" * 100)
    for it in catalog:
        dims = " x ".join(f"{d:.1f}" for d in it.dims_mm)
        ks = ", ".join(f"{p}={k:.3f}" for p, k in it.k_values.items()) or "(gabarit fail - not evaluated)"
        print(f"{it.name:28s} {dims:28s} {it.category_true:8s} {ks}")

    meshok = next(it for it in catalog if it.name == "Мешок")
    k_max = max(meshok.k_values.values())
    print()
    print(f"Sanity check (Q&A #113): Мешок.stl K_max={k_max:.3f} "
          f"(organizers confirmed 0.817 -> category D), "
          f"our category={meshok.category_true} "
          f"({'OK' if meshok.category_true == 'D' else 'MISMATCH'})")


if __name__ == "__main__":
    main()
