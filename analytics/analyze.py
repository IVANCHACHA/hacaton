"""Calculate routing and timing metrics from a saved kinematic simulation run."""

import argparse
import json
import math
import sqlite3
from pathlib import Path


def analyze(items, duration_seconds):
    with sqlite3.connect(":memory:") as db:
        db.row_factory = sqlite3.Row
        db.execute("""
            CREATE TABLE items (
                category_true TEXT,
                routed_zone TEXT,
                sensor_t REAL,
                decision_ready_t REAL,
                late_classification INTEGER
            )
        """)
        db.executemany("""
            INSERT INTO items VALUES (
                :category_true, :routed_zone, :sensor_t,
                :decision_ready_t, :late_classification
            )
        """, items)
        counts = dict(db.execute("""
            SELECT COUNT(*) AS generated,
                   COUNT(sensor_t) AS captured,
                   COUNT(CASE WHEN decision_ready_t <= :duration THEN 1 END) AS ready,
                   COUNT(routed_zone) AS routed,
                   COUNT(CASE WHEN routed_zone IS NULL THEN 1 END) AS in_transit,
                   COUNT(CASE WHEN routed_zone IS NOT NULL AND category_true IS NOT NULL
                              THEN 1 END) AS routed_with_reference,
                   COUNT(CASE WHEN category_true = routed_zone THEN 1 END) AS correct_routes,
                   COUNT(CASE WHEN category_true IN ('C', 'D') AND routed_zone IS NOT NULL
                              THEN 1 END) AS completed_requiring_diversion,
                   COUNT(CASE WHEN category_true IN ('C', 'D') AND routed_zone = 'B'
                              THEN 1 END) AS unsafe_to_b,
                   COUNT(CASE WHEN late_classification THEN 1 END) AS late_classifications
            FROM items
        """, {"duration": duration_seconds}).fetchone())
        routes = [dict(row) for row in db.execute("""
            SELECT category_true, routed_zone, COUNT(*) AS items
            FROM items
            WHERE routed_zone IS NOT NULL
            GROUP BY category_true, routed_zone
            ORDER BY category_true, routed_zone
        """)]
        sensor_latency_ms = dict(db.execute("""
            WITH ranked AS (
                SELECT (decision_ready_t - sensor_t) * 1000.0 AS latency_ms,
                       CUME_DIST() OVER (ORDER BY decision_ready_t - sensor_t) AS quantile
                FROM items
                WHERE sensor_t IS NOT NULL AND decision_ready_t <= :duration
            )
            SELECT COUNT(*) AS observations,
                   ROUND(MIN(CASE WHEN quantile >= 0.50 THEN latency_ms END), 2) AS p50,
                   ROUND(MIN(CASE WHEN quantile >= 0.95 THEN latency_ms END), 2) AS p95
            FROM ranked
        """, {"duration": duration_seconds}).fetchone())

    return {
        "duration_seconds": duration_seconds,
        "counts": counts,
        "generated_per_hour": round(counts["generated"] / duration_seconds * 3600, 1),
        "routed_per_hour": round(counts["routed"] / duration_seconds * 3600, 1),
        "routes": routes,
        "sensor_latency_ms": sensor_latency_ms,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path, help="path to sim_metrics_run.json")
    parser.add_argument("--duration-seconds", type=float, required=True)
    args = parser.parse_args()
    if not math.isfinite(args.duration_seconds) or args.duration_seconds <= 0:
        parser.error("--duration-seconds must be a finite positive number")

    with args.run.open(encoding="utf-8") as source:
        run = json.load(source)
    print(json.dumps(analyze(run["items"], args.duration_seconds), indent=2))


if __name__ == "__main__":
    main()
