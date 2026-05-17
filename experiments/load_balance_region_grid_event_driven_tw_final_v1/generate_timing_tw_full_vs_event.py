"""Generate the full-vs-event TW timing comparison table."""

from __future__ import annotations

import csv
from pathlib import Path


HERE = Path(__file__).resolve().parent
FULL_SUMMARY_PATH = (
    HERE / "timing_tw_full_s60_s180_one_time" / "event_driven_tw_region_grid_summary.csv"
)
EVENT_SUMMARY_PATH = (
    HERE / "timing_tw_event_s60_s180_one_time" / "event_driven_tw_region_grid_summary.csv"
)
COMPARISON_PATH = HERE / "timing_tw_full_vs_event_s60_s180_one_time.csv"

OUTPUT_COLUMNS = [
    "S",
    "grid_states",
    "queue_states",
    "op_states",
    "policy_label",
    "full_walltime_s",
    "event_walltime_s",
    "walltime_saving_pct",
    "speedup_x",
    "full_states_computed",
    "event_states_computed",
    "event_computation_saving_pct",
    "full_avg_reward",
    "event_avg_reward",
    "full_cum_reward",
    "event_cum_reward",
]

KEY_COLUMNS = ["S", "grid_states", "queue_states", "op_states", "policy_label"]


def read_summary(path: Path) -> dict[tuple[str, ...], dict[str, str]]:
    with path.open(newline="") as file:
        rows = csv.DictReader(file)
        return {tuple(row[column] for column in KEY_COLUMNS): row for row in rows}


def generate_comparison(
    full_summary_path: Path = FULL_SUMMARY_PATH,
    event_summary_path: Path = EVENT_SUMMARY_PATH,
    comparison_path: Path = COMPARISON_PATH,
) -> None:
    full_rows = read_summary(full_summary_path)
    event_rows = read_summary(event_summary_path)

    missing_event = sorted(set(full_rows) - set(event_rows))
    missing_full = sorted(set(event_rows) - set(full_rows))
    if missing_event or missing_full:
        raise ValueError(
            "Full and event summaries do not have matching keys: "
            f"missing_event={missing_event}, missing_full={missing_full}"
        )

    comparison_rows = []
    for key in sorted(full_rows, key=lambda item: (int(item[0]), item[4])):
        full = full_rows[key]
        event = event_rows[key]
        full_walltime = float(full["mean_walltime_seconds"])
        event_walltime = float(event["mean_walltime_seconds"])
        full_states = float(full["mean_states_computed"])
        event_states = float(event["mean_states_computed"])

        comparison_rows.append(
            {
                "S": full["S"],
                "grid_states": full["grid_states"],
                "queue_states": full["queue_states"],
                "op_states": full["op_states"],
                "policy_label": full["policy_label"],
                "full_walltime_s": full["mean_walltime_seconds"],
                "event_walltime_s": event["mean_walltime_seconds"],
                "walltime_saving_pct": (full_walltime - event_walltime) / full_walltime * 100.0,
                "speedup_x": full_walltime / event_walltime,
                "full_states_computed": int(full_states),
                "event_states_computed": int(event_states),
                "event_computation_saving_pct": (full_states - event_states) / full_states * 100.0,
                "full_avg_reward": full["mean_avg_reward"],
                "event_avg_reward": event["mean_avg_reward"],
                "full_cum_reward": full["mean_cum_reward"],
                "event_cum_reward": event["mean_cum_reward"],
            }
        )

    with comparison_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(comparison_rows)


if __name__ == "__main__":
    generate_comparison()
