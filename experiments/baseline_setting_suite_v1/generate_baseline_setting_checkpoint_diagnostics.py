"""Generate averaged checkpoint diagnostics from per-seed diagnostics."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


HERE = Path(__file__).resolve().parent
BY_SEED_PATH = HERE / "baseline_setting_checkpoint_diagnostics_by_seed.csv"
SUMMARY_PATH = HERE / "baseline_setting_checkpoint_diagnostics.csv"

GROUP_COLUMNS = [
    "round",
    "strategy",
    "display_name",
    "role",
    "policy",
    "transition_variant",
    "gate_mode",
]


def generate_summary(by_seed_path: Path = BY_SEED_PATH, summary_path: Path = SUMMARY_PATH) -> None:
    by_seed = pd.read_csv(by_seed_path)
    summary = (
        by_seed.groupby(GROUP_COLUMNS, as_index=False)
        .agg(
            n=("seed_index", "count"),
            mean_transition_l1_error=("transition_l1_error", "mean"),
            std_transition_l1_error=("transition_l1_error", "std"),
            mean_off_support_leakage=("off_support_leakage", "mean"),
            std_off_support_leakage=("off_support_leakage", "std"),
            mean_reward_pct_oracle=("reward_pct_oracle", "mean"),
            mean_top1_agreement=("top1_agreement", "mean"),
            mean_top2_agreement=("top2_agreement", "mean"),
        )
        .sort_values(["round", "role", "strategy"])
    )
    summary.to_csv(summary_path, index=False)


if __name__ == "__main__":
    generate_summary()
