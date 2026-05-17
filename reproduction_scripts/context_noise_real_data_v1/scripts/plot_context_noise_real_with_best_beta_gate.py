"""Plot Sweep A real-data context noise with added best beta-gate variants."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-yifu")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "docs/research/rmab_vm_outputs/context_noise_real_data_v1"
SUMMARY_PATH = OUT_DIR / "context_noise_summary_with_best_beta_gate.csv"

BASELINE_LABELS = [
    "state_thompson",
    "tw_dense",
    "local_ucb_tw_dense",
    "global_ucb_tw_dense",
    "exp4_dense",
]

REFINED_LABELS = [
    "tm_tw_refined_dense",
    "tm_tw_refined_gated_offline",
    "tm_tw_refined_gated_offline_low_rank",
    "tm_tw_refined_support_offline",
    "tm_tw_refined_gated_offline_best_beta",
    "tm_tw_refined_gated_offline_low_rank_best_beta",
    "tm_tw_refined_support_gated_offline_low_rank_best_beta",
]

DISPLAY = {
    "state_thompson": "State Thompson",
    "tw_dense": "TW",
    "local_ucb_tw_dense": "Local UCB+TW",
    "global_ucb_tw_dense": "Global UCB+TW",
    "exp4_dense": "EXP4",
    "tm_tw_dense": "TM--TW",
    "tm_tw_refined_dense": "Adaptive TM--TW",
    "tm_tw_refined_gated_offline": "Adaptive TM--TW + gated prior",
    "tm_tw_refined_gated_offline_low_rank": "Adaptive TM--TW + gated prior + low-rank",
    "tm_tw_refined_support_offline": "Adaptive TM--TW + support/offline prior",
    "tm_tw_refined_gated_offline_best_beta": "Adaptive TM--TW + best beta gated prior",
    "tm_tw_refined_gated_offline_low_rank_best_beta": (
        "Adaptive TM--TW + best beta gated prior + low-rank"
    ),
    "tm_tw_refined_support_gated_offline_low_rank_best_beta": (
        "Adaptive TM--TW + best beta support-gated prior + low-rank"
    ),
}

METHODS = [
    ("best_baseline", "Best paper\nbaseline", "#7a8794", "--", "s"),
    ("tm_tw_dense", "Original\nTM-TW", "#2f5f8f", "--", "^"),
    ("tm_tw_refined_dense", "Adaptive\ntrust", "#2c7fb8", "-", "o"),
    ("tm_tw_refined_gated_offline", "Adaptive\n+gated", "#d9812e", "-", "o"),
    ("tm_tw_refined_gated_offline_low_rank", "Adaptive\n+gated+LR", "#c2410c", "-", "o"),
    ("tm_tw_refined_support_offline", "Adaptive\n+support", "#1f9a8a", "-", "o"),
    ("tm_tw_refined_gated_offline_best_beta", "Best beta\n+gated", "#9467bd", "-.", "D"),
    (
        "tm_tw_refined_gated_offline_low_rank_best_beta",
        "Best beta\n+gated+LR",
        "#6b21a8",
        "-.",
        "D",
    ),
    (
        "tm_tw_refined_support_gated_offline_low_rank_best_beta",
        "Best beta\n+support+gated+LR",
        "#0f766e",
        "-.",
        "D",
    ),
]


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.bbox": "tight",
        }
    )


def cn_val(df: pd.DataFrame, state: int, noise: float, label: str) -> float:
    mask = (
        (df["S"] == state)
        & (abs(df["context_noise_level"] - noise) < 1e-9)
        & (df["policy_label"] == label)
    )
    rows = df[mask]
    return float(rows["mean_reward_pct_oracle"].iloc[0]) if not rows.empty else float("nan")


def best_baseline(df: pd.DataFrame, state: int, noise: float) -> float:
    return float(np.nanmax([cn_val(df, state, noise, label) for label in BASELINE_LABELS]))


def best_baseline_with_label(df: pd.DataFrame, state: int, noise: float) -> tuple[str, float]:
    values = [(label, cn_val(df, state, noise, label)) for label in BASELINE_LABELS]
    return max(values, key=lambda item: item[1] if not np.isnan(item[1]) else -999.0)


def best_refined_with_label(df: pd.DataFrame, state: int, noise: float) -> tuple[str, float]:
    values = [(label, cn_val(df, state, noise, label)) for label in REFINED_LABELS]
    return max(values, key=lambda item: item[1] if not np.isnan(item[1]) else -999.0)


def plot_context_noise(df: pd.DataFrame) -> None:
    states = [8, 20, 50, 100]
    noises = [0.0, 0.1, 0.2, 0.3]

    all_values: list[float] = []
    for state in states:
        for label, *_ in METHODS:
            for noise in noises:
                value = best_baseline(df, state, noise) if label == "best_baseline" else cn_val(df, state, noise, label)
                if not np.isnan(value):
                    all_values.append(value)

    ymin = max(0.0, min(all_values) - 3.0)
    ymax = min(106.0, max(all_values) + 3.0)

    plt.rcParams.update({"font.size": 12})
    fig, axes = plt.subplots(2, 2, figsize=(11.7, 8.2), sharex=True, sharey=True)
    for ax, state in zip(axes.flat, states):
        for label, short, color, linestyle, marker in METHODS:
            values = [
                best_baseline(df, state, noise) if label == "best_baseline" else cn_val(df, state, noise, label)
                for noise in noises
            ]
            if all(np.isnan(value) for value in values):
                continue
            linewidth = 1.8 if label == "best_baseline" else 2.1
            ax.plot(
                noises,
                values,
                marker=marker,
                linewidth=linewidth,
                linestyle=linestyle,
                color=color,
                label=short,
            )
        ax.set_title(f"S={state}", loc="left", fontweight="bold", fontsize=13, pad=7)
        ax.grid(True, axis="both", color="#d7d7d7", linewidth=0.7, alpha=0.75)
        ax.set_xticks(noises)
        ax.set_ylim(ymin, ymax)

    axes[0, 0].set_ylabel("Reward (% oracle)", fontsize=12)
    axes[1, 0].set_ylabel("Reward (% oracle)", fontsize=12)
    axes[1, 0].set_xlabel("Contextual state-noise probability", fontsize=12)
    axes[1, 1].set_xlabel("Contextual state-noise probability", fontsize=12)

    legend_handles = [
        Line2D(
            [0],
            [0],
            color=color,
            linewidth=1.8 if label == "best_baseline" else 2.1,
            linestyle=linestyle,
            marker=marker,
            markersize=5,
        )
        for label, _, color, linestyle, marker in METHODS
    ]
    legend_labels = [short for _, short, *_ in METHODS]
    fig.legend(
        legend_handles,
        legend_labels,
        loc="lower center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.52, 0.0),
        handlelength=2.4,
        fontsize=10,
    )
    fig.tight_layout(rect=[0, 0.14, 1, 1.0])
    fig.savefig(OUT_DIR / "fig_context_noise_real_with_best_beta_gate.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT_DIR / "fig_context_noise_real_with_best_beta_gate.pdf", bbox_inches="tight")
    plt.close(fig)


def write_table(df: pd.DataFrame) -> None:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Sweep A (real VM data): contextual-noise robustness with best beta-gate variants. "
        r"For each $(|S|, \text{noise})$ cell the table reports the best paper-style baseline "
        r"(with its name), original TM--TW, and the best refined variant, including the added "
        r"best beta-gate policies where available.}",
        r"\label{tab:sweep_a_real_best_beta_gate}",
        r"\resizebox{\linewidth}{!}{%",
        r"\begin{tabular}{cclccclc}",
        r"\toprule",
        r"$|S|$ & Noise & Best baseline name & Best base & TM--TW "
        r"& Best refined & Best refined variant & Margin vs baseline \\",
        r"\midrule",
    ]
    for state in [8, 20, 50, 100]:
        for noise in [0.0, 0.1, 0.2, 0.3]:
            baseline_label, baseline_value = best_baseline_with_label(df, state, noise)
            tmtw_value = cn_val(df, state, noise, "tm_tw_dense")
            refined_label, refined_value = best_refined_with_label(df, state, noise)
            lines.append(
                f"{state} & {noise:.1f} & {DISPLAY.get(baseline_label, baseline_label)} & "
                f"{baseline_value:.2f} & {tmtw_value:.2f} & {refined_value:.2f} & "
                rf"\textbf{{{DISPLAY.get(refined_label, refined_label)}}} & "
                f"{refined_value - baseline_value:+.2f} \\\\"
            )
        if state != 100:
            lines.append(r"\addlinespace")
    lines += [r"\bottomrule", r"\end{tabular}%", r"}", r"\end{table}", ""]
    (OUT_DIR / "tab_context_noise_real_with_best_beta_gate.tex").write_text("\n".join(lines))


def main() -> None:
    setup_style()
    df = pd.read_csv(SUMMARY_PATH)
    plot_context_noise(df)
    write_table(df)
    print("Saved fig_context_noise_real_with_best_beta_gate.pdf/png")
    print("Saved tab_context_noise_real_with_best_beta_gate.tex")


if __name__ == "__main__":
    main()
