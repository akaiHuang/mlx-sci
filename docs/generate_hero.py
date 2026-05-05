"""Generate the hero benchmark figure for mlx-sci README.

Runs offline using already-measured numbers (M1 Max, see each sub-package's
benchmark_results.md). No GPU required to render.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# (label, speedup, category)
# Each row is the *headline* speedup the sub-package's benchmark_results.md
# reports. Use the most representative single number per module so the chart
# stays honest and not cherry-picked.
DATA: list[tuple[str, float, str]] = [
    # Special functions
    ("bessel\nN=525×10k",         579.0,  "special"),
    ("wigner\nbatch=100k",       2000.0,  "special"),
    ("hyp2f1 (Metal)\nN=1M",        7.4,  "special"),
    ("airy\nN=1M",                  6.7,  "special"),
    ("gamma\nN=1M",                10.4,  "special"),
    ("expm\nn=1024 real",           2.0,  "special"),
    # Signal
    ("stft\n30s audio",            10.0,  "signal"),
    # Linear algebra / info
    ("fisher\nlarge MM",           39.0,  "info"),
    # Quantum
    ("qre eigh\nN=1000",            1.93, "quantum"),
    ("qre lanczos\nN=2000 vs NumPy", 650.0, "quantum"),
]

CATEGORY_COLOR = {
    "special": "#3b82f6",   # blue
    "signal":  "#a855f7",   # purple
    "info":    "#10b981",   # green
    "quantum": "#ef4444",   # red
}

CATEGORY_LABEL = {
    "special": "Special functions",
    "signal":  "Signal processing",
    "info":    "Linear algebra / Info",
    "quantum": "Quantum information",
}


def main(out: Path) -> None:
    labels   = [r[0] for r in DATA]
    speedups = [r[1] for r in DATA]
    cats     = [r[2] for r in DATA]
    colors   = [CATEGORY_COLOR[c] for c in cats]

    fig, ax = plt.subplots(figsize=(11.5, 5.6), dpi=160)

    x = np.arange(len(labels))
    bars = ax.bar(x, speedups, color=colors, edgecolor="#0b1220",
                  linewidth=0.5, zorder=3)

    ax.set_yscale("log")
    ax.set_ylim(0.8, 4500)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9.0)
    ax.set_ylabel("Speedup vs CPU baseline (log scale)", fontsize=11)
    ax.axhline(1.0, color="#475569", linestyle="--", linewidth=0.9,
               label="1× (parity)", zorder=2)

    # Bar value labels
    for bar, sp in zip(bars, speedups):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h * 1.08,
                f"{sp:g}×",
                ha="center", va="bottom", fontsize=9.5, fontweight="bold",
                color="#0b1220")

    # Title only — explanatory caption lives in the README so the figure
    # stays readable.
    fig.suptitle(
        "mlx-sci  —  GPU-accelerated scientific computing on Apple Silicon",
        fontsize=14, fontweight="bold", y=0.99)

    # Category legend — placed above the chart on a single horizontal row
    # so it doesn't overlap the tallest bars.
    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, color=CATEGORY_COLOR[k]) for k in
        ["special", "signal", "info", "quantum"]
    ]
    legend_labels = [CATEGORY_LABEL[k] for k in
                     ["special", "signal", "info", "quantum"]]
    ax.legend(legend_handles, legend_labels,
              loc="lower center", bbox_to_anchor=(0.5, 1.04),
              ncol=4, fontsize=9.5, frameon=False)

    ax.grid(axis="y", which="both", alpha=0.25, zorder=1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    print(f"saved → {out}")


if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    main(here / "hero_benchmarks.png")
