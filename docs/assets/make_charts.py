"""Render the charts in docs/results.md (light and dark SVG variants).

    uv run --with matplotlib python docs/assets/make_charts.py

Numbers are copied from the run logs (sft.jsonl, grpo.jsonl) and eval/report.json.
"""

from pathlib import Path

import matplotlib

matplotlib.use("svg")
import matplotlib.pyplot as plt

OUT = Path(__file__).parent
THEMES = {
    "light": {
        "text": "#0b0b0b",
        "muted": "#52514e",
        "grid": "#e4e3df",
        "sft": "#2a78d6",
        "grpo": "#eb6834",
        "band": "#2a78d6",
    },
    "dark": {
        "text": "#f0efec",
        "muted": "#c3c2b7",
        "grid": "#3a3936",
        "sft": "#3987e5",
        "grpo": "#d95926",
        "band": "#3987e5",
    },
}

SFT_STEPS = [0, 250, 500, 750, 1000, 1250, 1500, 1750, 2000]
SFT_HELDOUT = [0, 0.0, 0.5, 1.5, 7.0, 16.5, 23.0, 32.0, 34.0]
GRPO_STEPS = [0, 25, 50, 75, 100, 125, 150]
GRPO_HELDOUT = [34.0, 40.0, 39.0, 38.0, 44.0, 42.5, 43.0]
METRICS = ["Greedy pass@1", "Sampled pass@1\n(T = 0.8)", "Sampled pass@8"]
SFT_FINAL = [32.0, 9.3, 48.6]
GRPO_FINAL = [46.6, 19.0, 63.8]


def style(ax, t, ylabel):
    ax.set_facecolor("none")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(t["grid"])
    ax.tick_params(colors=t["muted"], labelsize=9, length=0, pad=6)
    ax.yaxis.grid(True, color=t["grid"], linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_ylabel(ylabel, color=t["muted"], fontsize=9)


def save(fig, name, mode):
    fig.patch.set_alpha(0)
    fig.savefig(OUT / f"{name}-{mode}.svg", bbox_inches="tight", metadata={"Date": None})
    plt.close(fig)


def sft_curve(mode, t):
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    style(ax, t, "Held-out exact match (%)")
    ax.axhspan(15, 70, color=t["band"], alpha=0.08, linewidth=0)
    ax.text(60, 66, "GRPO go window (15-70%)", color=t["muted"], fontsize=8, va="top")
    ax.plot(SFT_STEPS, SFT_HELDOUT, color=t["sft"], linewidth=2, marker="o", markersize=4)
    ax.annotate(
        "34%",
        (2000, 34),
        xytext=(6, 0),
        textcoords="offset points",
        color=t["text"],
        fontsize=9,
        va="center",
    )
    ax.set_xlabel("SFT step", color=t["muted"], fontsize=9)
    ax.set_xlim(0, 2150)
    ax.set_ylim(0, 75)
    save(fig, "sft-heldout", mode)


def grpo_curve(mode, t):
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    style(ax, t, "Held-out exact match (%)")
    ax.axhline(34, color=t["sft"], linewidth=1.2, linestyle=(0, (4, 3)))
    ax.text(148, 33.4, "SFT start: 34%", color=t["muted"], fontsize=8, ha="right", va="top")
    ax.plot(GRPO_STEPS, GRPO_HELDOUT, color=t["grpo"], linewidth=2, marker="o", markersize=4)
    ax.annotate(
        "43%", (150, 43), xytext=(6, 4), textcoords="offset points", color=t["text"], fontsize=9
    )
    ax.set_xlabel("GRPO step", color=t["muted"], fontsize=9)
    ax.set_xlim(0, 160)
    ax.set_ylim(25, 50)
    save(fig, "grpo-heldout", mode)


def final_bars(mode, t):
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    style(ax, t, "Exact match (%)")
    width, gap = 0.36, 0.02
    xs = range(len(METRICS))
    for offset, values, color, label in (
        (-(width + gap) / 2, SFT_FINAL, t["sft"], "SFT"),
        ((width + gap) / 2, GRPO_FINAL, t["grpo"], "GRPO"),
    ):
        bars = ax.bar([x + offset for x in xs], values, width, color=color, label=label)
        for bar, value in zip(bars, values, strict=True):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 1.2,
                f"{value:.1f}",
                ha="center",
                color=t["text"],
                fontsize=8.5,
            )
    ax.set_xticks(list(xs), METRICS)
    ax.set_ylim(0, 75)
    legend = ax.legend(frameon=False, fontsize=9, loc="upper left")
    for text in legend.get_texts():
        text.set_color(t["text"])
    save(fig, "final-eval", mode)


for mode, theme in THEMES.items():
    plt.rcParams["svg.hashsalt"] = "nanogpt"
    sft_curve(mode, theme)
    grpo_curve(mode, theme)
    final_bars(mode, theme)
