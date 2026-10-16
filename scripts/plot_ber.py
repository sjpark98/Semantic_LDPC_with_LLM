#!/usr/bin/env python3
"""Regenerate the final overall and conditional BER figures from CSV files."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
METHODS = {
    "bp50": ("BP50", "#4d4d4d", "o", "-"),
    "bp100": ("Fresh BP100", "#2f6f9f", "s", "--"),
    "proposed": ("Proposed", "#d62728", "D", "-"),
    "without_verification": ("Proposed w/o verification", "#ff7f0e", "^", "-."),
    "voting": ("Voting", "#77b5d9", "v", ":"),
}


def read_csv(path: Path) -> list[dict[str, float]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [
            {key: float(value) for key, value in row.items()}
            for row in csv.DictReader(handle)
        ]


def draw(source: Path, destination: Path, title: str) -> None:
    rows = read_csv(source)
    snr = np.asarray([row["snr_db"] for row in rows])
    figure, axis = plt.subplots(figsize=(7.2, 5.0), constrained_layout=True)
    for key, (label, color, marker, line_style) in METHODS.items():
        values = np.asarray([row[key] for row in rows], dtype=float)
        values[values <= 0] = np.nan
        axis.semilogy(
            snr,
            values,
            label=label,
            color=color,
            marker=marker,
            linestyle=line_style,
            linewidth=2.0,
            markersize=6,
        )
    axis.set_xlabel(r"$E_b/N_0$ (dB)")
    axis.set_ylabel("Payload BER")
    axis.set_title(title)
    axis.set_xticks(snr)
    axis.grid(True, which="both", color="#d9d9d9", linewidth=0.7, alpha=0.75)
    axis.set_axisbelow(True)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(frameon=True, framealpha=0.95, fontsize=9)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=300, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT / "results")
    parser.add_argument("--output", type=Path, default=ROOT / "figures")
    args = parser.parse_args()
    draw(
        args.results / "overall_ber.csv",
        args.output / "overall_ber.png",
        "Overall BER (10,000 held-out sentences per SNR)",
    )
    draw(
        args.results / "conditional_ber.csv",
        args.output / "conditional_ber.png",
        "Conditional BER (first 100 BP50 failures per SNR)",
    )
    print(args.output / "overall_ber.png")
    print(args.output / "conditional_ber.png")


if __name__ == "__main__":
    main()
