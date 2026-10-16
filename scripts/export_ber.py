#!/usr/bin/env python3
"""Export BER tables from a completed experiment result.json."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


METHODS = ("bp50", "bp100", "proposed", "without_verification", "voting")


def write_table(results: dict, field: str, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("snr_db", *METHODS))
        writer.writeheader()
        for snr in sorted(results, key=float):
            values = results[snr][field]
            writer.writerow({"snr_db": snr, **{method: values[method] for method in METHODS}})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_json", type=Path)
    parser.add_argument("--output", type=Path, default=Path("results"))
    args = parser.parse_args()
    results = json.loads(args.result_json.read_text())
    write_table(results, "overall_payload_ber", args.output / "overall_ber.csv")
    write_table(results, "conditional_payload_ber", args.output / "conditional_ber.csv")


if __name__ == "__main__":
    main()
