#!/usr/bin/env python3
"""Create deterministic, duplicate-safe splits from one-sentence-per-line text."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nr_semantic.data import (  # noqa: E402
    assign_split,
    audit_line_delimited_text,
    read_line_delimited_text,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="UTF-8 text file, one sentence per line")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data/splits/pyldpc_n1500_k752_sentence_v1",
    )
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--min-length", type=int, default=20)
    parser.add_argument("--max-length", type=int, default=94)
    args = parser.parse_args()

    records = read_line_delimited_text(args.input)
    seen: set[str] = set()
    selected = []
    for record in records:
        if not (args.min_length <= len(record.text) <= args.max_length):
            continue
        if record.normalized_hash in seen:
            continue
        seen.add(record.normalized_hash)
        selected.append(record)

    ratios = (0.70, 0.15, 0.15)
    by_split = {name: [] for name in ("train", "validation", "test")}
    for record in selected:
        by_split[assign_split(record.normalized_hash, args.seed, ratios)].append(record)

    args.output.mkdir(parents=True, exist_ok=True)
    for split, values in by_split.items():
        values.sort(key=lambda value: value.normalized_hash)
        with (args.output / f"{split}.jsonl").open("w", encoding="utf-8") as handle:
            for record in values:
                handle.write(json.dumps({
                    "source_id": record.source_id,
                    "text": record.text,
                    "normalized_hash": record.normalized_hash,
                }, ensure_ascii=True) + "\n")

    manifest = {
        "dataset": audit_line_delimited_text(args.input),
        "framing": "one complete sentence per codeword",
        "eligible_length_characters": [args.min_length, args.max_length],
        "deduplication": "normalized-text SHA-256; first occurrence retained",
        "split": {
            "method": "sha256(seed:normalized_hash)",
            "seed": args.seed,
            "ratios": {"train": ratios[0], "validation": ratios[1], "test": ratios[2]},
        },
        "counts": {name: len(values) for name, values in by_split.items()},
        "length_histogram": {
            name: dict(sorted(Counter(len(record.text) for record in values).items()))
            for name, values in by_split.items()
        },
        "ordering": "normalized_hash ascending within each split",
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output": str(args.output), "counts": manifest["counts"]}, indent=2))


if __name__ == "__main__":
    main()
