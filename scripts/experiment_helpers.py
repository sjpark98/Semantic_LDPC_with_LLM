"""Shared helpers for the memory-bounded semantic LDPC runner."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ldpc import encode
from nr_semantic.baseline_vote import select_uncertain_bytes
from nr_semantic.llm import (
    BASELINE_PROMPT_VERSION,
    PROPOSED_PROMPT_VERSION,
    baseline_prompt,
    proposed_prompt,
)


METHODS = ("bp50", "bp100", "proposed", "without_verification", "voting")


def error(bits: np.ndarray, info: np.ndarray, valid: int) -> int:
    return int(np.count_nonzero(bits[:valid] != info[:valid]))


def add(destination: dict, source: dict) -> None:
    for key, value in source.items():
        destination[key] = destination.get(key, 0) + int(value)


def prior_stats(mask, target, info, bp50_bits, final_bits, valid: int) -> dict:
    mask = np.asarray(mask[:valid], dtype=bool)
    target = np.asarray(target[:valid], dtype=np.int8)
    wrong = bp50_bits[:valid] != info[:valid]
    final_ok = final_bits[:valid] == info[:valid]
    return {
        "injected_bits": int(mask.sum()),
        "prior_correct_bits": int(np.count_nonzero(mask & (target == info[:valid]))),
        "direct_injected_wrong_to_correct": int(
            np.count_nonzero(mask & wrong & (target == info[:valid]))
        ),
        "final_injected_wrong_to_correct": int(np.count_nonzero(mask & wrong & final_ok)),
        "bp_propagated_noninjected_wrong_to_correct": int(
            np.count_nonzero((~mask) & wrong & final_ok)
        ),
    }


def prepare_event(event: dict, h, g, decoder):
    """Reconstruct one stored BP50 failure event and verify reproducibility."""
    valid = int(event["valid_bits"])
    info = event["info"]
    llr = event["llr"]
    codeword = encode(g, info)
    bp50 = decoder.decode(llr, max_iter=50, msg_clip=15.0)
    if error(bp50["bits"], info, valid) != int(event["bp50_errors"]):
        raise RuntimeError("event pool / decoder reproducibility mismatch")
    received = event["bp50_received_text"]
    syndrome_weight = int((h.dot(bp50["bits"]) % 2).sum())
    mean_column_degree = float(h.sum() / h.shape[1])
    mask = select_uncertain_bytes(
        bp50["post_llr"], valid // 8, syndrome_weight, mean_column_degree
    )
    return codeword, bp50, received, mask


def load_pool(paths: list[Path]):
    """Load event pools while decompressing each NumPy member only once."""
    events: list[dict] = []
    summaries: dict = {}
    for directory in paths:
        manifest = json.loads((directory / "manifest.json").read_text())
        arrays = np.load(directory / "frames.npz")
        infos = arrays["info"]
        llrs = arrays["llr"]
        rows = [
            json.loads(line)
            for line in (directory / "records.jsonl").read_text().splitlines()
        ]
        for row in rows:
            index = row["record_index"]
            row["info"] = infos[index]
            row["llr"] = llrs[index]
            row["pool"] = directory.name
            events.append(row)
        summaries.update(manifest["summaries"])
    return events, summaries


__all__ = [
    "METHODS",
    "BASELINE_PROMPT_VERSION",
    "PROPOSED_PROMPT_VERSION",
    "add",
    "baseline_prompt",
    "error",
    "load_pool",
    "prepare_event",
    "prior_stats",
    "proposed_prompt",
]
