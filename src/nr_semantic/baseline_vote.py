"""Vote-gated posterior-LLR fusion baseline adapted to the common NR link."""

from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np


@dataclass(frozen=True)
class VoteResult:
    positions: tuple[int, ...]
    values: dict[int, int]
    votes: dict[int, int]


def select_uncertain_bytes(posterior_l0: np.ndarray, valid_bytes: int, syndrome_weight: int, mean_column_weight: float) -> list[int]:
    if syndrome_weight == 0: return []
    byte_llr = posterior_l0[:valid_bytes * 8].reshape(valid_bytes, 8)
    log_good = np.logaddexp(0.0, -np.abs(byte_llr)).sum(axis=1) * -1  # sum(log(sigmoid(abs(L))))
    error_proxy = -np.expm1(log_good)
    count = min(valid_bytes, math.ceil(syndrome_weight / mean_column_weight))
    return sorted(range(valid_bytes), key=lambda i: (-error_proxy[i], i))[:count]


def extract_aligned_full_text(raw_output: str, received_text: str, mask_positions: list[int],
                              min_unmasked_agreement: float = 0.8) -> str | None:
    """Extract a whole-text answer embedded in an explanatory completion.

    The paper-style LLM input/output remains a complete masked sentence. For a
    completion longer than the sentence (e.g. ``The corrected text is: \"...\"``),
    inspect printable windows of the expected length and retain only the one
    that best agrees with positions that were *not* masked. This removes a
    wrapper; it never asks the model to emit position-only replacements.

    Short completions are deliberately rejected: there is no position-preserving
    way to infer their missing suffix. Exact-length completions pass unchanged.
    """
    expected = len(received_text)
    if len(raw_output) == expected:
        return raw_output
    if len(raw_output) < expected:
        return None
    masked = set(mask_positions)
    known = [i for i in range(expected) if i not in masked]
    if not known:
        return None
    best_score, best = -1.0, None
    for start in range(len(raw_output) - expected + 1):
        candidate = raw_output[start:start + expected]
        if not all(0x20 <= ord(char) <= 0x7E for char in candidate):
            continue
        score = sum(candidate[i] == received_text[i] for i in known) / len(known)
        if score > best_score:
            best_score, best = score, candidate
    return best if best is not None and best_score >= min_unmasked_agreement else None


def gate_votes(samples: list[str | None], mask_positions: list[int], threshold: float = 0.8,
               abstain_value: int | None = None) -> VoteResult:
    """Vote only at masked positions.

    ``abstain_value`` is the marker used to mask those positions (currently
    ``_``). It is ignored as a proposed replacement *only at these positions*.
    The acceptance denominator stays the total requested sample count (three),
    rather than the number of non-abstaining outputs. Thus an abstention cannot
    inflate confidence. A genuine underscore at a masked position cannot be
    recovered by this baseline; that deliberate limitation is recorded in the
    experiment protocol.
    """
    values: dict[int, int] = {}; votes: dict[int, int] = {}
    for position in mask_positions:
        count: dict[int, int] = {}
        for sample in samples:
            if sample is None or len(sample) <= position: continue
            value = ord(sample[position])
            if abstain_value is not None and value == abstain_value: continue
            if 0x20 <= value <= 0x7E: count[value] = count.get(value, 0) + 1
        if not count: continue
        value, count_value = min(count.items(), key=lambda item: (-item[1], item[0]))
        if count_value / 3 >= threshold:  # denominator is always S=3, never non-abstaining count
            values[position], votes[position] = value, count_value
    return VoteResult(tuple(sorted(values)), values, votes)


def pseudo_llr(vote: VoteResult, payload_bits: int, probability_clip: tuple[float, float] = (0.01, 0.99)) -> np.ndarray:
    out = np.zeros(payload_bits, dtype=np.float32)
    for pos in vote.positions:
        phi = float(np.clip(vote.votes[pos] / 3, *probability_clip))
        magnitude = math.log(phi / (1 - phi))
        bits = np.unpackbits(np.array([vote.values[pos]], dtype=np.uint8), bitorder="big")
        out[pos * 8:pos * 8 + 8] = (1 - 2 * bits.astype(np.int8)) * magnitude
    return out


def adaptive_alpha(posterior_l0: np.ndarray, vote: VoteResult, beta_scale: float = 0.30, alpha_max: float = 1.5) -> float:
    if not vote.positions: return 0.0
    indexes = np.concatenate([np.arange(position * 8, position * 8 + 8) for position in vote.positions])
    l_ldpc = float(np.median(np.abs(posterior_l0[indexes])))
    phi_bar = float(np.clip(np.median([vote.votes[position] / 3 for position in vote.positions]), 0.51, 0.99))
    l_llm = math.log(phi_bar / (1 - phi_bar))
    return float(np.clip(beta_scale * l_ldpc / l_llm, 0.0, alpha_max))
