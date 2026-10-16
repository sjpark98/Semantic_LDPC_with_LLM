"""Layer-2 validator: per-character greedy sequential acceptance of LLM corrections.

For each character the LLM changed, in order:
  1. Channel consistency: if any bit that would flip has |L_ch| > tau_ch, reject.
  2. Syndrome improvement: apply the flip to a scratch codeword; accept iff the
     syndrome weight strictly decreases (greedy, sequential state update).
"""
import numpy as np
from bitmap import char_bit_range


def char_to_bits(ch: str) -> np.ndarray:
    return np.unpackbits(np.frombuffer(ch.encode("latin-1", errors="replace"), dtype=np.uint8))


def validate_and_apply(H: np.ndarray, c_hat: np.ndarray, L_ch: np.ndarray,
                        char_positions: list[int], proposed_chars: list[str],
                        tau_ch: float = 6.0, use_validator: bool = True):
    """Returns (accepted_char_positions, accepted_bit_flags (n,), updated c_hat, stats dict)."""
    c_hat = c_hat.copy()
    n = len(c_hat)
    accepted_bit_mask = np.zeros(n, dtype=bool)
    accepted_positions = []
    n_checked = 0
    n_channel_rejected = 0
    n_syndrome_rejected = 0

    cur_weight = int((H.dot(c_hat) % 2).sum())

    for char_i, new_char in zip(char_positions, proposed_chars):
        lo, hi = char_bit_range(char_i)
        if hi > n:
            continue
        new_bits = char_to_bits(new_char)
        flip_mask = new_bits != c_hat[lo:hi]
        if not flip_mask.any():
            continue  # LLM proposed the same bits already present, nothing to validate
        n_checked += 1

        if not use_validator:
            c_hat[lo:hi] = new_bits
            accepted_bit_mask[lo:hi] |= flip_mask
            accepted_positions.append(char_i)
            continue

        flip_bit_idx = np.nonzero(flip_mask)[0] + lo
        if np.any(np.abs(L_ch[flip_bit_idx]) > tau_ch):
            n_channel_rejected += 1
            continue

        c_trial = c_hat.copy()
        c_trial[lo:hi] = new_bits
        trial_weight = int((H.dot(c_trial) % 2).sum())
        if trial_weight < cur_weight:
            c_hat = c_trial
            cur_weight = trial_weight
            accepted_bit_mask[lo:hi] |= flip_mask
            accepted_positions.append(char_i)
        else:
            n_syndrome_rejected += 1

    stats = {
        "n_checked": n_checked,
        "n_accepted": len(accepted_positions),
        "n_channel_rejected": n_channel_rejected,
        "n_syndrome_rejected": n_syndrome_rejected,
    }
    return accepted_positions, accepted_bit_mask, c_hat, stats


def _weight(H, bits):
    return int((H.dot(bits) % 2).sum())


def validate_v1(H: np.ndarray, c_hat: np.ndarray, char_positions: list[int], proposed_chars: list[str],
                 weight_tolerance: float = 0.0):
    """Phase-2 D-1 V1: batch-first + leave-one-out harm removal. Apply every proposed
    correction at once; if that doesn't improve the global syndrome weight (within
    weight_tolerance), iteratively drop whichever correction's leave-one-out
    contribution is most negative (most harmful) until the weight is acceptable or no
    harmful candidate remains. No tau_ch veto: a correction the channel strongly
    disagrees with is only ever removed because it demonstrably raises the syndrome
    weight, not because of its LLR magnitude alone.
    weight_tolerance (>=0): the acceptance test's static syndrome-weight check is only
    a proxy for what BP2's actual iterative redecode will do; with few candidates
    (common at high SNR) that proxy is noisy and can reject genuinely-correct
    corrections. Raising this above 0 accepts a batch (or a partial one) even if its
    raw flipped-in-place weight is up to `weight_tolerance` worse than the original,
    trusting BP2 to resolve it."""
    n = len(c_hat)
    flips = {}
    for i, ch in zip(char_positions, proposed_chars):
        lo, hi = char_bit_range(i)
        if hi > n:
            continue
        new_bits = char_to_bits(ch)
        if np.array_equal(new_bits, c_hat[lo:hi]):
            continue
        flips[i] = (lo, hi, new_bits)

    if not flips:
        return [], np.zeros(n, dtype=bool), c_hat, {"n_checked": 0, "n_accepted": 0, "n_removed": 0}

    w_orig = _weight(H, c_hat)
    threshold = w_orig + weight_tolerance
    c_cur = c_hat.copy()
    for lo, hi, new_bits in flips.values():
        c_cur[lo:hi] = new_bits
    w_cur = _weight(H, c_cur)

    accepted = set(flips.keys())
    n_removed = 0
    if w_cur > threshold:
        while accepted:
            contribs = {}
            for i in accepted:
                lo, hi, new_bits = flips[i]
                c_wo = c_cur.copy()
                c_wo[lo:hi] = c_hat[lo:hi]
                contribs[i] = _weight(H, c_wo) - w_cur
            worst_i = min(contribs, key=contribs.get)
            if contribs[worst_i] >= 0:
                break  # no remaining candidate is net-harmful
            lo, hi, new_bits = flips[worst_i]
            c_cur[lo:hi] = c_hat[lo:hi]
            accepted.discard(worst_i)
            n_removed += 1
            w_cur = _weight(H, c_cur)
            if w_cur <= threshold:
                break

    # Hard fallback: a partial batch that still violates the epsilon criterion
    # is not a verified semantic prior.  Reject every remaining candidate so
    # the pipeline performs its ordinary channel-LLR second BP pass.
    threshold_satisfied = w_cur <= threshold
    if not threshold_satisfied:
        accepted.clear()
        c_cur = c_hat.copy()
        n_removed = len(flips)

    accepted_bit_mask = np.zeros(n, dtype=bool)
    for i in accepted:
        lo, hi, new_bits = flips[i]
        accepted_bit_mask[lo:hi] = True
    stats = {"n_checked": len(flips), "n_accepted": len(accepted), "n_removed": n_removed,
             "threshold_satisfied": threshold_satisfied,
             "original_weight": w_orig, "candidate_weight": w_cur,
             "threshold": threshold}
    return sorted(accepted), accepted_bit_mask, c_cur, stats


def validate_v2(H: np.ndarray, dec, c_hat: np.ndarray, char_positions: list[int], proposed_chars: list[str]):
    """Phase-2 D-1 V2: local-check acceptance. Judge each correction only by the
    checks incident to its own flipped bits (<=, not <), so a distant unrelated error
    elsewhere in the codeword can no longer veto a locally-improving correction --
    this directly targets the PoC's 73% false-rejection problem from the global
    syndrome-weight criterion."""
    n = len(c_hat)
    c_cur = c_hat.copy()
    unsatisfied = H.dot(c_cur) % 2
    accepted_positions = []
    accepted_bit_mask = np.zeros(n, dtype=bool)
    n_checked = 0

    for i, ch in zip(char_positions, proposed_chars):
        lo, hi = char_bit_range(i)
        if hi > n:
            continue
        new_bits = char_to_bits(ch)
        flip_mask = new_bits != c_cur[lo:hi]
        if not flip_mask.any():
            continue
        n_checked += 1
        flip_idx = np.nonzero(flip_mask)[0] + lo
        local_checks = np.unique(dec.col_idx[flip_idx].flatten())
        local_before = int(unsatisfied[local_checks].sum())

        c_trial = c_cur.copy()
        c_trial[lo:hi] = new_bits
        unsatisfied_trial = H.dot(c_trial) % 2
        local_after = int(unsatisfied_trial[local_checks].sum())

        if local_after <= local_before:
            c_cur = c_trial
            unsatisfied = unsatisfied_trial
            accepted_bit_mask[lo:hi] |= flip_mask
            accepted_positions.append(i)

    stats = {"n_checked": n_checked, "n_accepted": len(accepted_positions)}
    return accepted_positions, accepted_bit_mask, c_cur, stats
