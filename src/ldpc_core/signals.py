"""Phase 2, B-1: per-character confidence signals for the LLM correction validator/calibrator.

Six signals per LLM-proposed character correction c (flip bit set F_c):
  s_llm  : avg per-token logprob from the LLM itself (teacher-forced), via llm_corrector
  s_syn  : mean fraction of unsatisfied checks incident to F_c (BP1 hard decision)
  s_ch   : max |channel LLR| among bits in F_c whose sign opposes the correction (0 if none)
  s_post : mean |BP1 posterior LLR| over F_c
  s_flip : number of bits in F_c (1-8)
  s_gap  : distance (in reference chars) from c to the nearest NW-alignment gap
"""
import numpy as np
from bitmap import char_bit_range
from align import nw_align
from validator import char_to_bits


def unsatisfied_check_counts(H: np.ndarray, dec, bits: np.ndarray) -> np.ndarray:
    """u_all[j] = number of unsatisfied checks incident to variable bit j."""
    unsatisfied = H.dot(bits) % 2  # (m,)
    return unsatisfied[dec.col_idx].sum(axis=1)  # (n,)


def gap_distances(reference: str, llm_output: str, **kw) -> np.ndarray:
    aligned_ref, aligned_out, _ = nw_align(reference, llm_output, **kw)
    ref_pos_of_col = []
    ref_pos = -1
    for a in aligned_ref:
        if a != "-":
            ref_pos += 1
        ref_pos_of_col.append(max(ref_pos, 0))
    gap_cols = [i for i, (a, b) in enumerate(zip(aligned_ref, aligned_out)) if a == "-" or b == "-"]
    n = len(reference)
    dist = np.full(n, float(n), dtype=np.float64)
    if not gap_cols:
        return dist
    gap_ref_positions = np.array(sorted({ref_pos_of_col[c] for c in gap_cols}), dtype=np.float64)
    for c in range(n):
        dist[c] = np.min(np.abs(gap_ref_positions - c))
    return dist


def compute_char_signals(H, dec, cur_bits: np.ndarray, cur_llr: np.ndarray, cur_post_llr: np.ndarray,
                          corrupted_text: str, corrected_text: str, llm_out: str,
                          char_positions: list[int], char_logprobs: np.ndarray | None = None):
    """Returns a list of per-character signal dicts, one per entry in char_positions
    (positions where corrected_text differs from corrupted_text)."""
    n = len(cur_bits)
    u_all = unsatisfied_check_counts(H, dec, cur_bits)
    gaps = gap_distances(corrupted_text, llm_out)
    d_v = dec.d_v

    records = []
    for i in char_positions:
        lo, hi = char_bit_range(i)
        if hi > n:
            continue
        new_bits = char_to_bits(corrected_text[i])
        flip_mask = new_bits != cur_bits[lo:hi]
        flip_idx = np.nonzero(flip_mask)[0] + lo
        if len(flip_idx) == 0:
            continue

        s_syn = float(u_all[flip_idx].mean() / d_v)
        target_sign = 1 - 2 * new_bits[np.nonzero(flip_mask)[0]].astype(np.float64)  # b=0-> +1 (target), b=1 -> -1
        ch_vals = cur_llr[flip_idx]
        opposing = ch_vals * target_sign < 0  # channel sign disagrees with proposed correction
        s_ch = float(np.abs(ch_vals[opposing]).max()) if opposing.any() else 0.0
        s_post = float(np.abs(cur_post_llr[flip_idx]).mean())
        s_flip = int(len(flip_idx))
        s_gap = float(gaps[i]) if i < len(gaps) else float(len(corrupted_text))
        s_llm = float(char_logprobs[i]) if char_logprobs is not None and i < len(char_logprobs) else float("nan")

        records.append({
            "char_idx": i, "s_llm": s_llm, "s_syn": s_syn, "s_ch": s_ch,
            "s_post": s_post, "s_flip": s_flip, "s_gap": s_gap,
        })
    return records
