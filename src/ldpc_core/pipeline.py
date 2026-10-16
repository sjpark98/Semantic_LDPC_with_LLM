"""Stage 1-8: BP -> gate -> LLM correct -> NW align -> validate -> LLR inject -> BP redecode."""
import numpy as np
from bitmap import chars_per_frame, char_bit_range
from textcodec import bits_to_text
from align import project_correction
from validator import validate_and_apply, validate_v1, validate_v2, char_to_bits
from signals import compute_char_signals

LLR_MAX_DEFAULT = 20.0


def inject_llr(L_base: np.ndarray, accepted_bit_mask: np.ndarray, target_bits: np.ndarray,
                post_llr: np.ndarray, alpha: float = 1.0, beta: float = 3.0,
                theta: float = 8.0, L_max: float = LLR_MAX_DEFAULT) -> np.ndarray:
    """Add an accepted LLM prior to the supplied BP2 LLR base.

    The normal receiver supplies the original channel LLR. ``L_base`` is kept
    explicit so a posterior-input ablation can be selected without changing
    the injection rule itself.
    """
    L_in = L_base.copy()
    L_llm = beta * (1 - 2 * target_bits.astype(np.float64))  # b=0 -> +beta, b=1 -> -beta
    inject_mask = accepted_bit_mask & (np.abs(post_llr) <= theta)
    L_in[inject_mask] = np.clip(L_base[inject_mask] + alpha * L_llm[inject_mask], -L_max, L_max)
    return L_in, inject_mask


def run_frame(H, dec, x_true: np.ndarray, u_true: np.ndarray, llr_ch: np.ndarray, k: int,
              corrector_fn, max_iter: int = 50, gate_syndrome_frac: float = 0.15,
              alpha: float = 1.0, beta: float = 3.0, theta: float = 8.0,
              tau_ch: float = 6.0, L_max: float = LLR_MAX_DEFAULT,
              use_validator: bool = True, use_llm: bool = True, outer_loop: int = 1,
              decoder_kwargs: dict | None = None, beta_policy=None, llm_scorer=None,
              validator_mode: str = "v0", force_llm: bool = False, v1_weight_tolerance: float = 0.0,
              content_len: int | None = None, injection_base: str = "channel",
              alignment_mode: str = "nw", injection_mask_scope: str = "validator",
              unchanged_prior_beta: float = 0.0):
    """Runs the full pipeline for one received frame. corrector_fn(str)->str provides the
    text correction (real LLM, or an oracle/stub for ablations). decoder_kwargs are passed
    through to every dec.decode() call (mode/msg_clip/min_sum_alpha/diagnostics, Phase-2 A-1).
    beta_policy (Phase-2 C-1): callable(signal_dict)->beta|None: when given, this REPLACES
    the legacy fixed-beta + binary validator path with continuous per-character beta
    (beta<=0/None == skip, generalizing accept/reject into one axis). llm_scorer(corrupted,
    corrected)->per-char logprob array, used only to populate the s_llm signal for beta_policy.
    force_llm: when True, run the LLM even on a converged BP1 (syndrome=0) -- this is the
    only way to ever touch an undetected error (a wrong codeword that still satisfies every
    parity check), since syndrome=0 is otherwise treated as unconditional success. Note the
    syndrome-based validators (V0/V1) can't usefully judge a correction to an already-
    converged codeword: any single-bit change strictly increases the syndrome weight from
    its minimum of 0, so 'requires improvement' rejects everything -- only
    use_validator=False (or a CRC-based acceptance test, not implemented here) can accept
    anything in this regime. content_len: if given, assume the receiver knows the real
    message only occupies the first content_len characters (e.g. from a length header)
    and never show the LLM (or consider for correction) anything past that -- the
    trailing pad-character region is left untouched, driven by its channel LLR alone."""
    if injection_base not in {"channel", "posterior"}:
        raise ValueError("injection_base must be 'channel' or 'posterior'")
    if alignment_mode not in {"nw", "strict"}:
        raise ValueError("alignment_mode must be 'nw' or 'strict'")
    if injection_mask_scope not in {"validator", "full_character"}:
        raise ValueError("injection_mask_scope must be 'validator' or 'full_character'")
    n = len(x_true)
    nchars = chars_per_frame(k)
    L = content_len if content_len is not None else nchars
    m = H.shape[0]
    decoder_kwargs = decoder_kwargs or {}

    out1 = dec.decode(llr_ch, max_iter=max_iter, **decoder_kwargs)
    result = {
        "bp1_converged": out1["converged"],
        "bp1_bit_errs": int(np.sum(out1["bits"] != x_true)),
        "bp1_iters": out1["n_iter"],
        "llm_called": False,
        "gated_out": False,
        "final_bits": out1["bits"],
        "final_converged": out1["converged"],
        "validator_stats": None,
        "final_diag": out1.get("diag"),
        "h2_direct_flips": 0,
        "h2_propagated_flips": 0,
        "h2_ratio": float("nan"),
        "n_fixed_direct": 0,
        "n_fixed_propagated": 0,
        "n_broken_direct": 0,
        "n_broken_propagated": 0,
    }
    if out1["converged"] and not force_llm:
        # NOTE: converged (syndrome=0) does not imply bp1_bit_errs==0 -- that's exactly
        # the undetected-error case. Carry the real count through instead of assuming success.
        result["final_bit_errs"] = result["bp1_bit_errs"]
        return result

    cur_llr = llr_ch
    cur_bits = out1["bits"]
    cur_post_llr = out1["post_llr"]

    for loop_i in range(outer_loop):
        bits_pre_redecode = cur_bits.copy()
        syn_weight = int((H.dot(cur_bits) % 2).sum())
        if syn_weight / m > gate_syndrome_frac:
            result["gated_out"] = True
            break
        if not use_llm:
            break

        corrupted_text = bits_to_text(cur_bits[: nchars * 8])[:L]
        llm_out = corrector_fn(corrupted_text)
        # ``strict`` is the paired no-NW ablation: any output whose length
        # violates the required character correspondence is discarded, leaving
        # the received text unchanged. It never silently truncates/reindexes.
        corrected_text = (project_correction(corrupted_text, llm_out)
                          if alignment_mode == "nw"
                          else (llm_out if len(llm_out) == len(corrupted_text) else corrupted_text))
        if len(corrected_text) != L:
            corrected_text = (corrected_text[:L]).ljust(L)
        result["llm_called"] = True

        char_positions = [i for i in range(L) if corrected_text[i] != corrupted_text[i]]
        proposed_chars = [corrected_text[i] for i in char_positions]

        if beta_policy is not None:
            # --- Phase 2 continuous-beta path (B1/B2/B3): replaces validate_and_apply ---
            char_logprobs = llm_scorer(corrupted_text, corrected_text) if llm_scorer else None
            sigs = compute_char_signals(H, dec, cur_bits, cur_llr, cur_post_llr,
                                         corrupted_text, corrected_text, llm_out,
                                         char_positions, char_logprobs)
            L_in = cur_llr.copy()
            accepted_bit_mask = np.zeros(n, dtype=bool)
            n_checked = len(sigs)
            n_accepted = 0
            for sig in sigs:
                beta_c = beta_policy(sig)
                if beta_c is None or beta_c <= 0:
                    continue
                i = sig["char_idx"]
                lo, hi = char_bit_range(i)
                new_bits = char_to_bits(corrected_text[i])
                flip_mask = new_bits != cur_bits[lo:hi]
                flip_idx = np.nonzero(flip_mask)[0] + lo
                target = 1 - 2 * new_bits[np.nonzero(flip_mask)[0]].astype(np.float64)
                keep = np.abs(cur_post_llr[flip_idx]) <= theta
                flip_idx, target = flip_idx[keep], target[keep]
                if len(flip_idx) == 0:
                    continue
                L_in[flip_idx] = np.clip(cur_llr[flip_idx] + alpha * beta_c * target, -L_max, L_max)
                accepted_bit_mask[flip_idx] = True
                n_accepted += 1
            result["validator_stats"] = {"n_checked": n_checked, "n_accepted": n_accepted,
                                          "n_channel_rejected": 0, "n_syndrome_rejected": 0}
            inject_mask = accepted_bit_mask
        else:
            # --- legacy/D-1 path: binary validator (v0/v1/v2) + fixed beta ---
            if validator_mode == "v1":
                accepted_positions, accepted_bit_mask, cur_bits, vstats = validate_v1(
                    H, cur_bits, char_positions, proposed_chars, weight_tolerance=v1_weight_tolerance,
                )
            elif validator_mode == "v2":
                accepted_positions, accepted_bit_mask, cur_bits, vstats = validate_v2(
                    H, dec, cur_bits, char_positions, proposed_chars,
                )
            else:
                accepted_positions, accepted_bit_mask, cur_bits, vstats = validate_and_apply(
                    H, cur_bits, cur_llr, char_positions, proposed_chars, tau_ch=tau_ch,
                    use_validator=use_validator,
                )
            result["validator_stats"] = vstats
            result["accepted_char_positions"] = list(accepted_positions)

            # Matched-mask verification ablation: use the same character-level
            # prior support as V1 (all eight bits of each selected character).
            # The legacy/default path preserves the mask returned by each
            # validator, where V0/no-verification marks changed bits only.
            if injection_mask_scope == "full_character":
                accepted_bit_mask = np.zeros(n, dtype=bool)
                for i in accepted_positions:
                    lo, hi = char_bit_range(i)
                    if hi <= n:
                        accepted_bit_mask[lo:hi] = True

            target_bits = cur_bits.copy().astype(np.float64)
            for i in char_positions:
                lo, hi = char_bit_range(i)
                if hi <= n:
                    target_bits[lo:hi] = char_to_bits(corrected_text[i])

            base_llr = cur_llr if injection_base == "channel" else cur_post_llr
            L_in, inject_mask = inject_llr(base_llr, accepted_bit_mask, target_bits, cur_post_llr,
                                            alpha=alpha, beta=beta, theta=theta, L_max=L_max)
            changed_inject_mask = inject_mask.copy()
            unchanged_inject_mask = np.zeros(n, dtype=bool)
            if unchanged_prior_beta > 0:
                changed_char_set = set(char_positions)
                for i in range(L):
                    if i in changed_char_set:
                        continue
                    lo, hi = char_bit_range(i)
                    if hi <= n:
                        unchanged_inject_mask[lo:hi] = True
                unchanged_targets = cur_bits.astype(np.float64)
                unchanged_llr = unchanged_prior_beta * (1 - 2 * unchanged_targets)
                # PoC option: treat an LLM-unchanged character as known and
                # overwrite, rather than add to, the BP2 input LLR magnitude.
                L_in[unchanged_inject_mask] = unchanged_llr[unchanged_inject_mask]
                inject_mask |= unchanged_inject_mask
            result["changed_injection_mask"] = changed_inject_mask
            result["unchanged_injection_mask"] = unchanged_inject_mask
            # A verifier acceptance alone is insufficient: theta can still
            # remove every bit.  In either case, do not feed a semantic prior
            # into BP2; preserve the ordinary channel-LLR second pass instead.
            if not inject_mask.any():
                L_in = cur_llr.copy()
                result["semantic_injection_skipped"] = True
                result["semantic_injection_skip_reason"] = (
                    "no_accepted_candidates" if not accepted_bit_mask.any()
                    else "all_accepted_bits_filtered_by_theta"
                )
            else:
                result["semantic_injection_skipped"] = False
            # Kept in-memory for evaluation runners; callers must not serialize
            # these large arrays for every successful frame.
            result["injection_mask"] = inject_mask.copy()
            result["injection_target_bits"] = target_bits.copy()
        result[f"n_injected_bits_loop{loop_i}"] = int(inject_mask.sum())
        result["validated_bits"] = cur_bits.copy()  # post-validation, pre-BP2 redecode state

        out2 = dec.decode(L_in, max_iter=max_iter, **decoder_kwargs)
        cur_llr = L_in
        cur_bits = out2["bits"]
        cur_post_llr = out2["post_llr"]
        result["final_converged"] = out2["converged"]
        result["final_iters"] = out2["n_iter"]
        result["final_diag"] = out2.get("diag")

        # H2: correction propagation ratio (plan.md metric #3), ground-truth-aware so we
        # can tell a genuine fix from a bit that got newly broken by the injection.
        flipped = bits_pre_redecode != cur_bits
        direct_flips = int(np.sum(inject_mask & flipped))
        propagated_flips = int(np.sum((~inject_mask) & flipped))
        result["h2_direct_flips"] += direct_flips
        result["h2_propagated_flips"] += propagated_flips

        err_before = bits_pre_redecode != x_true
        err_after = cur_bits != x_true
        fixed = err_before & ~err_after
        broken = (~err_before) & err_after
        result["n_fixed_direct"] += int(np.sum(fixed & inject_mask))
        result["n_fixed_propagated"] += int(np.sum(fixed & ~inject_mask))
        result["n_broken_direct"] += int(np.sum(broken & inject_mask))
        result["n_broken_propagated"] += int(np.sum(broken & ~inject_mask))
        result["h2_ratio"] = (result["n_fixed_propagated"] / result["n_fixed_direct"]
                               if result["n_fixed_direct"] > 0 else float("nan"))

        if out2["converged"]:
            break

    result["final_bits"] = cur_bits
    result["final_bit_errs"] = int(np.sum(cur_bits != x_true))
    return result
