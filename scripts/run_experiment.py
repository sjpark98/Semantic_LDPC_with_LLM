"""Memory-bounded GPT-3.5 evaluation over pre-collected pyLDPC events.

Unlike the legacy runner, this keeps at most STREAM_BATCH reconstructed BP50
states alive. Metrics are appended immediately and a checkpoint is written
after every batch, so API responses and completed work survive interruption.
"""
from __future__ import annotations

import gc
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
CORE = ROOT / "src" / "ldpc_core"
sys.path[:0] = [str(CORE), str(ROOT / "src"), str(HERE)]

from ldpc import BPDecoder, build_ldpc
from pipeline import run_frame
from nr_semantic.baseline_vote import adaptive_alpha, gate_votes, pseudo_llr
from nr_semantic.openai_corrector import OpenAICorrector
from experiment_helpers import (
    METHODS, add, baseline_prompt, error, load_pool, prepare_event,
    prior_stats, proposed_prompt, BASELINE_PROMPT_VERSION,
    PROPOSED_PROMPT_VERSION,
)


def atomic_json(path: Path, obj: object) -> None:
    tmp = path.with_suffix(path.suffix + ".partial")
    tmp.write_text(json.dumps(obj, indent=2) + "\n")
    os.replace(tmp, path)


def fetch_gpt(llm: OpenAICorrector, received: str, mask_positions: list[int]):
    proposed = llm.generate(
        proposed_prompt(received), do_sample=False, seed=0,
        max_new_tokens=128, prompt_version=PROPOSED_PROMPT_VERSION,
    ).text
    masked = "".join("_" if i in mask_positions else c for i, c in enumerate(received))
    prompt = baseline_prompt(masked, mask_positions)
    votes = [
        llm.generate_freeform(
            prompt, do_sample=True, seed=sample, max_new_tokens=128,
            temperature=.7, top_p=.8, prompt_version=BASELINE_PROMPT_VERSION,
        ).text
        for sample in range(3)
    ]
    return proposed, votes


def main() -> None:
    out = ROOT / "runs" / os.environ["PYLDPC_RUN"]
    out.mkdir(parents=True, exist_ok=True)
    pool_dirs = [ROOT / "runs" / p for p in os.environ["PYLDPC_EVENT_POOLS"].split(":")]
    events, summaries = load_pool(pool_dirs)
    snrs = [float(x) for x in os.environ["PYLDPC_SNRS"].split(",")]
    batch_size = int(os.environ.get("PYLDPC_STREAM_BATCH", "16"))
    workers = int(os.environ.get("GPT_WORKERS", "4"))
    max_events = int(os.environ.get("PYLDPC_MAX_EVENTS", "0"))
    theta, beta, epsilon = 6.0, 3.0, 18.0

    h, g = build_ldpc(1500, 3, 6, 42)
    n, k = g.shape
    decoder = BPDecoder(h)
    llm = OpenAICorrector(
        os.environ.get("OPENAI_MODEL", "gpt-3.5-turbo"),
        Path(os.environ.get("PYLDPC_LLM_CACHE", out / "llm_cache")),
        os.environ.get("OPENAI_API_KEY_FILE"),
        few_shot=True,
    )
    all_results = {}

    for snr in snrs:
        key = str(snr)
        selected = [e for e in events if float(e["snr_db"]) == snr]
        shard = os.environ.get("PYLDPC_EVENT_SHARD")
        if shard:
            shard_index, shard_count = (int(x) for x in shard.split("/", 1))
            selected = [e for i, e in enumerate(selected) if i % shard_count == shard_index]
        if max_events:
            selected = selected[:max_events]
        if not selected:
            raise RuntimeError(f"no events for SNR {snr}")

        summary = summaries[key]
        overall_bits = int(summary["overall_payload_bits"])
        conditional_bits = int(summary["conditional_payload_bits"])
        overall = dict(summary["overall_bp_errors"])
        conditional = dict(summary["conditional_bp_errors"])
        for name in ("proposed", "without_verification", "voting"):
            overall[name] = overall["bp50"]
            conditional[name] = conditional["bp50"]
        injection = {x: {} for x in ("proposed", "without_verification", "voting")}
        mismatch = {"proposed_raw_length_mismatch_frames": 0,
                    "proposed_raw_length_mismatch_chars": 0,
                    "voting_raw_samples": 0,
                    "voting_raw_length_mismatch_samples": 0,
                    "voting_raw_length_mismatch_frames": 0}
        metrics_path = out / f"semantic_event_metrics_{str(snr).replace('.', 'p')}db.jsonl"
        with metrics_path.open("w", encoding="utf-8") as metric_file:
            for start in range(0, len(selected), batch_size):
                batch_events = selected[start:start + batch_size]
                prepared = []
                for event in batch_events:
                    cw, bp50, received, mask = prepare_event(event, h, g, decoder)
                    prepared.append((event, cw, bp50, received, mask))

                replies = [None] * len(prepared)
                with ThreadPoolExecutor(max_workers=min(workers, len(prepared))) as executor:
                    futures = {executor.submit(fetch_gpt, llm, item[3], item[4]): i
                               for i, item in enumerate(prepared)}
                    for future in as_completed(futures):
                        replies[futures[future]] = future.result()

                for item, reply in zip(prepared, replies):
                    event, cw, bp50, received, mask_positions = item
                    p_raw, vote_raw = reply
                    valid, info, llr = int(event["valid_bits"]), event["info"], event["llr"]
                    if len(p_raw) != len(received):
                        mismatch["proposed_raw_length_mismatch_frames"] += 1
                        mismatch["proposed_raw_length_mismatch_chars"] += abs(len(p_raw) - len(received))
                    pv = run_frame(h, decoder, cw, info, llr, k, lambda _x, r=p_raw: r,
                        max_iter=50, beta=beta, theta=theta, gate_syndrome_frac=1.,
                        decoder_kwargs={"msg_clip": 15.}, validator_mode="v1",
                        use_validator=True, v1_weight_tolerance=epsilon, content_len=valid // 8)
                    nv = run_frame(h, decoder, cw, info, llr, k, lambda _x, r=p_raw: r,
                        max_iter=50, beta=beta, theta=theta, gate_syndrome_frac=1.,
                        decoder_kwargs={"msg_clip": 15.}, validator_mode="v0",
                        use_validator=False, content_len=valid // 8,
                        injection_mask_scope="full_character")

                    samples, bad = [], False
                    for raw in vote_raw:
                        mismatch["voting_raw_samples"] += 1
                        if len(raw) != len(received):
                            mismatch["voting_raw_length_mismatch_samples"] += 1
                            bad = True; samples.append(None)
                        else:
                            samples.append(raw)
                    mismatch["voting_raw_length_mismatch_frames"] += int(bad)
                    vote = gate_votes(samples, mask_positions, abstain_value=ord("_"))
                    voted = bp50
                    vote_mask, vote_target = np.zeros(n, bool), np.zeros(n, np.int8)
                    alpha = 0.0
                    if vote.positions:
                        alpha = adaptive_alpha(bp50["post_llr"], vote)
                        if alpha >= 1e-3:
                            for pos in vote.positions:
                                lo = pos * 8; vote_mask[lo:lo + 8] = True
                                vote_target[lo:lo + 8] = np.unpackbits(
                                    np.array([vote.values[pos]], np.uint8), bitorder="big")
                            prior = pseudo_llr(vote, n)
                            voted = decoder.decode(np.clip(bp50["post_llr"] + alpha * prior, -20., 20.),
                                                   max_iter=50, msg_clip=15.)
                    finals = {"proposed": error(pv["final_bits"], info, valid),
                              "without_verification": error(nv["final_bits"], info, valid),
                              "voting": error(voted["bits"], info, valid)}
                    base = int(event["bp50_errors"])
                    for name, value in finals.items():
                        if event["in_overall"]: overall[name] += value - base
                        if event["in_conditional_100"]: conditional[name] += value - base
                    stats = {
                        "proposed": prior_stats(pv.get("injection_mask", np.zeros(n, bool)),
                            pv.get("injection_target_bits", np.zeros(n, np.int8)), info,
                            bp50["bits"], pv["final_bits"], valid),
                        "without_verification": prior_stats(nv.get("injection_mask", np.zeros(n, bool)),
                            nv.get("injection_target_bits", np.zeros(n, np.int8)), info,
                            bp50["bits"], nv["final_bits"], valid),
                        "voting": prior_stats(vote_mask, vote_target, info, bp50["bits"], voted["bits"], valid),
                    }
                    for name, values in stats.items(): add(injection[name], values)
                    metric_file.write(json.dumps({"sentence_index": event["sentence_index"],
                        "in_overall": event["in_overall"], "in_conditional_100": event["in_conditional_100"],
                        "valid_bits": valid, "bp50_errors": base, "bp100_errors": event["bp100_errors"],
                        **{f"{x}_errors": v for x, v in finals.items()}, "injection": stats,
                        "proposed_raw_length": len(p_raw), "received_length": len(received),
                        "voting_raw_lengths": [len(x) for x in vote_raw], "voting_alpha": alpha}) + "\n")
                metric_file.flush()
                completed = min(start + batch_size, len(selected))
                atomic_json(out / "checkpoint.json", {"snr_db": snr, "completed": completed,
                    "total": len(selected), "overall_errors": overall,
                    "conditional_errors": conditional})
                print(f"{snr} dB: streamed {completed}/{len(selected)}", flush=True)
                del replies, prepared, batch_events
                gc.collect()

        for values in injection.values():
            values["injection_precision"] = (values["prior_correct_bits"] / values["injected_bits"]
                                                if values.get("injected_bits") else None)
        mismatch["proposed_raw_length_mismatch_rate"] = mismatch["proposed_raw_length_mismatch_frames"] / len(selected)
        mismatch["voting_raw_length_mismatch_sample_rate"] = (mismatch["voting_raw_length_mismatch_samples"] /
                                                               mismatch["voting_raw_samples"])
        result = {"profile": {"n": n, "k": k, "rate": k/n, "modulation": "BPSK",
                    "snr_db": snr, "overall_frames": 10000, "conditional_bp50_failures": 100,
                    "stream_batch": batch_size, "workers": workers, "event_shard": shard},
                  "parameters": {"theta": theta, "beta": beta, "epsilon": epsilon,
                    "proposed_input": "channel_llr_plus_semantic_prior", "nw_alignment": True},
                  "overall_payload_bits": overall_bits, "conditional_payload_bits": conditional_bits,
                  "overall_payload_errors": overall,
                  "overall_semantic_error_delta": {x: overall[x]-overall["bp50"] for x in ("proposed","without_verification","voting")},
                  "overall_payload_ber": {x: overall[x]/overall_bits for x in METHODS},
                  "conditional_payload_errors": conditional,
                  "conditional_semantic_error_delta": {x: conditional[x]-conditional["bp50"] for x in ("proposed","without_verification","voting")},
                  "conditional_payload_ber": {x: conditional[x]/conditional_bits for x in METHODS},
                  "injection": injection, "raw_output_length_mismatch": mismatch,
                  "semantic_events": len(selected)}
        atomic_json(out / f"result_{str(snr).replace('.', 'p')}db.json", result)
        all_results[key] = result
        atomic_json(out / "progress.json", all_results)
    atomic_json(out / "result.json", all_results)


if __name__ == "__main__":
    main()
