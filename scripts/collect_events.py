"""Prepare held-out channel trials for GPT-3.5 all-baseline evaluation.

The first ``OVERALL_FRAMES`` trials/SNR define fixed-corpus overall BER.  The
same deterministic stream continues only when required to obtain exactly
``TARGET_FAILURES`` BP50-failure events for conditional BER.  No LLM is called.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "src" / "ldpc_core"
sys.path[:0] = [str(CORE), str(ROOT / "src")]
from channel import awgn, bpsk_modulate, llr_from_y
from ldpc import BPDecoder, build_ldpc, encode
from textcodec import frame_text


def printable(bits: np.ndarray, valid: int) -> str:
    raw = np.packbits(bits[:valid].astype(np.uint8), bitorder="big").tobytes()
    return "".join(chr(v) if 32 <= v <= 126 else "?" for v in raw)


def main() -> None:
    snrs = [float(x) for x in os.environ["PYLDPC_SNRS"].split(",")]
    out = ROOT / "runs" / os.environ["PYLDPC_RUN"]
    overall_frames = int(os.environ.get("OVERALL_FRAMES", "10000"))
    target = int(os.environ.get("TARGET_FAILURES", "100"))
    cap = int(os.environ.get("CONDITIONAL_FRAME_CAP", "20000"))
    out.mkdir(parents=True, exist_ok=True)
    split = Path(os.environ.get(
        "PYLDPC_SPLIT",
        ROOT / "data/splits/pyldpc_n1500_k752_sentence_v1/test.jsonl",
    ))
    rows = [json.loads(x) for x in split.read_text().splitlines()]
    rows = [r for r in rows if r["text"].isascii() and 20 <= len(r["text"].encode()) <= 94]
    if len(rows) < cap:
        raise RuntimeError(f"need {cap} usable held-out sentences")
    h, g = build_ldpc(1500, 3, 6, 42)
    n, k = g.shape
    dec, rate = BPDecoder(h), k / n
    records, infos, llrs, lengths = [], [], [], []
    summaries = {}
    for snr in snrs:
        rng = np.random.default_rng(1_500_752 + round(snr * 1000))
        overall_bits = 0
        overall_err = {"bp50": 0, "bp100": 0}
        conditional_bits = 0
        conditional_err = {"bp50": 0, "bp100": 0}
        failures = 0
        for idx, row in enumerate(rows[:cap]):
            text = row["text"]; valid = len(text.encode()) * 8
            info = frame_text(text, k); cw = encode(g, info)
            y, var = awgn(bpsk_modulate(cw), snr, rate, rng); llr = llr_from_y(y, var)
            bp50 = dec.decode(llr, max_iter=50, msg_clip=15.0)
            e50 = int(np.count_nonzero(bp50["bits"][:valid] != info[:valid]))
            e100 = None
            in_overall = idx < overall_frames
            is_fail = not bp50["converged"]
            use_conditional = is_fail and failures < target
            if in_overall or use_conditional:
                bp100 = dec.decode(llr, max_iter=100, msg_clip=15.0)
                e100 = int(np.count_nonzero(bp100["bits"][:valid] != info[:valid]))
            if in_overall:
                overall_bits += valid; overall_err["bp50"] += e50; overall_err["bp100"] += int(e100)
            if use_conditional:
                conditional_bits += valid; conditional_err["bp50"] += e50; conditional_err["bp100"] += int(e100)
                failures += 1
            if is_fail and (in_overall or use_conditional):
                records.append({"record_index": len(records), "snr_db": snr, "sentence_index": idx,
                                "text": text, "valid_bits": valid, "bp50_errors": e50,
                                "bp100_errors": int(e100), "bp50_received_text": printable(bp50["bits"], valid),
                                "in_overall": in_overall, "in_conditional_100": use_conditional})
                # BP decisions near a convergence boundary can change under
                # float32 rounding.  Retain the channel observations exactly
                # enough to reproduce the BP50 failure event at replay time.
                infos.append(info.astype(np.int8)); llrs.append(llr.astype(np.float64)); lengths.append(valid)
            if idx + 1 >= overall_frames and failures >= target:
                searched = idx + 1
                break
        else:
            raise RuntimeError(f"{snr}: only {failures}/{target} failures by cap {cap}")
        summaries[str(snr)] = {"frames_for_conditional": searched, "conditional_failures": failures,
                               "overall_frames": overall_frames, "overall_payload_bits": overall_bits,
                               "overall_bp_errors": overall_err, "conditional_payload_bits": conditional_bits,
                               "conditional_bp_errors": conditional_err}
        print(json.dumps({"snr": snr, **summaries[str(snr)]}), flush=True)
    np.savez_compressed(out / "frames.npz", info=np.stack(infos), llr=np.stack(llrs),
                        valid_bits=np.asarray(lengths, dtype=np.int32))
    with (out / "records.jsonl").open("w", encoding="utf-8") as f:
        for r in records: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    (out / "manifest.json").write_text(json.dumps({"profile":{"n":n,"k":k,"rate":rate,"modulation":"BPSK"},
        "snrs_db":snrs,"overall_frames":overall_frames,"target_conditional_failures":target,
        "summaries":summaries,"records":len(records)}, indent=2)+"\n")


if __name__ == "__main__": main()
