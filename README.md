# Semantic LDPC with LLM

Code-only release for reproducing BER curves from an LLM-aided LDPC text
transmission experiment. No previous numerical results, datasets, figures, LLM
response caches, or MATLAB files are included.

## Experiment implemented by this code

- systematic `(3,6)`-regular pyLDPC code, `n=1500`, `k=752`
- BPSK over AWGN
- BP50 and fresh BP100 baselines with syndrome-zero early stopping
- Proposed: BP50, whole-sentence LLM repair, Needleman--Wunsch alignment,
  syndrome-based candidate verification, semantic-prior injection, fresh BP50
- matched Proposed without verification
- three-sample masked-character Voting baseline
- overall BER over a fixed number of test sentences
- conditional BER over the first fixed number of BP50 syndrome failures

This is a regular pyLDPC experiment, not a 3GPP NR LDPC implementation.

## Installation

Python 3.11 is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The input must be a UTF-8 text file containing one sentence per line. Despite
the filename used in the original experiment, it is not parsed as a CSV table.
Only normalized printable-ASCII sentences of 20--94 characters are used.

## One-command BER run

Set the API key only at runtime, then pass the sentence file to `run_all.sh`.

```bash
export OPENAI_API_KEY='your-key-at-runtime'
./run_all.sh /path/to/sentences.txt
```

This creates, locally and outside Git tracking:

```text
data/       deterministic train/validation/test split
runs/       channel-event pools, LLM cache, checkpoints, and raw metrics
results/    overall_ber.csv and conditional_ber.csv
figures/    overall_ber.png and conditional_ber.png
```

The default full evaluation sends 10,000 test sentences at each SNR in
`1.0,1.2,...,2.0` dB and accumulates the first 100 BP50 failures for the
conditional result. It can require many LLM calls and incur substantial API cost.

For a small pipeline check:

```bash
export OPENAI_API_KEY='your-key-at-runtime'
export PYLDPC_SNRS=1.0
export OVERALL_FRAMES=20
export TARGET_FAILURES=2
export CONDITIONAL_FRAME_CAP=100
export PYLDPC_MAX_EVENTS=2
./run_all.sh /path/to/sentences.txt
```

The API key may instead be stored in a user-owned file:

```bash
export OPENAI_API_KEY_FILE=/secure/path/openai_key
```

Do not place the key inside this repository.

## Run individual stages

```bash
python scripts/prepare_dataset.py /path/to/sentences.txt

export PYLDPC_RUN=events_final
export PYLDPC_SNRS=1.0,1.2,1.4,1.6,1.8,2.0
export OVERALL_FRAMES=10000
export TARGET_FAILURES=100
python scripts/collect_events.py

export PYLDPC_RUN=semantic_final
export PYLDPC_EVENT_POOLS=events_final
export OPENAI_API_KEY='your-key-at-runtime'
python scripts/run_experiment.py

python scripts/export_ber.py runs/semantic_final/result.json --output results
python scripts/plot_ber.py --results results --output figures
```

OpenAI model availability and sampled Voting outputs may change over time. LLM
responses are therefore cached by request content under the selected run directory.
