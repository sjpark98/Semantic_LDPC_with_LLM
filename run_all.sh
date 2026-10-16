#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 /path/to/sentences.txt" >&2
  exit 2
fi
if [[ -z "${OPENAI_API_KEY:-}" && -z "${OPENAI_API_KEY_FILE:-}" ]]; then
  echo "Set OPENAI_API_KEY or OPENAI_API_KEY_FILE before running." >&2
  exit 2
fi

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
dataset_path="$1"

export PYLDPC_SNRS="${PYLDPC_SNRS:-1.0,1.2,1.4,1.6,1.8,2.0}"
export OVERALL_FRAMES="${OVERALL_FRAMES:-10000}"
export TARGET_FAILURES="${TARGET_FAILURES:-100}"
export CONDITIONAL_FRAME_CAP="${CONDITIONAL_FRAME_CAP:-20000}"
export PYLDPC_STREAM_BATCH="${PYLDPC_STREAM_BATCH:-16}"
export GPT_WORKERS="${GPT_WORKERS:-4}"
export OPENAI_MODEL="${OPENAI_MODEL:-gpt-3.5-turbo}"

python "$repo_dir/scripts/prepare_dataset.py" "$dataset_path"

export PYLDPC_RUN="${EVENT_RUN:-events_final}"
python "$repo_dir/scripts/collect_events.py"

event_run="$PYLDPC_RUN"
export PYLDPC_RUN="${SEMANTIC_RUN:-semantic_final}"
export PYLDPC_EVENT_POOLS="$event_run"
python "$repo_dir/scripts/run_experiment.py"

python "$repo_dir/scripts/export_ber.py" \
  "$repo_dir/runs/$PYLDPC_RUN/result.json" --output "$repo_dir/results"
python "$repo_dir/scripts/plot_ber.py" \
  --results "$repo_dir/results" --output "$repo_dir/figures"
