#!/usr/bin/env bash
# GenRec EvalScope 压测（TTFT / TPOT / E2E）+ SID 合法率
# 用法: bash benchmark/run_genrec_evalscope_perf.sh
set -euo pipefail

BENCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$BENCH_DIR"

MODEL_PATH="${MODEL_PATH:-/mnt/nas_new_1/mcu-multimodal-rec/model/online/mcu-genrec-v1.1.23-fp8/1784718542}"
URL="${URL:-http://127.0.0.1:8904/v1/chat/completions}"
# shellcheck disable=SC2206
PARALLEL=(${PARALLEL:-1 8})
# shellcheck disable=SC2206
NUMBER=(${NUMBER:-5000 5000})
N="${N:-50}"
WARMUP="${WARMUP:-10}"
PRINT_SAMPLES="${PRINT_SAMPLES:-3}"
PRINT_TOP_K="${PRINT_TOP_K:-5}"
NAME="${NAME:-genrec_v1.1.23_fp8}"
SHARED_SYSTEM_PROMPT="${SHARED_SYSTEM_PROMPT:-}"
SHARED_SYSTEM_PROMPT_FILE="${SHARED_SYSTEM_PROMPT_FILE:-}"
VALID_SID_FILE="${VALID_SID_FILE:-$BENCH_DIR/../data/sid2vid.json}"
SID_VALID_NUM="${SID_VALID_NUM:-100}"

EXTRA_ARGS=()
if [[ -n "$SHARED_SYSTEM_PROMPT_FILE" ]]; then
  EXTRA_ARGS+=(--shared-system-prompt-file "$SHARED_SYSTEM_PROMPT_FILE")
elif [[ -n "$SHARED_SYSTEM_PROMPT" ]]; then
  EXTRA_ARGS+=(--shared-system-prompt "$SHARED_SYSTEM_PROMPT")
fi
if [[ -n "$VALID_SID_FILE" ]]; then
  EXTRA_ARGS+=(--valid-sid-file "$VALID_SID_FILE" --sid-valid-num "$SID_VALID_NUM")
fi

python "$BENCH_DIR/run_genrec_evalscope_perf.py" \
  --url "$URL" \
  --model "$MODEL_PATH" \
  --tokenizer-path "$MODEL_PATH" \
  --dataset-path "$BENCH_DIR/genrec_perf_5000.jsonl" \
  --parallel "${PARALLEL[@]}" \
  --number "${NUMBER[@]}" \
  --n "$N" \
  --warmup-num "$WARMUP" \
  --print-samples "$PRINT_SAMPLES" \
  --print-top-k "$PRINT_TOP_K" \
  --outputs-dir "$BENCH_DIR/outputs" \
  --name "$NAME" \
  "${EXTRA_ARGS[@]}"
