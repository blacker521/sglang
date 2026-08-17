#!/usr/bin/env bash
# A/B EvalScope perf: baseline (SGLANG_BEAM_DISABLE_VECTORIZED_SELECT=1) vs optimized.
# Uses run_genrec_evalscope_perf.py against a locally launched 0801 server.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

GPU="${GPU:-0}"
PORT="${PORT:-8907}"
MODEL_PATH="${MODEL_PATH:-/mnt/nas_new_1/mcu-multimodal-rec/model/online/mcu-genrec-v1.1.23-fp8/1784718542}"
# Keep runs tractable; override with PARALLEL / NUMBER for full 5000.
PARALLEL="${PARALLEL:-1 8}"
NUMBER="${NUMBER:-500 500}"
N="${N:-50}"
WARMUP="${WARMUP:-5}"
PRINT_SAMPLES="${PRINT_SAMPLES:-1}"
SID_VALID_NUM="${SID_VALID_NUM:-0}"
VALID_SID_FILE="${VALID_SID_FILE:-/mnt/datadisk0/mcu/chongyangwang/sglang-beam-search/data/sid2vid.json}"
OUT_ROOT="${OUT_ROOT:-$ROOT/outputs/ab_vectorized_select}"
VENV_PY="${VENV_PY:-$ROOT/.venv/bin/python}"
CLIENT_PY="${CLIENT_PY:-/home/chongyangwang/.venv/bin/python}"
URL="http://127.0.0.1:${PORT}/v1/chat/completions"

mkdir -p "$OUT_ROOT"
export PYTHONPATH="$ROOT/python${PYTHONPATH:+:$PYTHONPATH}"

kill_server() {
  local pids
  pids="$(ss -lntp 2>/dev/null | awk -v p=":$PORT" '$4 ~ p {print}' | sed -n 's/.*pid=\([0-9]*\).*/\1/p' | sort -u || true)"
  if [[ -n "${pids}" ]]; then
    echo "[ab] killing listeners on :$PORT -> $pids"
    # shellcheck disable=SC2086
    kill $pids 2>/dev/null || true
    sleep 2
    # shellcheck disable=SC2086
    kill -9 $pids 2>/dev/null || true
  fi
  pkill -f "sglang.launch_server.*--port ${PORT}" 2>/dev/null || true
  sleep 2
}

wait_ready() {
  local i
  for i in $(seq 1 180); do
    if curl -sf "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; then
      echo "[ab] server ready on :$PORT (${i}s)"
      return 0
    fi
    sleep 2
  done
  echo "[ab] server failed to become ready" >&2
  return 1
}

start_server() {
  local tag="$1"
  local disable="$2"
  local log="$OUT_ROOT/server_${tag}.log"
  kill_server
  echo "[ab] starting server tag=$tag DISABLE_VECTORIZED_SELECT=$disable (log=$log)"
  CUDA_VISIBLE_DEVICES="$GPU" \
  SGLANG_BEAM_DISABLE_VECTORIZED_SELECT="$disable" \
  "$VENV_PY" -m sglang.launch_server \
    --model-path "$MODEL_PATH" \
    --port "$PORT" \
    --host 127.0.0.1 \
    --enable-beam-search \
    --mem-fraction-static 0.7 \
    --quantization fp8 \
    --lm-head-special-token-ids 151669:153206,151645 \
    --attention-backend flashinfer \
    --kv-cache-dtype fp8_e4m3 \
    --fp8-gemm-backend flashinfer_cutlass \
    --schedule-policy lpm \
    >"$log" 2>&1 &
  echo $! >"$OUT_ROOT/server_${tag}.pid"
  wait_ready
}

run_client() {
  local tag="$1"
  local out_dir="$OUT_ROOT/$tag"
  mkdir -p "$out_dir"
  echo "[ab] client tag=$tag -> $out_dir"
  local extra=()
  if [[ "$SID_VALID_NUM" == "0" ]]; then
    extra+=(--no-sid-valid-check --sid-valid-num 0)
  else
    extra+=(--valid-sid-file "$VALID_SID_FILE" --sid-valid-num "$SID_VALID_NUM")
  fi
  # shellcheck disable=SC2086
  "$CLIENT_PY" "$ROOT/run_genrec_evalscope_perf.py" \
    --url "$URL" \
    --model "$MODEL_PATH" \
    --tokenizer-path "$MODEL_PATH" \
    --dataset-path "$ROOT/genrec_perf_5000.jsonl" \
    --parallel $PARALLEL \
    --number $NUMBER \
    --n "$N" \
    --warmup-num "$WARMUP" \
    --print-samples "$PRINT_SAMPLES" \
    --print-top-k 3 \
    --outputs-dir "$out_dir" \
    --name "genrec_vectorized_select_${tag}" \
    "${extra[@]}" \
    2>&1 | tee "$out_dir/client.log"
}

summarize() {
  "$CLIENT_PY" - <<'PY'
import json, re, sys
from pathlib import Path
root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("outputs/ab_vectorized_select")
# argv injected below
PY
}

echo "[ab] OUT_ROOT=$OUT_ROOT GPU=$GPU PORT=$PORT PARALLEL='$PARALLEL' NUMBER='$NUMBER'"

# Baseline first (old Python path), then optimized.
start_server baseline 1
run_client baseline
kill_server

start_server optimized 0
run_client optimized
kill_server

"$CLIENT_PY" - "$OUT_ROOT" <<'PY'
import json, sys
from pathlib import Path

root = Path(sys.argv[1])

def load_summaries(tag: str):
    base = root / tag
    rows = []
    for p in sorted(base.rglob("benchmark_summary.json")):
        d = json.loads(p.read_text())
        # EvalScope summary shapes vary; normalize common fields.
        conc = None
        m = re_search = __import__("re").search(r"parallel_(\d+)_number_(\d+)", str(p))
        if m:
            conc = int(m.group(1))
            number = int(m.group(2))
        else:
            number = None
        # Try several known layouts
        rps = (
            d.get("Request throughput (req/s)")
            or d.get("request_throughput")
            or d.get("Throughput")
            or d.get("rps")
        )
        lat = (
            d.get("Mean latency (s)")
            or d.get("mean_latency")
            or d.get("Average latency (s)")
            or d.get("latency_avg_s")
        )
        p50 = d.get("Median latency (s)") or d.get("latency_p50_s") or d.get("P50 latency (s)")
        p99 = d.get("P99 latency (s)") or d.get("latency_p99_s")
        ttft = d.get("Mean TTFT (ms)") or d.get("ttft_avg_ms") or d.get("Average TTFT (ms)")
        tpot = d.get("Mean TPOT (ms)") or d.get("tpot_avg_ms") or d.get("Average TPOT (ms)")
        rows.append({
            "path": str(p),
            "concurrency": conc,
            "number": number,
            "raw": d,
            "rps": rps,
            "lat": lat,
            "p50": p50,
            "p99": p99,
            "ttft": ttft,
            "tpot": tpot,
        })
    return rows

def fmt(x):
    if x is None:
        return "n/a"
    try:
        return f"{float(x):.4f}"
    except Exception:
        return str(x)

print("\n===== VECTORIZED SELECT A/B (EvalScope) =====")
for tag in ("baseline", "optimized"):
    rows = load_summaries(tag)
    print(f"\n--- {tag} ({len(rows)} summary file(s)) ---")
    if not rows:
        # fallback: show performance_summary.txt
        for p in sorted((root / tag).rglob("performance_summary.txt")):
            print(p.read_text()[:2000])
        continue
    for r in rows:
        print(
            f"conc={r['concurrency']} n={r['number']} "
            f"rps={fmt(r['rps'])} lat={fmt(r['lat'])} "
            f"p50={fmt(r['p50'])} p99={fmt(r['p99'])} "
            f"ttft={fmt(r['ttft'])} tpot={fmt(r['tpot'])}"
        )
        # also dump key raw keys for debugging once
        if r is rows[0]:
            keys = list(r["raw"].keys())[:40]
            print(f"  raw_keys={keys}")

base = {r["concurrency"]: r for r in load_summaries("baseline")}
opt = {r["concurrency"]: r for r in load_summaries("optimized")}
print("\n--- delta (optimized vs baseline) ---")
for conc in sorted(set(base) | set(opt)):
    if conc is None:
        continue
    b, o = base.get(conc), opt.get(conc)
    if not b or not o:
        continue
    def rel(nv, ov):
        try:
            nv, ov = float(nv), float(ov)
            if ov == 0:
                return "n/a"
            return f"{(nv/ov - 1)*100:+.2f}%"
        except Exception:
            return "n/a"
    print(
        f"conc={conc}: RPS {rel(o['rps'], b['rps'])}, "
        f"lat {rel(o['lat'], b['lat'])}, "
        f"p50 {rel(o['p50'], b['p50'])}, "
        f"p99 {rel(o['p99'], b['p99'])}"
    )
print("===== END =====\n")
PY

echo "[ab] done. artifacts under $OUT_ROOT"
