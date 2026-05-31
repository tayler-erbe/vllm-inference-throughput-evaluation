#!/usr/bin/env bash
# Experiment 4 — Qwen 2.5 14B AWQ · chunk_size=5000 · concurrency=4
set -u

PIPELINE_DIR="/var/advanalytics/datashare/Admin-pipeline/pipelines/Legislation_Pipeline"
cd "$PIPELINE_DIR" || exit 1

export TIKTOKEN_CACHE_DIR=$HOME/tiktoken-cache
VLLM_PYTHON="/var/advanalytics/anaconda3/envs/vllm/bin/python"
HF_MODELS_DIR="/var/advanalytics/datashare/hf_models"
SOURCE_SESSION=999995
N_BILLS=50
PROFILE="qwen14b_5000"
CONCURRENCY=4
REPEATS=3
MAX_MODEL_LEN=16384
MODEL_PATH="$HF_MODELS_DIR/Qwen2.5-14B-Instruct-AWQ"
MODEL_SERVED_NAME="qwen25-14b-awq"

SWEEP_ID="vllm_qwen14b_exp4_$(date +%Y%m%d_%H%M%S)"
SWEEP_LOG_DIR="logs/sweep_${SWEEP_ID}"
mkdir -p "$SWEEP_LOG_DIR" "logs/perf" "benchmarks/${SWEEP_ID}"
SWEEP_RESULTS_DIR="benchmarks/${SWEEP_ID}"
VLLM_LOG="$SWEEP_LOG_DIR/vllm_qwen14b.log"
PERF_CSV="logs/perf/perf_${SWEEP_ID}.csv"
MASTER_LOG="$SWEEP_LOG_DIR/master.log"
VLLM_PID=""
PERF_PID=""

log() { local ts; ts="$(date '+%Y-%m-%d %H:%M:%S')"; printf '[%s] %s\n' "$ts" "$*" | tee -a "$MASTER_LOG"; }

stop_vllm() {
  if [ -n "$VLLM_PID" ] && kill -0 "$VLLM_PID" 2>/dev/null; then
    log "Stopping vLLM..."; kill -TERM "$VLLM_PID" 2>/dev/null
    for i in $(seq 1 30); do if ! kill -0 "$VLLM_PID" 2>/dev/null; then break; fi; sleep 1; done
    if kill -0 "$VLLM_PID" 2>/dev/null; then kill -KILL "$VLLM_PID" 2>/dev/null; fi
    VLLM_PID=""; sleep 8
  fi
}
stop_perf() {
  if [ -n "$PERF_PID" ] && kill -0 "$PERF_PID" 2>/dev/null; then
    kill -TERM "$PERF_PID" 2>/dev/null; sleep 2; PERF_PID=""
  fi
}
cleanup() {
  log "===== CLEANUP ====="; stop_vllm; stop_perf
  nvidia-smi --query-gpu=memory.used,memory.free,utilization.gpu,temperature.gpu --format=csv | tee -a "$MASTER_LOG"
  log "===== EXITED ====="
}
trap cleanup EXIT INT TERM

# Write extractor config to configs dir (avoids --extractor-config flag issues)
cat > "configs/extractor_${PROFILE}.yaml" << 'YAML'
profile: qwen14b_5000
sessions:
  - 999995
chunk_limit: null
backend: vllm
prompt_path: prompts/feature_extraction_v1.txt
prompt_version: v1
flush_every_n_chunks: 25
vllm:
  base_url: http://localhost:8000/v1
  model_name: qwen25-14b-awq
  timeout_seconds: 300
  temperature: 0.0
  top_p: 1.0
  max_tokens: 1024
  system_prompt: null
ollama:
  path: /usr/local/bin/ollama
  model_name: qwen2.5:14b
  timeout_seconds: 300
slos:
  throughput_min:
    chunks_extracted_per_second: 0.05
  latency_max:
    api_time.p50: 60.0
    api_time.p99: 120.0
  counts_max:
    chunks_failed: 0
  ratio_max:
    - numerator: parse_partial
      denominator: chunks_extracted
      max: 0.30
YAML
log "Wrote configs/extractor_${PROFILE}.yaml"

# Write chunker config pointing to qwen_5000 chunks (already in silver)
cat > "configs/chunker_${PROFILE}.yaml" << 'YAML'
profile: qwen14b_5000
sessions:
  - 999995
mode: live
bill_limit: null
max_tokens: 5000
overlap_tokens: 500
massive_threshold_tokens: 80000
massive_first_n: 5
massive_last_n: 5
massive_middle_group_size: 10
hard_truncate_tokens: 50000
slos:
  throughput_min:
    tokens_processed_per_second: 5000
    bills_chunked_per_second: 5
YAML

start_vllm() {
  log "Starting vLLM for Qwen 2.5 14B AWQ..."
  log "NOTE: 14B model takes 2-4 minutes to load — this is normal."
  : > "$VLLM_LOG"
  nohup setsid "$VLLM_PYTHON" -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" --served-model-name "$MODEL_SERVED_NAME" \
    --quantization awq_marlin --max-model-len $MAX_MODEL_LEN \
    --gpu-memory-utilization 0.92 --enable-prefix-caching \
    --port 8000 --host 0.0.0.0 > "$VLLM_LOG" 2>&1 &
  VLLM_PID=$!
  log "vLLM PID $VLLM_PID"
  log "Waiting up to 300s..."
  for i in $(seq 1 300); do
    if curl -sf "http://localhost:8000/v1/models" 2>/dev/null | grep -q "$MODEL_SERVED_NAME"; then
      log "vLLM ready (took ${i}s)"; return 0; fi
    if ! kill -0 "$VLLM_PID" 2>/dev/null; then
      log "ERROR: vLLM died. Last 20 lines:"; tail -20 "$VLLM_LOG" | tee -a "$MASTER_LOG"; return 1; fi
    if [ $((i % 30)) -eq 0 ]; then log "  ...loading (${i}s)"; fi
    sleep 1
  done
  log "ERROR: vLLM not ready after 300s."; return 1
}

# Also create a symlink so qwen14b_5000 chunks point to qwen_5000 chunks
/bin/python3 -c "
import sys; sys.path.insert(0,'.')
from services.azure_blob_service import BlobClient, TIER_SILVER
b = BlobClient()
src = 'legislation/chunks_qwen_5000/session=999995/_chunks.parquet'
dst = 'legislation/chunks_qwen14b_5000/session=999995/_chunks.parquet'
if b.blob_exists(TIER_SILVER, src) and not b.blob_exists(TIER_SILVER, dst):
    import pandas as pd
    df = b.read_parquet(TIER_SILVER, src)
    # Update profile name in the chunks
    df['chunk_strategy'] = 'standard'
    b.write_parquet(df, TIER_SILVER, dst)
    print(f'Copied {len(df)} chunks to qwen14b_5000 profile')
elif b.blob_exists(TIER_SILVER, dst):
    print('qwen14b_5000 chunks already exist')
else:
    print('ERROR: source chunks not found')
" 2>/dev/null

run_cell() {
  local repeat=$1
  local label="${PROFILE}_c${CONCURRENCY}_rep${repeat}"
  log ""; log "----- CELL: $label -----"
  /bin/python3 -c "
import sys; sys.path.insert(0,'.')
from services.azure_blob_service import BlobClient, TIER_SILVER
b = BlobClient()
path = 'legislation/features_qwen14b_5000/session=999995/_features.parquet'
if b.blob_exists(TIER_SILVER, path):
    b.delete_blob(TIER_SILVER, path); print('Wiped features')
" >> "$MASTER_LOG" 2>&1 || true
  local t0; t0=$(date +%s)
  /bin/python3 -m tools.compare_extractors \
    --session "$SOURCE_SESSION" --bills $N_BILLS \
    --profiles "$PROFILE" --concurrency "$CONCURRENCY" \
    --repeats 1 --keep-data \
    --output "$SWEEP_RESULTS_DIR/${label}.json" \
    >> "$SWEEP_LOG_DIR/extract_${label}.log" 2>&1
  local rc=$? dur=$(( $(date +%s) - t0 ))
  if [ $rc -eq 0 ]; then
    log "  Cell $label done in ${dur}s"
    local sz; sz=$(stat -c%s "$SWEEP_RESULTS_DIR/${label}.json" 2>/dev/null || echo 0)
    [ "$sz" -lt 1000 ] && log "  WARNING: result file small (${sz}b)"
  else
    log "  WARNING: Cell $label exit code $rc after ${dur}s"
  fi
}

# PRE-FLIGHT
log "===================================================="; log "EXPERIMENT 4 — QWEN 2.5 14B AWQ"
log "===================================================="; log "Sweep: $SWEEP_ID"
log "Profile: $PROFILE | Session: $SOURCE_SESSION | Conc: $CONCURRENCY | Reps: $REPEATS"
[ ! -x "$VLLM_PYTHON" ] && log "ERROR: vLLM Python not found" && exit 2
[ ! -d "$MODEL_PATH" ] && log "ERROR: model missing: $MODEL_PATH" && exit 2
pgrep -f "vllm.entrypoints.openai.api_server" > /dev/null && {
  log "Killing stray vLLM..."
  pkill -TERM -f "vllm.entrypoints.openai.api_server" 2>/dev/null; sleep 8
  pkill -KILL -f "vllm.entrypoints.openai.api_server" 2>/dev/null; sleep 5; }
log "Pre-flight passed."

# LAUNCH
nohup /bin/python3 tools/perf_monitor.py \
  --output "$PERF_CSV" --interval 10 --duration 86400 \
  --vllm-url "http://localhost:8000" \
  >> "$SWEEP_LOG_DIR/perf_monitor.log" 2>&1 &
PERF_PID=$!; sleep 3

start_vllm || { log "FAILED to start vLLM"; exit 1; }

log "Running 3-call warmup (14B is slow — expect 5-10 min)..."
/bin/python3 -m tools.compare_extractors \
  --session "$SOURCE_SESSION" --bills 1 \
  --profiles "$PROFILE" --concurrency 1 --repeats 3 \
  --output /dev/null >> "$SWEEP_LOG_DIR/warmup.log" 2>&1
log "Warmup complete."

T0=$(date +%s)
for repeat in $(seq 1 $REPEATS); do run_cell "$repeat"; done
DUR=$(( $(date +%s) - T0 ))

log ""; log "===================================================="
log "EXPERIMENT 4 COMPLETE — ${DUR}s ($(printf '%dh:%02dm' $((DUR/3600)) $((DUR%3600/60))))"
log "===================================================="; log "Results: $SWEEP_RESULTS_DIR"
ls -la "$SWEEP_RESULTS_DIR" | tee -a "$MASTER_LOG"
exit 0
