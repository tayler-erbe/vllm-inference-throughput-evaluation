#!/usr/bin/env bash
# =====================================================================
# Legislation Pipeline — Qwen 3000 Concurrency Sweep
# =====================================================================
#
# Experiment 2 follow-up: runs Qwen 2.5 7B AWQ at its quality-best
# chunk size (3000 tokens) across 6 concurrency levels to complete
# the concurrency picture. The original Exp 2 tested qwen_500;
# this run tests qwen_3000 — Qwen's actual quality-best per Exp 1b.
#
# Design mirrors run_3model_vllm_sweep.sh / run_mistral_extended_chunks.sh
# exactly so results are directly comparable to Exp 2.
#
# Matrix: 1 model × 1 chunk size × 6 concurrency levels × 3 repeats = 18 cells
# Source: bronze session 999995 (50 controlled-length bills, 4500-4972 tokens)
# Estimated runtime: ~2-3 hours unattended
#
# Usage:
#   nohup setsid bash run_qwen_concurrency_sweep.sh \
#       > logs/qwen_conc_sweep_master.log 2>&1 &
#   echo $! > /tmp/qwen_conc_pid.txt
#
# To check progress:
#   tail -f logs/qwen_conc_sweep_master.log
#
# To stop cleanly:
#   kill -TERM $(cat /tmp/qwen_conc_pid.txt)
# =====================================================================

set -u

PIPELINE_DIR="/var/advanalytics/datashare/Admin-pipeline/pipelines/Legislation_Pipeline"
cd "$PIPELINE_DIR" || exit 1

export TIKTOKEN_CACHE_DIR=$HOME/tiktoken-cache

VLLM_PYTHON="/var/advanalytics/anaconda3/envs/vllm/bin/python"
HF_MODELS_DIR="/var/advanalytics/datashare/hf_models"

SOURCE_SESSION=999995
N_BILLS=50

MODEL="qwen"
CHUNK_SIZE=3000
PROFILE="qwen_3000"
REPEATS=3
CONCURRENCY_LEVELS=(1 4 8 12 16 24)

# Match main sweep vLLM args exactly
MAX_MODEL_LEN=8192
MODEL_PATH="$HF_MODELS_DIR/Qwen2.5-7B-Instruct-AWQ"
MODEL_SERVED_NAME="qwen25-7b-awq"

SWEEP_ID="vllm_qwen_conc_$(date +%Y%m%d_%H%M%S)"
SWEEP_LOG_DIR="logs/sweep_${SWEEP_ID}"
PERF_LOG_DIR="logs/perf"
mkdir -p "$SWEEP_LOG_DIR" "$PERF_LOG_DIR"
SWEEP_RESULTS_DIR="benchmarks/${SWEEP_ID}"
mkdir -p "$SWEEP_RESULTS_DIR"

VLLM_LOG="$SWEEP_LOG_DIR/vllm_qwen.log"
PERF_CSV="$PERF_LOG_DIR/perf_${SWEEP_ID}.csv"
MASTER_LOG="$SWEEP_LOG_DIR/master.log"

VLLM_PID=""
PERF_PID=""

# ---- Helpers ----
log() {
  local ts
  ts="$(date '+%Y-%m-%d %H:%M:%S')"
  printf '[%s] %s\n' "$ts" "$*" | tee -a "$MASTER_LOG"
}

stop_vllm() {
  if [ -n "$VLLM_PID" ] && kill -0 "$VLLM_PID" 2>/dev/null; then
    log "Stopping vLLM (PID $VLLM_PID)..."
    kill -TERM "$VLLM_PID" 2>/dev/null
    for i in $(seq 1 30); do
      if ! kill -0 "$VLLM_PID" 2>/dev/null; then break; fi
      sleep 1
    done
    if kill -0 "$VLLM_PID" 2>/dev/null; then
      kill -KILL "$VLLM_PID" 2>/dev/null
    fi
    VLLM_PID=""
    sleep 5
  fi
}

stop_perf_monitor() {
  if [ -n "$PERF_PID" ] && kill -0 "$PERF_PID" 2>/dev/null; then
    log "Stopping perf_monitor (PID $PERF_PID)..."
    kill -TERM "$PERF_PID" 2>/dev/null
    sleep 2
    if kill -0 "$PERF_PID" 2>/dev/null; then
      kill -KILL "$PERF_PID" 2>/dev/null
    fi
    PERF_PID=""
  fi
}

cleanup() {
  log ""
  log "===== CLEANUP (exit trap fired) ====="
  stop_vllm
  stop_perf_monitor
  log "Final GPU state:"
  nvidia-smi --query-gpu=memory.used,memory.free,utilization.gpu,temperature.gpu --format=csv | tee -a "$MASTER_LOG"
  log "===== SWEEP EXITED ====="
}
trap cleanup EXIT INT TERM

start_vllm() {
  log "Starting vLLM for qwen_3000 (path=$MODEL_PATH)..."
  : > "$VLLM_LOG"
  nohup setsid "$VLLM_PYTHON" -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" \
    --served-model-name "$MODEL_SERVED_NAME" \
    --quantization awq_marlin \
    --max-model-len $MAX_MODEL_LEN \
    --gpu-memory-utilization 0.90 \
    --enable-prefix-caching \
    --port 8000 \
    --host 0.0.0.0 \
    > "$VLLM_LOG" 2>&1 &
  VLLM_PID=$!
  log "vLLM launched with PID $VLLM_PID"

  log "Waiting for vLLM to become ready..."
  for i in $(seq 1 180); do
    if curl -sf "http://localhost:8000/v1/models" 2>/dev/null | grep -q "$MODEL_SERVED_NAME"; then
      log "vLLM ready (took ${i}s)"
      cp "$VLLM_LOG" "$SWEEP_LOG_DIR/vllm_startup.log"
      return 0
    fi
    if ! kill -0 "$VLLM_PID" 2>/dev/null; then
      log "ERROR: vLLM process died during startup. Last 30 lines:"
      tail -30 "$VLLM_LOG" | tee -a "$MASTER_LOG"
      return 1
    fi
    if [ $((i % 15)) -eq 0 ]; then log "  ...still waiting (${i}s)"; fi
    sleep 1
  done
  log "ERROR: vLLM did not become ready within 180s."
  tail -30 "$VLLM_LOG" | tee -a "$MASTER_LOG"
  return 1
}

start_perf_monitor() {
  log "Starting perf_monitor → $PERF_CSV"
  nohup "$VLLM_PYTHON" tools/perf_monitor.py \
    --output "$PERF_CSV" \
    --interval 10 \
    --duration 86400 \
    --vllm-url "http://localhost:8000" \
    >> "$SWEEP_LOG_DIR/perf_monitor.log" 2>&1 &
  PERF_PID=$!
  log "perf_monitor launched with PID $PERF_PID"
  sleep 3
}

ensure_chunks_exist() {
  # Chunks for qwen_3000/session=999995 confirmed present in silver:
  # legislation/chunks_qwen_3000/session=999995/_chunks.parquet
  # Created during original Exp 1 chunk-size sweep. Chunker not needed.
  log "Chunks confirmed present in silver for $PROFILE session=$SOURCE_SESSION"
  return 0
}

# Patch extractor config to use session 999995 at runtime
TEMP_EXTRACTOR_CONFIG="$SWEEP_LOG_DIR/extractor_${PROFILE}_999995.yaml"
sed 's/- 2176/- 999995/' "configs/extractor_${PROFILE}.yaml" > "$TEMP_EXTRACTOR_CONFIG"

run_cell() {
  local conc=$1
  local repeat=$2
  local label="${PROFILE}_c${conc}_rep${repeat}"

  log ""
  log "----- CELL: $label (conc=$conc repeat=$repeat) -----"

  # Wipe features parquet for clean repeat (system python has pandas)
  /bin/python3 -c "
import sys; sys.path.insert(0,'.')
from services.azure_blob_service import BlobClient, TIER_SILVER
b = BlobClient()
path = f'legislation/features_qwen_3000/session=999995/_features.parquet'
if b.blob_exists(TIER_SILVER, path):
    b.delete_blob(TIER_SILVER, path)
    print('Wiped features parquet for clean repeat')
" >> "$MASTER_LOG" 2>&1 || log "  (features wipe skipped — not critical)"

  local cell_start
  cell_start=$(date +%s)

  /bin/python3 -m tools.compare_extractors \
    --session "$SOURCE_SESSION" \
    --bills $N_BILLS \
    --profiles "$PROFILE" \
    --concurrency "$conc" \
    --repeats 1 \
    --keep-data \
    --output "$SWEEP_RESULTS_DIR/${label}.json" \
    >> "$SWEEP_LOG_DIR/extract_${label}.log" 2>&1
  local rc=$?
  local dur=$(( $(date +%s) - cell_start ))

  if [ $rc -eq 0 ]; then
    log "  Cell $label complete in ${dur}s (exit 0)"
    # Quick sanity check: file should be >1KB
    local fsize
    fsize=$(stat -c%s "$SWEEP_RESULTS_DIR/${label}.json" 2>/dev/null || echo 0)
    if [ "$fsize" -lt 1000 ]; then
      log "  WARNING: result file suspiciously small (${fsize} bytes) — check extract_${label}.log"
    fi
  else
    log "  WARNING: Cell $label exited with code $rc after ${dur}s"
  fi
  return $rc
}

# =====================================================================
# Pre-flight
# =====================================================================
log "===================================================="
log "QWEN 3000 CONCURRENCY SWEEP"
log "===================================================="
log "Sweep ID:     $SWEEP_ID"
log "Profile:      $PROFILE (chunk_size=3000, overlap=300)"
log "Source:       session=$SOURCE_SESSION, $N_BILLS bills"
log "Matrix:       ${#CONCURRENCY_LEVELS[@]} concurrency levels × $REPEATS repeats = $((${#CONCURRENCY_LEVELS[@]} * REPEATS)) cells"
log "Concurrency:  ${CONCURRENCY_LEVELS[*]}"
log "Perf CSV:     $PERF_CSV"
log "Results dir:  $SWEEP_RESULTS_DIR"
log ""
log "Pre-flight checks..."

if [ ! -x "$VLLM_PYTHON" ]; then
  log "ERROR: vLLM Python not found at $VLLM_PYTHON"
  exit 2
fi
if [ ! -d "$MODEL_PATH" ]; then
  log "ERROR: model dir missing: $MODEL_PATH"
  exit 2
fi
if [ ! -f "configs/chunker_${PROFILE}.yaml" ]; then
  log "ERROR: missing configs/chunker_${PROFILE}.yaml"
  exit 2
fi
if [ ! -f "configs/extractor_${PROFILE}.yaml" ]; then
  log "ERROR: missing configs/extractor_${PROFILE}.yaml"
  exit 2
fi

# Kill any stray vLLM
if pgrep -f "vllm.entrypoints.openai.api_server" > /dev/null; then
  log "WARNING: existing vLLM process detected — killing before start"
  pkill -TERM -f "vllm.entrypoints.openai.api_server" 2>/dev/null || true
  sleep 5
  pkill -KILL -f "vllm.entrypoints.openai.api_server" 2>/dev/null || true
  sleep 3
fi

log "Pre-flight passed."

# =====================================================================
# Main: start vLLM once, loop over concurrency levels
# =====================================================================
start_perf_monitor

if ! start_vllm; then
  log "FAILED to start vLLM. Aborting."
  exit 1
fi

# Ensure chunks exist (only needs to run once)
ensure_chunks_exist

# 5-call warmup before first cell (matches main sweep convention)
log ""
log "Running 5-call warmup before first cell..."
/bin/python3 -m tools.compare_extractors \
  --session "$SOURCE_SESSION" \
  --bills 1 \
  --profiles "$PROFILE" \
  --concurrency 1 \
  --repeats 5 \
  --output /dev/null \
  >> "$SWEEP_LOG_DIR/warmup.log" 2>&1
log "Warmup complete."

SWEEP_START=$(date +%s)

for conc in "${CONCURRENCY_LEVELS[@]}"; do
  log ""
  log "===================================================="
  log "CONCURRENCY = $conc"
  log "===================================================="
  for repeat in $(seq 1 $REPEATS); do
    run_cell "$conc" "$repeat"
  done
done

SWEEP_DUR=$(( $(date +%s) - SWEEP_START ))
log ""
log "===================================================="
log "SWEEP COMPLETE"
log "===================================================="
log "Total duration: ${SWEEP_DUR}s ($(printf '%dh:%02dm' $((SWEEP_DUR/3600)) $((SWEEP_DUR%3600/60))))"
log "Results:        $SWEEP_RESULTS_DIR"
log "Perf CSV:       $PERF_CSV"
log ""
log "Result files:"
ls -la "$SWEEP_RESULTS_DIR" | tee -a "$MASTER_LOG"

exit 0
