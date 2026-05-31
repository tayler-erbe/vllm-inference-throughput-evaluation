#!/usr/bin/env bash
# =====================================================================
# Legislation Pipeline — Qwen Extended Chunk-Size Hunch Test
# =====================================================================
#
# Companion to run_mistral_extended_chunks.sh. Tests whether Qwen's
# parse-OK U-shape continues climbing past chunk_size=3000 (67.7%).
#
# Qwen showed a U-shape in the original sweep:
#   500: 62.5%  700: 60.5%  900: 62.7%  1200: 57.0%  2000: 63.3%  3000: 67.7%
# The right side of the U could keep climbing — analogous to Mistral.
#
# Runs 1 model × 2 chunk sizes × 3 repeats = 6 extraction cells.
# Source data: bronze session 999995 (same 50 controlled-length bills).
#
# Key design choices (matching run_mistral_extended_chunks.sh exactly):
#   - vLLM 0.6.6, AWQ-Marlin, --max-model-len=16384
#   - Concurrency=4, 3 repeats per cell
#   - perf_monitor.py polling every 10s
#   - Trap-based cleanup
#
# IMPORTANT: This script must NOT run concurrently with the Mistral
# extended sweep. Both need exclusive access to the L4 GPU. Wait for
# the Mistral sweep to finish before launching this.
#
# Expected runtime: ~20-25 minutes wall-clock.
#
# Usage:
#   nohup setsid bash run_qwen_extended_chunks.sh \
#       > logs/qwen_extended_master.log 2>&1 &
#   echo $! > /tmp/qwen_extended_pid.txt
#
# To check progress:
#   tail -f logs/qwen_extended_master.log
#
# To stop cleanly:
#   kill -TERM $(cat /tmp/qwen_extended_pid.txt)
# =====================================================================

set -u

PIPELINE_DIR="/var/advanalytics/datashare/Admin-pipeline/pipelines/Legislation_Pipeline"
cd "$PIPELINE_DIR" || exit 1

export TIKTOKEN_CACHE_DIR="$HOME/tiktoken-cache"

VLLM_PYTHON="/var/advanalytics/anaconda3/envs/vllm/bin/python"
SYSTEM_PYTHON="/bin/python3"
HF_MODELS_DIR="/var/advanalytics/datashare/hf_models"

SOURCE_SESSION=999995
N_BILLS=50

MODELS=("qwen")
CHUNK_SIZES=(4000 5000)
REPEATS=3
CONCURRENCY=4
MAX_MODEL_LEN=16384

declare -A MODEL_PATH=(
  ["qwen"]="$HF_MODELS_DIR/Qwen2.5-7B-Instruct-AWQ"
)
declare -A MODEL_SERVED_NAME=(
  ["qwen"]="qwen25-7b-awq"
)

SWEEP_ID="vllm_qwen_extended_$(date +%Y%m%d_%H%M%S)"
SWEEP_LOG_DIR="logs/sweep_${SWEEP_ID}"
PERF_LOG_DIR="logs/perf"
mkdir -p "$SWEEP_LOG_DIR" "$PERF_LOG_DIR"
SWEEP_RESULTS_DIR="benchmarks/${SWEEP_ID}"
mkdir -p "$SWEEP_RESULTS_DIR"

VLLM_LOG="$SWEEP_LOG_DIR/vllm_current.log"
PERF_CSV="$PERF_LOG_DIR/perf_${SWEEP_ID}.csv"
MASTER_LOG="$SWEEP_LOG_DIR/master.log"

VLLM_PID=""
PERF_PID=""

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
      log "vLLM did not exit cleanly, sending SIGKILL"
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

# Hard guard: refuse to start if Mistral extended sweep is still running.
# Both jobs need exclusive L4 access; running both concurrently would
# OOM the GPU within seconds.
check_no_other_sweep() {
  if pgrep -f "run_mistral_extended_chunks.sh" > /dev/null; then
    log "ERROR: run_mistral_extended_chunks.sh is still running. Wait for it"
    log "       to finish before launching the Qwen sweep."
    log "       (Both need exclusive access to the L4 GPU.)"
    exit 3
  fi
  if pgrep -f "vllm.entrypoints.openai.api_server" > /dev/null; then
    log "WARNING: existing vLLM process detected from prior sweep."
    log "         Killing it before we start."
    pkill -TERM -f "vllm.entrypoints.openai.api_server" 2>/dev/null || true
    sleep 5
    if pgrep -f "vllm.entrypoints.openai.api_server" > /dev/null; then
      pkill -KILL -f "vllm.entrypoints.openai.api_server" 2>/dev/null || true
      sleep 3
    fi
  fi
}

start_vllm() {
  local model_name=$1
  local model_path="${MODEL_PATH[$model_name]}"
  local served="${MODEL_SERVED_NAME[$model_name]}"

  log "Starting vLLM for model=$model_name (path=$model_path, served-as=$served)..."
  log "  Using --max-model-len=$MAX_MODEL_LEN for 5000-token chunk support"

  : > "$VLLM_LOG"

  nohup setsid "$VLLM_PYTHON" -m vllm.entrypoints.openai.api_server \
    --model "$model_path" \
    --served-model-name "$served" \
    --quantization awq_marlin \
    --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization 0.90 \
    --enable-prefix-caching \
    --port 8000 \
    --host 0.0.0.0 \
    > "$VLLM_LOG" 2>&1 &
  VLLM_PID=$!
  log "vLLM launched with PID $VLLM_PID"

  log "Waiting for vLLM to become ready..."
  for i in $(seq 1 180); do
    if curl -sf "http://localhost:8000/v1/models" 2>/dev/null | grep -q "$served"; then
      log "vLLM ready (took ${i}s)"
      cp "$VLLM_LOG" "$SWEEP_LOG_DIR/vllm_${model_name}_startup.log"
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

  log "ERROR: vLLM did not become ready within 180s. Last 30 lines:"
  tail -30 "$VLLM_LOG" | tee -a "$MASTER_LOG"
  return 1
}

start_perf_monitor() {
  log "Starting perf_monitor → $PERF_CSV"
  nohup "$SYSTEM_PYTHON" tools/perf_monitor.py \
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
  local model_name=$1
  local size=$2
  local profile="${model_name}_${size}"

  local exists
  exists=$("$SYSTEM_PYTHON" -c "
from services.azure_blob_service import BlobClient, TIER_SILVER
b = BlobClient()
path = f'legislation/chunks_${profile}/session=${SOURCE_SESSION}/_chunks.parquet'
print('YES' if b.blob_exists(TIER_SILVER, path) else 'NO')
" 2>/dev/null)

  if [ "$exists" = "YES" ]; then
    log "  Chunks already exist for profile=$profile, skipping chunker"
    return 0
  fi

  log "  Chunks missing for profile=$profile, running chunker..."
  "$SYSTEM_PYTHON" -m stages.chunker \
    --profile "$profile" \
    --session "$SOURCE_SESSION" \
    --config "configs/chunker_${profile}.yaml" \
    >> "$SWEEP_LOG_DIR/chunker_${profile}.log" 2>&1
  local rc=$?
  if [ $rc -ne 0 ]; then
    log "  WARNING: chunker exited with code $rc — check chunker_${profile}.log"
    return $rc
  fi
  log "  Chunker complete for profile=$profile"
  return 0
}

run_extraction_cell() {
  local model_name=$1
  local size=$2
  local repeat=$3
  local profile="${model_name}_${size}"
  local label="${profile}_rep${repeat}"

  log ""
  log "----- CELL: $label -----"

  "$SYSTEM_PYTHON" -c "
from services.azure_blob_service import BlobClient, TIER_SILVER
b = BlobClient()
path = f'legislation/features_${profile}/session=${SOURCE_SESSION}/_features.parquet'
if b.blob_exists(TIER_SILVER, path):
    b.delete_blob(TIER_SILVER, path)
    print(f'Wiped features parquet for $profile (clean slate for repeat $repeat)')
" >> "$MASTER_LOG" 2>&1

  local cell_start
  cell_start=$(date +%s)

  "$SYSTEM_PYTHON" -m tools.compare_extractors \
    --session "$SOURCE_SESSION" \
    --bills $N_BILLS \
    --profiles "$profile" \
    --concurrency $CONCURRENCY \
    --repeats 1 \
    --keep-data \
    --output "$SWEEP_RESULTS_DIR/${label}.json" \
    >> "$SWEEP_LOG_DIR/extract_${label}.log" 2>&1
  local rc=$?
  local cell_end
  cell_end=$(date +%s)
  local cell_dur=$((cell_end - cell_start))

  if [ $rc -eq 0 ]; then
    log "  Cell $label complete in ${cell_dur}s (exit 0)"
  else
    log "  WARNING: Cell $label exited with code $rc after ${cell_dur}s — check extract_${label}.log"
  fi
  return $rc
}

# =====================================================================
# Main
# =====================================================================
log "===================================================="
log "QWEN EXTENDED CHUNK-SIZE HUNCH TEST"
log "===================================================="
log "Sweep ID:        $SWEEP_ID"
log "Hypothesis:      Qwen parse-OK climbs past chunk_size=3000 (67.7%)"
log "                 — testing the right side of the observed U-shape"
log "Source:          session=$SOURCE_SESSION, $N_BILLS bills"
log "Matrix:          1 model × ${#CHUNK_SIZES[@]} chunk sizes × $REPEATS repeats = $((${#CHUNK_SIZES[@]} * REPEATS)) cells"
log "Concurrency:     $CONCURRENCY"
log "max-model-len:   $MAX_MODEL_LEN"
log "vLLM env:        $VLLM_PYTHON"
log "Perf CSV:        $PERF_CSV"
log "Results dir:     $SWEEP_RESULTS_DIR"
log "Log dir:         $SWEEP_LOG_DIR"
log ""

# Hard guard against running concurrent with Mistral sweep
check_no_other_sweep

log "Pre-flight checks..."

if [ ! -x "$VLLM_PYTHON" ]; then
  log "ERROR: vLLM Python not executable at $VLLM_PYTHON"
  exit 2
fi

for model in "${MODELS[@]}"; do
  if [ ! -d "${MODEL_PATH[$model]}" ]; then
    log "ERROR: model dir missing: ${MODEL_PATH[$model]}"
    exit 2
  fi
done

for model in "${MODELS[@]}"; do
  for size in "${CHUNK_SIZES[@]}"; do
    if [ ! -f "configs/chunker_${model}_${size}.yaml" ]; then
      log "ERROR: missing config: configs/chunker_${model}_${size}.yaml"
      exit 2
    fi
    if [ ! -f "configs/extractor_${model}_${size}.yaml" ]; then
      log "ERROR: missing config: configs/extractor_${model}_${size}.yaml"
      log "  → create it before running (template: configs/extractor_qwen_3000.yaml)"
      exit 2
    fi
  done
done

start_perf_monitor

SWEEP_START=$(date +%s)

for model in "${MODELS[@]}"; do
  log ""
  log "===================================================="
  log "MODEL: $model"
  log "===================================================="

  if ! start_vllm "$model"; then
    log "FAILED to start vLLM for $model. Skipping all $model cells."
    stop_vllm
    continue
  fi

  for size in "${CHUNK_SIZES[@]}"; do
    # ensure_chunks_exist removed - compare_extractors.py handles chunking

    for repeat in $(seq 1 $REPEATS); do
      run_extraction_cell "$model" "$size" "$repeat"
    done
  done

  log ""
  log "Model $model complete. Stopping vLLM..."
  stop_vllm
  sleep 10
done

SWEEP_END=$(date +%s)
SWEEP_DUR=$((SWEEP_END - SWEEP_START))

log ""
log "===================================================="
log "SWEEP COMPLETE"
log "===================================================="
log "Total duration: ${SWEEP_DUR}s ($(printf '%dh:%02dm' $((SWEEP_DUR/3600)) $((SWEEP_DUR%3600/60))))"
log "Results:        $SWEEP_RESULTS_DIR"
log "Perf CSV:       $PERF_CSV"
log "Logs:           $SWEEP_LOG_DIR"
log ""
log "Per-cell JSON results:"
ls -la "$SWEEP_RESULTS_DIR" | tee -a "$MASTER_LOG"

exit 0
