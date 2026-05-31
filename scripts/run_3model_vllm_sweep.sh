#!/usr/bin/env bash
# =====================================================================
# Legislation Pipeline vLLM Re-Evaluation — Master Sweep
# =====================================================================
#
# Runs 3 models × 6 chunk sizes × 3 repeats = 54 extraction cells.
# Source data: bronze session 999995 (50 controlled-length bills, 4500-4972 tokens).
#
# Key design choices (documented in docs/methodology.md):
#   - vLLM 0.6.6 (after documented 0.11 upgrade attempt fell back)
#   - AWQ-Marlin quantization on L4
#   - --max-model-len 8192 (avoids context overflow at chunk_size=3000)
#   - --enable-prefix-caching (template is identical across chunks)
#   - Concurrency=4 (production-realistic batching, the Red Hat answer)
#   - 3 repeats per cell, with 5-call warmup before each (industry-standard)
#   - Chunk caching DISABLED for primary sweep (A/B comparison after)
#   - perf_monitor.py polling every 10s with safety alarms
#   - vLLM started via nohup setsid for signal isolation
#
# Expected runtime: ~15-20 hours at concurrency=4.
# Designed for Memorial Day long weekend unattended execution.
#
# Usage:
#   nohup setsid bash run_3model_vllm_sweep.sh > logs/sweep_master.log 2>&1 &
#   echo $! > /tmp/sweep_pid.txt
#
# To check progress:
#   tail -f logs/sweep_master.log
#   ls -la logs/perf/perf_*.csv
#
# To stop cleanly:
#   kill -TERM $(cat /tmp/sweep_pid.txt)
#   # The script's exit trap will stop vLLM + perf_monitor cleanly.
# =====================================================================

set -u  # error on unset vars (but NOT -e — we want to keep going on cell failures)

PIPELINE_DIR="/var/advanalytics/datashare/Admin-pipeline/pipelines/Legislation_Pipeline"
cd "$PIPELINE_DIR" || exit 1

# Environment
VLLM_PYTHON="/var/advanalytics/anaconda3/envs/vllm/bin/python"
SYSTEM_PYTHON="/bin/python3"  # System Python has the full pipeline deps (pandas, azure-storage-blob, etc.)
SYSTEM_PYTHON="/bin/python3"  # System Python has the full pipeline deps (pandas, azure-storage-blob, etc.)
HF_MODELS_DIR="/var/advanalytics/datashare/hf_models"

# Source session for bills (bronze 999995, 50 controlled-length bills)
SOURCE_SESSION=999995
N_BILLS=50

# Sweep matrix
MODELS=("mistral" "qwen" "llama31")
CHUNK_SIZES=(500 700 900 1200 2000 3000)
REPEATS=3
CONCURRENCY=4

# Model → vLLM startup args mapping
declare -A MODEL_PATH=(
  ["mistral"]="$HF_MODELS_DIR/Mistral-7B-Instruct-v0.3-AWQ"
  ["qwen"]="$HF_MODELS_DIR/Qwen2.5-7B-Instruct-AWQ"
  ["llama31"]="$HF_MODELS_DIR/Meta-Llama-3.1-8B-Instruct-AWQ-INT4"
)
declare -A MODEL_SERVED_NAME=(
  ["mistral"]="mistral-7b-awq"
  ["qwen"]="qwen25-7b-awq"
  ["llama31"]="llama31-8b-awq"
)

# Output paths
SWEEP_ID="vllm_sweep_$(date +%Y%m%d_%H%M%S)"
SWEEP_LOG_DIR="logs/sweep_${SWEEP_ID}"
PERF_LOG_DIR="logs/perf"
mkdir -p "$SWEEP_LOG_DIR" "$PERF_LOG_DIR"
SWEEP_RESULTS_DIR="benchmarks/${SWEEP_ID}"
mkdir -p "$SWEEP_RESULTS_DIR"

VLLM_LOG="$SWEEP_LOG_DIR/vllm_current.log"
PERF_CSV="$PERF_LOG_DIR/perf_${SWEEP_ID}.csv"
MASTER_LOG="$SWEEP_LOG_DIR/master.log"

# Track child PIDs for cleanup
VLLM_PID=""
PERF_PID=""

# =====================================================================
# Helpers
# =====================================================================
log() {
  # Append to master log AND stdout (which the outer nohup redirects to file)
  local ts
  ts="$(date '+%Y-%m-%d %H:%M:%S')"
  printf '[%s] %s\n' "$ts" "$*" | tee -a "$MASTER_LOG"
}

stop_vllm() {
  if [ -n "$VLLM_PID" ] && kill -0 "$VLLM_PID" 2>/dev/null; then
    log "Stopping vLLM (PID $VLLM_PID)..."
    kill -TERM "$VLLM_PID" 2>/dev/null
    # Wait up to 30 seconds for clean shutdown
    for i in $(seq 1 30); do
      if ! kill -0 "$VLLM_PID" 2>/dev/null; then break; fi
      sleep 1
    done
    if kill -0 "$VLLM_PID" 2>/dev/null; then
      log "vLLM did not exit cleanly, sending SIGKILL"
      kill -KILL "$VLLM_PID" 2>/dev/null
    fi
    VLLM_PID=""
    # Wait for GPU memory to release
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
  local model_name=$1
  local model_path="${MODEL_PATH[$model_name]}"
  local served="${MODEL_SERVED_NAME[$model_name]}"

  log "Starting vLLM for model=$model_name (path=$model_path, served-as=$served)..."

  # Truncate the current vLLM log (we keep all logs per-model in archives below)
  : > "$VLLM_LOG"

  # nohup + setsid for full process-group isolation from any signals
  # (including the daily yum-update cron that killed Tuesday's run)
  nohup setsid "$VLLM_PYTHON" -m vllm.entrypoints.openai.api_server \
    --model "$model_path" \
    --served-model-name "$served" \
    --quantization awq_marlin \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.90 \
    --enable-prefix-caching \
    --port 8000 \
    --host 0.0.0.0 \
    > "$VLLM_LOG" 2>&1 &
  VLLM_PID=$!
  log "vLLM launched with PID $VLLM_PID (detached via nohup setsid)"

  # Wait for vLLM to come up — poll /v1/models, timeout 180s
  log "Waiting for vLLM to become ready..."
  for i in $(seq 1 180); do
    if curl -sf "http://localhost:8000/v1/models" 2>/dev/null | grep -q "$served"; then
      log "vLLM ready (took ${i}s)"
      # Archive the startup log for this model
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
  # 24 hours max duration as safety belt. Interval 10s.
  # nohup so it survives any parent-shell hiccup.
  nohup "$SYSTEM_PYTHON" tools/perf_monitor.py \
    --output "$PERF_CSV" \
    --interval 10 \
    --duration 86400 \
    --vllm-url "http://localhost:8000" \
    >> "$SWEEP_LOG_DIR/perf_monitor.log" 2>&1 &
  PERF_PID=$!
  log "perf_monitor launched with PID $PERF_PID"
  sleep 3  # give it a moment to write the header
}

ensure_chunks_exist() {
  local model_name=$1
  local size=$2
  local profile="${model_name}_${size}"

  # Check whether chunks for this profile already exist in silver
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

  # Wipe features parquet so the extractor runs fresh (not idempotency-skipped).
  # This is the same pattern the harness uses for repeats.
  "$SYSTEM_PYTHON" -c "
from services.azure_blob_service import BlobClient, TIER_SILVER
b = BlobClient()
path = f'legislation/features_${profile}/session=${SOURCE_SESSION}/_features.parquet'
if b.blob_exists(TIER_SILVER, path):
    b.delete_blob(TIER_SILVER, path)
    print(f'Wiped features parquet for $profile (clean slate for repeat $repeat)')
" >> "$MASTER_LOG" 2>&1

  # Time the cell
  local cell_start
  cell_start=$(date +%s)

  # Extract via the harness. Important flags:
  #   --session 999995 (our bronze source)
  #   --bills 50 (full sample)
  #   --profiles ${profile} (one cell per call)
  #   --concurrency 4 (continuous batching exercise)
  #   --repeats 1 (we manage repeats in the outer loop for fault tolerance)
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
# Main sweep
# =====================================================================
log "===================================================="
log "LEGISLATION PIPELINE vLLM RE-EVALUATION"
log "===================================================="
log "Sweep ID:     $SWEEP_ID"
log "Source:       session=$SOURCE_SESSION, $N_BILLS bills"
log "Matrix:       ${#MODELS[@]} models × ${#CHUNK_SIZES[@]} chunk sizes × $REPEATS repeats = $((${#MODELS[@]} * ${#CHUNK_SIZES[@]} * REPEATS)) cells"
log "Concurrency:  $CONCURRENCY"
log "vLLM env:     $VLLM_PYTHON"
log "Perf CSV:     $PERF_CSV"
log "Results dir:  $SWEEP_RESULTS_DIR"
log "Log dir:      $SWEEP_LOG_DIR"
log ""
log "Pre-flight checks..."

# Sanity check: vLLM Python exists
if [ ! -x "$VLLM_PYTHON" ]; then
  log "ERROR: vLLM Python not executable at $VLLM_PYTHON"
  exit 2
fi

# Sanity check: each model dir exists
for model in "${MODELS[@]}"; do
  if [ ! -d "${MODEL_PATH[$model]}" ]; then
    log "ERROR: model dir missing: ${MODEL_PATH[$model]}"
    exit 2
  fi
done

# Sanity check: each chunker + extractor config exists for the 7B cells
for model in "${MODELS[@]}"; do
  for size in "${CHUNK_SIZES[@]}"; do
    if [ ! -f "configs/chunker_${model}_${size}.yaml" ]; then
      log "ERROR: missing config: configs/chunker_${model}_${size}.yaml"
      exit 2
    fi
    if [ ! -f "configs/extractor_${model}_${size}.yaml" ]; then
      log "ERROR: missing config: configs/extractor_${model}_${size}.yaml"
      exit 2
    fi
  done
done

# Make sure no stray vLLM is running before we start
if pgrep -f "vllm.entrypoints.openai.api_server" > /dev/null; then
  log "WARNING: existing vLLM process detected. Killing it before we start."
  pkill -TERM -f "vllm.entrypoints.openai.api_server" 2>/dev/null || true
  sleep 5
  if pgrep -f "vllm.entrypoints.openai.api_server" > /dev/null; then
    pkill -KILL -f "vllm.entrypoints.openai.api_server" 2>/dev/null || true
    sleep 3
  fi
fi

# Start perf_monitor BEFORE vLLM so we capture vLLM startup
start_perf_monitor

# =====================================================================
# Outer loop: per model, start vLLM, run all cells, stop vLLM
# =====================================================================
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

  # Run all (chunk_size, repeat) cells for this model
  for size in "${CHUNK_SIZES[@]}"; do
    # Ensure chunks exist before extraction loop. The chunker is deterministic
    # so we run it once per profile, then re-use across repeats.
    ensure_chunks_exist "$model" "$size"

    for repeat in $(seq 1 $REPEATS); do
      run_extraction_cell "$model" "$size" "$repeat"
    done
  done

  log ""
  log "Model $model complete. Stopping vLLM before next model..."
  stop_vllm
  # Brief pause to ensure VRAM is fully released
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

# Cleanup runs via trap
exit 0
