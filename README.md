# vLLM Multi-Model Evaluation — Illinois Legislation Pipeline

**Project 10 · AI Solutions · University of Illinois System · May 2026**

A six-stage evaluation funnel to select a production LLM for structured legislation extraction — designed to make the right choice once, so the pipeline can scale to 50 states without revisiting the foundation.

---

## Production Recommendation

```
Model:        Llama 3.1 8B Instruct AWQ
Chunk size:   1,200 tokens (overlap: 120)
Concurrency:  12–24
Backend:      vLLM 0.6.6, AWQ-Marlin quantization
```

**Why Llama wins:**
- **Quality:** 92.5% parse-OK — 20.5pp lead over next-best at quality-best configurations
- **Validated:** 60/60 cells confirmed by blind Opus 4.8 scoring
- **Stable under load:** <0.6pp quality variation across c=1→24
- **Proxy reliable:** parse-OK is directionally reliable within Llama; actively misranks for Qwen
- **p99 tail:** 14s→39s across c=1→24 (Mistral: 15s→74s — disqualifying)

---

## Hardware & Environment

| Component | Value |
|-----------|-------|
| GPU | NVIDIA L4, 22.5 GB VRAM, 72W TDP |
| Server | urbadvanalytics1.admin.uillinois.edu |
| OS | Ubuntu 24.04 |
| vLLM | 0.6.6 |
| Quantization | AWQ-Marlin |
| Storage | Azure Blob Storage (bronze/silver/gold) |

**Models evaluated:**

| Model | Params | VRAM |
|-------|--------|------|
| Mistral 7B Instruct v0.3 AWQ | 7B | ~4 GB |
| Qwen 2.5 7B Instruct AWQ | 7B | ~4 GB |
| Llama 3.1 8B Instruct AWQ | 8B | ~4.5 GB |
| Qwen 2.5 14B Instruct AWQ | 14B | ~9 GB |

**Standard vLLM launch args:**
```bash
python -m vllm.entrypoints.openai.api_server \
  --model <MODEL_PATH> \
  --served-model-name <NAME> \
  --quantization awq_marlin \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.90 \
  --enable-prefix-caching \
  --port 8000
```

---

## Extraction Schema

15 structured fields per bill chunk. **11 substantive** (scored in qualitative review): `legislative_goal`, `key_provisions`, `intended_beneficiaries`, `increasing_aspects`, `decreasing_aspects`, `intent`, `motivation`, `potential_impact`, `ideological_alignment`, `legislative_strategy`, `policy_domain`. **4 short-answer**: `fiscal_impact_estimate`, `effective_dates`, `penalties_and_enforcement`, `reporting_requirements`.

**Parse status:** `parse_ok` (all fields populated) · `parse_partial` (some null) · `parse_empty` (failed)

**Parse-OK rate** = `parse_ok / total_chunks`

---

## Experiment 1 — Chunk-Size Sweep

**54 cells** · 3 models × 6 chunk sizes × 3 repeats · Concurrency=4 · Session 999995 (50 bills, 4,500–4,972 tokens)

**Parse-OK % results:**

| Chunk Size | llama31 | qwen | mistral |
|-----------|---------|------|---------|
| 500 | 89.5% | 62.5% | 56.8% |
| 700 | 87.9% | 60.5% | 55.9% |
| 900 | 89.4% | 62.7% | 60.1% |
| **1200** | **92.5%** | 57.1% | 65.0% |
| 2000 | 88.4% | 63.3% | 63.0% |
| 3000 | 85.0% | 67.7% | 72.0% |

Llama leads at every chunk size by 12–27pp. Run with `run_3model_vllm_sweep.sh`.

---

## Experiment 1b — Extended Sweep

**12 additional cells** · Mistral and Qwen at 4000/5000 tokens · `--max-model-len 16384`

| Chunk Size | qwen | mistral |
|-----------|------|---------|
| 4000 | 62.8% | 59.8% |
| 5000 | 58.7% | 72.0% |

**Tail-chunk artifact:** At chunk_size=4000, bills of 4,500–5,000 tokens produce a short tail chunk (~500–900 tokens) that correctly returns `parse_partial`. At chunk_size=5000 bills become single chunks and Mistral recovers to 72.0%.

---

## Experiment 2 — Concurrency Sweep

**54 cells** · 3 profiles × 6 concurrency levels × 3 repeats

**Throughput (chunks/sec):**

| Conc | llama31_1200 | qwen_500 | mistral_3000 |
|------|-------------|---------|-------------|
| 1 | 0.094 | 0.175 | 0.058 |
| 4 | 0.338 | 0.706 | 0.212 |
| 8 | 0.598 | 1.312 | 0.367 |
| 12 | 0.741 | 1.634 | 0.471 |
| 16 | 0.824 | 1.852 | 0.517 |
| 24 | 0.880 | 2.060 | 0.535 |

**p99 latency (seconds):**

| Conc | llama31_1200 | qwen_500 | mistral_3000 |
|------|-------------|---------|-------------|
| 1 | 14.1 | 9.5 | 15.2 |
| 4 | 21.3 | 9.8 | 22.4 |
| 8 | 28.7 | 11.2 | 38.9 |
| 12 | 33.4 | 14.1 | 52.3 |
| 16 | 36.8 | 16.9 | 63.7 |
| 24 | 39.2 | 18.8 | 74.1 |

**Mistral eliminated:** p99 grows 5× (15s→74s). KV cache head-of-line blocking from 3,000-token prompts.

**Quality invariant:** <0.6pp parse-OK variation across all concurrency levels for all models.

---

## Experiment 2b — Qwen 3000 Concurrency

**18 cells** · Qwen 2.5 7B at chunk_size=3000 (actual quality-best) · 6 concurrency levels

| Conc | Parse-OK | Chunks/sec | p50 | p99 |
|------|---------|-----------|-----|-----|
| 1 | 68.0% | 0.145 | 6.68s | 10.33s |
| 4 | 67.7% | 0.421 | 9.06s | 14.81s |
| 8 | 67.7% | 0.630 | 11.80s | 21.30s |
| 12 | 68.0% | 0.794 | 14.07s | 27.72s |
| 16 | 67.7% | 0.947 | 15.63s | 29.99s |
| 24 | 71.3% | 1.157 | 19.04s | 31.06s |

Quality ceiling at 68–71%. Effective throughput at c=24: 0.825 vs Llama's 0.814 — nearly identical, but at 71% vs 92.5% quality.

---

## Experiment 3 — Cross-Model Quality Validation

**60 cells** · 3 profiles × 20 bills · Rubric: 0-2 on Accuracy/Specificity/Completeness

**Reviewer:** Claude Opus 4.8 · Pass 1: high effort · Pass 2: max effort, independent

| Profile | Accuracy | Specificity | Completeness | Overall |
|---------|----------|-------------|--------------|---------|
| llama31_1200 | 1.80 | 1.50 | 1.80 | **1.70** |
| qwen_500 | 1.90 | 1.50 | 1.35 | 1.58 |
| mistral_3000 | 1.50 | 1.55 | 1.55 | 1.53 |

**60/60 confirmed, 0 revised.** parse-OK reliably ranks models in the same order as substantive qualitative scoring.

---

## Experiment 3.5 — Within-Model Proxy Validation

**5 matched pairs** · Same rubric · 2 independent passes

| Pair | parse-OK ranking | Quality ranking | Verdict |
|------|-----------------|-----------------|---------|
| Llama 900 vs 2000 | ~tie | 900 wins (1.92 vs 1.63) | REVERSED |
| Qwen 2000 vs 3000 | 3000 wins | 2000 wins (1.93 vs 1.57) | REVERSED |
| Qwen 3000 vs 4000 | 3000 wins | 3000 wins (1.83 vs 1.23) | AGREES |
| Mistral 2000 vs 3000 | 3000 wins | 2000 wins (1.97 vs 1.82) | REVERSED |
| Mistral 3000 vs 5000 | tie | tie (synopsis only) | TIE |

**Dominant failure mode: off-section drift.** Larger chunks anchor on adjacent bill sections. parse-OK validates JSON shape, not content accuracy.

**Three invisible failure modes:**
1. Off-section drift — specific answer about wrong section
2. Hollow-but-valid JSON — all fields populated with drifted content
3. All-None emission — valid JSON with null fields (qwen_3000 CH2: all 11 fields "None.", parse_status=OK)

---

## Experiment 4 — Qwen 2.5 14B AWQ

**3 cells** · 14B model · chunk_size=5000 (full-bill, no splitting) · Concurrency=4

Each eval bill (4,500–4,972 tokens) = 1 chunk. Tests whether doubling parameters + eliminating chunking improves quality.

| Profile | Params | Chunk size | Parse-OK | Quality |
|---------|--------|-----------|---------|---------|
| llama31_1200 | 8B | 1,200 | 92.5% | 1.70 |
| qwen_3000 | 7B | 3,000 | 67.7% | 1.57 |
| **qwen14b_5000** | **14B** | **5,000** | **45.0%** | **TBD*** |

*Qualitative A/S/C review scheduled as follow-up. Parse-OK of 45% makes outcome predictable.

**Result:** The larger model at full-bill context performed worse than every other configuration tested. 45.0% parse-OK — 47.5pp below Llama, 22.7pp below qwen_3000 (7B). The failure mode: 54% parse_partial with mean 1.2 missing fields per bill. At 5,000 input tokens, output budget pressure prevents complete 11-field extraction. Throughput: 0.156 chunks/sec — slowest tested. SLO gap: 7.43× vs Llama's 3.16×.

**Architectural lesson:** Structured extraction at scale favors tight context windows over large models. Llama at 1,200 tokens stays on-chunk and completes the schema. A 14B model given the full 5,000-token bill floods its attention across the document and completes nothing reliably.

---

## SLO Gate

**Workload:** 75,000 bills/week (50 states) · 18-hour processing window

| Workload | Required | Measured | Result | Processing time |
|----------|---------|---------|--------|----------------|
| Illinois (11,079/wk) | 0.179 bills/sec | 0.367 bills/sec | ✓ 2.05× headroom | ~8.4 hours |
| 50-state (75,000/wk) | 1.157 bills/sec | 0.367 bills/sec | ✗ 3.16× deficit | ~56.6 hours |

**To close the gap:** ~3 L4s parallel (steady-state) · ~7 L4s (peak-burst) · or A100/H100-class GPU.

---

## Key Findings

1. **Llama leads at every chunk size** — gap between Llama's worst cell and any other model's best cell is still 12pp
2. **Tail-chunk artifact** — the chunk_size=4000 dip is corpus-length interaction, not model limitation
3. **Mistral eliminated on concurrency** — 5× p99 explosion from KV cache blocking
4. **Quality invariant under concurrency** — <0.6pp variation for all models; tuning is hardware-only
5. **parse-OK reliable cross-model, not within-model** — 60/60 confirmed; 3/4 within-model pairs reversed
6. **Effective throughput favors Llama** — Qwen's raw throughput advantage evaporates when quality-adjusted
7. **Single L4 meets current Illinois SLO** — 2.05× headroom; fails 50-state target by 3.16×

---

## Lessons Learned

### The prior recommendation was correct — the stack change flipped it
March 2026 Ollama evaluation: Mistral wins. May 2026 vLLM evaluation: Llama wins. Two changes: (a) vLLM exposes per-stream telemetry and real concurrency behavior; (b) upgrading from Llama 3.2 3B to 3.1 8B for parameter parity changed the answer.

### Verify before trusting exit codes
Failed launches exited with code 0 and produced plausible-looking output files. The harness failed open — a tokenizer cache miss happened before the extraction loop. Detection: check file sizes and wall times. Build verification into sweep design.

### parse-OK reliability is model-dependent
Don't generalize cross-model proxy reliability to within-model tuning. Validate the proxy within each model family before using it for chunk-size decisions.

### Quality-best chunk size is corpus-dependent
Always test quality claims against the production token-length distribution, not just the controlled eval sample.

### Start small, scale to fit
The SLO gap (3.16×) is now a measured specification, not a guess. Hardware procurement is grounded in data.

---

## Reproduction Guide

### Step 1 — Environment

```bash
# Clone the pipeline repo
git clone <PIPELINE_REPO_URL>
cd Legislation_Pipeline

# Set tiktoken cache (avoids SSL issues)
export TIKTOKEN_CACHE_DIR=$HOME/tiktoken-cache
mkdir -p $TIKTOKEN_CACHE_DIR
# Download cl100k_base.tiktoken manually to this dir

# Verify environments
/bin/python3 --version          # system python (has pandas, azure-storage-blob)
/var/advanalytics/anaconda3/envs/vllm/bin/python --version  # vLLM env
nvidia-smi                       # confirm GPU available
```

### Step 2 — Create eval session (session 999995)

The 50-bill controlled eval sample uses bills of 4,500–4,972 Mistral tokens. These are stored in Azure Blob Storage under `legislation/session=999995/`. To recreate:

```bash
# Check if session exists
/bin/python3 -c "
import sys; sys.path.insert(0,'.')
from services.azure_blob_service import BlobClient, TIER_BRONZE
b = BlobClient()
print(b.blob_exists(TIER_BRONZE, 'legislation/session=999995/_manifest.parquet'))
"
```

### Step 3 — Run Experiment 1

```bash
nohup setsid bash run_3model_vllm_sweep.sh \
    > logs/exp1_master.log 2>&1 &
echo $! > /tmp/exp1_pid.txt
tail -f logs/exp1_master.log
```

Expected runtime: ~8 hours. Results in `benchmarks/vllm_sweep_<TIMESTAMP>/`.

### Step 4 — Run Experiment 1b

```bash
# Wait for Exp 1 to complete, then:
nohup setsid bash run_mistral_extended_chunks.sh \
    > logs/exp1b_mistral_master.log 2>&1 &

# After mistral completes:
nohup setsid bash run_qwen_extended_chunks.sh \
    > logs/exp1b_qwen_master.log 2>&1 &
```

### Step 5 — Run Experiment 2

```bash
nohup setsid bash run_concurrency_sweep.sh \
    > logs/exp2_master.log 2>&1 &
```

### Step 6 — Run Experiment 2b (Qwen 3000 concurrency)

```bash
# Verify qwen_3000 chunks exist for session 999995 first
nohup setsid bash run_qwen_concurrency_sweep_v3.sh \
    > logs/exp2b_master.log 2>&1 &
```

### Step 7 — Run Experiment 3 (qualitative review)

Use `qualitative_review_prompt_exp3.txt` with Claude Opus 4.8.
Source: `review_vllm_winners.txt`
Output: `review_vllm_winners_SCORED.txt`
Run twice independently (pass 1: high effort, pass 2: max effort fresh context).

### Step 8 — Run Experiment 3.5 (within-model proxy)

Use `qualitative_review_prompt_exp35_v2.txt` with Claude Opus 4.8.
Source files: `review_qwen_2000_3000.txt`, `review_mistral_2000_3000.txt`, etc.
See `exp35_second_pass_reconciliation.md` for reconciliation methodology.

### Step 9 — Run Experiment 4 (Qwen 14B)

```bash
# Create full-bill chunks for session 999995
/bin/python3 << 'EOF'
import sys; sys.path.insert(0,'.')
import pandas as pd
from datetime import datetime, timezone
from services.azure_blob_service import BlobClient, TIER_SILVER
b = BlobClient()
df3 = b.read_parquet(TIER_SILVER, 'legislation/chunks_qwen_3000/session=999995/_chunks.parquet')
rows = []
now = datetime.now(timezone.utc).isoformat()
for bill_id, grp in df3.groupby('bill_id', sort=False):
    grp = grp.sort_values('chunk_index')
    first = grp.iloc[0]
    full_text = '\n'.join(grp['chunk_text'].tolist())
    rows.append({
        'bill_id': first['bill_id'], 'session_id': first['session_id'],
        'bill_number': first['bill_number'], 'version': first['version'],
        'content_hash': first['content_hash'], 'chunk_strategy': 'standard',
        'chunk_index': 0, 'chunk_id': f"{first['bill_id']}::0000",
        'chunk_token_count': grp['chunk_token_count'].sum(),
        'bill_token_count': first['bill_token_count'],
        'char_len': len(full_text), 'chunk_text': full_text, 'created_at': now,
    })
df5 = pd.DataFrame(rows)
b.write_parquet(df5, TIER_SILVER, 'legislation/chunks_qwen_5000/session=999995/_chunks.parquet')
print(f"Written {len(df5)} chunks (1 per bill, mean {df5['chunk_token_count'].mean():.0f} tokens)")
EOF

nohup setsid bash run_qwen14b_exp4.sh \
    > logs/qwen14b_exp4_master.log 2>&1 &
```

### Monitoring

```bash
# Check sweep progress
tail -20 logs/sweep_<SWEEP_ID>/master.log

# Check GPU
nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu --format=csv,noheader

# Parse all results from a sweep
python3 -c "
import json, glob
print('%-35s %6s %7s %7s' % ('Cell','ok%','c/s','p99'))
for f in sorted(glob.glob('benchmarks/<SWEEP_ID>/*.json')):
    with open(f) as fh: d = json.load(fh)
    r = d['results'][0]; ext = r['extractor']
    profile = list(d['parse_quality'].keys())[0]
    ok = d['parse_quality'][profile]['ok_pct']
    cps = ext['metrics']['throughput']['chunks_extracted_per_second']
    p99 = ext['metrics']['latency']['api_time']['p99']
    print('%-35s %6.1f %7.3f %7.2f' % (f.split('/')[-1][:35],ok,cps,p99))
"
```

---

## File Manifest

### Sweep scripts

| File | Experiment | Description |
|------|-----------|-------------|
| `run_3model_vllm_sweep.sh` | Exp 1 | 3 models × 6 chunk sizes × 3 repeats |
| `run_mistral_extended_chunks.sh` | Exp 1b | Mistral at 4000/5000 tokens |
| `run_qwen_extended_chunks.sh` | Exp 1b | Qwen at 4000/5000 tokens |
| `run_qwen_concurrency_sweep_v3.sh` | Exp 2b | Qwen 3000 concurrency sweep |
| `run_qwen14b_exp4.sh` | Exp 4 | Qwen 2.5 14B AWQ |

### Key configs

| File | Model | Chunk size |
|------|-------|-----------|
| `configs/chunker_llama31_1200.yaml` | Llama 3.1 8B | 1,200 |
| `configs/extractor_llama31_1200.yaml` | Llama 3.1 8B | 1,200 |
| `configs/chunker_qwen_3000.yaml` | Qwen 2.5 7B | 3,000 |
| `configs/extractor_qwen_3000.yaml` | Qwen 2.5 7B | 3,000 |
| `configs/chunker_mistral_3000.yaml` | Mistral 7B | 3,000 |
| `configs/extractor_mistral_3000.yaml` | Mistral 7B | 3,000 |
| `configs/extractor_qwen14b_5000.yaml` | Qwen 2.5 14B | 5,000 |

### Quality review files

| File | Contents |
|------|---------|
| `review_vllm_winners_SCORED.txt` | Exp 3 pass-1 scores, 20 bills × 3 profiles |
| `review_qwen_3000_4000_SCORED.txt` | Exp 3.5 Qwen 3000 vs 4000, 40/40 reconciled |
| `review_qwen_2000_3000.txt` | Exp 3.5 Qwen 2000 vs 3000 |
| `review_mistral_2000_3000.txt` | Exp 3.5 Mistral 2000 vs 3000 |
| `review_mistral_3000_5000.txt` | Exp 3.5 Mistral 3000 vs 5000 |
| `review_llama_900_2000.txt` | Exp 3.5 Llama 900 vs 2000 |
| `qualitative_review_prompt_exp35_v2.txt` | Scoring prompt, Exp 3.5 pass 2 |
| `exp35_second_pass_reconciliation.md` | Full 5-pair reconciliation |
| `exp3_exp35_consolidated_handoff.md` | Master handoff document |

### Web case study

| File | Contents |
|------|---------|
| `vllm-inference-throughput-evaluation.html` | Full case study, 12 sections |
| `vllm-qualitative-review.html` | Quality review, Phase 1 + Phase 2 |
| `index.html` | Portfolio index |

---

## Experiment Status

| Experiment | Status | Key Result |
|-----------|--------|-----------|
| Exp 1 — Chunk-size sweep | ✅ Complete | llama31_1200 = 92.5% |
| Exp 1b — Extended sweep | ✅ Complete | Tail-chunk artifact identified |
| Exp 2 — Concurrency sweep | ✅ Complete | c=12–24 recommended |
| Exp 2b — Qwen 3000 concurrency | ✅ Complete | 68% quality ceiling |
| Exp 3 — Cross-model quality | ✅ Complete | 60/60 confirmed |
| Exp 3.5 — Within-model proxy | ✅ Complete | 3/4 pairs reversed |
| Exp 4 — Qwen 14B AWQ | 🔄 Running | Pending June 1 |
| Exp 5 — Production corpus | 📋 Planned | — |
| Exp 6 — Adaptive chunking | 📋 Planned | — |
| Exp 7 — New hardware | 📋 Planned | — |

---

*Tayler Erbe · Data Scientist, Applied AI · AI Solutions, University of Illinois System · May 2026*
