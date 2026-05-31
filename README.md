# Legislation_Pipeline

End-to-end data pipeline for the AITS Legislative Intelligence Platform.

Takes raw bills from LegiScan and the Illinois General Assembly, extracts
structured features using locally-hosted Ollama, builds embeddings for
RAG, and lands the result in Oracle for the production intelligence
platform (LegislationTrends2026 App Service).

## Stages

The pipeline runs in this order. Each stage is an independent module
under `stages/` that can be invoked alone for testing.

1. **ingestion** — pull bills from LegiScan + IL GA APIs → bronze blob (raw JSON)
2. **chunker** — token-based chunking with overlap → silver blob (parquet)
3. **extractor** — Ollama subprocess calls per chunk → silver blob (raw LLM output)
4. **parser** — labeled-field parser → silver blob (structured rows)
5. **standardizer** — column rename, type coercion → gold blob (final features parquet)
6. **embedder** — sentence embeddings + FAISS index → indexes blob
7. **db_writer** — gold features → Oracle production tables

## Storage model

- **Bronze** (Azure Blob): raw bills, immutable, partitioned by ingestion date
- **Silver** (Azure Blob): chunked and intermediate parquets
- **Gold** (Azure Blob + Oracle): final standardized features
- **Local `storage/`**: ephemeral working files during a pipeline run

## How to run

Single stage (for development/debug):
```bash
python -m stages.chunker --config configs/pipeline.yaml
```

Full pipeline (what cron runs):
```bash
python -m orchestration.main
```

## Setup

See [SETUP.md](SETUP.md) for one-time server setup (env vars, Oracle client,
Ollama paths, etc.).

## Plug-ins

- **Pipeline logger** (`utils/pipeline_logger.py`) — writes parquet logs to
  `logs/pipeline_log.parquet`. The job execution monitor reads these
  automatically; no registration step needed.
- **Azure Blob** (`services/azure_blob_service.py`) — read/write to
  bronze/silver/gold containers in `leganalyticsstorage`.
- **Ollama** (`services/ollama_service.py`) — subprocess wrapper for the
  local Ollama CLI at `/usr/local/bin/ollama`.
- **Oracle** (`services/oracle_service.py`) — connection + write helpers
  for the dsstag01 Oracle instance.
