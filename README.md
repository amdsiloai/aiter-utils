# aiter-utils

Utilities for tuning, benchmarking, and analyzing [AITER](https://github.com/ROCm/aiter) kernels on AMD Instinct GPUs.

## Tools

| Script | Description |
|--------|-------------|
| [`fetch_run_logs.py`](fetch_run_logs.py) | Download container logs from Databricks MLflow artifacts |
| [`extract_gemm_shapes.py`](extract_gemm_shapes.py) | Extract untuned GEMM shapes from container logs |
| [`mlflow_client.py`](mlflow_client.py) | Lightweight Databricks MLflow client (shared library) |

## Documentation

- [AITER Kernel Tuning Guide](docs/aiter_tuning_guide.md) -- Step-by-step instructions for tuning blockscale GEMM and Fused MoE kernels, including per-shape microbenchmarking before/after.

## Typical Workflow

```
1. Run benchmarks       ->  Results logged to Databricks MLflow
2. fetch_run_logs.py    ->  Download container logs locally
3. extract_gemm_shapes  ->  Find untuned GEMM shapes from logs
4. AITER tuner          ->  Tune kernels for discovered shapes
5. Re-benchmark         ->  Verify improvement
```

### Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Download logs from a benchmark run
python fetch_run_logs.py \
    --experiment-pattern "Qwen--Qwen3-Next*" \
    --run-pattern "oob_" \
    -o ./logs

# Extract shapes that fell back to default kernels
python extract_gemm_shapes.py ./logs -o shapes/untuned_blockscale.csv

# Feed into AITER tuner (run inside AITER container)
python csrc/ck_gemm_a8w8_blockscale/gemm_a8w8_blockscale_tune.py \
    -i shapes/untuned_blockscale.csv \
    -o aiter/configs/a8w8_blockscale_tuned_gemm.csv \
    --libtype both
```

## Configuration

### Databricks credentials

`fetch_run_logs.py` and `benchmark_analysis.py` require Databricks access.
Create a `.env` file or set environment variables:

```bash
DATABRICKS_WORKSPACE_URL=https://your-workspace.azuredatabricks.net
DATABRICKS_API_KEY=dapi...
```

## Repository Structure

```
aiter-utils/
├── README.md
├── requirements.txt
├── .env.template
├── fetch_run_logs.py          # Download MLflow container logs
├── extract_gemm_shapes.py     # Parse logs for untuned shapes
├── mlflow_client.py           # Lightweight Databricks MLflow client
├── docs/
│   └── aiter_tuning_guide.md  # Full tuning guide
├── logs/                      # Downloaded logs (gitignored)
└── shapes/                    # Extracted shape CSVs (gitignored)
```
