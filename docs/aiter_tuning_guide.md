# AITER Kernel Tuning Guide: Blockscale GEMM & Fused MoE

Target hardware: **AMD Instinct MI300X / MI355X**
Quantization: **FP8 (A8W8) with 128x128 block scaling**

---

## Overview

AITER ships with default kernel selections that work across all shapes but are not optimal
for any specific model. Tuning selects the fastest CK/CK-Tile kernel variant per (M, N, K)
shape for two operator families:

| Operator | What it tunes | Config files | AITER reference |
|----------|--------------|--------------|-----------------|
| **Blockscale GEMM** | Dense linear projections (QKV, output, gate/up/down for shared expert) | `a8w8_blockscale_{un,}tuned_gemm.csv` | [csrc/ck_gemm_a8w8_blockscale](https://github.com/ROCm/aiter/tree/main/csrc/ck_gemm_a8w8_blockscale) |
| **Fused MoE (FMoE)** | Two-stage MoE dispatch (gate + expert GEMM) | `{un,}tuned_fmoe.csv` | [csrc/ck_gemm_moe_2stages_codegen](https://github.com/ROCm/aiter/tree/main/csrc/ck_gemm_moe_2stages_codegen) |

Tuning is **per-GPU-architecture** (gfx942, gfx950, ...) and should be re-run when
switching hardware or significantly changing AITER versions.

---

## Prerequisites

```bash
# Working AITER install (develop mode recommended for tuning)
pip uninstall -y amd-aiter
rm -rf aiter/jit/build/* aiter/jit/__pycache__ aiter/jit/*.so
AITER_REBUILD=1 python3 setup.py develop
```

---

## 1. Blockscale GEMM Tuning

> AITER upstream docs: [ck_gemm_a8w8_blockscale README](https://github.com/ROCm/aiter/tree/main/csrc/ck_gemm_a8w8_blockscale)

### 1.1 Input shapes

The untuned shape list lives at `aiter/configs/a8w8_blockscale_untuned_gemm.csv`:

```
M,N,K
16, 1536, 7168
16, 576, 7168
16, 7168, 256
...
```

Each row is a (batch, output_dim, input_dim) shape that appears in the model's dense
linear layers. See `aiter/configs/model_configs/` for model-specific shape list examples.

### 1.2 Microbenchmark BEFORE tuning

Run the per-shape benchmark with the **default** (untuned) kernel selection to establish
a baseline. This tests every shape in the CSV and reports throughput (TFLOPS) and
bandwidth (GB/s) per shape:

```bash
python3 op_tests/test_gemm_a8w8_blockscale.py --ck_preshuffle False \
    2>&1 | tee blockscale_before_tuning.log
```

The output contains per-shape lines like:

```
[CK  ] (  M=  16, N= 1536, K= 7168): 0.45 TFLOPS, 312.5 GB/s, err=0.0
[TILE] (  M=  16, N= 1536, K= 7168): 0.52 TFLOPS, 356.2 GB/s, err=0.0
```

### 1.3 Run the tuner

The tuner sweeps all candidate CK and CK-Tile kernels for each shape and writes the
winner to the output CSV:

```bash
python3 csrc/ck_gemm_a8w8_blockscale/gemm_a8w8_blockscale_tune.py \
    -i aiter/configs/a8w8_blockscale_untuned_gemm.csv \
    -o aiter/configs/a8w8_blockscale_tuned_gemm.csv \
    --libtype both
```

**Options:**

| Flag | Description |
|------|-------------|
| `--libtype both` | Tune across both CK and CK-Tile candidates (recommended) |
| `--libtype ck` | Only tune CK kernels (faster, may miss CK-Tile wins at high M) |
| `--libtype cktile` | Only tune CK-Tile kernels |

The tuned CSV gains additional columns (`cu_num`, `libtype`, `kernelId`, `splitK`, `us`,
`kernelName`, `tflops`, `bw`, `errRatio`).

**Runtime:** ~10-30 minutes depending on the number of shapes.

### 1.4 Rebuild AITER with tuned config

After tuning, rebuild so the JIT modules pick up the new CSV:

```bash
AITER_REBUILD=1 python3 setup.py develop
```

### 1.5 Microbenchmark AFTER tuning

Re-run the same benchmark to measure improvement:

```bash
python3 op_tests/test_gemm_a8w8_blockscale.py --ck_preshuffle False \
    2>&1 | tee blockscale_after_tuning.log
```

### 1.6 Compare results

```bash
diff blockscale_before_tuning.log blockscale_after_tuning.log
```

Or extract TFLOPS per shape for a side-by-side comparison:

```bash
grep -E "^\[" blockscale_before_tuning.log > /tmp/before.txt
grep -E "^\[" blockscale_after_tuning.log  > /tmp/after.txt
paste /tmp/before.txt /tmp/after.txt | column -t
```

---

## 2. Fused MoE (FMoE) Tuning

> AITER upstream docs: [ck_gemm_moe_2stages_codegen](https://github.com/ROCm/aiter/tree/main/csrc/ck_gemm_moe_2stages_codegen)

### 2.1 Input shapes

The untuned shape list lives at `aiter/configs/untuned_fmoe.csv`:

```
token,model_dim,inter_dim,expert,topk,act_type,dtype,q_dtype_a,q_dtype_w,q_type,use_g1u1,doweight_stage1
512,6144,4096,8,2,ActivationType.Silu,torch.bfloat16,torch.float8_e4m3fn,...
```

For FP8 blockscale MoE models, the relevant rows typically use:
- `q_dtype_a=torch.float8_e4m3fn`, `q_dtype_w=torch.float8_e4m3fn`
- `q_type=QuantType.per_1x128` (blockscale)
- `use_g1u1=1` (gate-up fused), `act_type=ActivationType.Silu`

### 2.2 Microbenchmark BEFORE tuning

Use the MoE blockscale test to benchmark per-shape before tuning:

```bash
python3 op_tests/test_moe_blockscale.py \
    2>&1 | tee fmoe_before_tuning.log
```

For a more targeted benchmark matching a specific model configuration, pass the
model's MoE dimensions directly:

```bash
python3 op_tests/test_moe_blockscale.py \
    --E <num_experts> --topk <top_k> --model_dim <dim> --inter_dim <inter_dim> \
    --dtype torch.bfloat16 --quant_type per_1x128 \
    2>&1 | tee fmoe_model_before_tuning.log
```

### 2.3 Run the tuner

```bash
AITER_REBUILD=1 python3 csrc/ck_gemm_moe_2stages_codegen/gemm_moe_tune.py \
    -i aiter/configs/untuned_fmoe.csv \
    -o aiter/configs/tuned_fmoe.csv
```

The tuner evaluates CK 2-stage codegen kernels and ASM kernels (32x128, 64x256) for
each shape, selecting the best combination for stage-1 (gate+up GEMM) and stage-2
(down GEMM).

**Runtime:** ~30-60 minutes. The FMoE tuner is significantly slower than the GEMM tuner
due to the combinatorial search over two stages.

### 2.4 Rebuild AITER with tuned config

```bash
AITER_REBUILD=1 python3 setup.py develop
```

### 2.5 Microbenchmark AFTER tuning

```bash
python3 op_tests/test_moe_blockscale.py \
    2>&1 | tee fmoe_after_tuning.log
```

### 2.6 Compare results

```bash
diff fmoe_before_tuning.log fmoe_after_tuning.log
```

---

## 3. Full Tuning Workflow (Quick Reference)

```bash
# ── Step 1: Baseline microbenchmarks ──────────────────────────
python3 op_tests/test_gemm_a8w8_blockscale.py --ck_preshuffle False \
    2>&1 | tee blockscale_before.log

python3 op_tests/test_moe_blockscale.py \
    2>&1 | tee fmoe_before.log

# ── Step 2: Tune blockscale GEMMs ────────────────────────────
python3 csrc/ck_gemm_a8w8_blockscale/gemm_a8w8_blockscale_tune.py \
    -i aiter/configs/a8w8_blockscale_untuned_gemm.csv \
    -o aiter/configs/a8w8_blockscale_tuned_gemm.csv \
    --libtype both

# ── Step 3: Tune Fused MoE ───────────────────────────────────
AITER_REBUILD=1 python3 csrc/ck_gemm_moe_2stages_codegen/gemm_moe_tune.py \
    -i aiter/configs/untuned_fmoe.csv \
    -o aiter/configs/tuned_fmoe.csv

# ── Step 4: Rebuild AITER ────────────────────────────────────
AITER_REBUILD=1 python3 setup.py develop

# ── Step 5: Post-tuning microbenchmarks ──────────────────────
python3 op_tests/test_gemm_a8w8_blockscale.py --ck_preshuffle False \
    2>&1 | tee blockscale_after.log

python3 op_tests/test_moe_blockscale.py \
    2>&1 | tee fmoe_after.log

# ── Step 6: Back up tuned configs ────────────────────────────
cp aiter/configs/a8w8_blockscale_tuned_gemm.csv \
    aiter/configs/model_configs/a8w8_blockscale_tuned_gemm_$(hostname)_$(date +%Y%m%d).csv
cp aiter/configs/tuned_fmoe.csv \
    aiter/configs/model_configs/tuned_fmoe_$(hostname)_$(date +%Y%m%d).csv
```

---

## 4. End-to-End Validation

After tuning, validate that accuracy is preserved and measure serving throughput.

### 4.1 Accuracy (lm_eval)

Start the vLLM server with AITER enabled:

```bash
VLLM_ROCM_USE_AITER=1 \
vllm serve <MODEL> \
    --gpu-memory-utilization 0.95 \
    --host 0.0.0.0 \
    --port 8989 \
    --tensor-parallel-size 1
```

Run lm_eval against the server:

```bash
lm_eval --model local-completions \
    --tasks gsm8k \
    --model_args model=<MODEL>,base_url=http://localhost:8989/v1/completions,num_concurrent=16,max_retries=3,tokenized_requests=False
```

Compare the accuracy score against the pre-tuning baseline to confirm no regression.

### 4.2 Serving throughput

```bash
vllm bench serve \
    --backend vllm \
    --base-url http://localhost:8989 \
    --model <MODEL> \
    --dataset-name random \
    --input-len 1024 \
    --output-len 1024 \
    --num-prompts 32 \
    --num-warmup-requests 4 \
    --max-concurrency 4 \
    --percentile-metrics ttft,tpot,itl,e2el
```

Repeat with `--max-concurrency 8,16,32,64,128` and `--input-len/--output-len` combinations
(1k/1k, 1k/8k, 8k/1k) for a full sweep.

---

## 5. Tuning Tips

- **Always benchmark before AND after** -- some shapes may regress if the tuner
  picks a kernel that is fast in isolation but causes cache/register pressure under
  concurrent execution.
- **CK-Tile wins at large M** (high batch/concurrency), CK wins at small M (decode).
  Use `--libtype both` to get the best of both worlds.
- **FMoE tuning is more impactful** than GEMM tuning for MoE models because MoE
  layers dominate compute time.
- **Back up tuned CSVs** -- they are hardware-specific and take significant time to
  regenerate.
- **Accuracy check is mandatory** -- CK-Tile kernels have occasionally produced
  incorrect results for certain shapes (see the quality table in the notes). Always
  run lm_eval after tuning.

---

## 6. Discovering Untuned Shapes from Production Logs

When running vLLM in production, AITER logs a warning for every GEMM shape that
has no entry in the tuned CSV. These warnings can be harvested to build a
model-specific tuning shape list.

### 6.1 Fetch container logs from Databricks MLflow

If benchmark runs are tracked in Databricks, use `fetch_run_logs.py` to bulk-download
the server container logs:

```bash
python fetch_run_logs.py \
    --experiment-pattern "Qwen--Qwen3-Next*" \
    --run-pattern "oob_" \
    --log-pattern "docker_container_*.log" \
    -o ./logs
```

This downloads all matching `docker_container_*.log` artifacts organized by
experiment / run / child-run into `./logs/`.

### 6.2 Extract tuned & untuned shapes

Scan the downloaded logs for both the startup tuning table (already tuned shapes)
and "not found tuned config" warnings (untuned shapes):

```bash
python extract_gemm_shapes.py ./logs \
    --gemm-type a8w8_blockscale \
    -o ./shapes
```

This produces three CSVs in the output directory:

| File | Contents |
|------|----------|
| `a8w8_blockscale_tuned_shapes.csv` | Already tuned shapes with full kernel details (cu_num, M, N, K, libtype, kernelId, splitK, us, kernelName, tflops, bw, errRatio) |
| `a8w8_blockscale_untuned_shapes.csv` | Shapes that fell back to default config (M, N, K only) |
| `a8w8_blockscale_all_shapes.csv` | Combined view with a `status` column (`tuned`, `untuned`, or `both`) |

The tuned shapes are collected so they can be **re-tuned after AITER updates** to
pick up kernel improvements (new CK/CK-Tile variants, better heuristics, etc.).

To re-tune all shapes (tuned + untuned) after an AITER upgrade:

```bash
# Strip the status column to get a clean M,N,K CSV for the tuner
cut -d, -f1-3 shapes/a8w8_blockscale_all_shapes.csv > /tmp/all_shapes.csv

python csrc/ck_gemm_a8w8_blockscale/gemm_a8w8_blockscale_tune.py \
    -i /tmp/all_shapes.csv \
    -o aiter/configs/a8w8_blockscale_tuned_gemm.csv \
    --libtype both
```

To tune only the untuned shapes (the `_untuned_shapes.csv` has `M,N,K` only and
can be fed directly):

```bash
python csrc/ck_gemm_a8w8_blockscale/gemm_a8w8_blockscale_tune.py \
    -i shapes/a8w8_blockscale_untuned_shapes.csv \
    -o aiter/configs/a8w8_blockscale_tuned_gemm.csv \
    --libtype both
```

### 6.3 Merge with existing tuned configs

If you have an existing tuned CSV and only want to add newly-discovered shapes:

```bash
# Existing tuned shapes
cut -d, -f2-4 aiter/configs/a8w8_blockscale_tuned_gemm.csv | tail -n+2 | sort -u > /tmp/existing.txt

# Newly discovered shapes
tail -n+2 shapes/a8w8_blockscale_untuned_shapes.csv | sort -u > /tmp/new.txt

# Shapes that need tuning (not already covered)
comm -23 /tmp/new.txt /tmp/existing.txt > /tmp/missing.txt

echo "Shapes needing tuning: $(wc -l < /tmp/missing.txt)"
```

---

## Appendix: Config File Reference

| File | Description |
|------|-------------|
| [`csrc/ck_gemm_a8w8_blockscale/`](https://github.com/ROCm/aiter/tree/main/csrc/ck_gemm_a8w8_blockscale) | Blockscale GEMM tuning infrastructure (tuner, kernels, README) |
| [`csrc/ck_gemm_moe_2stages_codegen/`](https://github.com/ROCm/aiter/tree/main/csrc/ck_gemm_moe_2stages_codegen) | Fused MoE tuning infrastructure |
| `aiter/configs/a8w8_blockscale_untuned_gemm.csv` | Input shapes for blockscale GEMM tuning (M, N, K) |
| `aiter/configs/a8w8_blockscale_tuned_gemm.csv` | Tuned blockscale GEMM results |
| `aiter/configs/untuned_fmoe.csv` | Input shapes for FMoE tuning |
| `aiter/configs/tuned_fmoe.csv` | Tuned FMoE results |
| `aiter/configs/model_configs/` | Model-specific tuned configs |
| `op_tests/test_gemm_a8w8_blockscale.py` | Blockscale GEMM microbenchmark |
| `op_tests/test_moe_blockscale.py` | Fused MoE microbenchmark |
