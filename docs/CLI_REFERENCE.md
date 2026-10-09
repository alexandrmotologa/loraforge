# LoRAForge CLI Reference

The LoRAForge command-line interface provides tools for predicting memory allocations, inspecting accelerators, auditing training datasets, supervising fine-tuning runs, and launching the web control plane.

```bash
loraforge [OPTIONS] COMMAND [ARGS]...
```

## Commands

### `loraforge vram` / `loraforge estimate`

Calculates component-level video memory consumption and assesses hardware feasibility.

```bash
loraforge vram \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --gpu rtx-3060-12gb \
  --seq-len 2048 \
  --batch-size 1 \
  --grad-acc 8 \
  --rank 16 \
  --quant nf4 \
  --optimizer paged_adamw_8bit \
  --grad-ckpt \
  --flash-attn
```

**Options:**
- `-m, --model TEXT`: Model architecture identifier or path to directory with `config.json` [default: `meta-llama/Llama-3.1-8B-Instruct`]
- `-g, --gpu TEXT`: Target GPU identifier or reference preset [default: `rtx-3060-12gb`]
- `-s, --seq-len INTEGER`: Maximum training sequence length [default: `2048`]
- `-b, --batch-size INTEGER`: Per-device micro-batch size [default: `1`]
- `-a, --grad-acc INTEGER`: Gradient accumulation steps [default: `8`]
- `-r, --rank INTEGER`: LoRA decomposition rank [default: `16`]
- `-q, --quant TEXT`: Quantization method (`nf4`, `fp4`, `int8`, `none`) [default: `nf4`]
- `-o, --optimizer TEXT`: Optimizer implementation (`paged_adamw_8bit`, `adamw_8bit`, `adamw_torch`, `lion_8bit`) [default: `paged_adamw_8bit`]
- `--grad-ckpt / --no-grad-ckpt`: Enable or disable gradient checkpointing [default: `--grad-ckpt`]
- `--flash-attn / --no-flash-attn`: Enable or disable FlashAttention-2 [default: `--flash-attn`]

---

### `loraforge autotune`

Finds the highest-performing feasible hyperparameter configuration for a target GPU.

```bash
loraforge autotune \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --gpu rtx-3060-12gb \
  --effective-batch 16 \
  --seq-len 2048
```

---

### `loraforge hardware`

Inspects locally attached NVIDIA GPUs via NVML/SMI and lists supported reference presets.

```bash
loraforge hardware
```

---

### `loraforge models`

Displays built-in foundation model architecture specifications (layers, hidden sizes, MLP dimensions, head ratios).

```bash
loraforge models
```

---

### `loraforge data audit`

Analyzes token distribution, percentiles (p50, p90, p95, p99), truncation rates, and estimated step milestones.

```bash
loraforge data audit dataset.jsonl --format sharegpt --seq-len 2048
```

---

### `loraforge train`

Starts a supervised training job with live console telemetry and early OOM tripwires.

```bash
loraforge train \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --gpu rtx-3060-12gb \
  --steps 100 \
  --dry-run
```

---

### `loraforge export`

Consolidates adapters and produces deployment manifests for InferOps.

```bash
loraforge export \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --out ./exports/llama3-fine-tuned \
  --engine vllm
```

---

### `loraforge serve`

Launches the FastAPI control plane and responsive Web Dashboard.

```bash
loraforge serve --host 127.0.0.1 --port 8765
```
