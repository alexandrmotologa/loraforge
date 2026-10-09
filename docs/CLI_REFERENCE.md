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

### `loraforge presets`

Lists turnkey fine-tuning task presets (reasoning, coding, agentic, chat) with recommended models and hyperparameters.

```bash
loraforge presets
```

---

### `loraforge init`

Initializes a reproducible fine-tuning job YAML configuration from a task preset.

```bash
loraforge init --preset reasoning --gpu rtx-4090-24gb --out loraforge.yaml
```

---

### `loraforge cloud`

Generates a containerized deployment bundle (CUDA 12.4 Dockerfile and entrypoint script) for RunPod or Vast.ai.

```bash
loraforge cloud --model meta-llama/Llama-3.1-8B-Instruct --gpu rtx-4090-24gb --out ./cloud_deploy --provider runpod
```

---

### `loraforge data clean`

Sanitizes dataset files: removes exact duplicates via SHA-256 fingerprinting, strips invisible control characters, and prunes degenerate single-turn samples.

```bash
loraforge data clean dataset.jsonl --output cleaned.jsonl --format sharegpt
```

---

### `loraforge data preview`

Previews formatted prompts rendered with foundation model chat templates (Llama 3, ChatML, Mistral, DeepSeek-R1, Gemma 2, Phi 3).

```bash
loraforge data preview dataset.jsonl --template llama3 --samples 2
```

---

### `loraforge merge`

Fuses multiple specialized LoRA adapters or compresses adapter rank using TIES, DARE, SVD rank truncation, or Linear interpolation:

```bash
# TIES merging across two specialized adapters
loraforge merge \
  --adapters ./runs/coding_lora,./runs/reasoning_lora \
  --out ./merged_adapters \
  --method ties \
  --density 0.2

# DARE stochastic drop-and-rescale
loraforge merge \
  --adapters ./runs/adapter_a,./runs/adapter_b \
  --out ./merged_adapters \
  --method dare \
  --drop-rate 0.5

# SVD rank truncation (compress rank 64 to rank 16)
loraforge merge \
  --adapters ./runs/heavy_lora \
  --out ./compressed_adapter \
  --method svd \
  --target-rank 16
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

