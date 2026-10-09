<p align="center">
  <img src="docs/images/logo.png?raw=true" alt="LoRAForge Logo" width="140" style="border-radius: 28px;" />
</p>

<h1 align="center">LoRAForge</h1>

<p align="center">
  <b>Bare-metal fine-tuning control plane, predictive VRAM supervisor, and Multi-LoRA fusion engine for consumer & workstation GPUs.</b>
  <br />
  <i>Training companion to <a href="https://github.com/alexandrmotologa/inferops">InferOps</a>.</i>
</p>

<p align="center">
  <a href="https://github.com/alexandrmotologa/loraforge/actions"><img src="https://img.shields.io/badge/build-passing-brightgreen.svg" alt="Build Status" /></a>
  <a href="https://github.com/alexandrmotologa/loraforge/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-Apache--2.0-blue.svg" alt="License" /></a>
  <img src="https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-3776AB.svg?logo=python&logoColor=white" alt="Python Versions" />
  <img src="https://img.shields.io/badge/cuda-12.1%20%7C%2012.4-76B900.svg?logo=nvidia&logoColor=white" alt="CUDA Ready" />
  <img src="https://img.shields.io/badge/coverage-90%25-success.svg" alt="Test Coverage" />
</p>

<p align="center">
  <img src="docs/images/loraforge_demo.gif?raw=true" alt="LoRAForge Demo Walkthrough" width="940" style="border-radius: 12px; box-shadow: 0 16px 36px rgba(0,0,0,0.35);" />
</p>

---

## Visual Control Plane & CLI Walkthrough

<table>
  <tr>
    <td width="50%">
      <p align="center"><b>Interactive Web Control Plane</b></p>
      <a href="docs/images/dashboard.png"><img src="docs/images/dashboard.png?raw=true" alt="LoRAForge Web Dashboard" /></a>
      <p align="center"><i>Real-time tensor budget allocation bar & model feasibility verdict.</i></p>
    </td>
    <td width="50%">
      <p align="center"><b>Predictive VRAM Profiler (CLI)</b></p>
      <a href="docs/images/cli_vram.png"><img src="docs/images/cli_vram.png?raw=true" alt="CLI VRAM Profiling" /></a>
      <p align="center"><i>Exact mathematical memory breakdown before tensors touch VRAM.</i></p>
    </td>
  </tr>
  <tr>
    <td width="50%">
      <p align="center"><b>Hardware Auto-Tuner</b></p>
      <a href="docs/images/cli_autotune.png"><img src="docs/images/cli_autotune.png?raw=true" alt="CLI Hardware Auto-Tuner" /></a>
      <p align="center"><i>Binary-search optimization for safe micro-batch and sequence lengths.</i></p>
    </td>
    <td width="50%">
      <p align="center"><b>Multi-LoRA Adapter Fusion</b></p>
      <a href="docs/images/loraforge_adapter_merger.png"><img src="docs/images/loraforge_adapter_merger.png?raw=true" alt="Multi-LoRA Adapter Merger" /></a>
      <p align="center"><i>Consolidate adapters via TIES, DARE, and SVD rank truncation.</i></p>
    </td>
  </tr>
</table>

---

## Overview

Fine-tuning modern large language models on consumer hardware (such as RTX 3060 12GB, RTX 4070 12GB, RTX 3090/4090 24GB) frequently fails due to silent out-of-memory crashes caused by activation surges, optimizer allocation spikes, and fragmentation.

LoRAForge eliminates trial-and-error fine-tuning through:
1. **Mathematical Pre-Flight VRAM Profiling**: Computes exact video memory requirements before allocating tensors or spawning worker processes.
2. **Subprocess Process Supervisor with Watchdog**: Spawns isolated PyTorch + TRL `SFTTrainer` worker processes, streams live JSON metrics, and halts execution via $dV/dt$ tripwires before driver crashes occur.
3. **Multi-LoRA Adapter Fusion Engine**: Merges specialized domain adapters without catastrophic parameter interference using TIES sign-election (NeurIPS 2023), DARE stochastic drop-and-rescale, and SVD rank truncation.
4. **Sequence Multi-Packing**: First-Fit Decreasing (FFD) sequence concatenation groups dialogue turns into fixed context windows with reset position IDs and attention boundaries.
5. **InferOps Handoff**: Generates deployment-ready serving manifests for vLLM, SGLang, and llama.cpp.

---

## Architecture

```
+----------------------------------------------------------------------------------+
|                                    LoRAForge                                     |
|                                                                                  |
|  +-------------------+  +-----------------------+  +--------------------------+  |
|  | GPU Hardware Probe|  | Predictive VRAM Engine|  | Dataset Sequence Packing |  |
|  | NVML / CUDA arch  |  | Exact tensor budgets  |  | Multi-pack, zero-padding |  |
|  +---------+---------+  +-----------+-----------+  +------------+-------------+  |
|            |                        |                           |                |
|            +------------------------+---------------------------+                |
|                                     |                                            |
|                                     v                                            |
|                    +----------------------------------+                          |
|                    |   Fine-Tuning Process Supervisor |                          |
|                    |   OOM Watchdog / Tripwire        |                          |
|                    |   Real-time Telemetry & Metrics  |                          |
|                    +----------------+-----------------+                          |
|                                     |                                            |
|            +------------------------+---------------------------+                |
|            |                                                    |                |
|            v                                                    v                |
|  +-------------------+                                +-----------------------+  |
|  | Multi-LoRA Fusion |                                | InferOps Handoff      |  |
|  | TIES / DARE / SVD |                                | Manifests & configs   |  |
|  +-------------------+                                +-----------------------+  |
+----------------------------------------------------------------------------------+
```

---

## Key Capabilities

* **Predictive Memory Sizing**: Analytical formulas for base weights (NF4 with double-quantization constants, INT8, FP16), LoRA matrices ($A \times B$), optimizer overhead (`paged_adamw_8bit`, `adamw_8bit`, `adamw_torch`, `lion_8bit`), activations under FlashAttention-2 vs. SDPA, and gradient checkpointing.
* **Multi-GPU Distributed Sharding**: Models tensor distributions under DDP, ZeRO-1 ($M_{\text{opt}} / N$), ZeRO-2 ($(M_{\text{opt}} + M_{\text{grad}}) / N$), and ZeRO-3 / FSDP ($M_{\text{base}} / N$, $M_{\text{lora}} / N$, $M_{\text{opt}} / N$, $M_{\text{grad}} / N$ with single-layer gathering buffers).
* **Bare-Metal Worker Execution**: Generates production PyTorch `SFTTrainer` worker scripts with `BitsAndBytesConfig`, `prepare_model_for_kbit_training`, and real adapter saving.
* **Hardware Watchdog Protection**: Monitors VRAM utilization and growth rates ($dV/dt$), tripping safety halts before unrecoverable kernel panics.
* **Turnkey Task Presets**: Hyperparameter configurations tailored for reasoning (DeepSeek-R1 CoT style), code generation, agentic tool calls, and dialogue.
* **Dataset Sanitization**: Regex Unicode control-character stripping, empty turn pruning, and SHA-256 deduplication.
* **Specialized Chat Templates**: Formats data for Meta Llama 3, ChatML (Qwen, Hermes), Mistral, DeepSeek-R1, Google Gemma 2, and Microsoft Phi 3.
* **Multi-LoRA Adapter Merger**: Combines weights in NumPy with zero-copy Safetensors I/O using TIES sign-election, DARE rescaled masking, and SVD truncation.
* **Dual Control Plane**: Fully featured Typer CLI and responsive FastAPI + WebSocket control plane.

---

## Installation

```bash
git clone https://github.com/alexandrmotologa/loraforge.git
cd loraforge
python -m venv .venv

# On Linux / macOS:
source .venv/bin/activate
# On Windows:
.\.venv\Scripts\activate

pip install -e ".[dev]"
```

---

## Quick Start

### 1. Profile Training VRAM on Target GPU

```bash
loraforge vram \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --gpu rtx-3060-12gb \
  --seq-len 2048 \
  --batch-size 1 \
  --grad-acc 8 \
  --rank 16 \
  --quant nf4
```

### 2. Auto-Tune Hyperparameters for Zero-OOM

```bash
loraforge autotune \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --gpu rtx-3060-12gb \
  --effective-batch 16 \
  --seq-len 2048
```

### 3. Launch Supervised Training Run

```bash
# Bare-metal execution with live telemetry:
loraforge train \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --gpu rtx-3060-12gb \
  --dataset data/train.jsonl \
  --no-dry-run \
  --steps 100

# Or dry-run simulation mode for CI / systems without local GPUs:
loraforge train --steps 25 --dry-run
```

### 4. Merge Multi-LoRA Adapters (TIES / DARE / SVD)

```bash
loraforge merge \
  --adapters ./runs/adapter_code ./runs/adapter_math \
  --method ties \
  --density 0.2 \
  --out ./runs/merged_agent
```

### 5. Launch the Web Control Plane

```bash
loraforge serve --port 8000
```
Open [http://localhost:8000](http://localhost:8000) to view the live dashboard.

---

## Programmatic Python API

```python
from loraforge.core.models import PRESET_MODELS
from loraforge.core.hardware import PRESET_GPUS
from loraforge.core.config import (
    LoRAConfig,
    QuantizationConfig,
    QuantizationType,
    TrainingConfig,
)
from loraforge.vram import VRAMCalculator

model = PRESET_MODELS["meta-llama/Llama-3.1-8B-Instruct"]
gpu = PRESET_GPUS["rtx-3060-12gb"]

calc = VRAMCalculator()
report = calc.estimate(
    model=model,
    lora=LoRAConfig(r=16),
    quant=QuantizationConfig(quant_type=QuantizationType.NF4, double_quant=True),
    training=TrainingConfig(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        max_seq_length=2048,
        gradient_checkpointing=True,
    ),
    gpu=gpu,
)

print(f"Status: {report.status}")
print(f"Peak VRAM: {report.peak_vram_mb:.1f} MB / {gpu.total_vram_mb} MB")
print(f"Headroom: {report.headroom_mb:.1f} MB")
```

---

## Brand Identity: The Vulcan Forge Ram

<p align="center">
  <img src="docs/images/logo.png?raw=true" alt="LoRAForge Vulcan Ram" width="160" style="border-radius: 32px;" />
</p>

LoRAForge uses the **Forge Ram (The Vulcan Ram)** as its mascot. In fine-tuning, massive parameter spaces are shaped into dense, resilient low-rank matrices through computational force. The curved low-poly horns symbolize the dual low-rank adapter matrices ($A \times B$) wrapping around the foundation model, while the obsidian and cyan armor reflects mathematical stability under thermal and memory constraints.

---

## Documentation

* [Architecture & Subsystems](docs/ARCHITECTURE.md)
* [Mathematical VRAM Formulations](docs/VRAM_FORMULAS.md)
* [CLI Reference](docs/CLI_REFERENCE.md)
* [REST & WebSocket API Reference](docs/API_REFERENCE.md)
* [InferOps Integration Guide](docs/INFEROPS_INTEGRATION.md)

---

## Testing

Run the test suite with coverage:

```bash
pytest --cov=loraforge --cov-report=term-missing tests
```

---

## License

Apache-2.0
