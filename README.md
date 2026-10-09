# LoRAForge

Bare-metal local fine-tuning control plane and QLoRA supervisor for consumer and workstation GPUs. Training companion to [InferOps](https://github.com/alexandrmotologa/inferops).

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
|  | Safetensors Merge |                                | InferOps Handoff      |  |
|  | FP16 / BF16       |                                | Manifests & configs   |  |
|  +-------------------+                                +-----------------------+  |
+----------------------------------------------------------------------------------+
```

## Overview

Fine-tuning large language models on consumer GPUs (RTX 3060 12GB, RTX 4070 12GB, RTX 3090/4090 24GB) often fails due to unexpected out-of-memory errors caused by dynamic activation spikes and unoptimized optimizer states.

LoRAForge provides analytical memory sizing before any tensors touch GPU memory, supervises training execution with early memory tripwires, packs dataset sequences to eliminate padding waste, and exports deployment-ready manifests directly into InferOps.

## Capabilities

- **Predictive VRAM Calculator**: Computes exact video memory requirements before training starts. Models base weights (4-bit NF4 with Double Quantization, 8-bit, 16-bit), LoRA adapter decomposition matrices across all attention and MLP projections, gradient buffers, optimizer states (`paged_adamw_8bit`, `adamw_8bit`, `adamw_torch`, `lion_8bit`), and FlashAttention-2 activation memory under gradient checkpointing.
- **Hardware Telemetry & Arch Detection**: Probes NVIDIA GPUs via NVML and CUDA runtime. Detects compute capability, Tensor Core generation, FlashAttention-2 compatibility, and physical headroom. Includes reference presets for remote workstation planning.
- **Sequence Multi-Packing**: First-Fit Decreasing (FFD) bin packing groups short dialogue samples into fixed `max_seq_length` blocks with reset position IDs and cumulative length attention boundaries, eliminating padding waste.
- **Process Supervisor & Watchdog**: Tracks VRAM growth rate ($dV/dt$) and halts or warns before an unrecoverable CUDA out-of-memory driver crash occurs.
- **InferOps Integration**: Exports merged Safetensors (FP16/BF16) or GGUF quantization plans alongside native InferOps service definitions (`inferops_manifest.yaml`) for instant multi-engine serving.
- **Typer CLI & Web Control Plane**: Run interactive CLI audits or monitor loss curves and parameter sweeps through a local FastAPI dashboard on port 8765.

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

## Quick Start

### 1. Profile Training VRAM on Target Hardware

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

Example output:
```
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━┓
┃ Component                        ┃ Formula / Source                 ┃ Allocation (MB) ┃   % of Peak ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━┩
│ Base Model Weights               │ NF4 Quantized Weights            │      4894.7 MB  │       57.6% │
│ Quantization Metadata            │ Block scales + Double-Quant      │       155.1 MB  │        1.8% │
│ LoRA Adapter Matrices            │ Rank=16 (41,943,040 params)      │        80.0 MB  │        0.9% │
│ Gradient Buffers                 │ FP16/BF16 Trainable Gradients    │        80.0 MB  │        0.9% │
│ Optimizer States                 │ paged_adamw_8bit Moments Buffer  │        80.3 MB  │        0.9% │
│ Activation Memory                │ Checkpointing=True, FlashAttn=True│      2234.4 MB  │       26.3% │
│ CUDA Context & Workspace         │ CUDA CC 8.6 Runtime Baseline     │       680.0 MB  │        8.0% │
│ Allocator Fragmentation          │ 12% Dynamic Tensor Headroom      │       287.3 MB  │        3.4% │
├──────────────────────────────────┼──────────────────────────────────┼─────────────────┼─────────────┤
│ Total Predicted Peak VRAM        │ Sum of components                │      8491.8 MB  │ 69.1% of GPU│
│ Physical GPU Memory              │ Available: 11520.0 MB safe       │     12288.0 MB  │      100.0% │
│ Free Headroom                    │ Remaining margin before OOM      │      3796.2 MB  │       30.9% │
└──────────────────────────────────┴──────────────────────────────────┴─────────────────┴─────────────┘
```

### 2. Auto-Tune Hyperparameters for a Specific GPU

```bash
loraforge autotune --model meta-llama/Llama-3.1-8B-Instruct --gpu rtx-3060-12gb
```

### 3. Audit Dataset Token Lengths and Packing

```bash
loraforge data audit dataset.jsonl --format sharegpt --seq-len 2048
```

### 4. Supervise a Fine-Tuning Run

```bash
loraforge train \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --gpu rtx-3060-12gb \
  --steps 100 \
  --dry-run
```

### 5. Export to InferOps

```bash
loraforge export \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --out ./exports/llama3-custom \
  --engine vllm
```

Launch directly in InferOps:
```bash
inferops serve --manifest ./exports/llama3-custom/inferops_manifest.yaml --engine vllm --port 8000
```

### 6. Start the Web Control Plane

```bash
loraforge serve --host 127.0.0.1 --port 8765
```

Open `http://127.0.0.1:8765` in your browser to inspect interactive stacked memory bars, adjust rank and context sliders in real time, and monitor loss curves over WebSockets.

## Python API

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

## Documentation

- [Architecture & Subsystems](docs/ARCHITECTURE.md)
- [Mathematical VRAM Formulations](docs/VRAM_FORMULAS.md)
- [CLI Reference](docs/CLI_REFERENCE.md)
- [REST & WebSocket API Reference](docs/API_REFERENCE.md)
- [InferOps Integration Guide](docs/INFEROPS_INTEGRATION.md)

## Testing

Run the test suite:

```bash
pytest --cov=loraforge --cov-report=term-missing tests
```

## License

Apache-2.0
