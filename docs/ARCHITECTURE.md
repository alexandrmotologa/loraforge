# LoRAForge Architecture

LoRAForge provides a local fine-tuning control plane and QLoRA supervisor for consumer and workstation GPUs. It serves as the training companion to [InferOps](https://github.com/alexandrmotologa/inferops), which handles inference serving across vLLM, SGLang, and llama.cpp.

```
                                 +-----------------------+
                                 |  Dataset Ingestion    |
                                 |  Alpaca / ShareGPT    |
                                 +-----------+-----------+
                                             |
                                             v
+-----------------------+        +-----------------------+
| GPU Hardware Probe    |        | Sequence Multi-Packer |
| NVML / SMI / CC arch  |        | FFD Bin Packing       |
+-----------+-----------+        +-----------+-----------+
            |                                |
            +----------------+---------------+
                             |
                             v
                 +-----------------------+
                 | Predictive VRAM Engine|
                 | Exact tensor budgets  |
                 +-----------+-----------+
                             |
                             v
                 +-----------------------+
                 | Fine-Tuning Supervisor|
                 | Watchdog & Tripwires  |
                 +-----------+-----------+
                             |
            +----------------+---------------+
            |                                |
            v                                v
+-----------------------+        +-----------------------+
| Merged Safetensors    |        | InferOps Manifest     |
| FP16 / BF16 Weights   |        | Serving Config YAML   |
+-----------------------+        +-----------------------+
```

## System Components

### 1. Hardware Probe (`loraforge.core.hardware`)
Queries attached NVIDIA GPUs through direct NVML bindings or `nvidia-smi`. Extracts:
- Physical VRAM capacity and current driver reservation.
- CUDA compute capability (e.g. 8.6 for Ampere, 8.9 for Ada Lovelace).
- Hardware support for BFloat16 math and FlashAttention-2 kernels.
- Curated reference specifications for remote and simulated planning (RTX 3060 12GB through RTX 4090 24GB and A100/H100 80GB).

### 2. Predictive VRAM Engine (`loraforge.vram.calculator`)
Computes video memory allocation before any tensors are loaded into GPU memory. Evaluates:
- Quantized base model parameters in 4-bit NormalFloat (NF4) with Double Quantization.
- Trainable low-rank decomposition matrices across attention and MLP projections.
- Trainable gradient buffers.
- Optimizer momentum and variance states across `paged_adamw_8bit`, `adamw_8bit`, and `adamw_torch`.
- Activation memory comparing FlashAttention-2 and standard SDPA, with and without activation gradient checkpointing.
- CUDA runtime context baseline and PyTorch caching allocator fragmentation overhead.

### 3. Hyperparameter Optimizer (`loraforge.vram.optimizer`)
Finds the largest safe sequence length, batch size, and LoRA rank that fit within the target GPU memory envelope while keeping a safety headroom for display servers.

### 4. Dataset Packing Pipeline (`loraforge.data`)
Parses heterogeneous datasets in ShareGPT, Alpaca, ChatML, and raw completion formats. Applies First-Fit Decreasing (FFD) bin packing to group short conversations into full-length context windows, avoiding padding tokens and increasing GPU training throughput.

### 5. Fine-Tuning Supervisor (`loraforge.supervisor`)
Monitors training subprocesses, parses stdout metrics (loss, learning rate, step speed), and routes them to a `TelemetryEventBus`. Includes a memory watchdog that tracks VRAM consumption rate ($dV/dt$) and halts the run if memory pressure threatens to trigger a kernel-level display crash.

### 6. InferOps Export Bridge (`loraforge.export.inferops`)
Generates deployment manifests and quickstart launcher scripts for [InferOps](https://github.com/alexandrmotologa/inferops). Supports handoff to vLLM, SGLang, or llama.cpp with automated scale-to-zero settings.
