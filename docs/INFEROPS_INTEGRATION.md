# InferOps Integration Guide

LoRAForge and [InferOps](https://github.com/alexandrmotologa/inferops) provide complementary control planes for managing local LLM workloads on consumer and workstation hardware.

| Tool | Core Responsibility | Supported Engines / Backends |
|---|---|---|
| **LoRAForge** | Training control plane, predictive VRAM sizing, sequence packing, QLoRA supervision | PyTorch, BitsAndBytes, PEFT, TRL |
| **InferOps** | Inference control plane, scale-to-zero serverless, token analytics, multi-engine routing | vLLM, SGLang, llama.cpp |

## End-to-End Workflow

```
[1. Dataset Prep]        --> Ingest & pack sequences with LoRAForge
         |
[2. VRAM Profiling]      --> Calculate exact memory budget with LoRAForge
         |
[3. Supervised Training] --> Run fine-tuning loop under LoRAForge OOM watchdog
         |
[4. Model Export]        --> Export merged Safetensors or GGUF + InferOps manifest
         |
[5. Inference Serving]   --> Deploy to InferOps with scale-to-zero serverless
```

## Step 1: Fine-Tune with LoRAForge

Configure and train your model using the LoRAForge CLI or web dashboard:

```bash
loraforge train \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --gpu rtx-3060-12gb \
  --steps 200
```

## Step 2: Export InferOps Manifest

Generate an InferOps deployment manifest:

```bash
loraforge export \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --out ./exports/llama3-custom \
  --engine vllm
```

This generates `inferops_manifest.yaml` inside the output directory:

```yaml
model_id: llama3-custom-agent-ft
display_name: llama3-custom-agent (LoRAForge Fine-Tuned)
engine: vllm
model_path: ./exports/llama3-custom
base_model_reference: meta-llama/Llama-3.1-8B-Instruct
context_window: 8192
gpu_memory_utilization: 0.90
max_model_len: 2048
dtype: bfloat16
scale_to_zero_idle_minutes: 15
tags:
  - fine-tuned
  - loraforge
  - Llama-3.1-8B-Instruct
```

## Step 3: Launch in InferOps

Serve the fine-tuned model directly using the generated shell script or InferOps CLI:

```bash
inferops serve \
  --manifest ./exports/llama3-custom/inferops_manifest.yaml \
  --engine vllm \
  --port 8000
```

InferOps automatically handles model warm-up, scale-to-zero idling after 15 minutes of inactivity, GPU telemetry tracking, and OpenAI-compatible API endpoints.
