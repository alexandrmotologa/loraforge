# LoRAForge REST & WebSocket API Reference

The LoRAForge web control plane exposes a REST API and WebSocket stream on port `8765` by default.

Start the API server:
```bash
loraforge serve --host 127.0.0.1 --port 8765
```

## REST Endpoints

### `GET /api/health`
Returns service status and version information.

**Response:**
```json
{
  "status": "healthy",
  "service": "loraforge",
  "version": "0.1.0"
}
```

---

### `GET /api/hardware`
Returns live GPU telemetry and reference presets.

**Response:**
```json
{
  "telemetry": [],
  "detected": [],
  "presets": {
    "rtx-3060-12gb": {
      "name": "NVIDIA GeForce RTX 3060 12GB",
      "total_vram_mb": 12288,
      "compute_capability_major": 8,
      "compute_capability_minor": 6,
      "supports_bf16": true,
      "supports_flash_attn_2": true,
      "safe_vram_headroom_mb": 768
    }
  }
}
```

---

### `GET /api/models`
Returns specifications for registered foundation models.

---

### `POST /api/vram/estimate`
Calculates component-level memory breakdown and evaluates GPU feasibility.

**Request Body:**
```json
{
  "model_name": "meta-llama/Llama-3.1-8B-Instruct",
  "gpu_name": "rtx-3060-12gb",
  "seq_len": 2048,
  "batch_size": 1,
  "grad_acc": 8,
  "rank": 16,
  "quant": "nf4",
  "optimizer": "paged_adamw_8bit",
  "gradient_checkpointing": true,
  "flash_attention": true
}
```

**Response:**
```json
{
  "model_name": "meta-llama/Llama-3.1-8B-Instruct",
  "gpu_name": "NVIDIA GeForce RTX 3060 12GB",
  "total_gpu_vram_mb": 12288,
  "usable_gpu_vram_mb": 11520.0,
  "breakdown": {
    "cuda_context_mb": 680.0,
    "base_model_weights_mb": 4894.67,
    "quantization_metadata_mb": 155.07,
    "lora_weights_mb": 80.0,
    "gradients_mb": 80.0,
    "optimizer_states_mb": 80.32,
    "activations_mb": 2234.38,
    "fragmentation_buffer_mb": 287.33
  },
  "peak_vram_mb": 8491.77,
  "headroom_mb": 3796.23,
  "utilization_percentage": 69.1,
  "status": "OPTIMAL",
  "recommendations": []
}
```

---

### `POST /api/vram/autotune`
Finds safe hyperparameter settings for a model and target accelerator.

**Request Body:**
```json
{
  "model_name": "meta-llama/Llama-3.1-8B-Instruct",
  "gpu_name": "rtx-3060-12gb",
  "effective_batch": 16,
  "target_seq_len": 2048
}
```

---

### `POST /api/jobs`
Launches a supervised fine-tuning run.

---

### `GET /api/jobs`
Lists all tracked training jobs and current statuses.

---

### `POST /api/jobs/{job_id}/cancel`
Terminates an ongoing fine-tuning job.

---

### `POST /api/export/inferops`
Generates an InferOps service manifest and launcher script.

---

## WebSocket Stream

### `WS /ws/telemetry`
Subscribes to live step metrics during fine-tuning runs.

**Sample Message:**
```json
{
  "step": 14,
  "epoch": 0.42,
  "loss": 1.4821,
  "learning_rate": 0.000186,
  "grad_norm": 0.75,
  "tokens_per_second": 2400.0,
  "vram_used_mb": 8240.5,
  "vram_peak_mb": 8240.5,
  "elapsed_seconds": 1.15,
  "timestamp": 1712678900.12
}
```
