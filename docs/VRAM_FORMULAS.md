# Predictive Training VRAM Formulations

This document describes the analytical formulas used by LoRAForge to estimate peak GPU memory requirements prior to launching training runs.

## 1. Base Model Weights ($M_{\text{base}}$)

Let $N_{\text{base}}$ be the total parameter count of the model, $d_{\text{model}}$ the hidden dimension, and $V$ the vocabulary size.

### Quantization Schemes

1. **Full Precision (16-bit FP16 / BF16)**:
   $$M_{\text{base}} = \frac{N_{\text{base}} \times 2.0}{1024^2} \text{ MB}$$

2. **4-bit NormalFloat with Double Quantization (NF4 DQ)**:
   In standard QLoRA implementations (`bitsandbytes`), embedding layers and language modeling heads remain in 16-bit unless explicitly requested.
   $$N_{\text{embed}} = V \times d_{\text{model}}$$
   $$N_{\text{head}} = V \times d_{\text{model}} \quad (\text{if un-tied})$$
   $$N_{\text{quant}} = N_{\text{base}} - N_{\text{embed}} - N_{\text{head}}$$
   
   Weights consume 4 bits ($0.5$ bytes) per parameter:
   $$M_{\text{weights}} = (N_{\text{quant}} \times 0.5) + ((N_{\text{embed}} + N_{\text{head}}) \times 2.0) \text{ bytes}$$
   
   Block quantization constants (block size $B = 64$):
   $$\text{Blocks} = \frac{N_{\text{quant}}}{64}$$
   Double quantization compresses first-level block scales to 8-bit (1 byte) with 32-bit second-level scales per 256 blocks:
   $$M_{\text{meta}} = (\text{Blocks} \times 1.0) + \left(\frac{\text{Blocks}}{256} \times 4.0\right) \text{ bytes} \approx 0.01587 \text{ bytes/param}$$
   $$M_{\text{base}} = \frac{M_{\text{weights}} + M_{\text{meta}}}{1024^2} \text{ MB}$$

## 2. LoRA Adapter Trainable Parameters ($N_{\text{lora}}$)

For a target linear matrix $W \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$, rank decomposition creates matrices $A \in \mathbb{R}^{r \times d_{\text{in}}}$ and $B \in \mathbb{R}^{d_{\text{out}} \times r}$.
$$\text{Params}(W) = r \cdot (d_{\text{in}} + d_{\text{out}})$$

In a transformer block with hidden size $d_{\text{model}}$, intermediate dimension $d_{\text{mlp}}$, head dimension $d_{\text{head}}$, query heads $h_q$, and key-value heads $h_{\text{kv}}$:

| Target Projection | $d_{\text{in}}$ | $d_{\text{out}}$ | Parameters per Layer |
|---|---|---|---|
| `q_proj` | $d_{\text{model}}$ | $d_{\text{model}}$ | $2 \cdot r \cdot d_{\text{model}}$ |
| `k_proj` | $d_{\text{model}}$ | $h_{\text{kv}} \cdot d_{\text{head}}$ | $r \cdot (d_{\text{model}} + h_{\text{kv}} \cdot d_{\text{head}})$ |
| `v_proj` | $d_{\text{model}}$ | $h_{\text{kv}} \cdot d_{\text{head}}$ | $r \cdot (d_{\text{model}} + h_{\text{kv}} \cdot d_{\text{head}})$ |
| `o_proj` | $d_{\text{model}}$ | $d_{\text{model}}$ | $2 \cdot r \cdot d_{\text{model}}$ |
| `gate_proj` | $d_{\text{model}}$ | $d_{\text{mlp}}$ | $r \cdot (d_{\text{model}} + d_{\text{mlp}})$ |
| `up_proj` | $d_{\text{model}}$ | $d_{\text{mlp}}$ | $r \cdot (d_{\text{model}} + d_{\text{mlp}})$ |
| `down_proj` | $d_{\text{mlp}}$ | $d_{\text{model}}$ | $r \cdot (d_{\text{mlp}} + d_{\text{model}})$ |

Across $L$ transformer layers:
$$N_{\text{lora}} = L \sum_{m \in \text{targets}} \text{Params}(m)$$
$$M_{\text{lora}} = \frac{N_{\text{lora}} \times 2.0}{1024^2} \text{ MB}$$

## 3. Gradients ($M_{\text{grad}}$)

Base model weights are frozen ($\text{requires\_grad}=\text{False}$). Gradients are accumulated only for trainable parameters:
$$M_{\text{grad}} = \frac{N_{\text{lora}} \times \text{bytes}_{\text{grad}}}{1024^2} \text{ MB}$$
Where $\text{bytes}_{\text{grad}} = 2.0$ for 16-bit mixed precision gradient accumulation, and $4.0$ for FP32.

## 4. Optimizer Memory ($M_{\text{opt}}$)

Stored optimizer states depend on the chosen implementation:

- **`adamw_torch` (FP32)**: Momentum ($m$) + Variance ($v$) in 32-bit:
  $$\text{bytes}_{\text{opt}} = 8.0 \text{ bytes/param}$$
- **`paged_adamw_8bit` / `adamw_8bit`**: Quantized 8-bit states + block scale tables:
  $$\text{bytes}_{\text{opt}} = 2.0 + \frac{2 \times 4.0}{2048} \approx 2.008 \text{ bytes/param}$$
- **`lion_8bit`**: Single momentum state in 8-bit:
  $$\text{bytes}_{\text{opt}} = 1.0 + \frac{4.0}{2048} \approx 1.002 \text{ bytes/param}$$

$$M_{\text{opt}} = \frac{N_{\text{lora}} \times \text{bytes}_{\text{opt}}}{1024^2} \text{ MB}$$

## 5. Activation Memory ($M_{\text{act}}$)

Let $b$ be the micro-batch size and $s$ the sequence length.

### Case A: Gradient Checkpointing Enabled

Intermediate activations inside each layer are recomputed during backward execution. Only the layer input tensors $[b, s, d_{\text{model}}]$ are stashed:
$$M_{\text{stashed}} = L \cdot b \cdot s \cdot d_{\text{model}} \cdot 2.0 \text{ bytes}$$

Peak intra-block recomputation memory during the backward pass of a single layer:
- **With FlashAttention-2**: The $O(s^2)$ attention matrix is never materialized in HBM.
  $$M_{\text{attn}} = b \cdot s \cdot (2 d_{\text{model}} + 2 h_{\text{kv}} d_{\text{head}}) \cdot 2.0 \text{ bytes}$$
- **With Standard Attention (SDPA / Eager)**: Stores full query-key score matrices:
  $$M_{\text{attn}} = (b \cdot h_q \cdot s^2 + b \cdot s \cdot (d_{\text{model}} + 2 h_{\text{kv}} d_{\text{head}})) \cdot 2.0 \text{ bytes}$$
- **SwiGLU MLP Block**:
  $$M_{\text{mlp}} = b \cdot s \cdot (d_{\text{model}} + 3 d_{\text{mlp}}) \cdot 2.0 \text{ bytes}$$
- **Logits & Loss**: Chunked cross-entropy buffer for token chunks:
  $$M_{\text{logits}} = \min(b \cdot s, 1024) \cdot V \cdot 2.0 \text{ bytes}$$

$$M_{\text{act}} = \frac{M_{\text{stashed}} + \max(M_{\text{attn}}, M_{\text{mlp}}) + M_{\text{logits}}}{1024^2} \text{ MB}$$

### Case B: Gradient Checkpointing Disabled

Activations across all $L$ layers remain in memory simultaneously:
$$M_{\text{act}} = \frac{L \cdot (M_{\text{attn}} + M_{\text{mlp}} + 2 b s d_{\text{model}} \cdot 2) + b s V \cdot 2}{1024^2} \text{ MB}$$

## 6. CUDA Context and Allocator Fragmentation

- **CUDA Runtime Baseline ($M_{\text{cuda}}$)**: Fixed buffer reserved by driver initialization and cuBLAS handles (~680 MB on Ampere/Ada).
- **Allocator Fragmentation Buffer ($M_{\text{frag}}$)**: Safety buffer for dynamic tensor reuse patterns:
  $$M_{\text{frag}} = 0.12 \times (M_{\text{act}} + M_{\text{grad}} + M_{\text{lora}})$$

## 7. Peak Memory Budget

$$M_{\text{peak}} = M_{\text{cuda}} + M_{\text{base}} + M_{\text{lora}} + M_{\text{grad}} + M_{\text{opt}} + M_{\text{act}} + M_{\text{frag}}$$
