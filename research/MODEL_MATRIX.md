# MODEL SELECTION & EVALUATION MATRIX

## 1. Selected Open-Weight Language Models

We evaluate public open-weight language models from the Qwen2.5 and Llama-3.2 model families, choosing model scales suitable for MPS Apple Silicon GPU hardware acceleration.

| Model Identifier | Parameters | License | Context Length | Base / Instruct | GPU Requirement | Target Role in EAR-GRPO Phase II |
|---|---|---|---|---|---|---|
| `Qwen/Qwen2.5-0.5B-Instruct` | 490M | Apache 2.0 | 32,768 | Instruct | ~2GB VRAM (MPS) | **Tier 0 Pipeline Validation & Fast Controls** |
| `Qwen/Qwen2.5-1.5B-Instruct` | 1.54B | Apache 2.0 | 32,768 | Instruct | ~4GB VRAM (MPS) | **Tier 1 Scientific Pilot & Main Evidence** |
| `Qwen/Qwen2.5-Math-1.5B-Instruct` | 1.54B | Apache 2.0 | 4,096 | Math Specialization | ~4GB VRAM (MPS) | **Tier 1 Math Domain Benchmark** |
| `meta-llama/Llama-3.2-1B-Instruct` | 1.23B | Llama 3.2 Community | 128,000 | Instruct | ~3.5GB VRAM (MPS) | **Cross-Model Architecture Generalization** |

---

## 2. License & Availability Verification

* **Qwen2.5 Family**: Released under permissive **Apache 2.0** license. Fully permits research, redistribution, modification, and evaluation without commercial restriction.
* **Llama-3.2 Family**: Released under **Llama 3.2 Community License**. Permits free academic research and commercial deployment up to 700M monthly active users.

---

## 3. Hardware Compute Context
* Device: **Apple Silicon MPS (Metal Performance Shaders) GPU**
* VRAM: Shared Unified Memory (36GB System RAM)
* Execution Mode: Full precision `float32` / mixed `bfloat16`
