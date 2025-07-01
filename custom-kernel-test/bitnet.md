
# BitNet b1.58 2B4T Technical Report

**Shuming Ma\*, Hongyu Wang\*, Shaohan Huang, Xingxing Zhang, Ying Hu, Ting Song, Yan Xia, Furu Wei**

**Link:** [https://aka.ms/GeneralAI](https://aka.ms/GeneralAI)  
**arXiv:** [2504.12285v2 [cs.CL] 25 Apr 2025](https://arxiv.org/abs/2504.12285)

---

## Abstract

We introduce `BitNet b1.58 2B4T`, the first open-source, native 1-bit Large Language Model (LLM) at the 2-billion parameter scale. Trained on a corpus of 4 trillion tokens, the model has been rigorously evaluated across benchmarks covering language understanding, mathematical reasoning, coding proficiency, and conversational ability. Our results demonstrate that `BitNet b1.58 2B4T` achieves performance on par with leading open-weight, full-precision LLMs of similar size, while offering significant advantages in computational efficiency, including substantially reduced memory footprint, energy consumption, and decoding latency. To facilitate further research and adoption, the model weights are released via Hugging Face along with open-source inference implementations for both GPU and CPU architectures.

- ✅ **BitNet b1.58 2B4T (1.58-bit):** `bitnet-b1.58-2B-4T`
  - The packed weight of `BitNet b1.58 2B4T`, used for inference only
- ✅ **BitNet b1.58 2B4T (bf16):** `bitnet-b1.58-2B-4T-bf16`
  - The master weight of `BitNet b1.58 2B4T`, used for training only
- ✅ **BitNet b1.58 2B4T (gguf):** `bitnet-b1.58-2B-4T-gguf`
  - The GGUF format of `BitNet b1.58 2B4T`, used for `bitnet.cpp`
- ✅ **BitNet b1.58 2B4T Code:** `bitnet.cpp` | **Demo:** [aka.ms/bitnet-demo](https://aka.ms/bitnet-demo)

> **Figure 1:** `BitNet b1.58 2B4T` advances the Pareto frontier defined by leading open-weight LLMs under 3B parameters in terms of performance versus memory, demonstrating superior efficiency.

---

## 1. Introduction

Open-source large language models (LLMs) have become pivotal in democratizing access to advanced AI capabilities. However, a significant barrier hinders their broader adoption: the substantial computational resources required for deployment and inference.

1-bit LLMs, representing an extreme yet promising form of model quantization, offer a compelling solution to efficiency challenges. While prior work has explored 1-bit models, existing efforts often fall into two categories: 1) post-training quantization (PTQ) methods, which can lead to significant performance degradation, or 2) native 1-bit models developed at smaller scales. This performance gap has limited the practical impact of 1-bit LLMs thus far.

To bridge this gap, we introduce `BitNet b1.58 2B4T`, the first open-source, native 1-bit LLM trained at scale. The core contribution of this work is to demonstrate that a native 1-bit LLM, when trained effectively at scale, can achieve performance comparable to leading open-weight, full-precision models of similar size.

## 2. Architecture

The architecture of `BitNet b1.58 2B4T` is derived from the standard Transformer model, incorporating significant modifications based on the BitNet framework. The core innovation is replacing standard linear layers (`torch.nn.Linear`) with custom `BitLinear` layers.

-   **Weight Quantization:**
 Model weights are quantized to 1.58 bits (ternary values `{-1, 0, +1}`) during the forward pass using an absmean quantization scheme.
-   **Activation Quantization:** Activations are quantized to 8-bit integers using an absmax quantization strategy, applied per-token.
-   **Normalization:** We incorporate `subln` normalization to enhance training stability.

Other established LLM techniques are also integrated:

-   **Activation Function (FFN):** Employs squared ReLU (ReLU²) instead of SwiGLU.
-   **Positional Embeddings:** Uses Rotary Position Embeddings (RoPE).
-   **Bias Removal:** All bias terms are removed from linear and normalization layers.
-   **Tokenizer:** Adopts the LLaMA 3 tokenizer (BPE, 128,256 vocab size).

## 3. Training

The training process involved three phases: pre-training, supervised fine-tuning (SFT), and direct preference optimization (DPO).

### 3.1 Pre-training

#### 3.1.1 Learning Rate Schedule
A two-stage schedule was used:
1.  **Stage 1 (High Learning Rate):** An initial phase with a high peak learning rate and cosine decay.
2.  **Stage 2 (Cooldown):** An abrupt decay midway through training, followed by a cosine schedule with a lower peak value.

#### 3.1.2 Weight Decay Schedule
1.  **Stage 1:** Weight decay followed a cosine schedule, peaking at 0.1.
2.  **Stage 2:** Weight decay was disabled (set to zero).

### 3.2 Supervised Fine-tuning (SFT)

The model was fine-tuned on a diverse collection of public instruction-following datasets (e.g., WildChat, LMSYS-Chat-1M, WizardLM Evol-Instruct).

#### 3.2.2 Chat Template

```
<|begin_of_text|>System: {system_message}<|eot_id|>
User: {user_message_1}<|eot_id|>
Assistant: {assistant_message_1}<|eot_id|>
User: {user_message_2}<|eot_id|>
Assistant: {assistant_message_2}<|eot_id|>...
```

### 3.3 Direct Preference Optimization (DPO)

DPO was applied to align the model with human preferences using datasets like UltraFeedback and MagPie. The training ran for 2 epochs with a learning rate of 2 × 10⁻⁷ and a beta of 0.1.

## 4. Evaluation

### 4.1 Main Results

`BitNet b1.58 2B4T` shows remarkable resource efficiency and highly competitive task performance compared to leading full-precision models in its size class.

**Table 1: Comparison with Full-Precision LLMs (1B-2B)**

| Benchmark (Metric) | LLaMA 3.2 1B | Gemma-3 1B | Qwen2.5 1.5B | SmolLM2 1.7B | MiniCPM 2B | **BitNet b1.58 2B** |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Memory** (Non-emb) | 2GB | 1.4GB | 2.6GB | 3.2GB | 4.8GB | **0.4GB** |
| **Latency** (CPU; TPOT) | 48ms | 41ms | 65ms | 67ms | 124ms | **29ms** |
| **Energy** (Estimated) | 0.258J | 0.186J | 0.347J | 0.425J | 0.649J | **0.028J** |
| **Training Tokens** | 9T | 2T | 18T | 11T | 1.1T | **4T** |
| **Average Score** | 44.90 | 43.74 | 55.23 | 48.70 | 42.05 | **54.19** |
| GSM8K | 38.21 | 31.16 | 56.79 | 45.11 | 4.40 | **58.38** |
| ARC-Challenge | 37.80 | 38.40 | 46.67 | 43.52 | 44.80 | **49.91** |
| WinoGrande | 59.51 | 58.48 | 62.83 | 68.98 | 61.80 | **71.90** |

### 4.2 Comparison with Post-training Quantized Models

`BitNet b1.58 2B4T` achieves a lower memory footprint than INT4 quantized models while maintaining stronger overall performance.

**Table 2: Comparison with PTQ Models**

| Benchmark (Metric) | Qwen2.5 1.5B-bf16 | Qwen2.5 1.5B-GPTQ-int4 | Qwen2.5 1.5B-AWQ-int4 | **BitNet b1.58 2B** |
| :--- | :--- | :--- | :--- | :--- |
| **Memory** (Non-emb) | 2.6GB | 0.7GB | 0.7GB | **0.4GB** |
| **Activation** | bf16 | bf16 | bf16 | **int8** |
| **Average Score** | 55.72 | 52.15 | 51.17 | **55.01** |
| GSM8K | 56.79 | 50.57 | 50.64 | **58.38** |
| IFEval | 50.12 | 47.84 | 45.44 | **53.48** |

### 4.3 Comparison with Open-weight 1-bit Models

`BitNet b1.58 2B4T` is the leading model among other 1-bit models, outperforming both smaller natively trained models and larger models quantized to 1-bit.

**Table 3: Comparison with Other 1-bit Models**

| Benchmark (Metric) | Bonsai 0.5B | OLMO-Bitnet 1B | Falcon3-1.58bit 7B | Llama3-8B-1.58 8B | **BitNet b1.58 2B** |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Native 1-bit** | ✓ | ✓ | ✗ | ✗ | **✓** |
| **Average Score** | 41.06 | 32.22 | 50.76 | 49.75 | **60.68** |
| MMLU | 25.74 | 25.47 | 42.79 | 35.04 | **53.17** |
| ARC-Challenge | 33.19 | 26.54 | 37.80 | 43.69 | **49.91** |
| CommonsenseQA | 18.43 | 19.49 | 67.08 | 28.50 | **71.58** |


## 5. Inference Implementation

We developed and open-sourced dedicated inference libraries for both GPU and CPU platforms, available at [https://aka.ms/bitnet](https://aka.ms/bitnet).

### 5.1 GPU Inference
A custom CUDA kernel was developed for the W1.58A8 matrix multiplication. It packs four ternary weights into a single `int8` for storage and unpacks them in on-chip Shared Memory for computation.

### 5.2 CPU Inference
`bitnet.cpp` is a C++ library serving as a reference implementation for efficient CPU inference of 1-bit LLMs. It provides optimized kernels and ensures lossless inference.

## 6. Conclusion

`BitNet b1.58 2B4T` is a significant step towards highly efficient yet capable LLMs. Our work demonstrates the viability of extreme quantization directly within the training process, achieving performance comparable to full-precision models with dramatically reduced computational requirements. It represents a compelling proof-of-concept that challenges the necessity of full-precision weights for high performance in LLMs at scale.

## 7. Future Directions

-   **Scaling Laws and Larger Models:** Exploring training of larger models (7B, 13B+) to test performance parity.
-   **Hardware Co-Design and Optimization:** Continued development of optimized kernels and co-designing hardware accelerators.
-   **Extended Sequence Length:** Enhancing the model to process longer contexts.
-   **Multilingual Capabilities:** Extending the training corpus to support multiple languages.
-   **Multimodal Integration:** Integrating 1-bit principles into multimodal architectures.
-   **Theoretical Understanding:** Analyzing the learning dynamics and representational properties of 1-bit models.

---

## Appendix

### A. Open-weight Baselines

-   **LLAMA 3.2 1B:** `meta-llama/Llama-3.2-1B-Instruct`
-   **Gemma-3 1B:** `google/gemma-3-1b-it`
-   **Qwen2.5 0.5B:** `Qwen/Qwen2.5-0.5B-Instruct`
-   **SmolLM2 1.7B:** `HuggingFaceTB/SmolLM2-1.7B-Instruct`
-   **OLMo-Bitnet 1B:** `NousResearch/OLMo-Bitnet-1B`
-   **Falcon3-1.58bit 7B:** `tiiuae/Falcon3-7B-Instruct-1.58bit`
-   **Llama3-8B-1.58 8B:** `HF1BitLLM/Llama3-8B-1.58-100B-tokens`

### B. Evaluation Pipeline Details

-   **HumanEval+:** `evalplus` toolkit.
-   **MATH-500:** Customized `math-evaluation-harness` toolkit.
-   **MT-Bench:** Official `LLM Judge` open-source codebase.
-   **Other benchmarks:** Standard `lm-evaluation-harness` framework.

**Table 4: ADD and MUL energy consumption (in pJ) at 7nm**

| Bits | ADD Energy | MUL Energy |
| :--- | :--- | :--- |
| FP16 | 0.16 | 0.34 |
| INT8 | 0.007 | 0.07 |