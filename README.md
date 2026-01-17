# Autoregressive Transformer with Rotary Positional Embeddings (RoPE)

A from-scratch PyTorch implementation of a decoder-only Transformer language model, optimized for linguistic coherence and training stability. This project demonstrates modern LLM architectural techniques used in state-of-the-art models like Llama 3.

## Key Features

* **Architecture:** Decoder-only Transformer with 6 layers and 6 attention heads (18.3M parameters)
* **Positional Encoding:** Implemented **Rotary Positional Embeddings (RoPE)** for enhanced relative position awareness
* **Tokenization:** Custom-trained **Byte-Pair Encoding (BPE)** tokenizer with a 10k vocabulary
* **Optimization:** Integrated **Cosine Learning Rate Annealing**, **AdamW with Weight Decay**, and **Gradient Clipping**
* **Performance:** Achieved a validation perplexity of **141.56** on the **WikiText-103** dataset

## Technical Stack

* **Framework:** PyTorch
* **Hardware:** Accelerated via Apple Silicon (MPS) / NVIDIA CUDA
* **Dataset:** WikiText-103 (Raw-v1)
* **Language:** Python 3.12+

## Training Dynamics

The model was trained for 5,000 iterations using a causal masking strategy. To prevent overfitting, **validation-based checkpointing** was used to capture weights at the global minimum of the loss curve.

| Metric | Initial | Final (Best) |
|:---|:---|:---|
| **Train Loss** | 9.27 | 3.48 |
| **Val Perplexity** | 10,662.5 | 141.56 |

## 💻 Usage

### 1. Setup Environment
```bash
pip install -r requirements.txt
```

### 2. Interactive Demo

Run the following to generate text from a custom prompt using the pre-trained weights:
```bash
python interact.py
```

## 📜 Acknowledgments

This project was developed as a deep dive into modern Transformer architectures, focusing on the mathematical transition from absolute positional encodings to relative rotary embeddings.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**Note:** For best results, ensure you have a CUDA-compatible GPU or Apple Silicon device for hardware acceleration during training and inference.
