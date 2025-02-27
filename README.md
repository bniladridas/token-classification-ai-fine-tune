# AI-Powered NER Beast

![Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)  
![CUDA](https://img.shields.io/badge/CUDA-Yes-brightgreen)  
![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Check%20it-blue)

---

## What’s This?

A fine-tuned [bert-base-uncased](https://huggingface.co/bert-base-uncased) for **NER** (Named Entity Recognition) on the [CoNLL-2003](https://huggingface.co/datasets/conll2003) dataset. It’s fast, GPU-ready, and hits a solid **0.0474** validation loss.

---

## Why It’s Cool

- Runs like a champ on NVIDIA GPUs (mixed precision, gradient accumulation, the works)
- Easy to track with TensorBoard
- Lives on [Hugging Face](https://huggingface.co/bniladridas/token-classification-ai-fine-tune) for quick grabs

---

## How to Use It

```python
from transformers import pipeline

ner = pipeline("token-classification", model="bniladridas/token-classification-ai-fine-tune")
print(ner("Apple is buying a U.K. startup for $1 billion"))
```

---

## Training Bits

- **Dataset**: CoNLL-2003
- **Learning Rate**: 2e-05
- **Batch Size**: 16
- **Epochs**: 3
- **Loss**: Dropped to 0.0160 training, 0.0474 validation

---

## Need More?

- [Model Page](https://huggingface.co/bniladridas/token-classification-ai-fine-tune)
- [Dataset Info](https://huggingface.co/datasets/conll2003)
- [CUDA Docs](https://docs.nvidia.com/cuda/)

---

## Setup

- NVIDIA GPU + CUDA
- PyTorch 2.0.1+
- Transformers 4.28.1+

---

Licensed under **Apache 2.0**. Tags: `token-classification`, `conll2003`, `generated_from_trainer`.
