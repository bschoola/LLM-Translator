# LLM Translator — EN → PT Fine-tuning Study

A hands-on study project exploring how to fine-tune a pre-trained translation model using **LoRA (Low-Rank Adaptation)** to specialize it for English → Brazilian Portuguese translation.

> This project is purely for learning purposes. The goal is to understand the mechanics of LLM fine-tuning, dataset preparation, and adapter-based training — not to build a production-ready translation system.

---

## What this project covers

- Loading and preprocessing a real translation dataset from Hugging Face
- Fine-tuning a pre-trained transformer model with **LoRA** (instead of updating all weights)
- Evaluating translation quality using the **BLEU score**
- Saving and loading a lightweight **adapter** (~few MB) instead of a full model copy
- Running everything **locally** with GPU acceleration

---

## What is LoRA?

Standard fine-tuning updates every parameter in the model, which is slow and produces a full copy of the model (~300 MB). **LoRA** instead inserts small trainable matrices inside the attention layers and freezes everything else:

```
Without LoRA:  input → [W_original (frozen)] → output
With LoRA:     input → [W_original (frozen)] + [A × B (trainable)] → output
                                                ↑
                                          this is the adapter
```

The result: only ~1% of parameters are trained, the adapter file is a few MB, and it can be shared independently from the base model.

---

## Stack

| Component | Tool |
|---|---|
| Base model | `Helsinki-NLP/opus-mt-en-ROMANCE` (MarianMT) |
| Fine-tuning | Hugging Face `transformers` + `Seq2SeqTrainer` |
| LoRA | Hugging Face `peft` |
| Dataset | `VanessaSchenkel/translation-en-pt` |
| Evaluation | `sacrebleu` (BLEU score) |
| Runtime | Python 3.11 · PyTorch · CUDA |

---

## Project structure

```
LLM-Translator/
├── train.py            # Fine-tuning script with LoRA
├── translate.py        # Load adapter and run translations
├── notebook.ipynb      # Step-by-step Jupyter guide (same content as train.py)
├── requirements.txt    # Python dependencies
├── checkpoints/        # Training checkpoints (saved per epoch)
└── outputs/
    └── adapter-en-pt/  # Final LoRA adapter saved here after training
```

---

## Getting started

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

> Requires a CUDA-compatible NVIDIA GPU. The `requirements.txt` pulls PyTorch with CUDA 12.1 support.

### 2. Verify GPU

```python
import torch
print(torch.cuda.is_available())       # should return True
print(torch.cuda.get_device_name(0))   # your GPU name
```

### 3. Train

```bash
python train.py
```

By default, training uses **5% of the dataset** for quick iteration. Edit the constant at the top of `train.py` to change this:

```python
TRAIN_SUBSET = 0.05   # 5%  → ~13k examples  (fast, good for testing)
TRAIN_SUBSET = 0.20   # 20% → ~52k examples  (better quality)
TRAIN_SUBSET = None   # 100% → full dataset   (full training run)
```

Training output will show something like:

```
trainable params: 884,736 || all params: 78,020,608 || trainable%: 1.13
```

That means only 1% of the model weights are actually updated — the rest stays frozen.

### 4. Translate

```bash
python translate.py
# or pass a sentence directly:
python translate.py "The sun is shining brightly."
```

---

## How the adapter is saved

After training, only the LoRA adapter is saved to `outputs/adapter-en-pt/`:

```
outputs/adapter-en-pt/
├── adapter_config.json     # LoRA configuration (rank, target modules, etc.)
├── adapter_model.safetensors  # the learned weights (~few MB)
└── tokenizer files
```

To use the adapter elsewhere, load the base model and apply it on top:

```python
from transformers import MarianMTModel, MarianTokenizer
from peft import PeftModel

base = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-en-ROMANCE")
model = PeftModel.from_pretrained(base, "./outputs/adapter-en-pt")

# Or merge into a single standalone model (no peft dependency needed at inference):
model = model.merge_and_unload()
```

---

## Key concepts learned

| Concept | Where it appears |
|---|---|
| Tokenization | `preprocess()` in `train.py` |
| LoRA adapter | `LoraConfig` + `get_peft_model()` |
| Seq2Seq training loop | `Seq2SeqTrainer` |
| BLEU score evaluation | `build_compute_metrics()` |
| Dataset normalization | `normalize_dataset()` — handles nested dict columns |
| Early stopping | `EarlyStoppingCallback` |

---

## Dataset

[VanessaSchenkel/translation-en-pt](https://huggingface.co/datasets/VanessaSchenkel/translation-en-pt) — a Hugging Face dataset with ~260k English/Portuguese sentence pairs, stored in a nested `translation: {en, pt}` format that is automatically flattened during preprocessing.

---

## References

- [Hugging Face — Fine-tuning Seq2Seq models](https://huggingface.co/docs/transformers/tasks/translation)
- [PEFT library documentation](https://huggingface.co/docs/peft)
- [LoRA paper — Hu et al., 2021](https://arxiv.org/abs/2106.09685)
- [Helsinki-NLP/opus-mt-en-ROMANCE](https://huggingface.co/Helsinki-NLP/opus-mt-en-ROMANCE)
