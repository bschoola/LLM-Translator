"""LoRA fine-tuning of MarianMT (EN→PT) — saves only the adapter (~a few MB)."""

import os
from datasets import load_dataset
from transformers import (
    MarianMTModel,
    MarianTokenizer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
)
from peft import LoraConfig, get_peft_model, TaskType
import evaluate
import numpy as np

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODEL_NAME = "Helsinki-NLP/opus-mt-en-ROMANCE"
ADAPTER_DIR = "./outputs/adapter-en-pt"   # only the LoRA adapter (~a few MB)
CHECKPOINT_DIR = "./checkpoints"
MAX_INPUT_LENGTH = 128
MAX_TARGET_LENGTH = 128
BATCH_SIZE = 16
NUM_EPOCHS = 3
LEARNING_RATE = 5e-5
TRAIN_SUBSET = 0.05   # use only 5% of the dataset — set to None to use everything

# LoRA hyperparameters
LORA_R = 16        # rank: how much "space" the adapter has to learn (↑ = more capacity, more MB)
LORA_ALPHA = 32    # scale: amplifier for the adapter signal (convention: 2 × r)
LORA_DROPOUT = 0.05
# Attention layers where LoRA is applied (q=query, v=value are the most impactful)
LORA_TARGET_MODULES = ["q_proj", "v_proj"]


# ---------------------------------------------------------------------------
# 1. Dataset
# ---------------------------------------------------------------------------
def load_translation_dataset():
    dataset = load_dataset("VanessaSchenkel/translation-en-pt", field="data")
    print(f"Dataset loaded: {dataset}")
    print(f"Available columns: {dataset['train'].column_names}")
    return dataset


def normalize_dataset(dataset):
    """If the 'translation' column is a dict {"en": ..., "pt": ...}, expands it to flat columns."""
    cols = dataset["train"].column_names
    if "translation" not in cols:
        return dataset

    sample = dataset["train"][0]["translation"]
    if not isinstance(sample, dict):
        return dataset

    lang_keys = list(sample.keys())
    en_key = next((k for k in ["en", "english"] if k in lang_keys), None)
    pt_key = next((k for k in ["pt", "portuguese"] if k in lang_keys), None)

    if not en_key or not pt_key:
        raise ValueError(
            f"Column 'translation' has keys {lang_keys}. Expected 'en' and 'pt'."
        )

    print(f"Expanding 'translation.{en_key}' and 'translation.{pt_key}' → flat columns 'en' and 'pt'")

    def flatten(batch):
        return {
            "en": [t[en_key] for t in batch["translation"]],
            "pt": [t[pt_key] for t in batch["translation"]],
        }

    return dataset.map(flatten, batched=True, remove_columns=cols, desc="Normalizing")


def get_column_names(dataset):
    """Auto-detects the source and target column names."""
    cols = dataset["train"].column_names
    candidates = [
        ("en", "pt"),
        ("english", "portuguese"),
        ("source", "target"),
        ("input", "output"),
    ]
    for src, tgt in candidates:
        if src in cols and tgt in cols:
            return src, tgt
    raise ValueError(
        f"Could not identify EN/PT columns. Available columns: {cols}\n"
        "Edit get_column_names() with the correct names."
    )


# ---------------------------------------------------------------------------
# 2. Tokenization
# ---------------------------------------------------------------------------
def preprocess(examples, tokenizer, src_col, tgt_col):
    inputs = [f">>pt<< {text}" for text in examples[src_col]]
    targets = examples[tgt_col]

    model_inputs = tokenizer(
        inputs, max_length=MAX_INPUT_LENGTH, truncation=True, padding=False
    )
    labels = tokenizer(
        text_target=targets, max_length=MAX_TARGET_LENGTH, truncation=True, padding=False
    )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


# ---------------------------------------------------------------------------
# 3. Metrics (BLEU)
# ---------------------------------------------------------------------------
def build_compute_metrics(tokenizer):
    bleu = evaluate.load("sacrebleu")

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        result = bleu.compute(
            predictions=[p.strip() for p in decoded_preds],
            references=[[l.strip()] for l in decoded_labels],
        )
        return {"bleu": round(result["score"], 2)}

    return compute_metrics


# ---------------------------------------------------------------------------
# 4. Training with LoRA
# ---------------------------------------------------------------------------
def main():
    print("=== Loading dataset ===")
    dataset = load_translation_dataset()
    dataset = normalize_dataset(dataset)

    if TRAIN_SUBSET is not None:
        n = int(len(dataset["train"]) * TRAIN_SUBSET)
        dataset["train"] = dataset["train"].select(range(n))
        print(f"Using {TRAIN_SUBSET*100:.0f}% of dataset → {n} examples")

    src_col, tgt_col = get_column_names(dataset)
    print(f"Using columns: source='{src_col}', target='{tgt_col}'")

    print("\n=== Loading model and tokenizer ===")
    tokenizer = MarianTokenizer.from_pretrained(MODEL_NAME)
    base_model = MarianMTModel.from_pretrained(MODEL_NAME)

    # Configure and apply LoRA
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES,
        bias="none",
    )
    model = get_peft_model(base_model, lora_config)
    model.print_trainable_parameters()
    # Expected output:
    # trainable params: 884,736 || all params: 78,020,608 || trainable%: 1.13%
    # Only ~1% of the weights are updated!

    print("\n=== Tokenizing dataset ===")
    tokenized = dataset.map(
        lambda ex: preprocess(ex, tokenizer, src_col, tgt_col),
        batched=True,
        remove_columns=dataset["train"].column_names,
        desc="Tokenizing",
    )

    if "validation" not in tokenized:
        split = tokenized["train"].train_test_split(test_size=0.05, seed=42)
        tokenized["train"] = split["train"]
        tokenized["validation"] = split["test"]

    print(f"Train: {len(tokenized['train'])} | Validation: {len(tokenized['validation'])}")

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True)

    training_args = Seq2SeqTrainingArguments(
        output_dir=CHECKPOINT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        warmup_steps=200,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="bleu",
        predict_with_generate=True,
        generation_max_length=MAX_TARGET_LENGTH,
        fp16=True,
        report_to="none",
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=build_compute_metrics(tokenizer),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    print("\n=== Starting training ===")
    trainer.train()

    # Save ONLY the LoRA adapter (not the full base model)
    print(f"\n=== Saving adapter to {ADAPTER_DIR} ===")
    model.save_pretrained(ADAPTER_DIR)
    tokenizer.save_pretrained(ADAPTER_DIR)
    print("Training complete!")
    print(f"Adapter saved to: {ADAPTER_DIR}")
    print("Run translate.py to test.")


if __name__ == "__main__":
    main()
