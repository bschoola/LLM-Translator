"""Testa o modelo com adapter LoRA. Carrega o base + aplica o adapter treinado."""

import sys
import os
from transformers import MarianMTModel, MarianTokenizer
from peft import PeftModel

BASE_MODEL = "Helsinki-NLP/opus-mt-en-ROMANCE"
ADAPTER_DIR = "./outputs/adapter-en-pt"


def load_model(merge: bool = False):
    """
    Carrega base + adapter LoRA.
    merge=True: funde os pesos (mais rápido na inferência, perde a modularidade).
    merge=False: mantém adapter separado (padrão para uso normal).
    """
    tokenizer = MarianTokenizer.from_pretrained(ADAPTER_DIR)
    base = MarianMTModel.from_pretrained(BASE_MODEL)

    if os.path.isdir(ADAPTER_DIR):
        print(f"Carregando adapter de: {ADAPTER_DIR}")
        model = PeftModel.from_pretrained(base, ADAPTER_DIR)
        if merge:
            # Funde adapter nos pesos base → modelo único, sem dependência do peft
            model = model.merge_and_unload()
            print("Adapter fundido no modelo base.")
    else:
        print(f"[aviso] Adapter não encontrado em '{ADAPTER_DIR}'. Usando modelo base.")
        model = base

    model.eval()
    return tokenizer, model


def translate(texts: list[str], tokenizer, model) -> list[str]:
    inputs = [f">>pt<< {t}" for t in texts]
    encoded = tokenizer(inputs, return_tensors="pt", padding=True, truncation=True, max_length=128)
    # Para PeftModel, precisa acessar o modelo base para generate
    base_model = model.base_model if hasattr(model, "base_model") and not isinstance(model, MarianMTModel) else model
    generated = base_model.generate(**encoded, num_beams=4, max_length=128)
    return tokenizer.batch_decode(generated, skip_special_tokens=True)


def main():
    tokenizer, model = load_model()

    sentences = [
        "Hello, how are you?",
        "The weather is beautiful today.",
        "I love learning new languages.",
        "Artificial intelligence is changing the world.",
        "Can you help me with this translation?",
    ]

    if len(sys.argv) > 1:
        sentences = [" ".join(sys.argv[1:])]

    print("\n=== Traduções EN → PT ===\n")
    translations = translate(sentences, tokenizer, model)
    for src, tgt in zip(sentences, translations):
        print(f"EN: {src}")
        print(f"PT: {tgt}")
        print()


if __name__ == "__main__":
    main()
