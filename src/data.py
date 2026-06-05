# src/data.py
# Activation extraction and summary generation

import torch
import random
from datasets import load_dataset
from transformers import AutoModelForCausalLM
from src.config import (
    MODEL_NAME, TARGET_LAYER, N_TEXTS, N_WARMUP,
    TRAIN_SIZE, MAX_LEN,
    ACTIVATIONS_PATH, SUMMARIES_PATH, TEXTS_PATH
)
from src.models import tokenizer

device = "cuda" if torch.cuda.is_available() else "cpu"


def load_texts():
    """Load wikitext-103 and filter short texts."""
    print("Loading dataset...")
    wiki  = load_dataset(
        "wikitext", "wikitext-103-raw-v1", split="train"
    )
    texts = [t for t in wiki["text"]
             if len(t.strip()) > 150][:N_TEXTS]
    print(f"Loaded {len(texts)} texts")
    return texts


def extract_activations(texts, batch_size=16):
    """
    Extract layer TARGET_LAYER residual stream activations.
    One activation per text, taken at last non-padding token.
    NOT L2-normalized (normalization collapses variance).
    """
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, output_hidden_states=True
    ).to(device)
    model.eval()

    all_acts = []
    print(f"Extracting activations from {len(texts)} texts...")

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        enc   = tokenizer(
            batch, return_tensors="pt",
            padding=True, truncation=True,
            max_length=MAX_LEN
        ).to(device)
        with torch.no_grad():
            out = model(**enc)
        hs      = out.hidden_states[TARGET_LAYER]
        lengths = enc.attention_mask.sum(dim=1) - 1
        acts    = hs[torch.arange(len(batch)), lengths]
        all_acts.append(acts.cpu())
        if i % 8000 == 0:
            print(f"  {i}/{len(texts)}")

    activations = torch.cat(all_acts, dim=0)
    print(f"Done. Shape: {activations.shape}")
    print(f"Mean norm  : {activations.norm(dim=1).mean():.2f}")
    print(f"Baseline MSE: "
          f"{((activations - activations.mean(0))**2).mean():.4f}")

    del model
    torch.cuda.empty_cache()
    return activations


def make_rich_summary(text, max_words=80):
    """
    Create structured summary capturing start, middle, and end
    of text. Better proxy than simple truncation.
    """
    words  = text.strip().split()
    n      = len(words)
    start  = " ".join(words[:30])
    middle = " ".join(
        words[n//2-15:n//2+15]
    ) if n > 60 else ""
    end    = " ".join(words[-30:])
    s      = f"Topic and context: {start}. "
    if middle:
        s  += f"Key content: {middle}. "
    s      += f"Conclusion: {end}"
    return s


def generate_summaries(texts):
    """Generate proxy summaries for SFT warm-start."""
    print(f"Generating {N_WARMUP} summaries...")
    summaries = []
    for i, t in enumerate(texts[:N_WARMUP]):
        summaries.append(make_rich_summary(t))
        if i % 1000 == 0:
            print(f"  {i}/{N_WARMUP}")
    return summaries


def save_data(activations, summaries, texts):
    torch.save(activations, ACTIVATIONS_PATH)
    torch.save(summaries,   SUMMARIES_PATH)
    torch.save(texts,       TEXTS_PATH)
    print("Data saved.")


def load_data():
    activations = torch.load(ACTIVATIONS_PATH, map_location="cpu")
    summaries   = torch.load(SUMMARIES_PATH)
    texts       = torch.load(TEXTS_PATH)
    return activations, summaries, texts