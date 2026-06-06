# src/evaluate.py
# FVE computation and qualitative evaluation

import torch
import torch.nn as nn
import random
import json
import os
from src.config import TARGET_LAYER, MODEL_NAME, HIDDEN_DIM
from src.models import tokenizer

device = "cuda" if torch.cuda.is_available() else "cpu"


def compute_fve(av, ar, activations, n_samples=60):
    """
    Compute Fraction of Variance Explained.

    FVE = 1 - E[||h - AR(AV(h))||^2] / E[||h - h_mean||^2]

    FVE = 0 : model predicts the mean (no information)
    FVE = 1 : perfect reconstruction
    FVE < 0 : worse than predicting the mean
    """
    av.eval()
    ar.eval()
    mse_fn  = nn.MSELoss()
    total   = 0.0
    indices = random.sample(range(len(activations)), n_samples)

    with torch.no_grad():
        for idx in indices:
            h      = activations[idx].unsqueeze(0).to(device)
            z      = av.inject_and_generate(h, max_new_tokens=80)
            h_pred = ar.forward(z)
            total += mse_fn(
                h_pred, activations[idx].unsqueeze(0)
            ).item()

    mean_mse = total / n_samples
    h_mean   = activations.mean(0)
    baseline = ((activations - h_mean)**2).mean().item()
    fve      = 1.0 - mean_mse / (baseline + 1e-8)

    av.train()
    ar.train()
    return round(fve, 4), round(mean_mse, 4), round(baseline, 4)


def qualitative_evaluation(av, ar, test_cases, save_path):
    """
    Run AV on test contexts and collect explanations.
    Saves results to JSON.
    """
    from transformers import AutoModelForCausalLM

    target_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, output_hidden_states=True
    ).to(device)
    target_model.eval()

    mse_fn  = nn.MSELoss()
    results = []

    print("Qualitative evaluation:")
    print("=" * 65)

    for text in test_cases:
        enc = tokenizer(
            text, return_tensors="pt",
            truncation=True, max_length=128
        ).to(device)
        with torch.no_grad():
            out = target_model(**enc)
        h = out.hidden_states[TARGET_LAYER][0,-1,:].unsqueeze(0)

        with torch.no_grad():
            explanation = av.inject_and_generate(
                h, max_new_tokens=80
            )
            h_pred  = ar.forward(explanation)
            mse_val = mse_fn(h_pred, h.cpu()).item()

        results.append({
            "context":     text,
            "explanation": explanation,
            "mse":         round(mse_val, 4)
        })
        print(f"\nContext    : {text[:60]}...")
        print(f"Explanation: {explanation[:110]}...")
        print(f"MSE        : {mse_val:.4f}")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(results, f, indent=2)

    del target_model
    torch.cuda.empty_cache()
    print(f"\nSaved to {save_path}")
    return results


def steganography_check(av, ar, activations, n_samples=50):
    """
    Test whether AR relies on semantic content or surface form.
    Paraphrase AV outputs and measure FVE drop.
    Low ratio (~1.0) = semantic content used (good).
    High ratio (>1.5) = surface form relied on (steganography).
    """
    mse_fn     = nn.MSELoss()
    orig_mses  = []
    para_mses  = []

    replacements = {
        "discusses": "covers",
        "text":      "passage",
        "about":     "regarding",
        "describes": "explains",
        "content":   "material",
        "shows":     "demonstrates",
        "important": "key",
        "the":       "this"
    }

    def paraphrase(text):
        words = text.split()
        return " ".join(
            [replacements.get(w.lower(), w) for w in words]
        )

    av.eval(); ar.eval()
    with torch.no_grad():
        for i in range(n_samples):
            h    = activations[i].unsqueeze(0).to(device)
            orig = av.inject_and_generate(h, max_new_tokens=80)
            para = paraphrase(orig)
            orig_mses.append(
                mse_fn(ar.forward(orig),
                       activations[i].unsqueeze(0)).item()
            )
            para_mses.append(
                mse_fn(ar.forward(para),
                       activations[i].unsqueeze(0)).item()
            )

    orig_mean = sum(orig_mses) / n_samples
    para_mean = sum(para_mses) / n_samples
    ratio     = para_mean / (orig_mean + 1e-8)

    print(f"Original MSE   : {orig_mean:.4f}")
    print(f"Paraphrased MSE: {para_mean:.4f}")
    print(f"Ratio          : {ratio:.3f}")
    if ratio < 1.3:
        print("Result: Low steganography — AR uses semantic content")
    else:
        print("Result: Possible steganography detected")

    return {
        "original_mse":   round(orig_mean, 4),
        "paraphrase_mse": round(para_mean, 4),
        "ratio":          round(ratio, 3)
    }

if __name__ == '__main__':
    print('Run via notebook: notebooks/nla_gpt2.ipynb')
    print('See src/evaluate.py for compute_fve, qualitative_evaluation functions.')
