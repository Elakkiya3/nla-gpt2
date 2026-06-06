# Natural Language Autoencoder on GPT-2

## Overview

This project reimplements the core methodology from Anthropic's paper *Natural Language Autoencoders Produce Unsupervised Explanations of LLM Activations* (Fraser-Taliente et al., 2026) on a small open-source language model.

The goal of a Natural Language Autoencoder (NLA) is to compress an internal activation into natural language and then reconstruct the original activation from that text.

The architecture consists of two components:

* **Activation Verbalizer (AV)**: converts a model activation into a natural language explanation.
* **Activation Reconstructor (AR)**: reconstructs the original activation from the generated explanation.

The central question is whether a natural language bottleneck can preserve meaningful information contained in model activations.

The primary evaluation metric is **Fraction of Variance Explained (FVE)**:

FVE = 1 − E[‖h − AR(AV(h))‖²] / E[‖h − h̄‖²]

where:

* h is the original activation
* h̄ is the mean activation
* AR(AV(h)) is the reconstructed activation

FVE = 0 corresponds to predicting the mean activation.

FVE = 1 corresponds to perfect reconstruction.

Anthropic reports approximately 0.6–0.8 FVE on Claude-scale models.

---

# Why This Approach Matters

Many interpretability techniques recover latent features that remain difficult for humans to understand directly.

Natural Language Autoencoders replace the latent bottleneck with natural language explanations. If activations can be reconstructed accurately after passing through text, then the generated explanations may provide insight into what information the model internally represents.

This makes NLAs an interesting bridge between mechanistic interpretability and human-readable explanations.

---

# Model Selection

I chose **GPT-2 (124M parameters)** because:

* It fits comfortably on a free Kaggle Tesla T4 GPU.
* Its architecture is well understood.
* It allows multiple experiments within a limited compute budget.
* It provides a realistic small-scale testbed for reproducing the paper's methodology.

Configuration:

| Setting      | Value        |
| ------------ | ------------ |
| Model        | GPT-2 (124M) |
| Layers       | 12           |
| Target Layer | 8            |
| Dataset      | WikiText-103 |
| GPU          | Tesla T4     |
| VRAM         | 15.6 GB      |

Layer 8 was selected because the paper focuses on middle-to-late transformer layers.

---

# Architecture

## Activation Verbalizer (AV)

The Activation Verbalizer receives a residual-stream activation from GPT-2 and generates a natural language explanation.

Implementation:

* GPT-2 language model
* Linear projection (768 → 768)
* Activation injection into embedding space
* Scaling factor α = 10
* Autoregressive text generation

The verbalizer attempts to describe information encoded in the activation using natural language.

---

## Activation Reconstructor (AR)

The Activation Reconstructor receives the generated explanation and predicts the original activation.

The paper uses a full language model as the reconstructor. Because two GPT-2 models exceed Kaggle T4 memory limits, I implemented a smaller reconstructor:

EmbeddingBag → Linear → Residual MLP Blocks → Linear

Architecture:

* EmbeddingBag(50257, 128)
* Linear(128 → 1024)
* Four residual blocks
* LayerNorm + GELU
* Linear(1024 → 768)

This is the largest architectural deviation from the original paper and likely a major reason for lower reconstruction performance.

---

# Training Procedure

During experimentation I found that joint optimization from random initialization consistently failed.

The reconstructor first needed to learn the activation space before meaningful verbalizer learning could occur.

I therefore used a three-stage training procedure.

## Phase A — Reconstructor Warmup

300 steps

The reconstructor was trained to match activation norms and avoid near-zero outputs.

Without this stage, subsequent training repeatedly collapsed.

## Phase B — Reconstructor Training

800 steps

The reconstructor was trained on matched activation-summary pairs.

This phase produced the first stable positive FVE values.

Peak FVE reached approximately 0.09.

## Phase C — Joint Supervised Fine-Tuning

600 steps

The verbalizer and reconstructor were trained jointly.

Surprisingly, reconstruction quality decreased during this stage.

This suggests that at small scale the verbalizer may introduce noise faster than the reconstructor can adapt.

## Reward-Weighted Fine-Tuning

500 steps

Inspired by the reinforcement-learning stage described in the paper.

Procedure:

1. Generate multiple candidate explanations.
2. Reconstruct activations.
3. Score explanations using reconstruction error.
4. Prefer explanations with higher reward.

Reward:

r = −log ||h − h'||²

This is a simplified reward-weighted optimization procedure rather than a full implementation of GRPO.

---

# Results

## Quantitative Results

| Stage                            | FVE       |
| -------------------------------- | --------- |
| Phase B Peak                     | ~0.09     |
| Phase C End                      | 0.014     |
| Reward-Weighted Fine-Tuning Best | 0.0455    |
| Anthropic (Claude-scale)         | 0.60–0.80 |

Final result:

**FVE = 0.0455**

Although substantially below the values reported by Anthropic, the system achieved positive reconstruction despite severe model-size and compute constraints.

---

# Qualitative Failure Modes

The most obvious limitation appears in the generated explanations.

Examples:

| Input Context                              | Generated Explanation                    |
| ------------------------------------------ | ---------------------------------------- |
| "The capital of France is Paris..."        | "Several of Zagreb's major cities..."    |
| "The mitochondria is the powerhouse..."    | "Henry VIII established a fleet..."      |
| "Neil Armstrong became the first human..." | "University of Cambridge in Oxford..."   |
| Python Fibonacci code                      | "The Usonian Center for Astrophysics..." |

The generated text is usually grammatical and fluent but often fails to preserve the semantic content of the source activation.

This suggests that the verbalizer learned the style of WikiText prose more strongly than the activation-conditioning signal.

---

# Steganography Check

A concern raised in the paper is whether reconstruction relies on hidden token-level patterns rather than semantic meaning.

To investigate this, I paraphrased generated explanations and measured reconstruction quality.

| Condition               | MSE   |
| ----------------------- | ----- |
| Original Explanation    | 9.71  |
| Paraphrased Explanation | 9.68  |
| Ratio                   | 0.997 |

The reconstruction error changed very little after paraphrasing.

This suggests that reconstruction is relatively insensitive to superficial wording changes.

However, because overall FVE remains low and generated explanations are often weakly aligned with source content, this result should be interpreted cautiously.

---

# Interesting Findings

## Three-Phase Training Was Essential

Every attempt at end-to-end training from random initialization failed.

Pretraining the reconstructor first was necessary for stable optimization.

This observation was one of the most important practical findings of the project.

## Joint Training Reduced Performance

Another surprising result was:

* Phase B FVE ≈ 0.09
* Phase C FVE ≈ 0.014

Joint optimization reduced reconstruction quality rather than improving it.

One possible explanation is that the verbalizer changed faster than the reconstructor could adapt.

## L2 Normalization Produced Unstable FVE

The paper normalizes activations before evaluation.

When applied directly to GPT-2 activations, I observed:

* Baseline MSE ≈ 0.001
* Extremely unstable FVE values
* FVE often below −2000

Using unnormalized activations produced stable measurements:

* Mean activation norm ≈ 110
* Baseline MSE ≈ 9.7

I did not investigate whether this effect persists for larger models.

---

# Why Results Differ From The Paper

Several factors likely explain the gap between 0.0455 FVE and Anthropic's reported 0.6–0.8.

### Reconstructor Capacity

The paper uses a full language model as the reconstructor.

This implementation uses a relatively small MLP.

This is likely the largest contributor to the performance gap.

### Model Scale

GPT-2 (124M) is far smaller than Claude-scale systems.

Larger models may contain more verbalizable internal representations.

### Training Budget

The paper trains for substantially longer.

My experiments used roughly 2000 optimization steps.

### Weak Supervision

The paper benefits from stronger explanation-generation procedures.

I relied on proxy summaries generated from the training corpus.

---

# Limitations and Open Questions

Several questions remain open:

* Is reconstructor capacity the dominant bottleneck?
* Would a full language-model reconstructor significantly improve FVE?
* Are proxy summaries sufficient for small-scale NLA training?
* Does the paraphrasing result reflect semantic understanding or broader distributional similarity?

These would be natural directions for future work.

---

# Reproducibility

All experiments were run on:

| Resource     | Value        |
| ------------ | ------------ |
| Platform     | Kaggle       |
| GPU          | Tesla T4     |
| VRAM         | 15.6 GB      |
| Dataset      | WikiText-103 |
| Model        | GPT-2 124M   |
| Target Layer | 8            |
| Runtime      | ~4 hours     |

Repository structure:

```text
src/
├── config.py
├── data.py
├── models.py
├── train.py
└── evaluate.py

data/
├── qualitative_results.json

figures/
├── fve_curve.png

README.md
requirements.txt
```

To reproduce:

```bash
git clone https://github.com/Elakkiya3/nla-gpt2
cd nla-gpt2

pip install -r requirements.txt

# Run the Kaggle notebook or execute the pipeline
# using the source files in src/
```

Final reported metrics:

* FVE = 0.0455
* Steganography ratio = 0.997

---

# Reference

Fraser-Taliente, Kantamneni, Ong et al. (2026).

Natural Language Autoencoders Produce Unsupervised Explanations of LLM Activations.

https://transformer-circuits.pub/2026/nla/index.html
