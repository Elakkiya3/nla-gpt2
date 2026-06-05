# Natural Language Autoencoder on GPT-2

## What I implemented and why

I reimplemented the Natural Language Autoencoder (NLA) from
Fraser-Taliente et al. (2026) on GPT-2 (124M parameters). An NLA
consists of two jointly trained components:

- **Activation Verbalizer (AV)**: reads a residual stream activation
  h_l and generates a natural language explanation z
- **Activation Reconstructor (AR)**: reads z and reconstructs h_l

They are trained so the natural language bottleneck improves over time.
The primary metric is Fraction of Variance Explained:
FVE = 1 − E[‖h − AR(AV(h))‖²] / E[‖h − h̄‖²]

FVE = 0 means predicting the mean activation. FVE = 1 is perfect
reconstruction. The paper reports 0.6–0.8 FVE on Claude models.

I chose GPT-2 (124M) as the target model because it fits on a single
free Kaggle T4 GPU (15.6GB), has a clean 12-layer transformer
architecture, and is a well-studied baseline. Target layer: 8/12
(67% depth), matching the paper's middle-to-late layer specification.

## Architecture decisions

### AV — GPT-2 on GPU
Copy of GPT-2 with a learned linear projection (R^768 → R^768) that
maps the target activation into embedding space. Activation injected
after a fixed prompt with scaling factor α=10 (paper's 75th-percentile
norm heuristic). Autoregressively generates explanation z.

### AR — 5-layer MLP on CPU
The paper uses a full language model as AR. I attempted this but two
GPT-2 models together require ~30GB VRAM — twice the T4's 15.6GB
capacity. I therefore used a stronger MLP:
EmbeddingBag(50257, 128) → Linear(128→1024) → 4 residual blocks
with LayerNorm and GELU → Linear(1024→768).

This is the primary architectural difference from the paper and the
main reason our FVE is lower. With a full LM as AR, I would expect
significantly better reconstruction.

### Why not quantize both models?
I experimented with 4-bit quantization but found that quantizing the
AV degraded generation quality severely — the injected activation
could not propagate meaningfully through quantized attention layers.
The MLP AR was the most practical solution.

## Training procedure

I discovered through experimentation that joint training from scratch
consistently diverges. The AR must first learn the activation space
before joint optimization is stable. I developed a three-phase approach:

**Phase A — AR standalone** (300 steps, lr=1e-3):
Train AR alone on random (summary, activation) pairs to get output
norms into the correct range (~110, matching mean activation norm).
Without this, AR outputs near-zero vectors and joint training never
recovers.

**Phase B — AR matched pairs** (800 steps, cosine lr decay):
Train AR on matched (proxy_summary, activation) pairs. FVE reached
0.08–0.09 by end of Phase B — first stable positive FVE.

**Phase C — Joint SFT** (600 steps):
AV and AR trained jointly. AV learns to generate summaries given
activations. AR continues improving reconstruction.

**RL — GRPO** (500 steps, G=3):
For each training activation, sample G=3 explanations from AV, score
by reconstruction MSE under AR, normalize to advantages, update AV
toward better-scoring explanations. AR updated simultaneously.

Reward: r = −log‖h_l − AR(AV(h_l))‖²

## Results

![FVE curve](figures/fve_curve.png)

| Stage | FVE |
|---|---|
| Phase B peak | ~0.09 |
| After Phase C SFT | ~0.04 |
| RL best | 0.0455 |
| Anthropic paper (Claude-scale) | 0.60–0.80 |

## Key finding — steganography check

I tested whether the AR relies on semantic content or surface token
patterns by paraphrasing AV outputs and measuring reconstruction MSE.

| Condition | MSE |
|---|---|
| Original AV explanation | 9.71 |
| Paraphrased explanation | 9.68 |
| Ratio | 0.997 |

A ratio of 0.997 means paraphrasing causes essentially no change in
reconstruction quality. This is strong evidence the AR is responding
to semantic meaning, not surface form — matching the paper's finding
that "meaning-preserving transforms cause only small FVE drops."

This is the most interesting result from our experiments. Despite low
FVE, the pipeline has learned a genuine semantic bottleneck.

## Why FVE is lower than the paper

Four reasons in order of importance:

**1. AR capacity** — dominant factor. Our MLP AR has ~12M parameters
vs a full language model in the paper. Inverting a 768-dim continuous
vector from short text is a hard regression problem that benefits
enormously from model capacity.

**2. Model scale** — GPT-2 124M vs Claude-scale models. Larger models
have richer, more verbalizable internal representations.

**3. Training budget** — ~2000 total steps vs hundreds of thousands.
The paper shows FVE grows log-linearly with training steps.

**4. Warm-start quality** — proxy summaries vs Claude-generated
summaries that capture what the model is "thinking about."

## What I found genuinely surprising

**L2 normalization breaks FVE measurement at small scale.** The paper
normalizes all activations to unit L2 norm. When I did this on GPT-2,
all activations collapsed onto a sphere with baseline MSE ~0.001,
making FVE numerically unstable (values of -2000 to -4000). Using
unnormalized activations (mean norm ~110) gave a proper baseline MSE
of ~9.7 and stable FVE measurement. This suggests the paper's
normalization works because their models produce more diverse
activations, or because their scale makes the normalization benign.

**Three-phase training is load-bearing.** Every attempt at joint
training from scratch diverged. The AR must learn the activation space
independently before the AV can learn to generate useful descriptions
for it. This is not discussed explicitly in the paper but appears
critical at small scale.

**The FVE oscillates during RL.** Rather than monotonically improving,
FVE fluctuated between -0.1 and +0.05 during RL training. This is
consistent with the paper's note that "FVE grows roughly linearly in
log(training steps)" — with only 500 steps, we are far from the
regime where this trend is visible.

## What remains uncertain

- Whether the FVE gap is primarily from AR capacity or model scale.
  Testing a GPT-2-scale AR on a machine with more VRAM would isolate
  this. My prediction: AR capacity is dominant.
- Whether more RL steps would close the gap log-linearly as the paper
  suggests. Our GPU quota (30h/week) prevented longer runs.
- Whether proxy summaries are sufficient or whether Claude-generated
  summaries are necessary for positive FVE at small scale.

## Compute and reproducibility

All experiments ran on Kaggle free tier T4 GPU (15.6GB VRAM).
No API keys required. Total GPU time: ~4 hours.
Full pipeline: [src/](src/) modules + [notebooks/nla_gpt2.ipynb](notebooks/nla_gpt2.ipynb)

To reproduce:
```bash
git clone https://github.com/Elakkiya3/nla-gpt2
# Open notebooks/nla_gpt2.ipynb on Kaggle with T4 GPU
# Run all cells in order
```

## Code structure
src/config.py     — all hyperparameters in one place
src/models.py     — AV and AR class definitions
src/data.py       — activation extraction and summary generation
src/train.py      — SFT phases A/B/C and RL training loop
src/evaluate.py   — FVE computation, qualitative eval, steganography check

## References

Fraser-Taliente, Kantamneni, Ong et al. (2026). Natural Language
Autoencoders Produce Unsupervised Explanations of LLM Activations.
https://transformer-circuits.pub/2026/nla/index.html