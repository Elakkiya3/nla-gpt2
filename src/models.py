# src/models.py
# Activation Verbalizer (AV) and Activation Reconstructor (AR)

import torch
import torch.nn as nn
from transformers import AutoTokenizer, GPT2LMHeadModel
from src.config import (
    MODEL_NAME, HIDDEN_DIM, ALPHA,
    AR_EMB_DIM, AR_HIDDEN, TARGET_LAYER
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token    = tokenizer.eos_token
tokenizer.padding_side = "left"

device = "cuda" if torch.cuda.is_available() else "cpu"


class ActivationVerbalizer(nn.Module):
    """
    Maps a residual stream activation h_l to a text explanation z.

    Architecture:
    - GPT-2 (124M) as the base language model
    - Learned linear projection: R^768 -> R^768
    - Activation injected at fixed position in prompt
    - Autoregressive generation produces explanation z

    Follows Fraser-Taliente et al. (2026) Section 3:
    'The AV is an LLM with the same architecture as M'
    """
    def __init__(self):
        super().__init__()
        self.lm       = GPT2LMHeadModel.from_pretrained(MODEL_NAME)
        self.act_proj = nn.Linear(HIDDEN_DIM, 768, bias=True)
        # Small init for stability (paper: warm-start prevents degenerate outputs)
        nn.init.normal_(self.act_proj.weight, std=0.01)
        nn.init.zeros_(self.act_proj.bias)
        self.alpha = ALPHA

    def inject_and_generate(self, h, max_new_tokens=80,
                             temperature=0.8):
        """
        h: (1, 768) activation tensor on GPU
        Returns: explanation string
        """
        prompt  = "The activation vector encodes the following:\n"
        enc     = tokenizer(prompt, return_tensors="pt").to(device)
        p_emb   = self.lm.transformer.wte(enc.input_ids)
        act_emb = self.act_proj(h * self.alpha).unsqueeze(1)
        full    = torch.cat([p_emb, act_emb], dim=1)
        with torch.no_grad():
            out = self.lm.generate(
                inputs_embeds=full,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                repetition_penalty=1.3,
                pad_token_id=tokenizer.eos_token_id
            )
        skip = enc.input_ids.shape[1] + 1
        return tokenizer.decode(
            out[0][skip:], skip_special_tokens=True
        ).strip()

    def forward_train(self, h, target_ids):
        """
        Teacher-forced cross-entropy for SFT training.
        h: (1, 768) on GPU
        target_ids: (1, T) token ids on GPU
        Returns: scalar loss
        """
        prompt  = "The activation vector encodes the following:\n"
        enc     = tokenizer(prompt, return_tensors="pt").to(device)
        p_emb   = self.lm.transformer.wte(enc.input_ids)
        act_emb = self.act_proj(h * self.alpha).unsqueeze(1)
        full    = torch.cat([p_emb, act_emb], dim=1)
        T       = full.shape[1]
        ignore  = torch.full(
            (1, T), -100, dtype=torch.long, device=device
        )
        labels  = torch.cat([ignore, target_ids], dim=1)
        s_emb   = self.lm.transformer.wte(target_ids)
        all_emb = torch.cat([full, s_emb], dim=1)
        return self.lm(inputs_embeds=all_emb, labels=labels).loss


class ActivationReconstructor(nn.Module):
    """
    Maps a text explanation z back to an activation h_l.

    Architecture:
    - EmbeddingBag for text encoding (mean pooling over tokens)
    - 5-layer MLP with residual connections and LayerNorm
    - Output: R^768 predicted activation

    Note: The paper uses a full LM as AR. We use a stronger MLP
    due to T4 GPU memory constraints (two GPT-2 models exceed 15.6GB).
    This is the primary factor limiting our FVE vs the paper's 0.6-0.8.
    """
    def __init__(self):
        super().__init__()
        self.embed = nn.EmbeddingBag(50257, AR_EMB_DIM, mode="mean")
        self.fc1   = nn.Linear(AR_EMB_DIM, AR_HIDDEN)
        self.fc2   = nn.Linear(AR_HIDDEN,  AR_HIDDEN)
        self.fc3   = nn.Linear(AR_HIDDEN,  AR_HIDDEN)
        self.fc4   = nn.Linear(AR_HIDDEN,  AR_HIDDEN)
        self.out   = nn.Linear(AR_HIDDEN,  HIDDEN_DIM)
        self.ln1   = nn.LayerNorm(AR_HIDDEN)
        self.ln2   = nn.LayerNorm(AR_HIDDEN)
        self.ln3   = nn.LayerNorm(AR_HIDDEN)
        self.ln4   = nn.LayerNorm(AR_HIDDEN)
        self.act   = nn.GELU()
        self.drop  = nn.Dropout(0.1)

    def forward(self, text):
        """
        text: string
        Returns: (1, 768) predicted activation on CPU
        """
        tids = tokenizer(
            text, return_tensors="pt",
            max_length=100, truncation=True
        ).input_ids
        x = self.embed(tids)
        x = self.act(self.ln1(self.fc1(x)))
        r = x
        x = self.drop(self.act(self.ln2(self.fc2(x)))) + r
        r = x
        x = self.drop(self.act(self.ln3(self.fc3(x)))) + r
        x = self.act(self.ln4(self.fc4(x)))
        return self.out(x)