# src/train.py
# SFT warm-start and RL training loops

import torch
import torch.nn as nn
import torch.optim as optim
import random
from src.config import (
    TRAIN_SIZE, VAL_SIZE,
    SFT_A_STEPS, SFT_B_STEPS, SFT_C_STEPS,
    LR_AV_SFT, LR_AR_SFT,
    RL_STEPS, G, LR_AV_RL, LR_AR_RL,
    AV_SFT_PATH, AR_SFT_PATH,
    AV_BEST_PATH, AR_BEST_PATH,
    FVE_HISTORY_PATH
)
from src.models import tokenizer
from src.evaluate import compute_fve

device = "cuda" if torch.cuda.is_available() else "cpu"
mse_fn = nn.MSELoss()


def phase_a_ar_standalone(ar, activations, summaries):
    """
    Phase A: Train AR alone to get output into correct range.
    Without this, joint training diverges immediately.
    """
    print(f"Phase A: AR standalone ({SFT_A_STEPS} steps)...")
    opt = optim.AdamW(ar.parameters(), lr=1e-3)

    for step in range(SFT_A_STEPS):
        idx    = random.randint(0, TRAIN_SIZE-1)
        h_cpu  = activations[idx].unsqueeze(0)
        s      = summaries[idx]
        h_pred = ar.forward(s)
        loss   = mse_fn(h_pred, h_cpu)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(ar.parameters(), 1.0)
        opt.step()

        if step % 100 == 0:
            print(f"  Step {step:3d} | loss={loss.item():.4f} "
                  f"| pred_norm={h_pred.norm():.2f} "
                  f"| target_norm={h_cpu.norm():.2f}")

    print("Phase A done\n")
    return ar


def phase_b_ar_matched(ar, activations, summaries):
    """
    Phase B: Train AR on matched (summary, activation) pairs.
    AR learns the actual summary->activation mapping.
    """
    print(f"Phase B: AR matched pairs ({SFT_B_STEPS} steps)...")
    opt = optim.AdamW(ar.parameters(), lr=5e-4)
    sch = optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=SFT_B_STEPS
    )

    for step in range(SFT_B_STEPS):
        idx    = random.randint(0, TRAIN_SIZE-1)
        h_cpu  = activations[idx].unsqueeze(0)
        s      = summaries[idx]
        h_pred = ar.forward(s)
        loss   = mse_fn(h_pred, h_cpu)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(ar.parameters(), 1.0)
        opt.step()
        sch.step()

        if step % 200 == 0:
            val_mse = 0.0
            ar.eval()
            with torch.no_grad():
                for vi in range(30):
                    vi_idx  = TRAIN_SIZE + vi
                    vh      = activations[vi_idx].unsqueeze(0)
                    vs      = summaries[vi_idx % len(summaries)]
                    val_mse += mse_fn(ar.forward(vs), vh).item()
            ar.train()
            val_mse /= 30
            baseline = ((activations[:TRAIN_SIZE] -
                         activations[:TRAIN_SIZE].mean(0)
                         )**2).mean().item()
            fve = 1.0 - val_mse/(baseline+1e-8)
            print(f"  Step {step:4d} | loss={loss.item():.4f} "
                  f"| val_FVE={fve:.4f}")

    print("Phase B done\n")
    return ar


def phase_c_joint_sft(av, ar, activations, summaries):
    """
    Phase C: Joint AV + AR supervised fine-tuning.
    AV learns to generate summaries given activations.
    AR continues to improve reconstruction.
    """
    print(f"Phase C: Joint SFT ({SFT_C_STEPS} steps)...")
    opt_av = optim.AdamW(av.parameters(), lr=LR_AV_SFT)
    opt_ar = optim.AdamW(ar.parameters(), lr=LR_AR_SFT)
    fve_sft = 0.0

    for step in range(SFT_C_STEPS):
        idx   = random.randint(0, TRAIN_SIZE-1)
        h_gpu = activations[idx].unsqueeze(0).to(device)
        h_cpu = activations[idx].unsqueeze(0)
        s     = summaries[idx]

        # AR update (CPU)
        for _ in range(2):
            h_pred  = ar.forward(s)
            loss_ar = mse_fn(h_pred, h_cpu)
            opt_ar.zero_grad()
            loss_ar.backward()
            torch.nn.utils.clip_grad_norm_(ar.parameters(), 1.0)
            opt_ar.step()

        # AV update (GPU)
        tids    = tokenizer(
            s, return_tensors="pt",
            max_length=80, truncation=True
        ).input_ids.to(device)
        loss_av = av.forward_train(h_gpu, tids)
        opt_av.zero_grad()
        loss_av.backward()
        torch.nn.utils.clip_grad_norm_(av.parameters(), 1.0)
        opt_av.step()
        torch.cuda.empty_cache()

        if step % 150 == 0:
            fve, m, b = compute_fve(
                av, ar,
                activations[TRAIN_SIZE:TRAIN_SIZE+VAL_SIZE],
                n_samples=40
            )
            fve_sft = fve
            print(f"  Step {step:4d} | AR={loss_ar.item():.4f} "
                  f"| AV={loss_av.item():.4f} "
                  f"| FVE={fve:.4f}")

    torch.save(av.state_dict(), AV_SFT_PATH)
    torch.save(ar.state_dict(), AR_SFT_PATH)
    print(f"Phase C done. FVE = {fve_sft:.4f}\n")
    return av, ar, fve_sft


def rl_training(av, ar, activations, fve_sft):
    """
    RL training using GRPO-style policy gradient.

    For each activation:
    1. Sample G explanations from AV
    2. Score each by reconstruction MSE under AR
    3. Normalize rewards to advantages
    4. Update AV toward higher-reward explanations
    5. Update AR with supervised regression

    Reward: r = -log||h - AR(AV(h))||^2
    """
    opt_av_rl  = optim.AdamW(av.parameters(), lr=LR_AV_RL)
    opt_ar_rl  = optim.AdamW(ar.parameters(), lr=LR_AR_RL)
    fve_history = []
    best_fve    = 0.0

    print(f"RL training ({RL_STEPS} steps, G={G})...")
    print("-" * 55)

    for step in range(RL_STEPS):
        idx   = random.randint(0, TRAIN_SIZE-1)
        h_gpu = activations[idx].unsqueeze(0).to(device)
        h_cpu = activations[idx].unsqueeze(0)

        # Sample G explanations and score
        explanations, rewards = [], []
        for _ in range(G):
            with torch.no_grad():
                z     = av.inject_and_generate(
                    h_gpu, max_new_tokens=80
                )
                h_hat = ar.forward(z)
                r     = -torch.log(
                    mse_fn(h_hat, h_cpu) + 1e-8
                ).item()
            explanations.append(z)
            rewards.append(r)

        # GRPO advantages
        r_mean = sum(rewards) / G
        r_std  = (
            sum((r-r_mean)**2 for r in rewards)/G
        )**0.5 + 1e-8
        advs   = [(r-r_mean)/r_std for r in rewards]

        # AR update (CPU)
        ar_loss = torch.tensor(0.0)
        for z in explanations:
            ar_loss = ar_loss + mse_fn(ar.forward(z), h_cpu)
        ar_loss = ar_loss / G
        opt_ar_rl.zero_grad()
        ar_loss.backward()
        torch.nn.utils.clip_grad_norm_(ar.parameters(), 1.0)
        opt_ar_rl.step()

        # AV update (GPU)
        av_loss = torch.tensor(0.0, device=device)
        for z, adv in zip(explanations, advs):
            tids = tokenizer(
                z, return_tensors="pt",
                max_length=80, truncation=True
            ).input_ids.to(device)
            if tids.shape[1] < 2:
                continue
            av_loss = av_loss + (
                -adv * av.forward_train(h_gpu, tids)
            )
        av_loss = av_loss / G
        opt_av_rl.zero_grad()
        av_loss.backward()
        torch.nn.utils.clip_grad_norm_(av.parameters(), 1.0)
        opt_av_rl.step()
        torch.cuda.empty_cache()

        if step % 50 == 0:
            fve, m, b = compute_fve(
                av, ar,
                activations[TRAIN_SIZE:TRAIN_SIZE+VAL_SIZE],
                n_samples=50
            )
            fve_history.append((step, fve))
            best_z = explanations[rewards.index(max(rewards))]
            mem    = torch.cuda.memory_allocated()/1e9

            print(f"\nStep {step:3d} | FVE={fve:.4f} "
                  f"| AR={ar_loss.item():.4f} "
                  f"| GPU={mem:.1f}GB")
            print(f"  → {best_z[:90]}...")

            if fve > best_fve:
                best_fve = fve
                torch.save(av.state_dict(), AV_BEST_PATH)
                torch.save(ar.state_dict(), AR_BEST_PATH)
                print(f"  *** Best FVE: {best_fve:.4f} ***")

    torch.save(fve_history, FVE_HISTORY_PATH)
    print(f"\nRL done. Best FVE = {best_fve:.4f}")
    return av, ar, fve_history, best_fve