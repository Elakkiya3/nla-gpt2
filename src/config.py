# src/config.py
# Central configuration — change settings here only

MODEL_NAME   = "gpt2"
TARGET_LAYER = 8        # 2/3 of 12 layers (paper spec)
HIDDEN_DIM   = 768      # GPT-2 hidden size
ALPHA        = 10.0     # activation scaling factor

# Data
N_TEXTS      = 60_000   # texts for activation extraction
N_WARMUP     = 5000     # texts for SFT warm-start
TRAIN_SIZE   = 3000     # training split
VAL_SIZE     = 500      # validation split
MAX_LEN      = 128      # max tokenizer length

# SFT
SFT_A_STEPS  = 300      # Phase A: AR standalone
SFT_B_STEPS  = 800      # Phase B: AR matched pairs
SFT_C_STEPS  = 600      # Phase C: joint AV + AR
LR_AV_SFT    = 3e-5
LR_AR_SFT    = 1e-4

# RL
RL_STEPS     = 500
G            = 3        # group size for GRPO
LR_AV_RL     = 3e-6
LR_AR_RL     = 5e-5

# AR architecture
AR_EMB_DIM   = 128
AR_HIDDEN    = 1024

# Paths
ACTIVATIONS_PATH = "/kaggle/working/activations.pt"
SUMMARIES_PATH   = "/kaggle/working/summaries.pt"
TEXTS_PATH       = "/kaggle/working/texts.pt"
AV_SFT_PATH      = "/kaggle/working/av_sft.pt"
AR_SFT_PATH      = "/kaggle/working/ar_sft.pt"
AV_BEST_PATH     = "/kaggle/working/av_best.pt"
AR_BEST_PATH     = "/kaggle/working/ar_best.pt"
FVE_HISTORY_PATH = "/kaggle/working/fve_history.pt"