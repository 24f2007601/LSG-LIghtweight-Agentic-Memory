"""
constants.py — Centralised configuration constants for LSG.
All magic numbers and domain strings are defined here.
"""

# ── Dataset ───────────────────────────────────────────────────────────────────
MAX_SAMPLES_PER_DOMAIN: int = 2_000
DOMAIN_TASKORIENTED: str = "taskoriented"
DOMAIN_PERSONACHAT: str = "personachat"

BITEXT_DATASET: str = "bitext/Bitext-customer-support-llm-chatbot-training-dataset"
PERSONACHAT_DATASET: str = "AlekseyKorshuk/persona-chat"

# ── Splits ────────────────────────────────────────────────────────────────────
TRAIN_SPLIT_SEED: int = 42
SOURCE_VAL_RATIO: float = 0.2
FEW_SHOT_POOL_SIZE: int = 200
INDOMAIN_TEST_RATIO: float = 0.4

# ── Model hyperparameters ──────────────────────────────────────────────────────
EMB_MODEL: str = "all-MiniLM-L6-v2"
EMB_DIM: int = 384
CONTEXT_WINDOW_K: int = 2

LR_C: float = 0.5
LR_MAX_ITER: int = 2_000

XGB_N_ESTIMATORS: int = 300
XGB_MAX_DEPTH: int = 5
XGB_LEARNING_RATE: float = 0.05
XGB_SUBSAMPLE: float = 0.8
XGB_COL_SAMPLE: float = 0.8
XGB_MIN_CHILD_WEIGHT: int = 3

MLP_HIDDEN_SIZES: tuple = (256, 64)
MLP_MAX_ITER: int = 400
MLP_ALPHA: float = 1e-3
MLP_VAL_FRACTION: float = 0.1

# ── Baseline ───────────────────────────────────────────────────────────────────
BASELINE_NOVELTY_THRESH: float = 0.85
BASELINE_RECENCY_DECAY: float = 0.95
BASELINE_REPORTED_LATENCY_MS: float = 8.614

# ── Calibration & evaluation ───────────────────────────────────────────────────
ECE_N_BINS: int = 10
UTILITY_LAMBDA: float = 0.3
BOOTSTRAP_N_ITER: int = 1_000
BOOTSTRAP_ALPHA: float = 0.05

# ── LLM Judge noise ───────────────────────────────────────────────────────────
JUDGE_NOISE_RATE: float = 0.03
JUDGE_SEED: int = 42

# ── Paths ─────────────────────────────────────────────────────────────────────
EMB_CACHE_DIR: str = ".emb_cache"
DEFAULT_RESULTS_DIR: str = "results2"
ASSETS_DIR: str = "assets"
