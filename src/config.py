from pathlib import Path
import torch


# --------------------------------------------------
# Project paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

OUTPUT_DIR = PROJECT_ROOT / "outputs"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
REPORT_DIR = OUTPUT_DIR / "reports"
EXPLANATION_DIR = OUTPUT_DIR / "explanations"
GENERATED_DATA_DIR = OUTPUT_DIR / "generated_samples"


# --------------------------------------------------
# Dataset
# --------------------------------------------------

IMDB_FILE = RAW_DATA_DIR / "IMDB Dataset.csv"

RANDOM_SEED = 42

TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15
TEST_RATIO = 0.15


# --------------------------------------------------
# Text processing
# --------------------------------------------------

# Single source of truth for sequence length: training, evaluation and
# every explainer read this value, so they cannot drift apart.
#
# 512 (BERT's architectural maximum) rather than 256 because IMDB reviews
# average roughly 230 words, which is ~300 word-piece tokens, so a 256-token
# window truncates over half the corpus. Substitutions past the cut-off are
# invisible to the model: the stego sample becomes byte-identical to its
# cover within the visible window while still carrying label 1. That is
# irreducible label noise, and it caps achievable ROC-AUC.
#
# Cost: roughly 2x the compute and activation memory of 256. If training
# runs out of VRAM, drop this to 256 and record the value alongside the
# result -- it is an experimental variable, not a free parameter.
MAX_LENGTH = 512

# Maximum number of substitutions per review.
# This sets the embedding rate together with review length, and is the
# dominant factor in how detectable the stego text is. Report it with
# every accuracy figure; accuracy without an embedding rate is unreadable.
MAX_SYNONYM_SUBSTITUTIONS = 8

# Minimum number of words in a review before modification
MIN_REVIEW_WORDS = 20


# --------------------------------------------------
# BERT
# --------------------------------------------------

BERT_MODEL_NAME = "bert-base-uncased"

BERT_NUM_LABELS = 2

BERT_BATCH_SIZE = 8
BERT_EPOCHS = 3
BERT_LEARNING_RATE = 2e-5
BERT_WEIGHT_DECAY = 0.01
BERT_WARMUP_RATIO = 0.1


# --------------------------------------------------
# Algorithm identification
# --------------------------------------------------

ALGORITHM_LABELS = {
    "cover": 0,
    "synonym": 1,
    "vae": 2,
}

ALGORITHM_LABEL_NAMES = {
    0: "cover",
    1: "synonym",
    2: "vae",
}

ALGORITHM_NUM_LABELS = len(ALGORITHM_LABELS)


# --------------------------------------------------
# Explainability
# --------------------------------------------------

IG_N_STEPS = 50
IG_INTERNAL_BATCH_SIZE = 1

TOP_K_EXPLANATION_WORDS = 20


# --------------------------------------------------
# Hardware
# --------------------------------------------------

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# RTX 3050 generally has limited VRAM.
# Keep this conservative.
USE_MIXED_PRECISION = torch.cuda.is_available()


def create_directories():
    directories = [
        DATA_DIR,
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        OUTPUT_DIR,
        CHECKPOINT_DIR,
        REPORT_DIR,
        EXPLANATION_DIR,
        GENERATED_DATA_DIR,
    ]

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    create_directories()
    print(f"Using device: {DEVICE}")