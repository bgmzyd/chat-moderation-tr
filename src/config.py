"""Central configuration. Every script reads paths and hyper-parameters from here,
so all models are trained and evaluated under exactly the same settings."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
SPLIT_DIR = DATA_DIR / "splits"
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"

# ---------------------------------------------------------------- data
# Official OffensEval-TR 2020 release (Çöltekin, 2020), CC-BY 2.0
DATA_URL = "https://coltekin.github.io/offensive-turkish/offenseval2020-turkish.zip"
SEED = 42
VAL_FRACTION = 0.10          # carved (stratified) from the official training set
LABELS = ["NOT", "OFF"]      # 0 = clean, 1 = offensive

# ---------------------------------------------------------------- models
MODEL_KEYS = ["lexicon", "tfidf", "w2v_ffn", "bert_fp32", "bert_int8"]
MODEL_NAMES = {
    "lexicon":   "Rule: edit-distance lexicon matcher",
    "tfidf":     "Baseline: tf-idf + logistic regression",
    "w2v_ffn":   "NN: own word2vec + feed-forward net",
    "bert_fp32": "Transformer: BERTurk (FP32)",
    "bert_int8": "Quantized: BERTurk (dynamic INT8)",
}
MODEL_DIRS = {k: MODELS_DIR / k for k in MODEL_KEYS}

BERT_BASE = os.environ.get("BERT_BASE", "dbmdz/bert-base-turkish-cased")
BERT_MAX_LEN = 128
BERT_EPOCHS = int(os.environ.get("BERT_EPOCHS", 3))
BERT_BATCH = 32
BERT_LR = 2e-5

W2V_DIM = 100
FFN_EPOCHS = int(os.environ.get("FFN_EPOCHS", 40))

# ---------------------------------------------------------------- evaluation
LATENCY_BUDGET_MS = 50.0     # our CPU budget per message (Use Case 4, requirement D)
LATENCY_RUNS = 200           # >= 100 runs required by R4
LATENCY_WARMUP = 10
CPU_THREADS = int(os.environ.get("CPU_THREADS", 2))  # fixed for every model

# Hugging Face Hub repo that stores the trained weights (R7: no weights in git).
# Change this to "<your-hf-username>/chat-moderation-tr-models".
HF_REPO_ID = os.environ.get("HF_REPO_ID", "bgmzydd/chat-moderation-tr-models")
