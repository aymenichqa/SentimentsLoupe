"""Configuration globale SentimentsLoupe."""
from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()

# ── Chemins ────────────────────────────────────────────────────────────────
ROOT_DIR    = Path(__file__).parent.parent.parent
DATA_DIR    = ROOT_DIR / "data"
RAW_DIR     = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR  = DATA_DIR / "models"
PLOTS_DIR   = MODELS_DIR / "plots"
LOG_DIR     = ROOT_DIR / "logs"

FLIPKART_RAW   = RAW_DIR  / "flipkart_data.csv"
FLIPKART_CLEAN = PROCESSED_DIR / "flipkart_cleaned.csv"

# Champion model path (ajouté)
CHAMPION_MODEL_PATH = MODELS_DIR / "champion_model.pkl"

# ── ML ─────────────────────────────────────────────────────────────────────
RANDOM_STATE = int(os.getenv("RANDOM_STATE", 42))
TEST_SIZE    = 0.20   # 80/20 split
VAL_SIZE     = 0.10   # sur le train (pour courbes d'apprentissage)
CV_FOLDS     = 5

# ── Sentiment mapping ──────────────────────────────────────────────────────
SENTIMENT_MAP = {1: "negative", 2: "negative", 3: "neutral", 4: "positive", 5: "positive"}
LABEL2ID      = {"negative": 0, "neutral": 1, "positive": 2}
ID2LABEL      = {0: "negative", 1: "neutral", 2: "positive"}
CLASS_NAMES   = ["negative", "neutral", "positive"]

# ── TF-IDF ─────────────────────────────────────────────────────────────────
TFIDF_MAX_FEATURES = 15000
TFIDF_NGRAM_RANGE  = (1, 3)   # unigrams + bigrams + trigrams
TFIDF_MIN_DF       = 2
TFIDF_SUBLINEAR_TF = True     # log(tf) → atténue les fréquences élevées

# ── Stratégies imbalance à comparer ───────────────────────────────────────
IMBALANCE_STRATEGIES = [
    "none",             # Baseline sans traitement
    "class_weight",     # Pénalisation dans la loss
    "random_over",      # RandomOverSampler (duplication)
    "smote",            # SMOTE (interpolation synthétique)
    "smotetomek",       # SMOTE + nettoyage Tomek Links
]
