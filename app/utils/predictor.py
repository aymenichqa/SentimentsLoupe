"""Fonctions de prédiction pour les deux modèles."""
from pathlib import Path
import sys

_PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import pandas as pd

from src.preprocessing.text_cleaner import TextCleaner
from src.config.settings import LABEL2ID, ID2LABEL
from app.utils.model_loader import ModelLoader

# ── Singleton global ──
_models = ModelLoader()
_text_cleaner = TextCleaner(lemmatize=False, remove_stopwords=False)


def get_models():
    _models.load_all()
    return _models


def get_text_cleaner():
    """Retourne le TextCleaner singleton."""
    return _text_cleaner


def _clean_text(text: str) -> str:
    return _text_cleaner.clean(text)


def _prepare_tfidf_input(text: str) -> pd.DataFrame:
    """Prépare un DataFrame pour le FeatureEngineer.

    Le FeatureEngineer attend un DataFrame avec les colonnes 'review' et
    'review_clean' (produit par TextCleaner).  Les features numériques
    (word_count, char_count, has_excl...) sont calculées automatiquement
    par la méthode transform() du FeatureEngineer.
    """
    clean = _clean_text(text)
    return pd.DataFrame({
        "review": [text],
        "review_clean": [clean],
    })


def predict_tfidf(text: str) -> dict:
    """Prédiction avec le modèle TF-IDF champion.

    Le champion LogisticRegression a été entraîné sur la sortie du
    FeatureEngineer (TF-IDF 15K tokens + 6 features numériques).
    On applique donc FeatureEngineer.transform() pour obtenir la
    matrice sparse attendue.
    """
    models = get_models()
    if models.champion_pipeline is None:
        return {"error": "Modèle TF-IDF non chargé", "label": None, "confidence": 0.0}

    if not models.fe_loaded or models.feature_engineer is None:
        return {"error": "FeatureEngineer non chargé", "label": None, "confidence": 0.0}

    df = _prepare_tfidf_input(text)
    try:
        # Le FeatureEngineer.transform() produit une matrice sparse
        # avec 15000 dims TF-IDF + 6 features numériques
        X = models.feature_engineer.transform(df)
        probs = models.champion_pipeline.predict_proba(X)[0]

        # classes_ = [0, 1, 2] → [negative, neutral, positive] (via ID2LABEL)
        pred_class = int(np.argmax(probs))
        confidence = float(np.max(probs))
        return {
            "label": ID2LABEL[pred_class],
            "confidence": confidence,
            "probabilities": {
                "negative": float(probs[0]),
                "neutral": float(probs[1]),
                "positive": float(probs[2]),
            }
        }
    except Exception as e:
        return {"error": f"Erreur TF-IDF: {e}", "label": "error", "confidence": 0.0}


def predict_xlm_roberta(text: str) -> dict:
    """Prédiction avec XLM-RoBERTa fine-tuné."""
    import torch
    models = get_models()
    if not models.xlm_loaded:
        return {"error": "Modèle XLM-RoBERTa non chargé", "label": None, "confidence": 0.0}

    try:
        inputs = models.xlm_tokenizer(
            text, return_tensors="pt", padding=True, truncation=True, max_length=64,
        )
        with torch.no_grad():
            logits = models.xlm_model(**inputs).logits
            probs = torch.softmax(logits, dim=-1).numpy()[0]
        pred_class = int(np.argmax(probs))
        confidence = float(np.max(probs))
        return {
            "label": ID2LABEL[pred_class],
            "confidence": confidence,
            "probabilities": {
                "negative": float(probs[0]),
                "neutral": float(probs[1]),
                "positive": float(probs[2]),
            }
        }
    except Exception as e:
        return {"error": f"Erreur XLM-RoBERTa: {e}", "label": "error", "confidence": 0.0}


def predict(text: str, model_type: str = "xlm_roberta") -> dict:
    """Prédiction selon le modèle choisi."""
    if not text or not text.strip():
        return {"error": "Texte vide", "label": "empty", "confidence": 0.0}
    if model_type == "tfidf":
        return predict_tfidf(text)
    return predict_xlm_roberta(text)


def predict_batch(texts: list, model_type: str = "xlm_roberta") -> list:
    """Prédiction sur une liste de textes."""
    return [predict(t, model_type) for t in texts]
