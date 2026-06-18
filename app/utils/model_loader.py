"""Charge les modèles entraînés (TF-IDF + XLM-RoBERTa) + FeatureEngineer."""
from pathlib import Path
import sys

_PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import joblib
import numpy as np

from src.config.settings import MODELS_DIR, LABEL2ID, ID2LABEL


class ModelLoader:
    """Charge et expose les deux modèles (TF-IDF champion + XLM-RoBERTa).

    Le modèle TF-IDF (LogisticRegression) a été entraîné sur la sortie du
    FeatureEngineer (TF-IDF 15K + 6 features numériques).  Le FeatureEngineer
    fitté est donc indispensable pour produire la matrice sparse attendue.
    """

    def __init__(self):
        self.champion_pipeline = None
        self.champion_obj = None
        self.feature_engineer = None   # FeatureEngineer fitté
        self.fe_loaded = False
        self.xlm_model = None
        self.xlm_tokenizer = None
        self.xlm_loaded = False
        self._loaded = False

    def load_all(self):
        if self._loaded:
            return
        self._load_feature_engineer()
        self._load_tfidf()
        self._load_xlm_roberta()
        self._loaded = True

    def _load_feature_engineer(self):
        """Charge le FeatureEngineer fitté (TF-IDF + features numériques)."""
        path = MODELS_DIR / "feature_engineer.pkl"
        if not path.exists():
            print(f"[MODEL] FeatureEngineer introuvable: {path}")
            self.feature_engineer = None
            self.fe_loaded = False
            return
        try:
            fe = joblib.load(str(path))
            # Vérifier que le TF-IDF est bien fitté
            if hasattr(fe, '_fitted') and fe._fitted:
                self.feature_engineer = fe
                self.fe_loaded = True
                print(f"[MODEL] FeatureEngineer chargé (TF-IDF: {len(fe.tfidf.vocabulary_)} tokens)")
            else:
                print("[MODEL] FeatureEngineer n'est pas fitté")
                self.feature_engineer = None
                self.fe_loaded = False
        except Exception as e:
            print(f"[MODEL] Erreur chargement FeatureEngineer: {e}")
            self.feature_engineer = None
            self.fe_loaded = False

    def _load_tfidf(self):
        path = MODELS_DIR / "champion_model.pkl"
        if not path.exists():
            print(f"[MODEL] Champion TF-IDF introuvable: {path}")
            self.champion_pipeline = None
            return

        obj = joblib.load(str(path))
        self.champion_obj = obj
        if isinstance(obj, dict) and 'model' in obj:
            self.champion_pipeline = obj['model']
        else:
            self.champion_pipeline = obj
        print(f"[MODEL] Champion TF-IDF chargé: {type(self.champion_pipeline).__name__}")

    def _load_xlm_roberta(self):
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        model_dir = MODELS_DIR / "xlm_roberta_finetuned"
        if not model_dir.exists():
            print(f"[MODEL] XLM-RoBERTa introuvable: {model_dir}")
            self.xlm_loaded = False
            return

        try:
            self.xlm_tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
            self.xlm_model = AutoModelForSequenceClassification.from_pretrained(
                str(model_dir),
                num_labels=3,
                id2label=ID2LABEL,
                label2id=LABEL2ID,
            )
            self.xlm_model.eval()
            self.xlm_loaded = True
            print(f"[MODEL] XLM-RoBERTa chargé depuis {model_dir}")
        except Exception as e:
            print(f"[MODEL] Erreur chargement XLM-RoBERTa: {e}")
            self.xlm_loaded = False
