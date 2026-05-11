"""Feature Engineering — TF-IDF + features numériques."""
import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix, issparse
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
from typing import Optional, List, Tuple
from pathlib import Path

from src.config.settings import (
    TFIDF_MAX_FEATURES, TFIDF_NGRAM_RANGE, TFIDF_MIN_DF, TFIDF_SUBLINEAR_TF
)


class FeatureEngineer:
    """Construit les features pour la classification de sentiment.

    Combine :
    - TF-IDF sur le texte nettoyé (review_clean)
    - Features numériques additionnelles (word_count, char_count, avg_word_len, has_excl, has_quest, caps_ratio)
    """

    def __init__(self, use_numeric_features: bool = True):
        self.use_numeric_features = use_numeric_features
        
        # TF-IDF vectorizer
        self.tfidf = TfidfVectorizer(
            max_features=TFIDF_MAX_FEATURES,
            ngram_range=TFIDF_NGRAM_RANGE,
            min_df=TFIDF_MIN_DF,
            sublinear_tf=TFIDF_SUBLINEAR_TF,
            strip_accents='unicode',
        )
        
        # Noms des features numériques
        self.numeric_feature_names = [
            'word_count', 'char_count', 'avg_word_len',
            'has_excl', 'has_quest', 'caps_ratio'
        ]
        self._fitted = False

    def _extract_numeric_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extrait les features numériques du DataFrame."""
        num_features = df[['review_clean']].copy()
        
        num_features['word_count'] = df['review_clean'].str.split().str.len()
        num_features['char_count'] = df['review_clean'].str.len()
        num_features['avg_word_len'] = num_features.apply(
            lambda r: r['char_count'] / max(r['word_count'], 1), axis=1
        )
        
        # Signaux lexicaux
        num_features['has_excl'] = df['review'].str.contains('!', regex=False).astype(float)
        num_features['has_quest'] = df['review'].str.contains(r'\?', regex=True).astype(float)
        num_features['caps_ratio'] = df['review'].apply(
            lambda t: sum(1 for c in str(t) if c.isupper()) / max(len(str(t)), 1)
        )
        
        return num_features[self.numeric_feature_names]

    def fit_transform(self, df: pd.DataFrame) -> csr_matrix:
        """Fitte le TF-IDF et transforme le DataFrame en matrice de features."""
        # TF-IDF sur review_clean
        texts = df['review_clean'].fillna('').values
        X_tfidf = self.tfidf.fit_transform(texts)
        
        # Features numériques
        if self.use_numeric_features:
            X_num = csr_matrix(self._extract_numeric_features(df).values.astype(float))
            X_combined = hstack([X_tfidf, X_num])
        else:
            X_combined = X_tfidf
        
        self._fitted = True
        return X_combined

    def transform(self, df: pd.DataFrame) -> csr_matrix:
        """Transforme le DataFrame en matrice de features (sans refitter)."""
        if not self._fitted:
            raise ValueError("FeatureEngineer n'est pas encore fitté. Appelez fit_transform d'abord.")
        
        texts = df['review_clean'].fillna('').values
        X_tfidf = self.tfidf.transform(texts)
        
        if self.use_numeric_features:
            X_num = csr_matrix(self._extract_numeric_features(df).values.astype(float))
            X_combined = hstack([X_tfidf, X_num])
        else:
            X_combined = X_tfidf
        
        return X_combined

    def save(self, path: Path) -> None:
        """Sauvegarde le FeatureEngineer entraîné."""
        joblib.dump(self, str(path))

    @staticmethod
    def load(path: Path) -> 'FeatureEngineer':
        """Charge un FeatureEngineer sauvegardé."""
        return joblib.load(str(path))