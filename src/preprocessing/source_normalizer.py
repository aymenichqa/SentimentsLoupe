"""Normalisation inter-sources — unifie les schemas de donnees."""

import pandas as pd
import numpy as np
from pathlib import Path

from src.config.settings import SENTIMENT_MAP
from src.preprocessing.text_cleaner import TextCleaner


class SourceNormalizer:
    """Normalise et unifie les donnees provenant de differentes sources.
    
    Convertit chaque source (Trustpilot, Jumia, Google Maps, Flipkart)
    vers un schema commun et reutilise le pipeline de nettoyage Phase 1.
    """
    
    def __init__(self):
        self.text_cleaner = TextCleaner(lemmatize=True, remove_stopwords=False)
        self.rating_map = SENTIMENT_MAP  # {1: "negative", 2: "negative", 3: "neutral", 4: "positive", 5: "positive"}
        self.label2id = {"negative": 0, "neutral": 1, "positive": 2}
    
    def normalize(self, df: pd.DataFrame, source: str) -> pd.DataFrame:
        """Normalise un DataFrame depuis n'importe quelle source.
        
        Args:
            df: DataFrame brut avec au minimum une colonne 'text' et 'rating'
            source: nom de la source (trustpilot, jumia, gmaps, flipkart)
        
        Returns:
            DataFrame normalise avec colonnes:
            - review: texte original
            - review_clean: texte nettoye
            - word_count: nombre de mots
            - rating: note (1-5)
            - sentiment: label texte (negative, neutral, positive)
            - label: label numerique (0, 1, 2)
            - source: nom de la source
        """
        df = df.copy()
        
        # Renommer les colonnes si necessaire
        col_map = {}
        if "text" in df.columns and "review" not in df.columns:
            col_map["text"] = "review"
        df = df.rename(columns=col_map)
        
        # S'assurer que review existe
        if "review" not in df.columns:
            raise ValueError(f"La source {source} doit avoir une colonne 'text' ou 'review'")
        
        # Nettoyage du texte (reutilisation Phase 1)
        df = self.text_cleaner.fit_transform(df, col="review")
        
        # Mapping rating -> sentiment
        if "sentiment" not in df.columns:
            df["sentiment"] = df["rating"].map(self.rating_map)
        
        # Mapping sentiment -> label numerique
        df["label"] = df["sentiment"].map(self.label2id)
        
        # Ajouter la source
        df["source"] = source
        
        # Garder les colonnes standardisees
        standard_cols = [
            "review", "review_clean", "word_count", "char_count",
            "avg_word_len", "rating", "sentiment", "label", "source"
        ]
        for col in standard_cols:
            if col not in df.columns:
                df[col] = None
        
        return df[standard_cols]
    
    def merge_sources(self, sources: dict) -> pd.DataFrame:
        """Fusionne plusieurs sources normalisees en un seul DataFrame.
        
        Args:
            sources: dict {nom_source: DataFrame_normalise}
        
        Returns:
            DataFrame unique avec toutes les sources concatennees
        """
        all_dfs = []
        for source_name, df in sources.items():
            df = self.normalize(df, source_name)
            all_dfs.append(df)
        
        return pd.concat(all_dfs, ignore_index=True)
    
    def summary(self, df: pd.DataFrame) -> pd.DataFrame:
        """Resume les statistiques par source."""
        summary = df.groupby("source").agg(
            count=("review", "count"),
            sentiment_neg=("sentiment", lambda x: (x == "negative").sum()),
            sentiment_neu=("sentiment", lambda x: (x == "neutral").sum()),
            sentiment_pos=("sentiment", lambda x: (x == "positive").sum()),
            rating_mean=("rating", "mean"),
            words_mean=("word_count", "mean"),
        ).round(2)
        
        summary["pct_neg"] = (summary["sentiment_neg"] / summary["count"] * 100).round(1)
        summary["pct_neu"] = (summary["sentiment_neu"] / summary["count"] * 100).round(1)
        summary["pct_pos"] = (summary["sentiment_pos"] / summary["count"] * 100).round(1)
        
        return summary