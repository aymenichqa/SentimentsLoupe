"""Base scraper class — reusable interface for all data sources."""

from abc import ABC, abstractmethod
import pandas as pd
import time
import random
from pathlib import Path


class BaseScraper(ABC):
    """Classe mere abstraite pour tous les scrapers.
    
    Fournit l'interface commune et les utilitaires (rate limiting, sauvegarde).
    Chaque source herite de BaseScraper et implemente scrape().
    """
    
    def __init__(self, source_name: str, delay_range: tuple = (1.0, 3.0)):
        """
        Args:
            source_name: identifiant unique de la source (trustpilot, jumia, gmaps)
            delay_range: intervalle de secondes entre requetes (rate limiting)
        """
        self.source_name = source_name
        self.delay_range = delay_range
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                          "AppleWebKit/537.36 (KHTML, like Gecko) "
                          "Chrome/120.0.0.0 Safari/537.36"
        }
    
    @abstractmethod
    def scrape(self) -> pd.DataFrame:
        """Collecte les avis depuis la source.
        
        Returns:
            DataFrame avec colonnes: text, rating, source, date (optionnel)
        """
        pass
    
    def _respectful_delay(self):
        """Pause entre les requetes pour eviter le blocage."""
        time.sleep(random.uniform(*self.delay_range))
    
    def save(self, df: pd.DataFrame, output_path: Path):
        """Sauvegarde les donnees collectees en CSV.
        
        Args:
            df: DataFrame a sauvegarder
            output_path: chemin du fichier CSV
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(str(output_path), index=False)
        print(f"[SAVE] {len(df)} avis sauvegardes dans {output_path}")
    
    def validate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Nettoie et valide le DataFrame collecte.
        
        - Supprime les lignes sans texte
        - Filtre les ratings invalides (1-5)
        - Supprime les doublons
        """
        if df.empty:
            return df
        
        df = df.copy()
        df = df.dropna(subset=["text"])
        df["text"] = df["text"].astype(str).str.strip()
        df = df[df["text"] != ""]
        
        if "rating" in df.columns:
            df = df[df["rating"].between(1, 5)]
        
        df = df.drop_duplicates(subset=["text"])
        df = df.reset_index(drop=True)
        
        return df