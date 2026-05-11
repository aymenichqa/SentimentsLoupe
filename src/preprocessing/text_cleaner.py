"""Nettoyeur de texte pour les avis Flipkart."""
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from typing import List, Optional
import pandas as pd


class TextCleaner:
    """Pipeline de nettoyage pour les avis produits.

    Étapes :
    1. Suppression de l'artefact 'READ MORE'
    2. Mise en minuscules
    3. Suppression des URLs, mentions, hashtags
    4. Suppression des emojis et caractères spéciaux
    5. Normalisation des espaces
    6. Lemmatisation optionnelle
    7. Suppression stopwords optionnelle
    8. Filtrage des mots trop courts (min_word_len)
    """

    def __init__(
        self,
        lemmatize: bool = True,
        remove_stopwords: bool = False,
        min_word_len: int = 2,
    ):
        self.lemmatize = lemmatize
        self.remove_stopwords = remove_stopwords
        self.min_word_len = min_word_len

        # Initialiser le lemmatizer si nécessaire
        if lemmatize:
            self.lemmatizer = WordNetLemmatizer()
        else:
            self.lemmatizer = None

        # Stopwords anglais
        try:
            self.stop_words = set(stopwords.words('english'))
        except LookupError:
            nltk.download('stopwords', quiet=True)
            self.stop_words = set(stopwords.words('english'))

    def clean(self, text: str) -> str:
        """Nettoie un seul texte."""
        if not isinstance(text, str):
            return ""

        # 1. Supprimer READ MORE
        text = re.sub(r'READ\s*MORE', '', text, flags=re.IGNORECASE)

        # 2. Supprimer les URLs
        text = re.sub(r'https?://\S+|www\.\S+', '', text)

        # 3. Supprimer les emojis (caractères Unicode au-delà de U+FFFF)
        text = re.sub(r'[^\x00-\x7F]+', '', text)

        # 4. Supprimer la ponctuation, garder lettres, chiffres, espaces
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)

        # 5. Mettre en minuscules
        text = text.lower()

        # 6. Tokenizer et filtrer
        tokens = text.split()
        
        # Filtrer les mots trop courts
        tokens = [t for t in tokens if len(t) >= self.min_word_len]

        # Supprimer les stopwords (optionnel)
        if self.remove_stopwords:
            tokens = [t for t in tokens if t not in self.stop_words]

        # Lemmatisation (optionnelle)
        if self.lemmatize and self.lemmatizer:
            tokens = [self.lemmatizer.lemmatize(t) for t in tokens]

        # 7. Rejoindre
        return ' '.join(tokens)

    def fit_transform(self, df: pd.DataFrame, col: str = 'review') -> pd.DataFrame:
        """Nettoie la colonne spécifiée du DataFrame et ajoute une colonne 'review_clean'."""
        df = df.copy()
        df['review_clean'] = df[col].apply(self.clean)
        
        # Ajouter les métriques de base
        df['word_count'] = df['review_clean'].str.split().str.len()
        df['char_count'] = df['review_clean'].str.len()
        df['avg_word_len'] = df.apply(
            lambda r: r['char_count'] / max(r['word_count'], 1), axis=1
        )
        
        return df