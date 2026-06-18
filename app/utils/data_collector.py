"""Collecte des avis analysés et feedback pour réentraînement futur."""
import csv
import json
import os
from pathlib import Path
from datetime import datetime
from typing import Optional

COLLECTED_DIR = Path(__file__).parent.parent.parent / "data" / "collected"
REVIEWS_FILE = COLLECTED_DIR / "reviews.csv"
CORRECTED_FILE = COLLECTED_DIR / "corrected.csv"
STATS_FILE = COLLECTED_DIR / "stats.json"


def _ensure_dirs():
    """Crée les dossiers de collecte si nécessaire."""
    COLLECTED_DIR.mkdir(parents=True, exist_ok=True)


def _init_csv(filepath: Path, fieldnames: list):
    """Initialise un fichier CSV avec en-tête si vide."""
    _ensure_dirs()
    if not filepath.exists():
        with open(filepath, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()


def save_review(
    text: str,
    predicted_sentiment: str,
    confidence: float,
    model_used: str,
    source: str = "manual",
    url: Optional[str] = None,
):
    """Sauvegarde un avis analysé dans la collection.

    Args:
        text: Texte de l'avis
        predicted_sentiment: Sentiment prédit (positive/neutral/negative)
        confidence: Score de confiance (0-1)
        model_used: Modèle utilisé (tfidf/xlm_roberta)
        source: Source (manual/jumia/trustpilot)
        url: URL d'origine si scraping
    """
    fieldnames = [
        "timestamp", "text", "predicted_sentiment", "confidence",
        "model_used", "source", "url", "corrected_sentiment"
    ]
    _init_csv(REVIEWS_FILE, fieldnames)

    with open(REVIEWS_FILE, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writerow({
            "timestamp": datetime.now().isoformat(),
            "text": text[:500],
            "predicted_sentiment": predicted_sentiment,
            "confidence": f"{confidence:.4f}",
            "model_used": model_used,
            "source": source,
            "url": url or "",
            "corrected_sentiment": "",
        })

    _update_stats()


def save_correction(text: str, corrected_sentiment: str):
    """Sauvegarde une correction feedback utilisateur.

    Cherche l'avis dans reviews.csv et met à jour le champ corrected_sentiment.
    Si l'avis n'existe pas, le sauvegarde dans corrected.csv.
    """
    fieldnames = [
        "timestamp", "text", "corrected_sentiment", "source"
    ]
    _init_csv(CORRECTED_FILE, fieldnames)

    with open(CORRECTED_FILE, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writerow({
            "timestamp": datetime.now().isoformat(),
            "text": text[:500],
            "corrected_sentiment": corrected_sentiment,
            "source": "user_feedback",
        })

    # Mettre à jour le corrected_sentiment dans reviews.csv
    _update_correction_in_reviews(text, corrected_sentiment)
    _update_stats()


def _update_correction_in_reviews(text: str, corrected: str):
    """Met à jour le champ corrected_sentiment pour l'avis correspondant."""
    if not REVIEWS_FILE.exists():
        return

    rows = []
    updated = False
    with open(REVIEWS_FILE, mode="r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Chercher l'avis par texte (première occurrence non corrigée)
            if (
                not updated
                and row["text"] == text[:500]
                and not row.get("corrected_sentiment", "")
            ):
                row["corrected_sentiment"] = corrected
                updated = True
            rows.append(row)

    if updated:
        with open(REVIEWS_FILE, mode="w", newline="", encoding="utf-8") as f:
            if rows:
                writer = csv.DictWriter(f, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)


def _update_stats():
    """Met à jour les statistiques de collecte."""
    stats = {
        "total_reviews": 0,
        "total_corrected": 0,
        "by_sentiment": {"positive": 0, "neutral": 0, "negative": 0},
        "by_source": {"manual": 0, "jumia": 0, "trustpilot": 0},
        "last_updated": datetime.now().isoformat(),
    }

    if REVIEWS_FILE.exists():
        with open(REVIEWS_FILE, mode="r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                stats["total_reviews"] += 1
                sent = row.get("predicted_sentiment", "")
                if sent in stats["by_sentiment"]:
                    stats["by_sentiment"][sent] += 1
                src = row.get("source", "manual")
                if src in stats["by_source"]:
                    stats["by_source"][src] += 1
                if row.get("corrected_sentiment", ""):
                    stats["total_corrected"] += 1

    _ensure_dirs()
    with open(STATS_FILE, mode="w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)


def get_stats() -> dict:
    """Retourne les statistiques actuelles."""
    if STATS_FILE.exists():
        with open(STATS_FILE, encoding="utf-8") as f:
            return json.load(f)
    return {
        "total_reviews": 0,
        "total_corrected": 0,
        "by_sentiment": {"positive": 0, "neutral": 0, "negative": 0},
        "by_source": {"manual": 0, "jumia": 0, "trustpilot": 0},
        "last_updated": "—",
    }


def get_recent_reviews(limit: int = 50) -> list:
    """Retourne les N derniers avis collectés."""
    if not REVIEWS_FILE.exists() or os.path.getsize(REVIEWS_FILE) == 0:
        return []
    with open(REVIEWS_FILE, mode="r", newline="", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))
    # Filtrer les lignes vides (sans timestamp)
    reader = [r for r in reader if r.get("timestamp", "").strip()]
    return reader[-limit:]


def export_reviews_csv() -> Path:
    """Retourne le chemin du fichier CSV principal."""
    return REVIEWS_FILE