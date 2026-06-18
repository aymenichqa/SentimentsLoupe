"""Intègre les scrapers Jumia et Trustpilot pour l'app Streamlit."""
from pathlib import Path
import sys

_PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import pandas as pd
import re
from urllib.parse import urlparse

from src.data_collection.jumia_fix_scraper import JumiaListingScraper
from src.data_collection.trustpilot_scraper import TrustpilotScraper


def detect_source(url: str) -> str:
    domain = urlparse(url).netloc.lower()
    if "jumia" in domain:
        return "jumia"
    elif "trustpilot" in domain:
        return "trustpilot"
    return "unknown"


def is_jumia_product_url(url: str) -> bool:
    """Vérifie si l'URL Jumia est une page produit individuelle."""
    return JumiaListingScraper.is_product_url(url)


def scrape_jumia(url: str, max_products: int = 10) -> pd.DataFrame:
    scraper = JumiaListingScraper(url, max_products=max_products)
    
    # Détecter si c'est une page produit ou un listing
    if JumiaListingScraper.is_product_url(url):
        print(f"[SCRAPER] URL produit détectée, mode singleton")
        df = scraper.scrape_product(url)
    else:
        print(f"[SCRAPER] URL listing détectée, mode multi-produits")
        df = scraper.scrape()
    
    if df.empty:
        return df
    if 'text' in df.columns and 'review' not in df.columns:
        df = df.rename(columns={'text': 'review'})
    return df


def scrape_trustpilot(
    url: str, max_pages: int = 3
) -> pd.DataFrame:
    # Extraire le domaine à reviewer depuis le path /review/...
    # Ex: https://www.trustpilot.com/review/www.electroplanet.ma
    # → domain = "www.electroplanet.ma", path = ""
    parsed = urlparse(url)
    path_raw = parsed.path
    if "/review/" in path_raw:
        site_domain = path_raw.split("/review/", 1)[1]
    else:
        site_domain = parsed.netloc.replace("www.", "")
    scraper = TrustpilotScraper(site_domain, path="", max_pages=max_pages)
    df = scraper.scrape()
    if df.empty:
        return df
    if 'text' in df.columns and 'review' not in df.columns:
        df = df.rename(columns={'text': 'review'})
    return df


def scrape_url(url: str, max_products: int = 10, max_pages: int = 3) -> pd.DataFrame:
    source = detect_source(url)
    if source == "jumia":
        return scrape_jumia(url, max_products)
    elif source == "trustpilot":
        return scrape_trustpilot(url, max_pages)
    else:
        return pd.DataFrame()


def format_url_example(source: str) -> str:
    examples = {
        "jumia": "https://www.jumia.ma/mlp-anniversaire/",
        "trustpilot": "https://www.trustpilot.com/review/www.example.com",
    }
    return examples.get(source, "")