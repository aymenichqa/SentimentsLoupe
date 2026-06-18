"""Scraper Jumia ciblé pour pages de listing (ex: mlp-anniversaire)."""
import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
import time
from pathlib import Path
from urllib.parse import urljoin

from src.data_collection.base_scraper import BaseScraper


class JumiaListingScraper(BaseScraper):
    """Scrape les avis des produits depuis une page listing Jumia.
    
    Utilisation:
        scraper = JumiaListingScraper("https://www.jumia.ma/mlp-anniversaire/")
        df = scraper.scrape()
    """
    
    BASE_URL = "https://www.jumia.ma"
    
    def __init__(self, listing_url: str, max_products: int = 15):
        super().__init__(source_name="jumia", delay_range=(1.0, 2.0))
        self.listing_url = listing_url
        self.max_products = max_products
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                          "AppleWebKit/537.36 (KHTML, like Gecko) "
                          "Chrome/124.0.0.0 Safari/537.36",
            "Accept-Language": "fr-FR,fr;q=0.9,en;q=0.8",
        })
    
    def _get_product_links(self) -> list:
        """Recupere les liens des produits depuis la page listing."""
        try:
            resp = self.session.get(self.listing_url, timeout=15)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")
            
            links = []
            for a in soup.select("a.core"):
                href = a.get("href", "")
                if href and "/catalog/" not in href:
                    full_url = urljoin(self.BASE_URL, href)
                    if full_url not in links:
                        links.append(full_url)
            
            print(f"[JUMIA] {len(links)} produits trouves sur la page listing")
            return links[:self.max_products]
            
        except Exception as e:
            print(f"[JUMIA] Erreur listing: {e}")
            return []
    
    def _scrape_product_reviews(self, product_url: str) -> list:
        """Scrape les avis d'un produit Jumia."""
        reviews = []
        
        try:
            resp = self.session.get(product_url, timeout=15)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")
            
            # Nom du produit
            name_el = soup.select_one("h1")
            product_name = name_el.get_text(strip=True) if name_el else "unknown"
            
            # Section feedback/avis
            feedback_section = soup.find("div", id="feedback")
            if not feedback_section:
                feedback_section = soup.find("section", class_=lambda c: c and "reviews" in " ".join(c).lower()) if hasattr(soup, 'find') else None
            
            if not feedback_section:
                # Chercher un article avec la classe des avis
                articles = soup.select("article.-pvm.-hr._bet")
            else:
                articles = feedback_section.select("article.-pvm.-hr._bet")
            
            if not articles:
                # Dernier fallback: chercher partout
                articles = soup.select("article.-pvm.-hr._bet")
            
            if not articles:
                return reviews
            
            for article in articles:
                try:
                    # Rating
                    stars_el = article.select_one(".stars._m._al")
                    rating = None
                    if stars_el:
                        stars_text = stars_el.get_text(strip=True)
                        match = re.search(r'(\d+)\s*out\s*of\s*5', stars_text)
                        if match:
                            rating = int(match.group(1))
                    
                    # Titre de l'avis
                    title_el = article.select_one("h3.-m.-fs16")
                    title = title_el.get_text(strip=True) if title_el else ""
                    
                    # Texte de l'avis
                    text_el = article.select_one("p.-pvm")
                    text = text_el.get_text(strip=True) if text_el else ""
                    
                    if not text and title:
                        text = title
                    elif not text:
                        continue
                    
                    # Date
                    date_el = article.select_one("span.-prm")
                    date = date_el.get_text(strip=True) if date_el else ""
                    
                    # Auteur
                    author_el = article.select_one("div.-pvm span:last-child")
                    author = author_el.get_text(strip=True) if author_el else ""
                    
                    reviews.append({
                        "text": text,
                        "rating": rating if rating and 1 <= rating <= 5 else None,
                        "date": date,
                        "product_name": product_name,
                        "product_url": product_url,
                        "source": "jumia"
                    })
                    
                except Exception:
                    continue
            
            if reviews:
                print(f"[JUMIA] {product_name[:50]}: {len(reviews)} avis")
            
        except Exception as e:
            print(f"[JUMIA] Erreur produit {product_url[:60]}: {e}")
        
        return reviews
    
    @staticmethod
    def is_product_url(url: str) -> bool:
        """Détecte si l'URL est une page produit individuelle (contient un ID)."""
        # Les URLs produits Jumia se terminent par -ID.html ou -NOM-XXXXX.html
        # Exemples : ...-n-67278164.html  ou  ...-if7370-65980317.html
        return bool(re.search(r'-[\w]+-\d+\.html$', url))

    def scrape_product(self, product_url: str) -> pd.DataFrame:
        """Scrape une seule page produit Jumia (URL en -ID.html)."""
        print(f"[JUMIA] Scraping page produit: {product_url}")
        reviews = self._scrape_product_reviews(product_url)
        df = pd.DataFrame(reviews)
        df = self.validate(df)
        print(f"[JUMIA] TOTAL: {len(df)} avis pour ce produit")
        return df

    def scrape(self) -> pd.DataFrame:
        """Scrape les avis de tous les produits de la page listing."""
        product_links = self._get_product_links()
        if not product_links:
            return pd.DataFrame()
        
        all_reviews = []
        for i, url in enumerate(product_links, 1):
            reviews = self._scrape_product_reviews(url)
            all_reviews.extend(reviews)
            
            if i % 5 == 0 or i == len(product_links):
                print(f"[JUMIA] {i}/{len(product_links)} produits, "
                      f"{len(all_reviews)} avis cumules")
            
            self._respectful_delay()
        
        df = pd.DataFrame(all_reviews)
        df = self.validate(df)
        print(f"\n[JUMIA] TOTAL: {len(df)} avis valides sur {len(product_links)} produits")
        return df
