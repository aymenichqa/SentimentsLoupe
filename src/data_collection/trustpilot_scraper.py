"""Scraper Trustpilot — avis consommateurs (FR/EN).
Utilise Playwright dans un processus separe pour contourner Cloudflare
et eviter les conflits asyncio de Jupyter.
"""

import pandas as pd
import subprocess
import sys
import json
import tempfile
import os
from pathlib import Path

from src.data_collection.base_scraper import BaseScraper


# ── Script Playwright temporaire ─────────────────────────────────────
# Ce script est ecrit dans un fichier temporaire et execute
# dans un processus Python separe (pas de conflit asyncio)
_SCRAPER_SCRIPT = '''
import sys, json, time
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright

domain = sys.argv[1]
path = sys.argv[2]
max_pages = int(sys.argv[3])
output_file = sys.argv[4]

BASE_URL = "https://www.trustpilot.com/review/{domain}"
all_reviews = []
full_domain = domain + path

print(f"[TRUSTPILOT] Scraping {full_domain} ({max_pages} pages)", flush=True)

with sync_playwright() as pw:
    browser = pw.chromium.launch(
        headless=False,
        args=["--disable-blink-features=AutomationControlled"]
    )
    page = browser.new_page()

    for page_num in range(1, max_pages + 1):
        url = BASE_URL.format(domain=domain)
        if path:
            url += path
        if page_num > 1:
            url += f"?page={page_num}"

        print(f"[TRUSTPILOT] Page {page_num}: {url}", flush=True)

        try:
            page.goto(url, timeout=60000, wait_until="domcontentloaded")
            page.wait_for_timeout(5000)
            
            # Detection et fermeture des popups/connexions
            popup_closed = False
            for attempt in range(3):  # max 3 tentatives de fermeture
                try:
                    popup_btns = page.query_selector_all(
                        '[aria-label="Close"], button[class*="close"], button[class*="dismiss"], '
                        'button:has-text("Continue"), a:has-text("Skip"), '
                        'button:has-text("Maybe later"), a:has-text("No thanks"), '
                        '[aria-label="Dismiss"], button[data-close]'
                    )
                    for btn in popup_btns:
                        if btn.is_visible():
                            btn.click()
                            page.wait_for_timeout(1000)
                            popup_closed = True
                            print("[TRUSTPILOT] Popup fermee", flush=True)
                except:
                    pass
                if not popup_closed:
                    break
            
            # Si on est sur une page de connexion, essayer de revenir aux avis
            if "login" in page.url.lower() or "signin" in page.url.lower() or "authenticate" in page.url.lower():
                print(f"[TRUSTPILOT] Page {page_num}: page de connexion detectee, tentative de contournement...", flush=True)
                # Essayer d'aller directement a l'URL sans redirection
                clean_url = url.split("?")[0]
                page.goto(clean_url, timeout=30000)
                page.wait_for_timeout(3000)
                # Re-essayer la fermeture de popup
                try:
                    skip_btn = page.query_selector('button:has-text("Continue"), a:has-text("Skip"), button:has-text("Maybe later")')
                    if skip_btn and skip_btn.is_visible():
                        skip_btn.click()
                        page.wait_for_timeout(2000)
                except:
                    pass
                if "login" in page.url.lower():
                    print(f"[TRUSTPILOT] Page {page_num}: toujours sur login, arret", flush=True)
                    break

            page.mouse.move(300, 400)
            time.sleep(0.5)
            page.keyboard.press("PageDown")
            time.sleep(1)
            page.mouse.move(500, 200)
            time.sleep(0.5)

            html = page.content()
            soup = BeautifulSoup(html, "html.parser")

            cards = soup.find_all("article")
            if not cards:
                cards = soup.select("section[data-service-review-card-paper]")

            if not cards:
                print(f"[TRUSTPILOT] Page {page_num}: aucun avis trouve", flush=True)
                continue

            page_count = 0
            for card in cards:
                try:
                    text_el = card.find("p")
                    text = text_el.get_text(strip=True) if text_el else ""
                    if not text or len(text) < 10:
                        continue

                    rating_el = card.select_one("[data-service-review-rating]")
                    rating = None
                    if rating_el:
                        r_val = rating_el.get("data-service-review-rating", "")
                        if r_val:
                            rating = int(r_val)

                    date_el = card.select_one("time")
                    date = date_el.get("datetime", "") if date_el else ""

                    all_reviews.append({
                        "text": text, "rating": rating,
                        "date": date, "source": "trustpilot",
                        "domain": full_domain
                    })
                    page_count += 1
                except:
                    continue

            print(f"[TRUSTPILOT] Page {page_num}: {page_count} avis", flush=True)

        except Exception as e:
            print(f"[TRUSTPILOT] Erreur page {page_num}: {e}", flush=True)
            continue

        time.sleep(2)

    browser.close()

with open(output_file, "w", encoding="utf-8") as f:
    json.dump(all_reviews, f, ensure_ascii=False)

print(f"[TRUSTPILOT] Sauvegarde: {len(all_reviews)} avis dans {output_file}", flush=True)
'''


class TrustpilotScraper(BaseScraper):
    """Scraper Trustpilot via Playwright (processus separe).
    
    Le scraping s'execute dans un processus Python independant,
    ce qui evite les conflits asyncio avec Jupyter Notebook.
    
    Installation:
        pip install playwright
        playwright install chromium
    """

    BASE_URL = "https://www.trustpilot.com/review/{domain}"

    def __init__(self, domain: str, path: str = "", max_pages: int = 5):
        super().__init__(source_name="trustpilot", delay_range=(2.0, 4.0))
        self.domain = domain
        self.path = path
        self.max_pages = max_pages

    def scrape(self) -> pd.DataFrame:
        """Execute le scraping dans un processus Python separe."""
        
        # Creer un fichier temporaire pour les resultats
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as tmp:
            output_file = tmp.name
        
        try:
            # Executer le script Playwright dans un processus separe
            print(f"[TRUSTPILOT] Scraping {self.domain}{self.path} ({self.max_pages} pages)")
            print("[TRUSTPILOT] Demarrage du processus Chrome (visible)...")
            
            result = subprocess.run(
                [
                    sys.executable, "-c", _SCRAPER_SCRIPT,
                    self.domain, self.path, str(self.max_pages), output_file
                ],
                capture_output=True, text=True, timeout=300
            )
            
            # Afficher la sortie du processus
            for line in result.stdout.split("\n"):
                if line.strip():
                    print(line)
            
            if result.returncode != 0:
                print(f"[TRUSTPILOT] Erreur processus: {result.stderr}")
                return pd.DataFrame()
            
            # Charger les resultats
            if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
                with open(output_file, "r", encoding="utf-8") as f:
                    reviews_data = json.load(f)
                
                df = pd.DataFrame(reviews_data)
                df = self.validate(df)
                print(f"[TRUSTPILOT] TOTAL: {len(df)} avis valides")
                return df
            
            return pd.DataFrame()
            
        except subprocess.TimeoutExpired:
            print("[TRUSTPILOT] TIMEOUT: le scraping a pris plus de 5 minutes")
            return pd.DataFrame()
        except Exception as e:
            print(f"[TRUSTPILOT] Erreur: {e}")
            return pd.DataFrame()
        finally:
            # Nettoyer le fichier temporaire
            if os.path.exists(output_file):
                os.unlink(output_file)