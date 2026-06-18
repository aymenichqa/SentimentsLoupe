"""Page Collection — Données collectées, statistiques, export."""
from pathlib import Path
import sys

_APP_ROOT = Path(__file__).parent.parent
_PROJECT_ROOT = _APP_ROOT.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import streamlit as st
import pandas as pd
from datetime import datetime

st.set_page_config(
    page_title="SentimentsLoupe — Collection",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

CSS_PATH = _APP_ROOT / "assets" / "style.css"
if CSS_PATH.exists():
    with open(CSS_PATH, encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

LOGO_PATH = _APP_ROOT / "assets" / "logo.png"
from app.utils.data_collector import get_stats, get_recent_reviews, export_reviews_csv

with st.sidebar:
    if LOGO_PATH.exists():
        st.image(str(LOGO_PATH), width=80)
    else:
        st.markdown("### 🔍")
    st.markdown(
        "<h2 style='margin-top:0;color:#3D3D3D;'>SentimentsLoupe</h2>"
        "<p style='color:#8D6E63;font-size:0.85rem;'>Analyse de sentiment</p>",
        unsafe_allow_html=True,
    )
    st.divider()
    st.page_link("app.py", label="   🏠 Accueil", use_container_width=True)
    st.page_link("pages/01_Analyse.py", label="   🔍 Analyse", use_container_width=True)
    st.page_link("pages/04_Collection.py", label="   📚 Collection", use_container_width=True)
    st.page_link("pages/02_Dashboard.py", label="   📊 Dashboard", use_container_width=True)
    st.page_link("pages/03_A_Propos.py", label="   ℹ️ À propos", use_container_width=True)
    st.divider()
    st.caption("Projet de fin d'études  •  Aymen Ich  •  Mai 2026")

st.markdown(
    "<div class='app-header'>"
    "<h1>📚 Collection de données</h1>"
    "<p class='subtitle'>Avis collectés pour le réentraînement futur des modèles</p>"
    "</div>",
    unsafe_allow_html=True,
)

stats = get_stats()

# ── KPIs ──
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(
        f"<div class='metric-box'>"
        f"<div class='metric-value'>{stats['total_reviews']}</div>"
        f"<div class='metric-label'>Avis collectés 🗂️</div></div>",
        unsafe_allow_html=True,
    )
with c2:
    st.markdown(
        f"<div class='metric-box'>"
        f"<div class='metric-value'>{stats['total_corrected']}</div>"
        f"<div class='metric-label'>Corrections feedback 👍</div></div>",
        unsafe_allow_html=True,
    )
with c3:
    pos = stats["by_sentiment"].get("positive", 0)
    rate = f"{pos / max(stats['total_reviews'], 1) * 100:.0f}%"
    st.markdown(
        f"<div class='metric-box'>"
        f"<div class='metric-value'>{rate}</div>"
        f"<div class='metric-label'>Taux de positivité 😊</div></div>",
        unsafe_allow_html=True,
    )
with c4:
    st.markdown(
        f"<div class='metric-box'>"
        f"<div class='metric-value'>{stats.get('last_updated', '—')[:10]}</div>"
        f"<div class='metric-label'>Dernière mise à jour</div></div>",
        unsafe_allow_html=True,
    )

st.divider()

# ── Répartition ──
st.markdown("### 📊 Répartition des avis collectés")
col1, col2 = st.columns(2)

with col1:
    st.markdown("**Par sentiment**")
    sent_data = pd.DataFrame({
        "Sentiment": list(stats["by_sentiment"].keys()),
        "Nombre": list(stats["by_sentiment"].values()),
    })
    st.bar_chart(sent_data.set_index("Sentiment"), height=250)

with col2:
    st.markdown("**Par source**")
    src_data = pd.DataFrame({
        "Source": list(stats["by_source"].keys()),
        "Nombre": list(stats["by_source"].values()),
    })
    st.bar_chart(src_data.set_index("Source"), height=250)

st.divider()

# ── Derniers avis ──
st.markdown("### 🕐 Derniers avis analysés")
reviews = get_recent_reviews(100)

if reviews:
    df_reviews = pd.DataFrame(reviews)
    
    # Vérifier que les colonnes attendues existent (évite les erreurs si CSV mal formaté)
    if "timestamp" not in df_reviews.columns:
        st.warning("⚠️ Le fichier de collecte est corrompu ou vide. Supprimez data/collected/reviews.csv et réessayez.")
    else:
        df_reviews["timestamp"] = pd.to_datetime(df_reviews["timestamp"]).dt.strftime("%d/%m/%Y %H:%M")

    # Colorer le sentiment
    def color_sentiment(val):
        colors = {"positive": "background:#E8F5E9;color:#2E7D32",
                  "neutral": "background:#FFF8E1;color:#F57F17",
                  "negative": "background:#FFEBEE;color:#C62828"}
        return colors.get(val, "")

    st.dataframe(
        df_reviews[["timestamp", "text", "predicted_sentiment", "confidence", "source", "corrected_sentiment"]],
        use_container_width=True,
        column_config={
            "timestamp": "Date",
            "text": "Texte",
            "predicted_sentiment": st.column_config.TextColumn("Prédiction", width="small"),
            "confidence": "Confiance",
            "source": "Source",
            "corrected_sentiment": "Corrigé",
        },
        hide_index=True,
    )

    # ── Export ──
    st.divider()
    st.markdown("### 📥 Export des données")
    csv_path = export_reviews_csv()
    if csv_path.exists():
        with open(csv_path, "rb") as f:
            st.download_button(
                label="📥 Télécharger le CSV des avis collectés",
                data=f,
                file_name=f"sentimentsloupe_collection_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                type="primary",
                use_container_width=True,
            )

    st.caption(
        "💡 Ces données peuvent être utilisées pour fine-tuner XLM-RoBERTa. "
        "Les avis corrigés via le feedback sont marqués dans la colonne 'Corrigé'."
    )
else:
    st.info("Aucun avis collecté pour le moment. Faites des analyses dans la page 🔍 Analyse !")

st.divider()
st.markdown(
    "<div class='app-footer'>"
    "SentimentsLoupe  •  Projet de fin d'études  •  Aymen Ich © 2026</div>",
    unsafe_allow_html=True,
)