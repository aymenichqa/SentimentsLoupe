"""Page À propos — Informations sur le projet."""
from pathlib import Path
import sys

_APP_ROOT = Path(__file__).parent.parent
_PROJECT_ROOT = _APP_ROOT.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import streamlit as st

st.set_page_config(
    page_title="SentimentsLoupe — À propos",
    page_icon="ℹ️",
    layout="wide",
    initial_sidebar_state="expanded",
)

CSS_PATH = _APP_ROOT / "assets" / "style.css"
if CSS_PATH.exists():
    with open(CSS_PATH, encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

LOGO_PATH = _APP_ROOT / "assets" / "logo.png"

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

col1, col2 = st.columns([1, 2])
with col1:
    if LOGO_PATH.exists():
        st.image(str(LOGO_PATH), width=180)
    else:
        st.markdown("# 🔍")
with col2:
    st.markdown(
        "<div class='app-header' style='border:none;padding-top:0;'>"
        "<h1>ℹ️ À propos</h1>"
        "<p class='subtitle'>SentimentsLoupe — Projet de fin d'études</p>"
        "</div>",
        unsafe_allow_html=True,
    )

st.divider()

st.markdown(
    "<div class='card'>"
    "<h3>🎯 Objectif du projet</h3>"
    "<p>Créer un système d'analyse de sentiment opérationnel pour le <b>contexte marocain</b>, "
    "capable de comprendre les avis clients en <b>anglais, français et darija</b>.</p>"
    "</div>",
    unsafe_allow_html=True,
)

c1, c2 = st.columns(2)
with c1:
    st.markdown(
        "<div class='card'>"
        "<h3>📊 Datasets</h3>"
        "<ul>"
        "<li><b>Flipkart</b> — 7 700 avis (anglais, produits)</li>"
        "<li><b>Trustpilot</b> — 1 235 avis (anglais, multidomaines)</li>"
        "<li><b>Jumia Maroc</b> — 60 avis (français/darija)</li>"
        "</ul></div>",
        unsafe_allow_html=True,
    )
with c2:
    st.markdown(
        "<div class='card'>"
        "<h3>🤖 Modèles</h3>"
        "<ul>"
        "<li><b>TF-IDF + LogisticRegression</b> — Pipeline classique</li>"
        "<li><b>XLM-RoBERTa</b> — 279M paramètres, 100 langues</li>"
        "</ul></div>",
        unsafe_allow_html=True,
    )

st.markdown(
    "<div class='card'>"
    "<h3>📜 Phase 1 — Baseline ML (TF-IDF)</h3>"
    "<ol>"
    "<li><b>EDA</b> : Analyse exploratoire de Flipkart (20K avis, 81% positifs)</li>"
    "<li><b>Nettoyage</b> : TextCleaner (URLs, emojis, READ MORE)</li>"
    "<li><b>Feature Engineering</b> : TF-IDF 15K features + 6 features numériques</li>"
    "<li><b>18 modèles testés</b> : 5 stratégies × 4 classifieurs</li>"
    "<li><b>Champion</b> : class_weight + LogisticRegression → F1 = 0.65</li>"
    "</ol></div>",
    unsafe_allow_html=True,
)

st.markdown(
    "<div class='card'>"
    "<h3>🔬 Phase 2 — Multi-sources & Multilingue</h3>"
    "<ul>"
    "<li><b>2A</b> : Test sur Trustpilot (gap domaine = 0.22) et Jumia (gap langue = 0.31)</li>"
    "<li><b>2B</b> : XLM-RoBERTa fine-tuné sur 8 929 avis → F1 Jumia = 0.57 🚀</li>"
    "<li><b>Leçon</b> : Le problème principal est la <b>langue</b>, pas le domaine</li>"
    "</ul></div>",
    unsafe_allow_html=True,
)

st.markdown(
    "<div class='card'>"
    "<h3>👨‍💻 Auteur</h3>"
    "<p><b>Aymen Ich</b> — Projet de fin d'études • Mai 2026</p>"
    "<p>Technologies : Python, scikit-learn, Hugging Face Transformers, Streamlit</p>"
    "</div>",
    unsafe_allow_html=True,
)

st.divider()
st.markdown(
    "<div class='app-footer'>"
    "SentimentsLoupe  •  Projet de fin d'études  •  Aymen Ich © 2026</div>",
    unsafe_allow_html=True,
)