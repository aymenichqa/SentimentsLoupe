"""SentimentsLoupe — Page d'accueil Streamlit."""
from pathlib import Path
import sys

_APP_ROOT = Path(__file__).parent
_PROJECT_ROOT = _APP_ROOT.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import streamlit as st

st.set_page_config(
    page_title="SentimentsLoupe — Accueil",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Préchargement des modèles au démarrage de l'app ──
@st.cache_resource
def _init_models():
    """Charge les modèles une seule fois (cache Streamlit)."""
    from app.utils.predictor import get_models
    return get_models()

_init_models()

CSS_PATH = _APP_ROOT / "assets" / "style.css"
if CSS_PATH.exists():
    with open(CSS_PATH, encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

LOGO_PATH = _APP_ROOT / "assets" / "logo.png"

with st.sidebar:
    if LOGO_PATH.exists():
        st.image(str(LOGO_PATH), width=80, use_container_width=False,
                 output_format="PNG", clamp=True)
        st.markdown(
            "<style>.stImage img{"
            "border-radius:50%;box-shadow:0 2px 8px rgba(0,0,0,0.12);"
            "border:2px solid #D7CCC8;transition:transform 0.25s ease;"
            "display:block;margin:0 auto;"
            "}"
            ".stImage img:hover{transform:scale(1.08) rotate(-2deg);"
            "box-shadow:0 4px 14px rgba(141,110,99,0.25);border-color:#8D6E63;"
            "}</style>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown("### 🔍")

    st.markdown(
        "<h2 style='margin-top:0;color:#3D3D3D;'>SentimentsLoupe</h2>"
        "<p style='color:#8D6E63;font-size:0.85rem;'>Analyse de sentiment cross-langue</p>",
        unsafe_allow_html=True,
    )
    st.divider()

    st.page_link("app.py", label="   🏠 Accueil", use_container_width=True)
    st.page_link("pages/01_Analyse.py", label="   🔍 Analyse", use_container_width=True)
    st.page_link("pages/04_Collection.py", label="   📚 Collection", use_container_width=True)
    st.page_link("pages/02_Dashboard.py", label="   📊 Dashboard", use_container_width=True)
    st.page_link("pages/03_A_Propos.py", label="   ℹ️ À propos", use_container_width=True)

    st.divider()
    st.markdown(
        "<div style='background:#F5F0EB;border-left:4px solid #8D6E63;"
        "padding:0.6rem 0.8rem;border-radius:4px;font-size:0.85rem;color:#5D4037;'>"
        "<b>Modèles disponibles</b><br>"
        "🔹 <b>TF-IDF</b> : F1 = 0.65<br>"
        "🔹 <b>XLM-RoBERTa</b> : F1 = 0.57"
        "</div>",
        unsafe_allow_html=True,
    )
    st.divider()
    st.caption("PFA EMSI — Aymen Ichqarrane & Ihssane Moutchou  •  Mai 2026")

col1, col2 = st.columns([1, 2.2])
with col1:
    if LOGO_PATH.exists():
        st.image(str(LOGO_PATH), width=200, use_container_width=False,
                 output_format="PNG", clamp=True)
        st.markdown(
            "<style>.stImage img{"
            "border-radius:16px;box-shadow:0 4px 14px rgba(0,0,0,0.10);"
            "border:2px solid #E8E0D8;transition:transform 0.2s;"
            "}"
            ".stImage img:hover{transform:scale(1.03);"
            "}</style>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown("# 🔍")

with col2:
    st.markdown(
        "<div style='padding:1rem 0;'>"
        "<h1 style='margin:0;color:#3D3D3D;'>SentimentsLoupe</h1>"
        "<p style='color:#8D6E63;font-size:1rem;margin:0.3rem 0;'>"
        "Analyse de sentiment cross-langue — Anglais • Français • Darija</p>"
        "</div>",
        unsafe_allow_html=True,
    )

st.divider()

c1, c2, c3 = st.columns(3)
with c1:
    st.markdown(
        "<div class='card'><h3>🔍 Analyser</h3>"
        "<p>Saisissez un avis ou une URL (Jumia / Trustpilot) "
        "pour une analyse instantanée.</p></div>",
        unsafe_allow_html=True,
    )
with c2:
    st.markdown(
        "<div class='card'><h3>🤖 Deux modèles</h3>"
        "<p>Comparez <b>TF-IDF</b> (classique) et "
        "<b>XLM-RoBERTa</b> (multilingue 100 langues).</p></div>",
        unsafe_allow_html=True,
    )
with c3:
    st.markdown(
        "<div class='card'><h3>📊 Dashboard</h3>"
        "<p>Visualisez les métriques complètes et "
        "l'évolution des performances.</p></div>",
        unsafe_allow_html=True,
    )

st.divider()
st.markdown("<h3 style='color:#3D3D3D;'>🏆 Résultats clés</h3>", unsafe_allow_html=True)

r1, r2, r3, r4 = st.columns(4)
for col, val, label in [
    (r1, "0.65", "F1 TF-IDF (Flipkart)"),
    (r2, "0.57", "F1 XLM-RoBERTa (Jumia)"),
    (r3, "+0.23", "Gain vs TF-IDF sur Jumia"),
    (r4, "100", "Langues (XLM-RoBERTa)"),
]:
    with col:
        st.markdown(
            f"<div class='metric-box'>"
            f"<div class='metric-value'>{val}</div>"
            f"<div class='metric-label'>{label}</div></div>",
            unsafe_allow_html=True,
        )

st.divider()
st.markdown(
    "<div style='text-align:center;padding:1rem 0;'>"
    "<p style='color:#8D6E63;font-size:1.05rem;'>"
    "👉 Commencez par la page <b>🔍 Analyse</b></p></div>",
    unsafe_allow_html=True,
)

st.markdown(
    "<div class='app-footer'>"
    "SentimentsLoupe  •  PFA EMSI  •  Aymen Ichqarrane & Ihssane Moutchou © 2026</div>",
    unsafe_allow_html=True,
)
