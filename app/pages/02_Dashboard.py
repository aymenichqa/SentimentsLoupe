"""Page Dashboard — Métriques et comparaison des modèles."""
from pathlib import Path
import sys

_APP_ROOT = Path(__file__).parent.parent
_PROJECT_ROOT = _APP_ROOT.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import streamlit as st
import json
import pandas as pd

st.set_page_config(
    page_title="SentimentsLoupe — Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

CSS_PATH = _APP_ROOT / "assets" / "style.css"
if CSS_PATH.exists():
    with open(CSS_PATH, encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

LOGO_PATH = _APP_ROOT / "assets" / "logo.png"
from src.config.settings import MODELS_DIR

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
    st.caption("PFA EMSI — Aymen Ichqarrane & Ihssane Moutchou  •  Mai 2026")

st.markdown(
    "<div class='app-header'>"
    "<h1>📊 Dashboard des métriques</h1>"
    "<p class='subtitle'>Comparaison TF-IDF vs XLM-RoBERTa</p>"
    "</div>",
    unsafe_allow_html=True,
)

metrics_p1_path = MODELS_DIR / "phase1_metrics.json"
metrics_p1 = {}
if metrics_p1_path.exists():
    with open(metrics_p1_path) as f:
        metrics_p1 = json.load(f)

f1_flipkart = metrics_p1.get("champion_metrics", {}).get("f1_macro", 0.6491)

# ── Chargement dynamique des métriques Phase 2 ──
metrics_p2_path = MODELS_DIR / "phase2_metrics.json"
metrics_p2 = {}
if metrics_p2_path.exists():
    with open(metrics_p2_path) as f:
        metrics_p2 = json.load(f)

p2_tfidf = metrics_p2.get("models", {}).get("tfidf_improved", {})
p2_xlm = metrics_p2.get("models", {}).get("xlm_roberta", {})

METRICS = {
    "Phase 1 — TF-IDF champion": {
        "Entraîné sur": "Flipkart (7 700)",
        "Trustpilot F1": 0.4297,
        "Jumia F1": 0.3395,
        "Flipkart F1": f1_flipkart,
        "Gain vs P1": "—",
    },
    "Phase 2A — TF-IDF amélioré": {
        "Entraîné sur": p2_tfidf.get("trained_on", "FK + Trustpilot (8 929)"),
        "Trustpilot F1": p2_tfidf.get("trustpilot_f1", "— (dans train)"),
        "Jumia F1": p2_tfidf.get("jumia_f1", 0.3764),
        "Flipkart F1": f1_flipkart,
        "Gain vs P1": p2_tfidf.get("gain_vs_p1", "+0.0369"),
    },
    "Phase 2B — XLM-RoBERTa": {
        "Entraîné sur": p2_xlm.get("trained_on", "FK + Trustpilot (8 929)"),
        "Trustpilot F1": p2_xlm.get("trustpilot_f1", 0.6017),
        "Jumia F1": p2_xlm.get("jumia_f1", 0.5660),
        "Flipkart F1": f1_flipkart,
        "Gain vs P1": p2_xlm.get("gain_vs_p1", "+0.2265"),
    },
}

st.markdown("### 🏆 Comparaison finale")
c1, c2, c3 = st.columns(3)
with c1:
    st.markdown(
        f"<div class='metric-box'>"
        f"<div class='metric-value'>{METRICS['Phase 1 — TF-IDF champion']['Jumia F1']:.3f}</div>"
        f"<div class='metric-label'>F1 TF-IDF → Jumia</div></div>",
        unsafe_allow_html=True,
    )
with c2:
    st.markdown(
        f"<div class='metric-box'>"
        f"<div class='metric-value'>{METRICS['Phase 2B — XLM-RoBERTa']['Jumia F1']:.3f}</div>"
        f"<div class='metric-label'>F1 XLM-RoBERTa → Jumia</div></div>",
        unsafe_allow_html=True,
    )
with c3:
    st.markdown(
        f"<div class='metric-box'>"
        f"<div class='metric-value' style='color:#2E7D32;'>+0.2265</div>"
        f"<div class='metric-label'>Gain absolu</div></div>",
        unsafe_allow_html=True,
    )

st.divider()
st.markdown("### 📋 Tableau détaillé")

df_metrics = pd.DataFrame.from_dict(METRICS, orient="index")
df_metrics.index.name = "Modèle"
df_metrics = df_metrics.reset_index()

for col in ["Trustpilot F1", "Jumia F1", "Flipkart F1"]:
    df_metrics[col] = df_metrics[col].apply(
        lambda x: f"{x:.4f}" if isinstance(x, float) else str(x)
    )

st.dataframe(
    df_metrics,
    use_container_width=True,
    column_config={
        "Modèle": st.column_config.TextColumn("Modèle"),
        "Entraîné sur": st.column_config.TextColumn("Dataset"),
        "Trustpilot F1": st.column_config.TextColumn("Trustpilot F1"),
        "Jumia F1": st.column_config.TextColumn("Jumia F1 🎯"),
        "Flipkart F1": st.column_config.TextColumn("Flipkart F1"),
        "Gain vs P1": st.column_config.TextColumn("Gain vs P1"),
    },
    hide_index=True,
)

st.divider()
st.markdown("### 📊 Visualisation")

chart_data = pd.DataFrame({
    "Modèle": ["TF-IDF", "XLM-RoBERTa"],
    "Trustpilot": [0.4297, 0.6017],
    "Jumia": [0.3395, 0.5660],
    "Flipkart": [f1_flipkart, f1_flipkart],
}).set_index("Modèle")

st.bar_chart(chart_data, height=400)
st.caption("Comparaison F1-Macro par source et par modèle")

st.divider()
st.markdown("### 🔬 Détail des 20 modèles Phase 1")

if "all_results" in metrics_p1:
    rows = []
    for r in metrics_p1["all_results"]:
        rows.append({
            "Stratégie": r["strategy"],
            "Modèle": r["model"],
            "F1-Macro": f"{r['f1_macro']:.4f}",
            "Accuracy": f"{r['accuracy']:.4f}",
        })
    df_all = pd.DataFrame(rows)
    st.dataframe(df_all, use_container_width=True, hide_index=True)
else:
    st.info("Données détaillées non disponibles. Exécutez le notebook 03 pour générer les métriques.")

st.divider()
st.markdown(
    "<div class='card'>"
    "<h3>💡 Interprétation pour la soutenance</h3>"
    "<blockquote style='border-left:4px solid #8D6E63;padding-left:1rem;margin:1rem 0;'>"
    "<b>Le gap DOMAINE (0.22) est inférieur au gap LANGUE (0.31).</b><br><br>"
    "Cela prouve que le vrai problème n'est pas le changement de site e-commerce "
    "(Flipkart → Trustpilot), mais bien la différence de langue "
    "(Anglais → Français/Darija).<br><br>"
    "XLM-RoBERTa résout ce problème en comprenant le français et l'arabe natifs :<br>"
    "• F1 sur Jumia passe de <b>0.34 → 0.57</b> (+67%)<br>"
    "• F1 sur Trustpilot passe de <b>0.43 → 0.60</b> (+40%)"
    "</blockquote></div>",
    unsafe_allow_html=True,
)

st.markdown(
    "<div class='app-footer'>"
    "SentimentsLoupe  •  PFA EMSI  •  Aymen Ichqarrane & Ihssane Moutchou © 2026</div>",
    unsafe_allow_html=True,
)
