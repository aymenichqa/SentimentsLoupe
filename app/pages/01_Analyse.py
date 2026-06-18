"""Page Analyse — Saisie texte, URL scraping, prédiction."""
from pathlib import Path
import sys

_APP_ROOT = Path(__file__).parent.parent
_PROJECT_ROOT = _APP_ROOT.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import streamlit as st
import pandas as pd
from app.utils.scraper_embed import is_jumia_product_url, scrape_url, detect_source

st.set_page_config(
    page_title="SentimentsLoupe — Analyse",
    page_icon="🔍",
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

    model_choice = st.radio(
        "🤖 Modèle",
        ["xlm_roberta", "tfidf"],
        format_func=lambda x: "XLM-RoBERTa 🚀" if x == "xlm_roberta" else "TF-IDF classique",
        index=0,
    )

    st.divider()
    st.caption("Projet de fin d'études  •  Aymen Ich  •  Mai 2026")

st.markdown(
    "<div class='app-header'>"
    "<h1>🔍 Analyse de sentiment</h1>"
    "<p class='subtitle'>Saisissez un avis ou collez une URL Jumia / Trustpilot</p>"
    "</div>",
    unsafe_allow_html=True,
)

input_mode = st.radio(
    "Mode d'entrée",
    ["📝 Texte libre", "🌐 URL (scraping automatique)"],
    horizontal=True,
    label_visibility="collapsed",
)

user_input = ""
url_input = ""

if "📝" in input_mode:
    user_input = st.text_area(
        "Saisissez un avis (EN, FR ou Darija) :",
        placeholder="Ex: \"had l produit zwine bzzaf, khdam mzyan w prix raisonnable\"",
        height=120,
    )
else:
    url_input = st.text_input(
        "Collez une URL :",
        placeholder="https://www.jumia.ma/mlp-anniversaire/",
    )
    is_product = url_input and is_jumia_product_url(url_input)
    
    if is_product:
        st.caption("📦 Page produit détectée ! Les avis de ce seul produit seront récupérés.")
        max_products = 1  # Pas besoin de slider pour un seul produit
    else:
        st.caption(
            "Exemples : Jumia → page listing **ou** page produit • "
            "Trustpilot → https://www.trustpilot.com/review/..."
        )
        max_products = st.slider(
            "Nombre max de produits à scraper (Jumia)",
            min_value=3, max_value=20, value=10, step=1,
        )

analyze_clicked = st.button("🔍 Analyser", type="primary", use_container_width=True)

result_placeholder = st.empty()

if analyze_clicked:
    from app.utils.predictor import predict, predict_batch

    if "🌐" in input_mode and url_input:
        with st.spinner("🔍 Scraping en cours..."):
            source = detect_source(url_input)
            if source == "unknown":
                st.error("❌ Source non reconnue. Utilisez une URL Jumia ou Trustpilot.")
            else:
                st.info(f"🌐 Source détectée : **{source.title()}**")
                df = scrape_url(url_input, max_products=max_products)

                if df.empty or len(df) == 0:
                    st.warning("⚠️ Aucun avis trouvé. Vérifiez l'URL.")
                else:
                    st.success(f"✅ {len(df)} avis récupérés depuis {source.title()}")
                    texts = df["review"].tolist()
                    results = predict_batch(texts, model_choice)
                    df["sentiment"] = [r.get("label", "error") for r in results]
                    df["confidence"] = [r.get("confidence", 0.0) for r in results]

                    # ── Sauvegarder les avis scrapés dans la collection ──
                    from app.utils.data_collector import save_review
                    for _, row in df.iterrows():
                        save_review(
                            text=row["review"],
                            predicted_sentiment=row["sentiment"],
                            confidence=row["confidence"],
                            model_used=model_choice,
                            source=source,
                            url=url_input,
                        )

                    st.divider()
                    st.markdown("### 📊 Résultats du scraping")
                    col_counts = df["sentiment"].value_counts()
                    c1, c2, c3, c4 = st.columns(4)
                    with c1:
                        st.markdown(
                            f"<div class='metric-box'>"
                            f"<div class='metric-value'>{len(df)}</div>"
                            f"<div class='metric-label'>Avis analysés</div></div>",
                            unsafe_allow_html=True,
                        )
                    with c2:
                        pos = col_counts.get("positive", 0)
                        st.markdown(
                            f"<div class='metric-box'>"
                            f"<div class='metric-value' style='color:#2E7D32;'>{pos}</div>"
                            f"<div class='metric-label'>Positifs 😊</div></div>",
                            unsafe_allow_html=True,
                        )
                    with c3:
                        neu = col_counts.get("neutral", 0)
                        st.markdown(
                            f"<div class='metric-box'>"
                            f"<div class='metric-value' style='color:#F57F17;'>{neu}</div>"
                            f"<div class='metric-label'>Neutres 😐</div></div>",
                            unsafe_allow_html=True,
                        )
                    with c4:
                        neg = col_counts.get("negative", 0)
                        st.markdown(
                            f"<div class='metric-box'>"
                            f"<div class='metric-value' style='color:#C62828;'>{neg}</div>"
                            f"<div class='metric-label'>Négatifs 😢</div></div>",
                            unsafe_allow_html=True,
                        )

                    if len(df) > 0:
                        st.progress(pos / len(df) if pos > 0 else 0.0,
                                    text=f"Taux de positivité : {pos / len(df) * 100:.0f}%")

                    with st.expander("📋 Voir le détail des avis", expanded=True):
                        for i, row in df.iterrows():
                            sent = row["sentiment"]
                            conf = row["confidence"]
                            emoji = {"positive": "😊", "neutral": "😐", "negative": "😢"}.get(sent, "❓")
                            color = {"positive": "#2E7D32", "neutral": "#F57F17", "negative": "#C62828"}.get(sent, "#6A1B9A")
                            bg = {"positive": "#E8F5E9", "neutral": "#FFF8E1", "negative": "#FFEBEE"}.get(sent, "#F3E5F5")
                            st.markdown(
                                f"<div style='background:{bg};border-radius:8px;"
                                f"padding:0.6rem 1rem;margin:0.3rem 0;"
                                f"border-left:4px solid {color};'>"
                                f"<span style='font-size:0.85rem;'>"
                                f"{emoji} <b>{sent}</b> ({conf:.0%}) "
                                f"<span style='color:#8D6E63;'>| {row['review'][:80]}{'...' if len(row['review']) > 80 else ''}</span>"
                                f"</span></div>",
                                unsafe_allow_html=True,
                            )

    elif "📝" in input_mode and user_input:
        with st.spinner("🧠 Analyse en cours..."):
            result = predict(user_input, model_choice)

        if "error" in result and result["error"]:
            st.error(f"❌ {result['error']}")
        else:
            label = result["label"]
            confidence = result["confidence"]
            probs = result.get("probabilities", {})

            # ── Sauvegarder l'analyse dans la collection ──
            from app.utils.data_collector import save_review
            save_review(
                text=user_input,
                predicted_sentiment=label,
                confidence=confidence,
                model_used=model_choice,
                source="manual",
            )

            emoji = {"positive": "😊", "neutral": "😐", "negative": "😢"}.get(label, "❓")
            css_class = f"result-{label}" if label in ("positive", "neutral", "negative") else "result-error"

            st.markdown(
                f"<div class='{css_class}' style='text-align:center;padding:1.2rem;font-size:1.3rem;font-weight:600;border-radius:12px;'>"
                f"{emoji} {label.title()} — {confidence:.1%}</div>",
                unsafe_allow_html=True,
            )

            st.markdown("### 📊 Distribution des probabilités")
            cols = st.columns(3)
            for col, (name, key, clr) in zip(cols, [
                ("Négatif", "negative", "#C62828"),
                ("Neutre", "neutral", "#F57F17"),
                ("Positif", "positive", "#2E7D32"),
            ]):
                val = probs.get(key, 0.0)
                with col:
                    st.metric(label=name, value=f"{val:.1%}", delta=None, delta_color="normal")
                    st.progress(val)

            # ── Feedback utilisateur ──
            st.markdown("### 📝 Feedback")
            st.caption("Cette analyse est-elle correcte ?")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("✅ Oui, c'est correct", key="fb_ok"):
                    from app.utils.data_collector import save_correction
                    save_correction(user_input, label)
                    st.success("Merci ! Votre feedback aide à améliorer le modèle.")
            with col2:
                if st.button("❌ Non, c'est faux", key="fb_no"):
                    st.session_state["show_correction"] = True

            if st.session_state.get("show_correction"):
                correct_sentiment = st.selectbox(
                    "Quel est le bon sentiment ?",
                    ["positive", "neutral", "negative"],
                    key="correction_select",
                )
                if st.button("Confirmer la correction", key="confirm_correction"):
                    from app.utils.data_collector import save_correction
                    save_correction(user_input, correct_sentiment)
                    st.session_state["show_correction"] = False
                    st.success(f"Merci ! Correction enregistrée : {correct_sentiment}")

            st.markdown(
                f"<div class='info-box'>"
                f"<b>Texte analysé :</b><br>{user_input[:200]}{'...' if len(user_input) > 200 else ''}"
                f"</div>",
                unsafe_allow_html=True,
            )
    else:
        st.info("👆 Saisissez un texte ou une URL, puis cliquez sur **🔍 Analyser**")

st.divider()
st.markdown(
    "<div class='app-footer'>"
    "SentimentsLoupe  •  Projet de fin d'études  •  Aymen Ich © 2026</div>",
    unsafe_allow_html=True,
)