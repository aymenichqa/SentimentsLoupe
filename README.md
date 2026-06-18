# 🔍 SentimentsLoupe — Analyse de Sentiment Cross-Langue

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.6.1-orange)](https://scikit-learn.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.58.0-red)](https://streamlit.io)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.12.0-ee4c2c)](https://pytorch.org)
[![Transformers](https://img.shields.io/badge/🤗_Transformers-5.9.0-yellow)](https://huggingface.co)

**SentimentsLoupe** est une application d'analyse de sentiment cross-langue développée dans le cadre d'un Projet de Fin d'Année (PFA) à l'EMSI. Le système compare deux approches de Machine Learning — classique (TF-IDF) et moderne (Transformers) — pour analyser des avis clients en anglais et en français.

> 👨‍💻 **Auteurs :** Aymen Ichqarrane & Ihssane Moutchou  
> 🎓 **Établissement :** EMSI — Filière IIR  
> 📅 **Année :** Mai 2026

---

## 📋 Table des matières

- [Aperçu du projet](#-aperçu-du-projet)
- [Fonctionnalités](#-fonctionnalités)
- [Résultats clés](#-résultats-clés)
- [Architecture du projet](#-architecture-du-projet)
- [Méthodologie](#-méthodologie)
- [Installation](#-installation)
- [Utilisation](#-utilisation)
- [Structure du projet](#-structure-du-projet)
- [Technologies utilisées](#-technologies-utilisées)
- [Roadmap](#-roadmap)
- [Limites et perspectives](#-limites-et-perspectives)

---

## 🎯 Aperçu du projet

Face à la diversité des sources d'avis clients et des langues utilisées, **SentimentsLoupe** propose un pipeline complet d'analyse de sentiment : de la collecte des données (scraping Jumia et Trustpilot) à la visualisation des résultats (dashboard interactif), en passant par la comparaison de deux approches de Machine Learning.

Le projet démontre que le **gap lié à la langue** (anglais → français : -0.31 F1) est plus grand que le **gap lié au domaine** (Flipkart → Trustpilot : -0.22 F1), et qu'un modèle Transformer multilingue (XLM-RoBERTa) permet de réduire significativement ce gap.

---

## ✨ Fonctionnalités

### 🖥️ Application Streamlit (5 pages)

| Page | Fonctionnalité |
|:-----|:---------------|
| **🏠 Accueil** | Présentation du projet avec scores clés |
| **🔍 Analyse** | Saisie de texte libre + Scraping automatique (Jumia / Trustpilot) + Feedback utilisateur (✅ / ❌) |
| **📚 Collection** | Statistiques des données collectées, graphiques de répartition, export CSV |
| **📊 Dashboard** | Comparaison des métriques TF-IDF vs XLM-RoBERTa, tableau des 18 modèles, graphiques |
| **ℹ️ À propos** | Présentation détaillée du projet, datasets, modèles, phases |

### 🤖 Double modèle

| Modèle | Description | Performance (F1) |
|:-------|:------------|:-----------------:|
| **TF-IDF + LogisticRegression** | Approche classique : 15 000 tokens TF-IDF + 6 features numériques | **0.65** sur Flipkart |
| **XLM-RoBERTa** | Transformer multilingue (279M params) fine-tuné sur 8 929 avis | **0.57** sur Jumia |

### 📊 Résultats du scraping

- Scraping Jumia (pages listing + pages produits individuelles)
- Scraping Trustpilot (via Playwright)
- Détection automatique du type d'URL (listing vs produit)
- Sauvegarde automatique des avis dans la collection

### 💾 Collecte de données

- Chaque analyse est sauvegardée dans `data/collected/reviews.csv`
- L'utilisateur peut corriger les prédictions via le feedback (✅ / ❌)
- Les corrections sont enregistrées pour un réentraînement futur
- Export CSV des données collectées

---

## 🏆 Résultats clés

### Phase 1 — TF-IDF Champion (18 modèles comparés)

| Stratégie | LogisticRegression | LinearSVC | MultinomialNB | SVM-Calibré |
|:----------|:-----------------:|:---------:|:-------------:|:-----------:|
| **none** (baseline) | 0.57 | 0.60 | 0.61 | 0.54 |
| **class_weight** | **0.65** 🏆 | 0.57 | 0.61 | 0.56 |
| **random_over** | 0.64 | 0.61 | 0.62 | 0.62 |
| **SMOTE** | 0.64 | 0.59 | 0.61 | 0.62 |
| **SMOTETomek** | 0.63 | 0.60 | 0.61 | 0.61 |

### Phase 2 — Gaps Domaine vs Langue

| Test | F1 | Gap | Interprétation |
|:-----|:--:|:---:|:---------------|
| Flipkart (entraînement) | 0.65 | — | Référence |
| Trustpilot (même langue) | 0.43 | -0.22 | **Gap DOMAINE** |
| Jumia (langue différente) | 0.34 | -0.31 | **Gap LANGUE** 🔑 |

> **🔑 Découverte clé :** Le gap **LANGUE** (0.31) est supérieur au gap **DOMAINE** (0.22).  
> Le vrai problème n'est pas le site, c'est la **langue**.

### Phase 2B — XLM-RoBERTa

| Modèle | Trustpilot F1 | Jumia F1 | Gain vs P1 |
|:-------|:------------:|:--------:|:----------:|
| **TF-IDF champion** | 0.43 | 0.34 | — |
| **TF-IDF amélioré** | — (dans train) | 0.38 | +0.04 |
| **XLM-RoBERTa 🚀** | **0.60** | **0.57** | **+0.23 (+67%)** |

---

## 🏗 Architecture du projet

L'architecture de SentimentsLoupe repose sur quatre couches :

```
┌──────────────────────────────────────────────────────────────┐
│             NOTEBOOKS (expérimentation)                       │
│  01_EDA → 02_Prep. → 03_Models → 04_Results → 05_Scrap. → 06_XLM-R │
│  Génèrent : champion_model.pkl, feature_engineer.pkl,        │
│  phase1/2_metrics.json, xlm_roberta_finetuned/               │
└──────────────────────────┬───────────────────────────────────┘
                           │ Modèles sauvegardés
                           ▼
┌──────────────────────────────────────────────────────────────┐
│                APPLICATION STREAMLIT (app/)                    │
│  ┌──────────┐  ┌────────────┐  ┌──────────┐  ┌───────────┐ │
│  │ Accueil  │  │ Analyse    │  │Collect.  │  │ Dashboard │ │
│  │ app.py   │  │ 01_Analyse │  │04_Collec │  │02_Dash.  │ │
│  └──────────┘  └─────┬──────┘  └──────────┘  └───────────┘ │
└──────────────────────┼───────────────────────────────────────┘
                       │
┌──────────────────────┼───────────────────────────────────────┐
│     COUCHE MÉTIER (app/utils/ + src/)                        │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────────────┐│
│  │predictor │ │ scraper_ │ │data_col. │ │Jumia + Trustpilot││
│  │   .py    │ │ embed.py │ │  .py     │ │   Scrapers       ││
│  └──────────┘ └──────────┘ └──────────┘ └──────────────────┘│
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────────────┐│
│  │TextClean │ │Feature   │ │ModelLoader│ │ ImbalanceExper. ││
│  │  er.py   │ │Engineer  │ │   .py     │ │    .py          ││
│  └──────────┘ └──────────┘ └──────────┘ └──────────────────┘│
└──────────────────────────────────────────────────────────────┘
                           │
┌──────────────────────────┼───────────────────────────────────┐
│     COUCHE STOCKAGE (data/)                                   │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────┐ │
│  │ models/  │  │  raw/    │  │processed/│  │ collected/   │ │
│  │ champion │  │ flipkart │  │ cleaned  │  │ reviews.csv  │ │
│  │ xlm_ro   │  │ jumia    │  │ .csv     │  │ stats.json   │ │
│  └──────────┘  └──────────┘  └──────────┘  └──────────────┘ │
└──────────────────────────────────────────────────────────────┘
```

### Flux de données (analyse texte libre)

1. **Utilisateur** saisit un texte dans l'interface Streamlit
2. **predict()** dans `predictor.py` charge le modèle choisi (TF-IDF ou XLM-RoBERTa)
3. **Si TF-IDF** : TextCleaner → FeatureEngineer (TF-IDF 15K + 6 features) → LogisticRegression
4. **Si XLM-RoBERTa** : Tokenizer → Modèle Transformer → softmax
5. **Résultat** affiché + sauvegardé dans `data_collector.py`
6. **Feedback** : l'utilisateur peut corriger → correction sauvegardée

---

## 📚 Méthodologie

La réalisation de SentimentsLoupe s'est appuyée sur une démarche progressive et itérative, combinant trois approches complémentaires :

### CRISP-DM (data science)

Le cycle CRISP-DM (Cross-Industry Standard Process for Data Mining) a structuré la partie data science en six étapes : compréhension du problème, exploration des données, préparation, modélisation, évaluation et déploiement. Son caractère itératif a permis d'améliorer progressivement les performances du système.

### Agile (travail en binôme)

Le projet a été découpé en 6 sprints de 2 à 3 semaines, facilitant la répartition des tâches entre les deux membres de l'équipe et l'adaptation continue aux difficultés rencontrées.

| Sprint | Durée | Contenu | Livrable |
|:------:|:-----:|:--------|:---------|
| **1 — Exploration** | 3 sem | EDA, définition des métriques | Notebook 01 |
| **2 — Preprocessing** | 2 sem | TextCleaner, FeatureEngineering | Notebook 02, pipeline |
| **3 — Modélisation** | 3 sem | 18 modèles TF-IDF, champion | Notebook 03, champion.pkl |
| **4 — Multi-sources** | 3 sem | Scraping Jumia + Trustpilot, gaps | Notebooks 04 & 05 |
| **5 — Phase 2B & App** | 4 sem | XLM-RoBERTa, Streamlit, dashboard | Notebook 06, app |
| **6 — Finalisation** | 2 sem | Tests, collecte, rapport | App finale, soutenance |

### Gantt & PERT (planification)

Le diagramme de Gantt a permis de visualiser les phases dans le temps, et l'analyse PERT a identifié le chemin critique pour garantir le respect des délais.

### Répartition des responsabilités

- **Aymen Ichqarrane** : préparation des données, développement des modèles de Machine Learning, expérimentation, évaluation et analyse des performances
- **Ihssane Moutchou** : développement des modules de scraping, intégration des données et développement de l'application Streamlit

---

## 💻 Installation

### Prérequis

- Python 3.10+
- Git
- (Optionnel) GPU pour accélérer le fine-tuning

### Étapes

```bash
# 1. Cloner le dépôt
git clone https://github.com/aymenichqa/sentimentsloupe.git
cd sentimentsloupe

# 2. Créer et activer l'environnement virtuel
python -m venv .venv
# Windows
.\.venv\Scripts\Activate.ps1
# Linux/Mac
source .venv/bin/activate

# 3. Installer les dépendances
pip install -r requirements.txt

# 4. (Optionnel) Installer Playwright pour le scraping Trustpilot
playwright install chromium

# 5. Lancer l'application
streamlit run app/app.py
```

---

## 🚀 Utilisation

### Lancer l'application

```bash
cd sentimentsloupe
streamlit run app/app.py
```

Puis ouvrez http://localhost:8501 dans votre navigateur.

### Tester les fonctionnalités

1. **Analyse texte libre** : allez dans "🔍 Analyse", tapez un avis, sélectionnez le modèle, cliquez "Analyser"
2. **Scraping Jumia** : collez une URL Jumia (listing ou produit), cliquez "Analyser"
3. **Scraping Trustpilot** : collez une URL Trustpilot, cliquez "Analyser" (nécessite `playwright install chromium`)
4. **Collection** : visualisez les données collectées, exportez en CSV
5. **Dashboard** : consultez les métriques comparatives des modèles

### Exemples de textes à tester

```text
# XLM-RoBERTa (comprend le français et l'anglais)
"this product is amazing" → positif
"had l produit zwine bzzaf" → positif (darija)
"c'est vraiment nul" → négatif

# TF-IDF (anglais uniquement)
"this product is amazing" → positif
"terrible quality" → négatif
"it is ok I guess not bad" → neutre
```

---

## 📁 Structure du projet

```
sentimentsloupe/
├── app/                          # Application Streamlit
│   ├── app.py                    # Page d'accueil (précharge les modèles)
│   ├── pages/
│   │   ├── 01_Analyse.py         # Analyse + Scraping + Feedback
│   │   ├── 02_Dashboard.py       # Métriques comparatives
│   │   ├── 03_A_Propos.py        # Informations projet
│   │   └── 04_Collection.py      # Données collectées + Export
│   ├── utils/
│   │   ├── predictor.py          # Chargement et prédiction (TF-IDF + XLM-R)
│   │   ├── scraper_embed.py      # Interface unifiée des scrapers
│   │   ├── model_loader.py       # Chargement des modèles sauvegardés
│   │   └── data_collector.py     # Sauvegarde, feedback, statistiques
│   └── assets/
│       ├── style.css             # Design personnalisé (thème beige/marron)
│       └── logo.png              # Logo de l'application
│
├── notebooks/                    # Notebooks Jupyter (expérimentation)
│   ├── 01_EDA_flipkart.ipynb     # Analyse exploratoire
│   ├── 02_preprocessing.ipynb    # Nettoyage et feature engineering
│   ├── 03_baseline_models.ipynb  # 18 modèles TF-IDF
│   ├── 04_results_analysis.ipynb # Analyse des gaps domaine vs langue
│   ├── 05_multisource_scraping.ipynb # Scraping Jumia + Trustpilot
│   └── 06_xlm_roberta_finetuning.ipynb # Fine-tuning XLM-RoBERTa
│
├── src/                          # Code source (pipelines ML + scraping)
│   ├── config/
│   │   └── settings.py           # Configuration globale (chemins, labels, hyperparams)
│   ├── preprocessing/
│   │   ├── text_cleaner.py       # Nettoyage des textes
│   │   ├── feature_engineering.py  # TF-IDF + features numériques
│   │   └── source_normalizer.py  # Normalisation multi-sources
│   ├── data_collection/
│   │   ├── base_scraper.py       # Classe abstraite pour les scrapers
│   │   ├── jumia_fix_scraper.py  # Scraping Jumia (listing + produit)
│   │   └── trustpilot_scraper.py # Scraping Trustpilot (Playwright)
│   └── models/
│       └── imbalance_strategies.py # 5 stratégies × 4 classifieurs = 18 modèles
│
├── data/                         # Données et modèles
│   ├── raw/                      # Données brutes (CSV, Parquet)
│   ├── processed/                # Données nettoyées
│   ├── models/                   # Modèles entraînés
│   │   ├── champion_model.pkl    # Meilleur modèle TF-IDF (F1=0.65)
│   │   ├── feature_engineer.pkl  # TF-IDF vectorizer fitté
│   │   ├── phase1_metrics.json   # Résultats des 18 modèles
│   │   ├── phase2_metrics.json   # Métriques XLM-RoBERTa
│   │   ├── xlm_roberta_finetuned/ # Modèle XLM-RoBERTa fine-tuné
│   │   └── plots/                # Graphiques générés par les notebooks
│   └── collected/                # Données collectées par l'app
│       ├── reviews.csv           # Toutes les analyses utilisateur
│       ├── corrected.csv         # Corrections feedback
│       └── stats.json            # Statistiques en temps réel
│
├── requirements.txt              # Dépendances Python
├── .env                          # Configuration (RANDOM_STATE=42)
├── .gitignore                    # Fichiers ignorés par Git
└── README.md                     # Ce fichier
```

---

## 🛠 Technologies utilisées

| Technologie | Version | Usage |
|:------------|:-------:|:------|
| **Python** | 3.10+ | Langage principal |
| **scikit-learn** | 1.6.1 | Pipeline TF-IDF, classifieurs, validation croisée |
| **imbalanced-learn** | 0.13.0 | SMOTE, RandomOverSampler, SMOTETomek |
| **PyTorch** | 2.12.0 | Moteur du modèle Transformer |
| **Transformers** | 5.9.0 | XLM-RoBERTa (Hugging Face) |
| **Streamlit** | 1.58.0 | Application web interactive |
| **BeautifulSoup** | 4.14.3 | Scraping Jumia (HTML parsing) |
| **Playwright** | 1.60.0 | Scraping Trustpilot (navigateur) |
| **pandas** | 3.0.3 | Manipulation des données |
| **numpy** | 2.4.6 | Calculs numériques |
| **joblib** | 1.5.3 | Sauvegarde des modèles |
| **matplotlib / seaborn** | — | Visualisations dans les notebooks |

---

## 🗺 Roadmap

### Version actuelle (MVP) ✅
- [x] Pipeline TF-IDF complet (18 modèles)
- [x] Fine-tuning XLM-RoBERTa (F1=0.57 sur données françaises)
- [x] Scraping Jumia (listing + pages produits)
- [x] Scraping Trustpilot (Playwright)
- [x] Application Streamlit (5 pages)
- [x] Système de collecte de données + feedback
- [x] Dashboard comparatif

### Version 1.0 (perspectives)
- [ ] Fine-tuning sur les données collectées via l'application
- [ ] Correction du modèle pour la reconnaissance du darija
- [ ] Déploiement cloud de l'application
- [ ] API REST d'analyse de sentiment
- [ ] Mode batch pour analyser des fichiers CSV complets

---

## ⚠️ Limites et perspectives

### Limites identifiées

| Limite | Impact | Cause |
|:-------|:------:|:------|
| **Reconnaissance du darija** limitée | "khayb" prédit "positif" ❌ | 0 avis darija dans le fine-tuning |
| **Dataset Jumia trop petit** | F1 limité à 0.57 | Seulement 60 avis pour le test |
| **Classe neutre peu détectée** | F1 neutre = 0.38 | Déséquilibre du dataset (81% positifs) |
| **Trustpilot bloqué** | Scraping impossible | Protection Cloudflare |

### Ce qu'il faut retenir pour la soutenance

> *"Notre projet montre qu'on peut faire de l'analyse de sentiment cross-langue avec des moyens limités. Le vrai défi est le manque de données pour le darija. Mais on a prouvé le concept : XLM-RoBERTa peut comprendre le français sans l'avoir vu à l'entraînement, et notre pipeline est prêt à intégrer plus de données via la collecte intégrée à l'application."*

---

## 📬 Contact

- **Aymen Ichqarrane** — Développement ML, modèles, expérimentation
- **Ihssane Moutchou** — Scraping, application Streamlit, collecte de données

Projet réalisé dans le cadre du **Projet de Fin d'Année** — EMSI, Filière IIR — Mai 2026.

---

<div align="center">
<strong>SentimentsLoupe</strong> — De TF-IDF aux Transformers, du scraping au dashboard.
</div>