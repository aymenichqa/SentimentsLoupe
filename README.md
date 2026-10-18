# SentimentsLoupe

**Une application d'analyse de sentiment qui ne s'arrête pas à une seule langue.**

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?logo=scikitlearn&logoColor=white)](https://scikit-learn.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-Transformer-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org)
[![GitHub](https://img.shields.io/badge/GitHub-aymenichqa-181717?logo=github&logoColor=white)](https://github.com/aymenichqa)

SentimentsLoupe est le projet de fin d'année (PFA) que j'ai réalisé à l'**EMSI**, en filière IIR, avec Ihssane Moutchou. L'idée de départ est simple : les outils d'analyse de sentiment marchent bien… tant que vous restez sur le site et la langue sur lesquels ils ont été entraînés. Nous avons voulu mesurer exactement **où** ça casse, et voir si un modèle multilingue pouvait réparer la casse.

> **En résumé :** changer de langue coûte plus cher (−0.31 F1) que changer de site (−0.22 F1). Le vrai obstacle n'est pas le domaine, c'est la langue — et XLM-RoBERTa réduit nettement cet écart.

---

## Sommaire

- [Le problème](#le-problème)
- [Ce que fait l'application](#ce-que-fait-lapplication)
- [Les résultats](#les-résultats)
- [Démarrage rapide](#démarrage-rapide)
- [Prendre en main l'app](#prendre-en-main-lapp)
- [Comment c'est organisé](#comment-cest-organisé)
- [La méthode derrière tout ça](#la-méthode-derrière-tout-ça)
- [Ce qui ne marche pas encore](#ce-qui-ne-marche-pas-encore)
- [Ce qui arrive](#ce-qui-arrive)
- [Auteurs et contact](#auteurs-et-contact)

---

## Le problème

Un avis écrit en anglais sur Flipkart, le même ressenti exprimé en français sur Trustpilot, ou en darija marocaine sur Jumia : pour un humain c'est la même chose. Pour un modèle entraîné sur le premier site, ce n'est pas du tout la même chose.

On a donc construit un pipeline complet pour y voir clair, de la collecte jusqu'au dashboard :

1. **Collecter** des avis depuis plusieurs sources (scraping Jumia et Trustpilot)
2. **Comparer** deux familles de modèles : classique (TF-IDF) et moderne (Transformers)
3. **Isoler** l'effet du site de l'effet de la langue, en les testant séparément
4. **Mesurer** ce que ça donne en conditions réelles, et l'afficher proprement dans une app

---

## Ce que fait l'application

L'interface Streamlit comporte cinq pages :

| Page | Ce qu'on y fait |
|:--|:--|
| 🏠 **Accueil** | Présentation du projet et des scores clés |
| 🔍 **Analyse** | Saisir un texte libre ou une URL (Jumia / Trustpilot), choisir le modèle, analyser |
| 📚 **Collection** | Consulter les statistiques des données collectées, exporter en CSV |
| 📊 **Dashboard** | Comparer TF-IDF et XLM-RoBERTa sur l'ensemble des métriques |
| ℹ️ **À propos** | Détails du projet, des datasets et des modèles |

Trois points méritent d'être signalés :

- **Le scraping s'adapte seul.** L'URL collée est analysée pour détecter s'il s'agit d'une page listing ou d'une page produit, et le bon extracteur se lance.
- **Le feedback alimente le modèle.** Chaque analyse est enregistrée dans `data/collected/reviews.csv`, et l'utilisateur peut corriger une prédiction avec ✅ / ❌. Ces corrections sont conservées pour un futur réentraînement.
- **Deux modèles cohabitent.** On bascule de l'un à l'autre dans l'interface sans rien redémarrer.

---

## Les résultats

### Phase 1 — Trouver un champion parmi 20 modèles TF-IDF

On croise 5 stratégies de gestion du déséquilibre de classes avec 4 classifieurs (F1) :

| Stratégie | LogisticRegression | LinearSVC | MultinomialNB | SVM calibré |
|:--|:-:|:-:|:-:|:-:|
| Aucune (baseline) | 0.57 | 0.60 | 0.61 | 0.54 |
| **Pondération des classes** | **0.65** 🏆 | 0.57 | 0.61 | 0.56 |
| RandomOverSampler | 0.64 | 0.61 | 0.62 | 0.62 |
| SMOTE | 0.64 | 0.59 | 0.61 | 0.62 |
| SMOTETomek | 0.63 | 0.60 | 0.61 | 0.61 |

**Champion :** `TF-IDF (15 000 tokens) + 6 features numériques + LogisticRegression` → F1 de **0.65** sur Flipkart.

### Phase 2 — Séparer l'effet du site de l'effet de la langue

C'est la partie la plus intéressante : on garde le même modèle, on le confronte à des données qu'il n'a jamais vues, et on regarde ce qu'il perd.

| Test | F1 | Écart vs référence | Ce que ça mesure |
|:--|:-:|:-:|:--|
| Flipkart (entraînement) | 0.65 | — | La référence |
| Trustpilot (même langue, autre site) | 0.43 | −0.22 | L'effet **domaine** |
| Jumia (autre site, autre langue) | 0.34 | −0.31 | L'effet **langue** 🔑 |

> **La découverte :** l'écart lié à la langue (0.31) dépasse celui lié au domaine (0.22). Affiner le modèle sur un seul site ne suffit pas tant que la barrière linguistique reste en place.

### Phase 2B — Ce que change un Transformer multilingue

XLM-RoBERTa (279 M de paramètres, pré-entraîné sur 100 langues) a été fine-tuné sur 8 929 avis — dont du français qui n'était pas dans l'entraînement du modèle classique.

| Modèle | Trustpilot F1 | Jumia F1 | Gain sur Jumia |
|:--|:-:|:-:|:-:|
| Champion TF-IDF | 0.43 | 0.34 | — |
| XLM-RoBERTa | **0.60** | **0.57** | **+0.23 (+67 %)** |

Le modèle comprend du français sans l'avoir vu à l'entraînement. C'est exactement l'effet recherché.

---

## Démarrage rapide

### Prérequis

- Python 3.10 ou supérieur
- Git
- Un GPU est un plus pour le fine-tuning, mais pas nécessaire pour faire tourner l'app

### Installation

```bash
# 1. Cloner le dépôt
git clone https://github.com/aymenichqa/Pfa_project.git
cd Pfa_project

# 2. Créer et activer un environnement virtuel
python -m venv venv

# Windows
venv\Scripts\activate
# Linux / macOS
source venv/bin/activate

# 3. Installer les dépendances
pip install -r requirements.txt

# 4. (Optionnel) nécessaire pour le scraping Trustpilot, qui passe par un vrai navigateur
playwright install chromium

# 5. Générer les modèles — obligatoire avant la première utilisation
python scripts/download_models.py

# 6. Lancer l'application
streamlit run app/app.py
```

Puis ouvrez [http://localhost:8501](http://localhost:8501).

### ⚠️ Les modèles ne sont pas dans le dépôt

Trois fichiers sont nécessaires au fonctionnement de l'app mais volontairement exclus du dépôt :

| Fichier | Taille réelle | Règle d'exclusion |
|:--|:-:|:--|
| `data/models/champion_model.pkl` | 354 Ko | `*.pkl` |
| `data/models/feature_engineer.pkl` | 602 Ko | `*.pkl` |
| `data/models/xlm_roberta_finetuned/model.safetensors` | 1.1 Go | `*.safetensors` (et bien au-delà des 100 Mo admis par GitHub) |

Les deux `.pkl` sont légers : rien ne les empêcherait techniquement d'être versionnés, ils sont simplement couverts par une règle `*.pkl` générique. Seul le poids de XLM-RoBERTa dépasse réellement la limite de GitHub.

`scripts/download_models.py` les recrée en exécutant les notebooks dans l'ordre (préprocessing → modèles → analyse → fine-tuning). C'est l'étape la plus longue du projet.

---

## Prendre en main l'app

Quelques textes pour vérifier que tout tourne, avec le comportement attendu :

```text
"this product is amazing"     →  positif   (les deux modèles)
"terrible quality"            →  négatif   (les deux modèles)
"c'est vraiment nul"          →  négatif   (XLM-RoBERTa uniquement)
"had l produit zwine bzzaf"   →  positif   (XLM-RoBERTa, en darija)
```

Le TF-IDF ne comprend que l'anglais — c'est voulu, c'est précisément ce que le projet illustre.

---

## Comment c'est organisé

```
Pfa_project/
├── app/                          # L'application Streamlit
│   ├── app.py                    # Page d'accueil (précharge les modèles)
│   ├── pages/                    # Les 5 pages de l'app
│   ├── utils/                    # predictor, scrapers, data_collector
│   └── assets/                   # style.css + logo
│
├── notebooks/                    # L'expérimentation, dans l'ordre
│   ├── 01_EDA_flipkart.ipynb     # Exploration des données
│   ├── 02_preprocessing.ipynb    # Nettoyage + feature engineering
│   ├── 03_baseline_models.ipynb  # Les 20 modèles TF-IDF
│   ├── 04_results_analysis.ipynb # Les écarts domaine vs langue
│   ├── 05_multisource_scraping.ipynb    # Jumia + Trustpilot
│   └── 06_xlm_roberta_finetuning.ipynb  # Fine-tuning du multilingue
│
├── src/                          # Le code réutilisable
│   ├── config/settings.py        # Chemins, labels, hyperparamètres
│   ├── preprocessing/            # TextCleaner, FeatureEngineer, normalisation
│   ├── data_collection/          # Scrapers (base abstraite, Jumia, Trustpilot)
│   ├── models/                   # Stratégies de déséquilibre
│   └── utils/
│
├── scripts/download_models.py    # Régénère les modèles volumineux
├── data/
│   ├── raw/                      # Données brutes
│   ├── processed/                # Données nettoyées
│   ├── models/                   # Modèles entraînés + métriques + plots
│   └── collected/                # Ce que l'app collecte (feedback inclus)
│
└── requirements.txt
```

**Le principe :** les notebooks servent à explorer et à entraîner, `src/` contient le code réellement réutilisé, et `app/` ne fait que consommer les artefacts produits. Vous changez un modèle, vous régénérez les `.pkl`, l'app suit.

---

## La méthode derrière tout ça

Au-delà du code, ce PFA s'est appuyé sur trois cadres :

- **CRISP-DM** pour structurer la partie data science — compréhension, exploration, préparation, modélisation, évaluation, déploiement. Son caractère itératif nous a servi plusieurs fois : la phase 2 n'aurait jamais existé sans ce que la phase 1 nous avait appris.
- **Agile** pour le travail en binôme — 6 sprints de 2 à 4 semaines, ce qui a permis de répartir les tâches et d'ajuster au fur et à mesure.
- **Gantt et PERT** pour la planification et l'identification du chemin critique.

| Sprint | Durée | Contenu | Livrable |
|:--:|:--:|:--|:--|
| 1 — Exploration | 3 sem | EDA, définition des métriques | Notebook 01 |
| 2 — Preprocessing | 2 sem | TextCleaner, feature engineering | Notebook 02 |
| 3 — Modélisation | 3 sem | 20 modèles TF-IDF | Notebook 03, champion.pkl |
| 4 — Multi-sources | 3 sem | Scraping Jumia + Trustpilot, écarts | Notebooks 04 & 05 |
| 5 — Phase 2B & App | 4 sem | XLM-RoBERTa, Streamlit, dashboard | Notebook 06, l'app |
| 6 — Finalisation | 2 sem | Tests, collecte, rapport | App finale, soutenance |

**Répartition du travail :**
- **Aymen Ichqarrane** — préparation des données, modèles, expérimentation, évaluation
- **Ihssane Moutchou** — modules de scraping, intégration des données, application Streamlit

---

## Ce qui ne marche pas encore

On préfère le dire clairement plutôt que de laisser croire que c'est résolu :

| Limite | Impact observé | Cause |
|:--|:--|:--|
| Darija mal reconnue | « khayb » prédit « positif » ❌ | Aucun avis darija dans le fine-tuning |
| Test Jumia réduit | F1 plafonné à 0.57 | Seulement 60 avis pour le test |
| Classe neutre peu détectée | F1 neutre = 0.38 | Dataset déséquilibré (81 % de positifs) |
| Scraping Trustpilot instable | Parfois bloqué | Protections Cloudflare côté site |

**Ce qu'on retient pour la soutenance :** on sait faire de l'analyse de sentiment cross-langue avec des moyens limités. Le vrai manque, c'est des données — surtout en darija. Mais le concept est démontré, et le pipeline est prêt à intégrer plus de données via la collecte intégrée à l'app.

---

## Ce qui arrive

- [ ] Réentraîner le modèle sur les données collectées via l'app
- [ ] Améliorer la reconnaissance du darija
- [ ] Mode batch pour analyser un fichier CSV en entier
- [ ] API REST d'analyse de sentiment
- [ ] Déploiement en ligne

---

## Technologies utilisées

| Technologie | Usage |
|:--|:--|
| **Python 3.10+** | Langage principal |
| **scikit-learn** | Pipeline TF-IDF, classifieurs, validation croisée |
| **imbalanced-learn** | SMOTE, RandomOverSampler, SMOTETomek |
| **PyTorch + Transformers** | Fine-tuning de XLM-RoBERTa |
| **Streamlit** | Application web interactive |
| **BeautifulSoup** | Scraping Jumia (parsing HTML) |
| **Playwright** | Scraping Trustpilot (navigateur headless) |
| **pandas / numpy** | Manipulation des données |
| **matplotlib / seaborn / plotly** | Visualisations |

---

## Auteurs et contact

**Aymen Ichqarrane** — développement ML, modèles, expérimentation
🐙 [github.com/aymenichqa](https://github.com/aymenichqa) · ✉️ [aymenichqa@gmail.com](mailto:aymenichqa@gmail.com)

**Ihssane Moutchou** — scraping, application Streamlit, collecte de données

Projet réalisé dans le cadre du **Projet de Fin d'Année** — EMSI, filière IIR — Mai 2026.

---

### Licence

Dépôt publié à des fins de présentation académique. N'hésitez pas à vous en inspirer pour vos propres projets.

<div align="center">
  <sub>SentimentsLoupe — De TF-IDF aux Transformers, du scraping au dashboard.</sub>
</div>
