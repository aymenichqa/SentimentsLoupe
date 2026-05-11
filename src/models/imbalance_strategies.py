"""Expérience comparative des stratégies anti-déséquilibre pour Phase 1."""
import numpy as np
import pandas as pd
from collections import defaultdict
from copy import deepcopy
import joblib
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    f1_score, accuracy_score, precision_score, recall_score,
    classification_report, confusion_matrix
)
from sklearn.model_selection import StratifiedKFold, cross_val_score

# Imbalance samplers
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.combine import SMOTETomek

from src.config.settings import RANDOM_STATE, CLASS_NAMES, IMBALANCE_STRATEGIES


class ImbalanceExperiment:
    """Compare plusieurs stratégies anti-déséquilibre sur plusieurs classifieurs."""

    def __init__(self):
        self.results_ = None
        self.champion_ = None
        self.models_ = defaultdict(dict)

    def _get_sampler(self, strategy: str):
        """Retourne un sampler selon la stratégie."""
        samplers = {
            'none': None,
            'random_over': RandomOverSampler(random_state=RANDOM_STATE),
            'smote': SMOTE(random_state=RANDOM_STATE),
            'smotetomek': SMOTETomek(random_state=RANDOM_STATE),
        }
        return samplers.get(strategy, None)

    def _get_classifier(self, model_name: str, strategy: str):
        """Retourne un classifieur avec ou sans class_weight."""
        if model_name == 'lr':
            if strategy == 'class_weight':
                return LogisticRegression(
                    C=1.0, max_iter=2000, class_weight='balanced',
                    random_state=RANDOM_STATE, n_jobs=-1
                )
            return LogisticRegression(
                C=1.0, max_iter=2000, random_state=RANDOM_STATE, n_jobs=-1
            )
        elif model_name == 'svm':
            if strategy == 'class_weight':
                return LinearSVC(
                    C=1.0, max_iter=5000, class_weight='balanced',
                    random_state=RANDOM_STATE
                )
            return LinearSVC(
                C=1.0, max_iter=5000, random_state=RANDOM_STATE
            )
        elif model_name == 'nb':
            return MultinomialNB(alpha=0.1)
        elif model_name == 'svm_calibrated':
            if strategy == 'class_weight':
                base = LinearSVC(
                    C=1.0, max_iter=5000, class_weight='balanced',
                    random_state=RANDOM_STATE
                )
            else:
                base = LinearSVC(
                    C=1.0, max_iter=5000, random_state=RANDOM_STATE
                )
            return CalibratedClassifierCV(base, cv=3, n_jobs=-1)
        else:
            raise ValueError(f"Modèle inconnu: {model_name}")

    def _compute_metrics(self, y_true, y_pred):
        """Calcule toutes les métriques."""
        f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
        return {
            'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
            'f1_weighted': f1_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1_negative': f1_per_class[0],
            'f1_neutral': f1_per_class[1],
            'f1_positive': f1_per_class[2],
            'accuracy': accuracy_score(y_true, y_pred),
            'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
            'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0),
            'classification_report': classification_report(
                y_true, y_pred, target_names=CLASS_NAMES, zero_division=0
            ),
            'confusion_matrix': confusion_matrix(y_true, y_pred).tolist(),
        }

    def run(self, X_train, y_train, X_test, y_test,
            strategies: list = None,
            models: list = None):
        """Exécute l'expérience complète.

        Args:
            X_train: sparse matrix features d'entraînement
            y_train: array labels d'entraînement
            X_test: sparse matrix features de test
            y_test: array labels de test
            strategies: liste des stratégies à comparer
            models: liste des modèles à comparer

        Returns:
            pd.DataFrame: résultats triés par F1-Macro descendant
        """
        if strategies is None:
            strategies = IMBALANCE_STRATEGIES
        if models is None:
            models = ['lr', 'svm', 'nb', 'svm_calibrated']

        results = []
        model_names = {
            'lr': 'LogisticRegression',
            'svm': 'LinearSVC',
            'nb': 'MultinomialNB',
            'svm_calibrated': 'SVM-Calibré',
        }

        print(f"{'Stratégie':<18} {'Modèle':<22} {'F1-Macro':<10} {'F1-Neg':<10} {'F1-Pos':<10}")
        print("-" * 70)

        for strategy in strategies:
            sampler = self._get_sampler(strategy)
            
            for model_name in models:
                clf = self._get_classifier(model_name, strategy)
                
                # Sampling (uniquement si pas class_weight)
                if sampler is not None and strategy not in ('none', 'class_weight'):
                    X_tr, y_tr = sampler.fit_resample(X_train, y_train)
                else:
                    X_tr, y_tr = X_train, y_train
                
                try:
                    # Entraînement
                    clf.fit(X_tr, y_tr)
                    
                    # Prédiction
                    y_pred = clf.predict(X_test)
                    
                    # Métriques
                    metrics = self._compute_metrics(y_test, y_pred)
                    
                    # Stockage
                    self.models_[strategy][model_name] = {
                        'model': clf,
                        'metrics': metrics
                    }
                    
                    results.append({
                        'strategy': strategy,
                        'model': model_names[model_name],
                        **metrics
                    })
                    
                    print(f"{strategy:<18} {model_names[model_name]:<22} "
                          f"{metrics['f1_macro']:.4f}     {metrics['f1_negative']:.4f}     "
                          f"{metrics['f1_positive']:.4f}")
                    
                except Exception as e:
                    print(f"{strategy:<18} {model_names[model_name]:<22} ERROR: {str(e)}")
                    continue

        self.results_ = pd.DataFrame(results)
        self.results_ = self.results_.sort_values('f1_macro', ascending=False).reset_index(drop=True)
        
        # Trouver le champion
        self._find_champion()
        
        return self.results_

    def _find_champion(self):
        """Identifie le meilleur modèle (F1-Macro max)."""
        if self.results_ is None or self.results_.empty:
            return
        
        best = self.results_.iloc[0]
        self.champion_ = {
            'strategy': best['strategy'],
            'model_name': best['model'],
            'metrics': {
                'f1_macro': best['f1_macro'],
                'f1_negative': best['f1_negative'],
                'f1_neutral': best['f1_neutral'],
                'f1_positive': best['f1_positive'],
                'accuracy': best['accuracy'],
                'precision_macro': best['precision_macro'],
                'recall_macro': best['recall_macro'],
                'classification_report': best['classification_report'],
                'confusion_matrix': best['confusion_matrix'],
            }
        }

    def get_champion(self):
        """Retourne (strategy, model_name, champion_dict)."""
        if self.champion_ is None:
            return None, None, None
        
        strategy = self.champion_['strategy']
        model_name = self.champion_['model_name']
        
        # Chercher le modèle dans self.models_
        # On utilise un mapping explicite pour éviter les collisions
        MODEL_KEY_MAP = {
            'LogisticRegression': 'lr',
            'LinearSVC': 'svm',
            'MultinomialNB': 'nb',
            'SVM-Calibré': 'svm_calibrated',
        }
        model_key = MODEL_KEY_MAP.get(model_name)
        
        if strategy in self.models_ and model_key in self.models_[strategy]:
            return strategy, model_name, self.models_[strategy][model_key]
        
        # Fallback: recherche par startswith (ancienne méthode)
        for s, models_dict in self.models_.items():
            if s == strategy:
                for name, m in models_dict.items():
                    if model_name.lower().startswith(name.lower()):
                        return strategy, model_name, m
        
        return strategy, model_name, self.champion_

    def cross_validate_champion(self, X_train, y_train, cv=5):
        """Cross-validation 5-fold du champion.
        
        CORRECTION: utilise le vrai modèle champion au lieu de toujours SVM-Calibré.

        Args:
            X_train: sparse matrix features
            y_train: array labels
            cv: nombre de folds

        Returns:
            dict: résumé CV
        """
        if self.champion_ is None:
            return {}
        
        strategy = self.champion_['strategy']
        model_name = self.champion_['model_name']
        
        # Mapping du nom du modèle vers la clé interne
        MODEL_KEY_MAP = {
            'LogisticRegression': 'lr',
            'LinearSVC': 'svm',
            'MultinomialNB': 'nb',
            'SVM-Calibré': 'svm_calibrated',
        }
        model_key = MODEL_KEY_MAP.get(model_name, 'svm_calibrated')
        
        # Reconstruire le classifieur champion (pas toujours svm_calibrated !)
        clf = self._get_classifier(model_key, strategy)
        
        # Sampling si nécessaire (hors CV pour compatibilité, mais idéalement dans un Pipeline)
        sampler = self._get_sampler(strategy)
        if sampler is not None and strategy not in ('none', 'class_weight'):
            X_cv, y_cv = sampler.fit_resample(X_train, y_train)
        else:
            X_cv, y_cv = X_train, y_train
        
        # Cross-validation
        cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=RANDOM_STATE)
        scores = cross_val_score(
            clf, X_cv, y_cv, cv=cv_splitter,
            scoring='f1_macro', n_jobs=-1
        )
        
        return {
            'cv_scores': scores.tolist(),
            'f1_macro_mean': float(scores.mean()),
            'f1_macro_std': float(scores.std()),
            'f1_weighted_mean': float(cross_val_score(
                clf, X_cv, y_cv, cv=cv_splitter,
                scoring='f1_weighted', n_jobs=-1
            ).mean()),
            'accuracy_mean': float(cross_val_score(
                clf, X_cv, y_cv, cv=cv_splitter,
                scoring='accuracy', n_jobs=-1
            ).mean()),
        }

    def save(self, path: Path):
        """Sauvegarde tous les modèles."""
        path = Path(str(path))
        path.mkdir(parents=True, exist_ok=True)
        
        for strategy, models_dict in self.models_.items():
            for model_name, m in models_dict.items():
                filename = f"model_{strategy}_{model_name}.pkl"
                joblib.dump(m, str(path / filename))
        
        # Résultats
        if self.results_ is not None:
            self.results_.to_csv(str(path / 'experiment_results.csv'), index=False)
        
        print(f"Modèles sauvegardés dans {path}")