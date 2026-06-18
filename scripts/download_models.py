"""
Script de téléchargement/génération des modèles volumineux.
Ces fichiers ne sont pas versionnés sur GitHub à cause de leur taille > 100 Mo.

Usage:
    python scripts/download_models.py
    
Ce script exécute les notebooks pour recréer les fichiers suivants :
    - data/models/champion_model.pkl
    - data/models/feature_engineer.pkl
    - data/models/xlm_roberta_finetuned/model.safetensors
"""
import os
import sys
from pathlib import Path

# Ajouter la racine du projet au path
sys.path.insert(0, str(Path(__file__).parent.parent))

ROOT = Path(__file__).parent.parent
MODELS_DIR = ROOT / "data" / "models"


def check_missing_files() -> list:
    """Liste les fichiers manquants."""
    required = [
        MODELS_DIR / "champion_model.pkl",
        MODELS_DIR / "feature_engineer.pkl",
        MODELS_DIR / "xlm_roberta_finetuned" / "model.safetensors",
        MODELS_DIR / "phase1_metrics.json",
        MODELS_DIR / "phase2_metrics.json",
    ]
    return [f for f in required if not f.exists()]


def generate_via_notebooks():
    """Exécute les notebooks pour générer les modèles."""
    import subprocess
    
    notebooks = [
        ("02_preprocessing.ipynb", "feature_engineer.pkl"),
        ("03_baseline_models.ipynb", "champion_model.pkl + phase1_metrics.json"),
        ("04_results_analysis.ipynb", "phase2_metrics.json"),
        ("06_xlm_roberta_finetuning.ipynb", "XLM-RoBERTa + model.safetensors"),
    ]
    
    print("=" * 60)
    print("📦 Génération des modèles à partir des notebooks...")
    print("=" * 60)
    
    for nb, desc in notebooks:
        nb_path = ROOT / "notebooks" / nb
        if not nb_path.exists():
            print(f"⚠️  Notebook {nb} introuvable, ignoré.")
            continue
        
        print(f"\n⚙️  Exécution de {nb} → {desc}")
        ret = subprocess.run(
            [sys.executable, "-m", "jupyter", "nbconvert", 
             "--to", "notebook", "--execute", 
             "--ExecutePreprocessor.timeout=600",
             "--output", str(nb_path),
             str(nb_path)],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        
        if ret.returncode == 0:
            print(f"✅  {nb} exécuté avec succès.")
        else:
            print(f"❌  {nb} a échoué. Erreur : {ret.stderr[:200]}")
    
    print("\n" + "=" * 60)
    print("✅ Génération terminée !")
    print("=" * 60)


def main():
    missing = check_missing_files()
    
    if not missing:
        print("✅ Tous les modèles sont déjà présents dans data/models/")
        print()
        for f in [
            MODELS_DIR / "champion_model.pkl",
            MODELS_DIR / "feature_engineer.pkl",
            MODELS_DIR / "xlm_roberta_finetuned" / "model.safetensors",
        ]:
            if f.exists():
                size_mb = f.stat().st_size / (1024 * 1024)
                print(f"   ✓ {f.name} ({size_mb:.1f} Mo)")
        return
    
    print(f"⚠️  {len(missing)} fichier(s) manquant(s) :")
    for f in missing:
        print(f"   - {f.relative_to(ROOT)}")
    
    print()
    resp = input("Voulez-vous exécuter les notebooks pour les générer ? (o/N) : ")
    if resp.lower().startswith("o"):
        generate_via_notebooks()
        still_missing = check_missing_files()
        if still_missing:
            print(f"\n⚠️  {len(still_missing)} fichier(s) toujours manquant(s).")
        else:
            print("\n✅ Tous les modèles sont prêts !")
    else:
        print("⏩ Opération annulée.")


if __name__ == "__main__":
    main()