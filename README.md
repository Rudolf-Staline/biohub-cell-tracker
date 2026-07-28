# BioHub Cell Tracker

Socle reproductible pour la compétition Kaggle **Biohub – Cell Tracking During Development**.

La V0 fournit une chaîne complète et auditable :

1. découverte et lecture des volumes Zarr 3D+t ;
2. détection de centroïdes par réponse LoG multi-échelle légère ;
3. association temporelle par Hungarian en unités physiques ;
4. hypothèses de division `1 → 2` sur les cellules non appariées ;
5. export au format `submission.csv` ;
6. audit strict des invariants du graphe.

Cette version n'est pas le modèle final. Elle sert de baseline de plomberie avant d'ajouter le détecteur appris, les embeddings d'identité, le modèle temporel et le solveur global.

## Installation locale

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
pytest
```

## Exécution

```bash
biohub-track run   --test-dir /chemin/vers/test   --output submission.csv
```

Le dossier de test doit contenir un ou plusieurs stores `*.zarr`.

## Notebook Kaggle

Le notebook `notebooks/biohub_submission_v0.ipynb` cherche le package dans :

- `/kaggle/input/biohub-cell-tracker-runtime` ;
- `/kaggle/working/biohub-cell-tracker` ;
- le dossier parent du notebook en local.

Pour Kaggle, empaqueter le dépôt comme Dataset et attacher ce Dataset au notebook. Le notebook écrit toujours `/kaggle/working/submission.csv` et refuse une sortie invalide.

## Étapes suivantes

- V1 : validation avec centroïdes GEFF réels et baseline oracle ;
- V2 : U-Net 3D anisotrope pour les centroïdes ;
- V3 : embeddings d'apparence et tracker temporel ;
- V4 : score explicite des divisions et optimisation globale ;
- V5 : redétection guidée des gaps et ensemble multi-fold.
