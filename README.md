# BioHub Cell Tracker

Socle reproductible pour la compétition Kaggle **Biohub – Cell Tracking During Development**.

## État actuel

### V0 — chaîne Kaggle complète

1. découverte et lecture des volumes Zarr 3D+t ;
2. détection de centroïdes par réponse LoG multi-échelle légère ;
3. association temporelle par Hungarian en unités physiques ;
4. hypothèses de division `1 → 2` ;
5. export `submission.csv` ;
6. audit strict des invariants du graphe.

### V1 — validation oracle GEFF

La V1 ajoute :

- lecture des graphes GEFF vers la représentation interne ;
- prise en charge de `t,z,y,x` ou d'un vecteur `position` décrit par les métadonnées ;
- normalisation du sens des arêtes selon le temps ;
- tracking sur les centroïdes réels ;
- métriques séparées pour les arêtes et les divisions ;
- export des prédictions oracle et rapport CSV ;
- CI GitHub avec Ruff et Pytest.

La baseline LoG n'est pas le modèle final. Elle valide la plomberie Kaggle pendant que le mode oracle
permet d'améliorer le tracker indépendamment du détecteur.

## Installation locale

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
pytest
```

Pour la validation GEFF :

```bash
pip install -e '.[oracle,dev]'
```

## Inférence Kaggle

```bash
biohub-track run \
  --test-dir /chemin/vers/test \
  --config configs/baseline.json \
  --output submission.csv
```

Le dossier de test doit contenir un ou plusieurs stores `*.zarr`.

## Validation oracle

```bash
biohub-track oracle-eval \
  --train-dir /chemin/vers/train \
  --config configs/baseline.json \
  --output experiments/oracle_v1.csv
```

Voir [`docs/oracle_validation.md`](docs/oracle_validation.md).

## Notebook Kaggle

Le notebook `notebooks/biohub_submission_v0.ipynb` cherche le package dans :

- `/kaggle/input/biohub-cell-tracker-runtime` ;
- `/kaggle/working/biohub-cell-tracker` ;
- le dossier parent du notebook en local.

Pour Kaggle, empaqueter le dépôt comme Dataset et attacher ce Dataset au notebook. Le notebook écrit
`/kaggle/working/submission.csv` et refuse une sortie invalide. Les dépendances GEFF sont optionnelles
et ne sont pas requises dans le runtime de soumission.

## Étapes suivantes

- V1.1 : exécuter le mode oracle sur tous les embryons et calibrer les seuils par validation groupée ;
- V2 : U-Net 3D anisotrope pour les centroïdes ;
- V3 : embeddings d'apparence et tracker temporel ;
- V4 : score explicite des divisions et optimisation globale ;
- V5 : redétection guidée des gaps et ensemble multi-fold.
