# Validation oracle GEFF

La validation oracle mesure le tracker sans erreur de détection : les nœuds et leurs coordonnées
proviennent du graphe GEFF de vérité terrain, puis seules les arêtes sont reconstruites.

## Installation

```bash
pip install -e '.[oracle,dev]'
```

## Exécution

```bash
biohub-track oracle-eval \
  --train-dir /chemin/vers/train \
  --config configs/baseline.json \
  --output experiments/oracle_v1.csv \
  --predictions-dir experiments/oracle_predictions
```

Le lecteur accepte les coordonnées GEFF sous forme de propriétés scalaires `t,z,y,x` ou d'un
vecteur `position` décrit par les axes de la métadonnée. Les arêtes orientées du futur vers le passé
sont automatiquement retournées. Les liens non adjacents sont refusés afin que l'évaluation ne
mélange pas le tracking standard et le gap closing.

## Métriques

- précision, rappel, F1 et Jaccard des arêtes ;
- précision, rappel, F1 et Jaccard des parents en division ;
- `proxy_score = edge_jaccard + 0.1 * division_jaccard`.

Ce score est un diagnostic local sur des nœuds identiques. Il ne prétend pas reproduire les éventuels
ajustements privés de la métrique Kaggle.
