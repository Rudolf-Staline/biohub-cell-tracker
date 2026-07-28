# Registre d'expériences

Chaque ligne de `ledger.csv` doit relier une configuration, un commit et des métriques reproductibles.

La première expérience V1 à lancer est la validation oracle :

```bash
biohub-track oracle-eval \
  --train-dir /chemin/vers/train \
  --config configs/baseline.json \
  --output experiments/oracle_v1.csv
```
