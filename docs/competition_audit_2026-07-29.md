# Audit de la compétition BioHub — 29 juillet 2026

## Décision principale

Les calculs lourds doivent être exécutés sur Kaggle. Le dépôt local reste réservé au code, aux tests unitaires, aux petites fixtures synthétiques et à la préparation des notebooks. Il n'est pas nécessaire de télécharger les 87,61 Go du jeu complet pour développer le moteur.

La stratégie retenue après inspection du dépôt officiel et des notebooks publics est :

> détecteur 3D appris de centres cellulaires + représentations de nœuds + score appris des liens et divisions + résolution globale du graphe + redétection ciblée des trous + calibrage du nombre total de nœuds.

Ultrack reste un benchmark expérimental prioritaire, mais ne devient pas le backbone par défaut avant d'avoir démontré trois choses sur un embryon réel : qualité des cartes foreground/contours disponibles, score officiel et temps CBC dans un notebook Kaggle.

## 1. Ce que la métrique change

Le dépôt officiel `royerlab/kaggle-cell-tracking-competition` contient l'implémentation de référence de la métrique. Notre métrique interne actuelle ne peut pas guider le choix du modèle, car elle compare directement les identifiants alors que l'évaluation officielle :

- apparie les nœuds prédits et annotés par affectation bipartite optimale dans un rayon de 7 µm ;
- tient compte du fait que les annotations sont clairsemées ;
- ignore certaines prédictions hors des régions annotées plutôt que de toutes les compter comme faux positifs ;
- pénalise le nombre total de nœuds prédits via l'Adjusted Edge Jaccard ;
- tolère une division un frame avant ou après, sous contraintes topologiques locales ;
- agrège les comptes TP/FP/FN par micro-moyenne.

Conséquence : l'intégration de l'évaluateur officiel est le prochain changement de code obligatoire. Le `proxy_score` actuel doit rester uniquement un test de cohérence sur graphes synthétiques.

## 2. Référence officielle à reproduire

Le dépôt du Royer Lab propose une baseline spécifique à la compétition :

1. `TemporalUNet3D` pour produire une carte de centres et des features par voxel ;
2. local-max suppression pour extraire les centroïdes ;
3. pooling des features au niveau des nœuds ;
4. `SimpleNodeTransformer` avec cross-attention pour scorer les associations entre `t` et `t+1` ;
5. supervision clairsemée sur les arêtes annotées.

Cette baseline doit être reproduite avant de développer un réseau entièrement différent. Elle donne un point de comparaison propre et fournit déjà les lecteurs, conversions GEFF/CSV, scripts d'entraînement et métrique.

## 3. Lecture des notebooks publics

Les scores seuls sont trompeurs. Les notebooks ont été classés selon la provenance de leurs entrées et leur capacité à reconstruire une soumission à partir des images brutes.

### Famille A — pipeline technique explicite

Le notebook de Sahil Deogade a atteint historiquement 0,884. Son sommaire indique :

- U-Net 3D ;
- seuil adaptatif ;
- raffinement subvoxel ;
- graphe sparse mutual-kNN ;
- division gating ;
- solveur ILP ;
- density scaling ;
- one-frame gap closing ;
- filtrage des tracks courts et nœuds isolés.

Sa dernière version affiche 0,042 : il faut donc reproduire la version historique correspondant au meilleur score, et non copier la dernière version sans contrôle.

### Famille B — détecteur appris + ensemble léger

Les notebooks suivants se situent autour de 0,90–0,911 avec des runtimes de 15 à 20 minutes :

- `Biohub Cell Tracking` : 0,900 ;
- `Clean Approach + Lightweight Local CV | No Hack` : meilleur score 0,908 ;
- `Biohub E016 | Embryo-Aware Dual Seed` : 0,908 ;
- `Two Seeds Logit Blend` : 0,911.

Les titres et entrées suggèrent que le gain vient surtout de l'ensemble de seeds, du calibrage par embryon et d'un support pack commun. Ce sont les candidats prioritaires pour une reproduction propre après la baseline officielle.

### Famille C — scores élevés mais faible valeur architecturale tant que les entrées ne sont pas auditées

- `Dark-AGI-Biohub Cell Tracking Solution` : meilleur score 0,952, notebook copié et dépendant du `Biohub Tracking Support Pack` ;
- `Biohub Modular Last Call-Turned` : meilleur score 0,965, runtime de 8 secondes.

Un runtime de huit secondes ne correspond pas à une inférence complète sur les volumes 3D. Ces notebooks peuvent être des post-traitements, des blends ou charger des prédictions pré-calculées. Ils ne doivent pas servir de preuve qu'une architecture donnée atteint 0,965. Ils seront étudiés uniquement pour leurs réparations de graphe et leur calibrage, après vérification de la provenance des fichiers d'entrée.

## 4. Position sur Ultrack

Ultrack est très pertinent conceptuellement : hypothèses multiples de segmentation, résolution globale, support 3D et mise à l'échelle sur des embryons de poisson-zèbre. Toutefois, il demande des cartes de foreground et de contours. La compétition ne fournit que des centres et des graphes clairsemés, pas des masques complets.

Le benchmark Ultrack ne sera lancé qu'après disponibilité d'une entrée crédible :

- foreground produit par un modèle appris ou par une transformation stable du signal ;
- contours ou carte d'arêtes utile dans les régions denses ;
- configuration CBC avec fenêtre temporelle pour respecter le budget Kaggle.

Critères d'adoption : score officiel supérieur au tracker appris de référence, mémoire compatible et temps d'exécution raisonnable sans licence Gurobi.

## 5. Architecture cible

```text
OME-Zarr 3D+t
    ↓
TemporalUNet3D ou détecteur 3D anisotrope
    ├── heatmap de centres
    ├── confiance de nœud
    ├── embeddings d'apparence
    └── score mitotique auxiliaire
    ↓
Génération de candidats par frame
    ↓
Graphe sparse t → t+1
    ├── distance physique
    ├── similarité d'embedding
    ├── cohérence du mouvement local
    ├── voisinage spatial
    └── triplets parent / deux filles
    ↓
Transformer ou MLP de scoring
    ↓
Solveur global
    ├── un parent maximum
    ├── deux enfants maximum
    ├── divisions explicites
    └── contrôle apparitions/disparitions
    ↓
Redétection d'un nœud manquant à t+1
    ↓
Nouvelle résolution
    ↓
Calibrage de densité et export CSV
```

Le gap recovery doit créer ou redétecter un nœud à la frame manquante. Une arête directe `t → t+2` est éliminée par la métrique officielle.

## 6. Plan d'expériences

### E000 — sonde des données sur Kaggle

- inventorier shapes, chunks, types et échelles sans charger les volumes complets ;
- lire trois frames sous-échantillonnées d'un seul embryon ;
- mesurer quantiles d'intensité et densité approximative ;
- inspecter le nombre de nœuds, arêtes et divisions des GEFF si les dépendances sont disponibles.

### E001 — métrique officielle

- convertir notre graphe interne vers le format officiel ;
- utiliser le code du Royer Lab épinglé à un commit ;
- vérifier le round-trip GEFF → CSV → GEFF ;
- ajouter des tests de non-régression sur les cas de division décalée et de faux nœuds.

### E002 — baseline officielle

- reproduire `TemporalUNet3D + SimpleNodeTransformer` ;
- split par embryon, jamais par nœud ;
- enregistrer le score officiel, le node recall, le total node ratio et le runtime.

### E003 — U-Net + graphe + ILP

- reproduire la meilleure version historique du notebook à 0,884 ;
- isoler les gains du seuil adaptatif, subvoxel refinement, ILP, gap recovery et density scaling.

### E004 — ensemble propre

- deux seeds du détecteur ;
- blend des logits avant extraction des centroïdes ;
- calibrage global et éventuellement par embryon ;
- pas de prédictions publiques pré-calculées.

### E005 — benchmark Ultrack

- un embryon ;
- CBC ;
- fenêtres temporelles avec chevauchement ;
- comparaison score officiel / temps / RAM avec E003.

## 7. Règles de validation

- Tous les scores décisionnels proviennent de la métrique officielle.
- Les splits sont groupés par embryon.
- La provenance de chaque poids, support pack et prédiction est documentée.
- Les fichiers dérivés du test public ne sont jamais utilisés comme vérité terrain.
- Chaque expérience enregistre commit, configuration, inputs Kaggle, runtime et score.
- La densité des nœuds est contrôlée explicitement à cause de la pénalité de surdétection.
- Le notebook final doit fonctionner sans Internet et découvrir les datasets dynamiquement.

## 8. Travail local contre travail Kaggle

### Local

- structures de données ;
- tests unitaires ;
- audit CSV ;
- interfaces des modèles ;
- fixtures synthétiques ;
- Git et documentation.

### Kaggle

- exploration des vrais volumes ;
- calcul de statistiques ;
- entraînement ;
- validation officielle ;
- inférence et soumission ;
- profilage mémoire/temps.

## Sources primaires

- Dépôt officiel de la compétition : `royerlab/kaggle-cell-tracking-competition`
- Description de la métrique : `metrics.md` du dépôt officiel
- Implémentation : `src/tracking_cellmot/metrics.py` et `division_metrics.py`
- Ultrack : `royerlab/ultrack`
- Pages Kaggle des notebooks répertoriés dans `public_notebooks_matrix.csv`

## Limite de cet audit

Le moteur de recherche expose correctement les métadonnées des notebooks publics, mais très peu de contenu des fils de discussion Kaggle. Aucun conseil de forum non vérifiable n'a donc été transformé en décision technique. La prochaine passe devra utiliser l'interface Kaggle connectée pour relever les discussions épinglées et les réponses des organisateurs, puis ajouter les informations confirmées à ce document.
