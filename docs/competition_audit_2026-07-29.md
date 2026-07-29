# Audit de la compétition BioHub — 29 juillet 2026

## Décision principale

Les calculs lourds doivent être exécutés sur Kaggle. Le dépôt local reste réservé au code, aux tests unitaires, aux petites fixtures synthétiques et à la préparation des notebooks. Il n'est pas nécessaire de télécharger les 87,61 Go du jeu complet.

Après inspection du dépôt officiel, des notebooks publics et d'un journal public détaillant plus de cent expériences, la stratégie de départ devient :

> mélange des heatmaps de deux `TemporalUNet3D` indépendants, extraction unique des centroïdes, scoring temporel appris, construction globale du graphe, post-traitement structurel, redétection ciblée des trous, puis calibrage du nombre de nœuds.

Ce choix est plus précis que « entraîner notre propre U-Net » ou « prendre Ultrack comme backbone ». Le meilleur point de départ public propre et documenté est un blend de détecteurs, pas un ensemble complet de deux trackers.

## 1. La métrique officielle est obligatoire

Le dépôt `royerlab/kaggle-cell-tracking-competition` contient l'implémentation de référence. Notre métrique interne actuelle ne peut pas guider le choix du modèle, car elle compare directement les identifiants alors que l'évaluation officielle :

- apparie les nœuds prédits et annotés par affectation bipartite optimale dans un rayon de 7 µm ;
- tient compte du caractère clairsemé des annotations ;
- ignore certaines prédictions hors des régions annotées plutôt que de toutes les compter comme faux positifs ;
- pénalise le nombre total de nœuds prédits via l'Adjusted Edge Jaccard ;
- tolère une division un frame avant ou après, sous contraintes topologiques locales ;
- agrège les comptes TP/FP/FN par micro-moyenne.

La version à épingler est au minimum le commit officiel `075fc5f`, qui contient le correctif supprimant l'exploitation par composantes faiblement connexes. Le journal d'expériences public indique que le patch a réduit les scores des solutions exploitant de faux forks, tandis que les solutions biologiquement plausibles ont peu changé côté arêtes.

Conséquence : le `proxy_score` actuel reste uniquement un test de cohérence synthétique. Le prochain changement de code obligatoire est l'intégration directe de l'évaluateur officiel et du round-trip CSV ↔ GEFF.

## 2. Limite majeure de la validation locale

Un journal d'expériences public a comparé le pipeline complet et une version ILP brute avec la métrique officielle sur les labels d'entraînement :

```text
local : ILP seul 0,9083 > pipeline complet 0,8877
public: ILP seul 0,877  < pipeline complet 0,909
```

Le classement est inversé. L'explication la plus plausible est la très faible couverture des labels : selon ce journal, certaines vidéos ont seulement 50 à 1 183 arêtes annotées pour 25 000 à 70 000 cellules, soit moins de 2 % de couverture. Les zones denses où le post-traitement apporte le plus sont donc mal représentées dans le score local.

La validation locale reste utile pour :

- détecter les régressions catastrophiques ;
- vérifier le schéma, les degrés et les arêtes consécutives ;
- confirmer qu'une branche de code est réellement active ;
- comparer les détecteurs sur le rappel des nœuds et la densité.

Elle ne suffit pas pour conclure qu'un post-traitement améliore ou dégrade le leaderboard. Les décisions fines sur le graphe nécessiteront de petits probes Kaggle contrôlés.

## 3. Baseline officielle et support pack public

Le dépôt du Royer Lab propose :

1. `TemporalUNet3D` pour produire une carte de centres et des features par voxel ;
2. local-max suppression pour extraire les centroïdes ;
3. pooling des features au niveau des nœuds ;
4. `SimpleNodeTransformer` avec cross-attention pour scorer les associations entre `t` et `t+1` ;
5. supervision clairsemée sur les arêtes annotées.

Le support pack public le plus utilisé est trompeusement nommé `50ep`. L'inspection publique de son checkpoint rapporte :

- epoch réel : 402 ;
- architecture complète dans un seul fichier : U-Net, tête de détection et transformer ;
- downsampling anisotrope `[1, 4, 4]` ;
- fenêtre temporelle 2 ;
- optimizer state présent.

Il ne faut donc pas repartir de zéro ni supposer que ce checkpoint est sous-entraîné. Des expériences publiques rapportent qu'une poursuite de l'entraînement a dégradé le leaderboard et que le rappel local des nœuds était déjà proche de 0,998. Le premier objectif est la reproduction exacte, pas un nouvel entraînement.

## 4. Meilleure stratégie publique propre identifiée

Le notebook `Two Seeds Logit Blend` n'exécute pas deux graphes complets. Son mécanisme documenté est :

```text
TemporalUNet3D seed A ─┐
                       ├─ blend des heatmaps de détection ─ centroïdes uniques
TemporalUNet3D seed B ─┘
                                      ↓
                           scoring des liens / ILP une fois
                                      ↓
                    post-traitement + DeepCenter ciblé
```

Détails rapportés :

- poids de la seconde heatmap autour de `0,475` ;
- score public observé autour de `0,912` après le patch ;
- un poids `0,40` a produit environ `0,910` ;
- DeepCenter ne valide que des nœuds marginaux proposés pour réparer un trou ;
- le graphe, l'ILP et le post-traitement ne sont exécutés qu'une fois.

Cette architecture évite deux échecs déjà observés :

- l'ensemble séquentiel de plusieurs pipelines dépasse le budget du rerun caché ;
- la moyenne des poids de modèles entraînés indépendamment affaiblit fortement le détecteur.

Le blend en espace des sorties devient donc notre incumbent à reproduire.

## 5. Le post-traitement est structurellement important

Des mesures publiques rapportent :

```text
ILP direct sans post-traitement : environ 0,877
pipeline complet             : environ 0,909
```

Le gain est important, mais les variations fines de paramètres sont souvent presque plates : modifier les coûts ILP, le rayon de motion relink ou certains poids n'a déplacé le leaderboard que d'environ 0,001. Le message n'est pas « supprimer le post-traitement », mais :

> conserver les étapes structurelles utiles et arrêter de sur-optimiser des constantes au millième.

Les briques qui méritent d'être reproduites sont :

- motion relinking ;
- insertion de nœuds pour les trous d'une frame ;
- filtrage des trajectoires très courtes ;
- divisions conservatrices ;
- contrôle de la densité de nœuds.

Un gap doit être réparé en créant ou redétectant un nœud à `t+1`. Une arête directe `t → t+2` est éliminée par la métrique officielle.

## 6. Densité, seuils et différences entre embryons

Le score n'est pas monotone avec le nombre de détections. Une série publique rapporte un maximum proche de :

- seuil de détection autour de `0,96875–0,97` ;
- longueur minimale de trajectoire autour de 6 ;
- environ 121 000 nœuds sur les quatre vidéos visibles pour le modèle simple de référence.

Pruner davantage ou réduire fortement les détections a dégradé le leaderboard. Ces nombres constituent un point initial à reproduire, pas des constantes universelles.

Les familles d'embryons semblent également différentes. Des validations publiques rapportent que le modèle appris améliore surtout le préfixe `6bba`, tandis qu'une méthode classique pouvait être meilleure sur `44b6`. Un signal forum, non encore vérifié par les organisateurs, indique en plus de nombreuses frames adjacentes identiques dans `6bba` et aucune dans `44b6`. Le notebook d'exploration devra donc mesurer explicitement les transitions figées et les statistiques par préfixe.

## 7. Lecture des notebooks publics

### Priorité 1 — reproductibles et utiles

- baseline officielle Royer Lab ;
- `Two Seeds Logit Blend` ;
- `Clean Approach + Lightweight Local CV | No Hack` ;
- variantes documentées du pipeline U-Net + graphe + ILP.

### Priorité 2 — ablations utiles

Le notebook de Sahil Deogade a atteint historiquement 0,884 et expose clairement : U-Net 3D, seuil adaptatif, raffinement subvoxel, mutual-kNN, division gating, ILP, density scaling, one-frame gap closing et filtrage. Sa dernière version affiche 0,042 : seule la version historique doit être étudiée.

### À ne pas prendre comme preuve architecturale

- notebooks copiés reposant sur un support pack opaque ;
- notebooks à 0,95+ avant le patch de métrique ;
- notebook à runtime de huit secondes, vraisemblablement post-traitement ou prédictions pré-calculées.

Les métadonnées Kaggle indexées peuvent encore afficher d'anciens meilleurs scores antérieurs au rescore. Le journal public indique que le haut du leaderboard est passé d'environ 0,982 à 0,929 après le correctif. Nous ne comparerons donc que des scores datés et post-patch.

## 8. Position sur Ultrack

Ultrack reste pertinent pour ses hypothèses multiples et son solveur global, mais il demande des cartes de foreground et de contours. Les données fournissent des centres et des graphes clairsemés, pas des masques complets.

Le benchmark Ultrack est reporté après reproduction du blend à deux seeds. Il devra démontrer :

- une entrée foreground/contours crédible ;
- un score officiel supérieur à l'incumbent ;
- une exécution CBC compatible avec le rerun caché ;
- une consommation mémoire acceptable.

Ultrack est donc un challenger, pas le point de départ.

## 9. Architecture cible révisée

```text
OME-Zarr 3D+t
    ↓
TemporalUNet3D seed A + seed B
    ↓
blend des heatmaps avant NMS
    ↓
centroïdes uniques + embeddings
    ↓
graphe sparse t → t+1
    ├── distance physique
    ├── score appris des liens
    ├── cohérence du mouvement local
    ├── voisinage spatial
    └── triplets parent / deux filles
    ↓
ILP / résolution globale
    ↓
post-traitement structurel
    ├── motion relink
    ├── short-track filtering
    ├── divisions conservatrices
    └── one-frame gap proposals
    ↓
DeepCenter sur les seuls nœuds de gap ambigus
    ↓
calibrage de densité + export CSV
```

## 10. Plan d'expériences

### E000 — sonde légère sur Kaggle

- inventorier shapes, chunks, types et échelles ;
- lire trois frames sous-échantillonnées d'un embryon ;
- mesurer les quantiles d'intensité ;
- détecter les frames adjacentes identiques par préfixe ;
- ne charger aucun volume complet.

### E001 — métrique officielle patchée

- épingler `royerlab/kaggle-cell-tracking-competition@075fc5f` ou plus récent vérifié ;
- convertir notre graphe interne vers le format officiel ;
- tester le round-trip GEFF → CSV → GEFF ;
- vérifier l'absence de l'ancienne fonction d'exploitation dans `division_metrics`.

### E002 — reproduction mono-seed

- checkpoint public complet ;
- seuil et post-traitement de l'incumbent ;
- notebook GPU unique ;
- mesure du runtime sur les quatre vidéos visibles.

### E003 — reproduction two-seed

- seed A + seed B ;
- blend `0,475` avant extraction des points ;
- graphe exécuté une seule fois ;
- DeepCenter ciblé ;
- soumission structurée et profilée.

### E004 — ablations sobres

- mono-seed contre two-seed ;
- sans DeepCenter ;
- sans motion relink ;
- sans gap repair ;
- densité globale contre calibration par préfixe.

Aucune grande grille. Chaque test doit répondre à une question unique.

### E005 — amélioration propre

- score de lien enrichi par mouvement collectif ;
- pseudo-labels sur les arêtes non annotées ;
- éventuellement flow/affinity head ;
- entraînement seulement si la nouvelle supervision dépasse les limites des labels clairsemés.

### E006 — benchmark Ultrack

- un embryon ;
- CBC ;
- fenêtres temporelles chevauchantes ;
- comparaison score officiel / temps / RAM avec E003.

## 11. Règles opérationnelles

- Le notebook soumis est réexécuté sur un test caché plus grand.
- La forme fiable rapportée publiquement est un modèle GPU unique ou un blend de détection léger ; les ensembles complets séquentiels ont expiré.
- Tous les datasets doivent être présents dans la sortie, même en cas de sous-détection extrême.
- Les `node_id` sont locaux à chaque dataset ; les validations doivent utiliser `(dataset, node_id)`.
- Les arêtes doivent relier des frames consécutives, avec degré entrant maximal 1 et degré sortant maximal 2.
- Chaque run enregistre le hash de la soumission, les nombres de nœuds/arêtes/divisions et les inputs Kaggle.
- Les résultats locaux sont séparés explicitement des résultats leaderboard.

## 12. Travail local contre travail Kaggle

### Local

- structures de données ;
- tests unitaires ;
- audit CSV ;
- interfaces des modèles ;
- fixtures synthétiques ;
- Git et documentation.

### Kaggle

- exploration des vrais volumes ;
- métrique officielle ;
- entraînement et inférence ;
- soumissions et probes ;
- profilage mémoire/temps.

## Sources

Sources primaires :

- `royerlab/kaggle-cell-tracking-competition` ;
- `royerlab/ultrack` ;
- pages Kaggle répertoriées dans `public_notebooks_matrix.csv`.

Source secondaire particulièrement détaillée :

- `dalloliogm/kaggle_competitions/competitions/biohub-cell-tracking-during-development/LEARNINGS.md` et `TASKS.md` au commit `66cddc0`.

Cette source secondaire contient des résultats leaderboard, logs d'exécution et références à des discussions Kaggle. Les informations qui n'ont pas été confirmées directement par le dépôt officiel sont signalées comme rapports empiriques, pas comme règles de la compétition.
