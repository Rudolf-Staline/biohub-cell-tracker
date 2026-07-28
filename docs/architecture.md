# Architecture V0 → architecture cible

## V0 livrée

```text
Zarr 3D+t
  → LoG 3D + maxima locaux
  → détections (t,z,y,x,score)
  → Hungarian frame à frame
  → divisions géométriques prudentes
  → graphe valide
  → submission.csv
```

## Architecture cible

```text
Zarr 3D+t
  → détecteur 3D multi-têtes
      centres + soft masks + embeddings + mitose
  → graphe de candidats
      nœuds + liens + gaps + triplets de division
  → modèle temporel bidirectionnel
  → optimisation globale
  → redétection guidée des cellules manquées
  → seconde optimisation
  → graphe de lignées final
```

## Règle de développement

Chaque nouvelle brique doit être remplaçable derrière une interface stable et mesurée indépendamment :

- détection seule ;
- tracking avec nœuds oracle ;
- divisions avec nœuds oracle ;
- pipeline complet.
