# Jour 4 : Deep Learning, Évaluation et Production
## CLEGBAZA Amos

---

## Plan de l'après-midi

### Phase 0 : Mise en route
Setup du notebook, imports, vérification des dépendances.

### Phase 1 : Séparer les données proprement en train / validation / test
Découpage en 3 jeux distincts avec stratification. Le test ne se touche qu'une seule fois, tout à la fin.

### Phase 2 : Bootstrap et bagging
Évaluation de la stabilité d'un modèle par rééchantillonnage avec remise. Visualisation de la dispersion des scores out-of-bag.

### Phase 3 : Validation croisée k-fold
Évaluation standard avec `cross_val_score`. Comparaison avec et sans `StratifiedKFold` sur données déséquilibrées.

### Phase 4 : Choisir la bonne métrique selon le coût métier
L'accuracy ne suffit pas. Démonstration du piège des 99% sur données déséquilibrées. Calcul du coût métier réel (FN × coût_fn + FP × coût_fp).

### Phase 5 : Sérialiser le modèle et le servir derrière une API
Sauvegarde avec `joblib`, endpoint Flask `/predict`, validation des entrées, gestion des erreurs.

### Phase 6 : Déployer une WebApp de prédiction
Interface Streamlit avec champs de saisie, bouton Prédire, affichage de la probabilité et avertissement hors plage.

### Phase 7 : Arbitrage final (phase ouverte)
Random Forest vs PMC Keras en validation croisée. SGD vs Adam. Leaderboard de l'arbitre : accuracy, recall, coût métier, temps d'entraînement, latence.

---

## Dataset
`load_breast_cancer` (scikit-learn): 569 patients, 30 features, classification binaire : tumeur maligne (0) ou bénigne (1).

---

## Dépendances
```
scikit-learn
tensorflow
joblib
flask
streamlit
```

---

## Livrable 
- `evaluation_production.ipynb` : notebook d'évaluation complet
- `modele.joblib` : modèle + scaler sérialisés
- `api.py` : endpoint Flask
- `app.py` : WebApp Streamlit
- README décrivant le plan de l'après-midi
