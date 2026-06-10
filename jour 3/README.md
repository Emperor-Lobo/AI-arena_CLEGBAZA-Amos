# Arène des Algos 
# CLEGBAZA Amos
## Jour 3 : Les algorithmes du ML et le Fight des IA

---

## Le projet

Construction d'une **Arène des Algos** : un notebook qui, sur quatre datasets réels, fait tourner et compare des algorithmes de Machine Learning. Chaque phase couvre une famille différente (régression, clustering, texte, classification binaire sur signal).

Notebook : `jour 3/fight_ia.ipynb`

---

## Structure du repo

```
jour 3/
├── fight_ia.ipynb      # notebook principal
├── Listings.zip            # dataset AirBnB Kaggle (zippé, voir Phase B)
├── SMSSpamCollection       # dataset SMS Spam UCI
└── sonar.all-data          # dataset Sonar UCI
```

---

## Phase A : Prédire les prix immobiliers (régression)

**Dataset** : `fetch_california_housing` (scikit-learn): 20 640 quartiers, 8 variables, cible = prix médian en centaines de milliers de $.

**Pipeline** : chargement → split 80/20 → StandardScaler → LinearRegression (baseline) → RandomForestRegressor.

**Résultats** :

| Modèle            | R2   | MAE  | RMSE |
|-------------------|------|------|------|
| LinearRegression  | 0.58 | 0.53 | 0.75 |
| RandomForest      | 0.81 | 0.33 | 0.51 |

**Checkpoints qualité** :
- Cas normal : les deux modèles tournent sur le dataset complet.
- Cas limite : sur 100 lignes, le R2 ne s'effondre pas systématiquement (dépend du tirage). Il faut donc tester sur plusieurs tirages pour confirmer l'instabilité.
- Cas adversarial : quartier fictif (revenu=0, population=9000) → prédiction de **-68.76** (×100k$). Le linéaire extrapole sans garde-fou. En production : valider les entrées (clipping, rejection) avant de prédire.

**Justification du choix de métrique** : R2 donne une lecture relative ("le modèle explique X% de la variance"), MAE est lisible dans l'unité du problème, RMSE pénalise les grosses erreurs. Les trois ensemble donnent une image complète.

---

## Phase B : Segmenter les annonces AirBnB (clustering)

### Difficulté rencontrée : choix du dataset

Le dataset uitlisé au départ  pour ce TP était un fichier `listings.csv` quelconque du site **Inside Airbnb**. Cependant, dans la version "visualisations" téléchargée, les colonnes `price` et `estimated_revenue_l365d` étaient entièrement vides. Le fichier allégé ne contient pas ces données financières essentielles au clustering.

**Solution** : remplacement par le dataset **Airbnb Listings & Reviews** disponible sur Kaggle (`mysarahmadbhat/airbnb-listings-reviews`), qui contient 279 712 annonces avec `price`, `minimum_nights`, `accommodates`, `review_scores_rating` et `bedrooms` bien renseignés.

**Contrainte technique** : le fichier `Listings.csv` (152 MB) dépasse la limite GitHub de 100 MB. Résolution en deux étapes :
1. Compression en `Listings.zip` (24.8 MB), sous la limite.
2. Purge de `Listings.csv` de l'historique Git via `git filter-branch` pour débloquer le push.

**Source** : https://www.kaggle.com/datasets/mysarahmadbhat/airbnb-listings-reviews
**Fichier commité** : `Listings.zip` (lire avec `zipfile.ZipFile` dans le notebook)

**Pipeline** : chargement → nettoyage NaN + filtre prix (0 < prix < 1000) → échantillon 20 000 lignes → StandardScaler → KMeans avec coude + silhouette.

**Résultats** : k=5 retenu (silhouette=0.427)

| Segment | Interprétation          | Prix moy. | Min nuits | Capacité | Note |
|---------|-------------------------|-----------|-----------|----------|------|
| 0       | Budget / standard       | 118 €     | 7         | 3        | 95   |
| 1       | Peu noté / inactif      | 212 €     | 6         | 3        | 52   |
| 2       | Longue durée            | 179 €     | 439       | 3        | 92   |
| 3       | Premium                 | 653 €     | 4         | 3        | 96   |
| 4       | Familial / groupe       | 269 €     | 6         | 7        | 94   |

**Checkpoints qualité** :
- Cas limite : sans standardisation, silhouette = 0.568 vs 0.427 avec. Paradoxe apparent : le score semble meilleur, mais `price` (0–1000) écrase `bedrooms` (1–5). Les segments ne reflètent que le prix, pas la structure métier réelle.
- Cas adversarial : injection d'une annonce à 100 000€/nuit → le centre price_scaled d'un segment explose de -0.165 à +134.99. Nettoyage J2 obligatoire avant tout KMeans.

---

## Phase C : Courriel vs spam (classification texte)

**Dataset** : SMS Spam Collection (UCI): 5 574 messages, 13.4% spam / 86.6% normal.
**Fichier** : `SMSSpamCollection` (tab-séparé, chargé en local)

**Pipeline** : chargement → split stratifié 80/20 → TF-IDF vectorisation → Naive Bayes (baseline texte) → Régression logistique.

**Résultats** :

| Modèle              | Precision spam | Recall spam | F1 spam | Accuracy |
|---------------------|---------------|-------------|---------|----------|
| Naive Bayes         | 1.00          | 0.72        | 0.84    | 0.96     |
| Régression logistique | 1.00        | 0.81        | 0.90    | 0.97     |

**Pourquoi F1 et pas accuracy** : le dataset est déséquilibré (86.6% normal). Un modèle qui répond toujours "normal" ferait déjà 86.6% sans rien apprendre. F1 équilibre precision et recall sur la classe minoritaire (spam).

**Checkpoints qualité** :
- Happy path : recall spam > 0.85 pour la LR 
- Edge case : message vide `""` → vecteur tout à zéro → proba spam = 0.134 (exactement le taux de base). Le vectorizer ne plante pas, le modèle renvoie la prior.
- Adversarial : `"salut, ton colis t attend, confirme ici"` → classé normal (proba spam = 0.126). Explication : le dataset est en anglais, les mots français ne sont pas dans le vocabulaire spam appris. Limite du bag-of-words monolingue.

**Observation** : Naive Bayes a une precision spam de 1.00. Quand il détecte un spam, il ne se trompe jamais. Mais son recall de 0.72 signifie qu'il rate 28% des vrais spams. La LR offre un meilleur équilibre (recall 0.81).

---

## Phase D : Décrypter les signaux d'un sonar (classification binaire)

**Dataset** : Sonar / Connectionist Bench (UCI): 208 échos, 60 variables de fréquence, cible : mine (M=1) ou rocher (R=0).
**Fichier** : `sonar.all-data` (chargé en local)

**Pipeline** : chargement → split stratifié 80/20 → StandardScaler → LogisticRegression + SVC rbf + RandomForest.

**Résultats** :

| Modèle             | Accuracy | F1    |
|--------------------|----------|-------|
| SVC (rbf)          | 92.86%   | 0.936 |
| LogisticRegression | 83.33%   | 0.844 |
| RandomForest       | 83.33%   | 0.844 |

**Checkpoints qualité** :

- Cas normal : SVM rbf domine, exactement ce que prédit la règle "quel algo en premier" : peu de données (208 lignes), beaucoup de variables (60) = terrain du SVM.
- Cas limite (sans standardisation) :

| Modèle             | Avec scaling | Sans scaling |
|--------------------|-------------|-------------|
| SVC (rbf)          | 92.86%      | 83.33%      |
| LogisticRegression | 83.33%      | 80.95%      |
| RandomForest       | 83.33%      | 83.33%      |

SVM chute de 9.5 points sans scaling. RandomForest non affecté — les arbres se moquent de l'échelle.

- Cas adversarial : capteur en panne (60 zéros) → LogReg prédit "rocher" avec 100% de confiance, SVM à 66.1%, RF à 71.5%. Les trois prédisent sans hésiter sur un signal nul. En production : détecter les entrées hors plage avant de prédire. Un capteur en panne ne devrait jamais atteindre le modèle.

---

## Récapitulatif des difficultés techniques

| Phase | Difficulté | Solution |
|-------|-----------|----------|
| B | Dataset Inside Airbnb : colonnes `price` et `revenue` vides | Remplacement par dataset Kaggle (279k annonces, prix renseigné) |
| B | `Listings.csv` (152 MB) dépasse la limite GitHub (100 MB) | Compression en `Listings.zip` (24.8 MB) + purge de l'historique Git |
| B | Silhouette meilleure sans standardisation (0.568 vs 0.427) | Le paradoxe vient de `price` qui écrase tout — les segments "compacts" ne reflètent que le prix, pas la structure métier |

---

## Leçons clés Jour 3 (phase 0 à phase D)

1. **L'accuracy seule ment** sur des classes déséquilibrées. Toujours regarder precision/recall/F1.
2. **KMeans exige une standardisation** obligatoire avant usage; une colonne à grande échelle écrase toutes les autres dans le calcul de distance.
3. **Le SVM brille sur peu de données / beaucoup de variables**; dataset sonar : 208 lignes, 60 features, 92.86% de F1.
4. **Les entrées hors plage sont dangereuses**. Un modèle linéaire extrapole sans garde-fou (prix immobilier négatif), un modèle de classification prédit avec assurance sur un signal nul.
5. **Le bag-of-words est monolingue**. Un modèle entraîné sur de l'anglais ne détecte pas les spams en français.
