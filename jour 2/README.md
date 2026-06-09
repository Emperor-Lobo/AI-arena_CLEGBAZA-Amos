# Arène des Algos  
# Auteur: CLEGBAZA Amos
## Jour 2 : Preprocessing du dataset Telco Customer Churn

---

## Le problème
Prédire si un client d'un opérateur télécom va résilier son contrat (churn).
Dataset : 7043 clients, 21 colonnes, cible : `Churn` (Oui/Non).

---

## Résumé du pipeline de nettoyage

| Étape              | Avant | Après                              |
|--------------------|-------|------------------------------------|
| Colonnes           | 21    | 23 (après encodage)                |
| Lignes             | 7043  | 7043 (aucune perdue)               |
| Trous cachés       | 11    | 0 (imputés par médiane)            |
| Colonnes VIF > 5   | 2     | 0 (TotalCharges supprimée)         |

---

## Décisions de nettoyage et justifications

### Phase 1 — Audit qualité
- Cible déséquilibrée : **73.5% Non / 26.5% Oui**
- Ce déséquilibre piégera l'accuracy : un modèle qui répond
  toujours "Non" fait déjà 73.5% sans rien apprendre.
- Aucun manquant visible au premier regard — mais des trous
  cachés existent (voir Phase 2).

### Phase 2 — Colonne piégée TotalCharges
- `TotalCharges` stockée en `object` au lieu de `float64`.
- Cause : 11 lignes contiennent un espace `" "` au lieu d'un nombre.
- **Décision : imputation par médiane** plutôt que suppression.
- Justification : 11 lignes sur 7043, soit 0.15% du dataset.
  Les supprimer ne changerait rien statistiquement, mais
  l'imputation est plus propre et ne perd aucune information.

### Phase 3 — Encodage des catégorielles
- **Colonnes binaires Yes/No** : encodées en 0/1 directement
  (1 colonne), pas en One-Hot (2 colonnes).
  Le One-Hot sur du binaire est redondant car la 2e colonne
  est toujours le complément de la 1ère.
- **Contract : ordinal** (0=month-to-month, 1=one year, 2=two year).
  Justification : il y a un ordre naturel de durée d'engagement.
  Un client en contrat 2 ans est plus engagé qu'un client mensuel.
- **InternetService, PaymentMethod : One-Hot** (nominales,
  3-4 modalités sans ordre).
- **customerID : supprimé** — identifiant unique, pas une feature.
  Si encodé en One-Hot : 7043 colonnes créées = explosion de dimensions.

### Phase 4 — Valeurs aberrantes
- **Aucun outlier détecté** sur `tenure`, `MonthlyCharges`,
  `TotalCharges` (règle IQR).
- Les valeurs extrêmes sont des cas réels (gros clients fidèles
  ou à forte facture), pas des erreurs de saisie.
- **Décision : garder toutes les valeurs** telles quelles.
- Note : la règle IQR ne s'applique pas aux colonnes binaires
  (ex: `SeniorCitizen`) — la fonction le détecte et refuse
  de calculer.

### Phase 5 — Multicolinéarité
- `TotalCharges` corrélée à `tenure` (0.83) et
  `MonthlyCharges` (0.65) → VIF = 8.1 (> seuil de 5).
- `tenure` avait un VIF = 6.3 à cause de cette redondance.
- **Décision : supprimer TotalCharges**.
- Justification : TotalCharges ≈ tenure × MonthlyCharges,
  elle ne porte pas d'information supplémentaire.
  Après suppression, les VIF de tenure et MonthlyCharges
  tombent en dessous de 5.

### Phase 6 — Features discriminantes
- Les deux méthodes (corrélation + Random Forest) s'accordent
  sur le podium : **Contract, tenure, MonthlyCharges**.
- `MonthlyCharges` : fort en Random Forest (0.236) mais modéré
  en corrélation (0.193) → **relation non linéaire** avec le churn.
  Le Random Forest capture ce signal mieux que la corrélation simple.

**Interprétation business :**
- Les clients en contrat **mensuel** partent bien plus que
  ceux en contrat longue durée.
- Les clients **récents** (tenure faible) sont les plus fragiles.
- Une **facture élevée** augmente le risque de départ,
  de façon non linéaire.
- Levier : fidéliser tôt et proposer des contrats longs.

### Phase 7 — Split, scaling, et data leakage
- **Règle d'or** : le scaler et l'imputer s'ajustent TOUJOURS
  sur le train seul, jamais sur le test.
- Version honnête vs triche : delta de 0.21% sur ce dataset
  (7000 lignes propres → médianes quasi identiques).
- Sur des données vraiment sales avec beaucoup de trous,
  l'écart peut être bien plus grand et transformer un faux
  champion en désastre de production.
- `stratify=y` utilisé pour garder l'équilibre 73.5/26.5
  dans train ET test.

---

## Résultat baseline
- Modèle : Régression logistique
- Accuracy : **79.84%**
- L'accuracy seule ne suffit pas sur un dataset déséquilibré.
  Il faudra regarder combien de vrais churners sont récupérés
  (rappel, F1-score) — sujet du Jour 4.

---

## Dataset final
- **23 colonnes**, **7043 lignes**
- 100% numérique, sans fuite de données
- Prêt pour l'Arène du Jour 3
