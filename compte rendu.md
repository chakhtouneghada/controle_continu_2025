# Détection de Fraude sur Cartes Bancaires

Projet de détection de fraude à la carte bancaire basé sur le dataset public **Credit Card Fraud Detection** (Kaggle – MLG ULB / Worldline).[web:2]

---

## 📌 Description du projet

Ce projet vise à construire un modèle de Machine Learning capable d’identifier automatiquement les transactions frauduleuses à partir de données réelles de cartes bancaires européennes.[web:2][web:9]  
Le jeu de données est extrêmement déséquilibré (≈ 0,172 % de fraudes), ce qui en fait un cas typique de classification avec classes rares.[web:2]

---

## 🗂️ Dataset

- Source : Kaggle – *Credit Card Fraud Detection* (MLG ULB / Worldline).[web:2]  
- Nombre de transactions : 284 807  
- Nombre de fraudes : 492 (≈ 0,172 %)  
- Variables :
  - `Time` : secondes depuis la première transaction.
  - `Amount` : montant de la transaction.
  - `V1` à `V28` : composantes issues d’une PCA (données anonymisées).[web:2]
  - `Class` : cible binaire (0 = normal, 1 = fraude).

> Le fichier CSV n’est pas inclus dans le dépôt et doit être téléchargé directement depuis Kaggle.[web:20]

---

## 🧹 Prétraitement des données

Principales étapes de préparation :  
- Suppression d’éventuels doublons et contrôle des valeurs manquantes.[web:2]  
- Standardisation de `Time` et `Amount` (centrage-réduction).[web:2]  
- Gestion du déséquilibre de classes avec :
  - **SMOTE** pour sur-échantillonner la classe fraude.[web:7]
  - `class_weight="balanced"` dans certains modèles pour pondérer la classe minoritaire.[web:7][web:9]

---

## 🔍 Analyse exploratoire (EDA)

- Étude de la distribution des montants (`Amount`) et du temps (`Time`).[web:2]  
- Visualisation de la répartition des classes (fraude vs normal).[web:2]  
- Analyse de certaines composantes PCA (ex. `V12`, `V14`, `V17`) montrant des patterns associés aux fraudes.[web:1][web:3]

---

## 🤖 Modèles utilisés

Plusieurs algorithmes de classification supervisée sont testés :  
- **Régression logistique** (baseline).  
- **Random Forest** (modèle d’arbres en ensemble, robuste au bruit).[web:5]  
- **XGBoost / LightGBM** (gradient boosting sur arbres, adaptés aux patterns rares et données déséquilibrées).[web:8]

Les modèles sont entraînés dans un pipeline intégrant prétraitement, gestion du déséquilibre et optimisation d’hyperparamètres.

---

## 📈 Métriques d’évaluation

L’accuracy seule n’est pas pertinente dans ce contexte déséquilibré.[web:9]  
Les métriques suivies sont :  
- **Recall** sur la classe fraude (priorité métier).  
- **Precision** sur la classe fraude.  
- **F1-score**.  
- **Matrice de confusion**.  
- **ROC-AUC** et **Precision–Recall AUC**, plus informative lorsque la classe positive est rare.[web:9]

---

## 🧠 Interprétabilité & améliorations

Pistes d’analyse et d’amélioration :  
- Utilisation de **SHAP** pour interpréter les décisions des modèles de type XGBoost / LightGBM.[web:8]  
- Exploration de modèles **deep learning** (autoencodeurs, LSTM) pour la détection d’anomalies et la prise en compte de la dimension temporelle.[web:4]  
- Intégration possible de nouvelles variables comportementales (device, localisation, historique client) dans un contexte réel.[web:6]

---

## 📦 Installation

