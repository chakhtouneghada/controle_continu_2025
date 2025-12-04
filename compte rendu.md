1. Introduction

La fraude par carte bancaire est un enjeu majeur pour les institutions financières. Chaque année, des milliers de transactions frauduleuses causent des pertes économiques importantes.
Ce projet a pour objectif de développer un modèle de Machine Learning capable de détecter automatiquement les transactions frauduleuses parmi un très grand volume de données, tout en minimisant les faux positifs.

Le dataset utilisé provient de Kaggle et contient des transactions bancaires anonymisées effectuées par des clients européens en 2013.

🎯 2. Objectifs du Projet

Détecter les transactions frauduleuses (Class = 1)

Gérer un dataset très déséquilibré (0.172% de fraudes)

Comparer plusieurs modèles de Machine Learning

Optimiser les performances en termes de :

Recall (priorité métier)

Precision

F1-score

ROC-AUC

Construire un pipeline complet : preprocessing → EDA → ML → évaluation

🗂️ 3. Dataset

Nombre de lignes : 284 807

Nombre de colonnes : 31

Nombre de fraudes : 492

Proportion de fraudes : 0.172%

Variables :

Time — secondes écoulées depuis la 1ère transaction

Amount — montant de la transaction

V1 à V28 — composantes PCA anonymisées

Class — 0 (normal), 1 (fraude)

➡️ Dataset officiel : https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

🧹 4. Prétraitement des données

Les étapes suivantes ont été appliquées :

✔ Nettoyage

Suppression des doublons

Vérification des valeurs manquantes

✔ Normalisation

Standardisation de Amount

Time optionnel selon la modélisation

✔ Gestion du déséquilibre

Deux stratégies testées :

SMOTE (oversampling)

Class weight = "balanced"

✔ Train / Test split

Séparation stratifiée en fonction de la classe

📊 5. Analyse Exploratoire (EDA)

L’EDA comprend :

Analyse de la distribution des montants

Visualisation du déséquilibre des classes

Étude des corrélations entre les composantes PCA

Boxplots / histogrammes par classe

Analyse temporelle des transactions (Time)

Chaque graphique est accompagné d’une interprétation détaillée.

🤖 6. Modélisation Machine Learning

Plusieurs modèles ont été testés :

🔹 Régression Logistique

Baseline simple

Performances correctes mais limitées

🔹 Random Forest

Bon équilibre précision / rappel

Résultats robustes

🔹 XGBoost (meilleur modèle)

Excellentes performances

Très bon rappel

Très bon F1-score

Capacité à repérer des patterns rares

✔ Optimisation

Validation croisée (5 folds)

GridSearchCV pour hyperparamètres

Test de différents seuils de classification (threshold tuning)

📈 7. Évaluation des modèles
Métriques utilisées :

Recall

Precision

F1-score

ROC-AUC

Confusion Matrix

Precision-Recall Curve (indispensable pour données déséquilibrées)

Résultat final :

XGBoost est le modèle le plus performant, capable de détecter la majorité des fraudes avec un faible taux de faux positifs.

Les métriques finales détaillées sont disponibles dans le notebook.

📁 8. Arborescence du projet
├── data/
│   └── creditcard.csv
├── notebooks/
│   └── credit_fraud_detection.ipynb
├── src/
│   ├── preprocessing.py
│   ├── modeling.py
│   └── evaluation.py
├── README.md
├── requirements.txt
└── rapport.md

📝 9. Installation & Exécution
🔧 1. Installer les dépendances :
pip install -r requirements.txt

▶️ 2. Lancer le notebook :
jupyter notebook notebooks/credit_fraud_detection.ipynb

📦 10. Technologies utilisées

Python

NumPy

Pandas

Matplotlib / Seaborn

Scikit-Learn

XGBoost

Imbalanced-Learn

Jupyter Notebook

🎥 11. Présentation (Data Storytelling)

Une vidéo (5–10 minutes) explique :

Le contexte métier

La démarche

Les choix techniques

Les résultats

Les limites & améliorations

🧾 12. Résultats principaux

Le modèle final atteint un rappel très élevé, essentiel pour la détection de fraude

Le dataset déséquilibré a été efficacement géré

La pipeline complète est reproductible à partir du notebook

Le projet montre la faisabilité d'un système intelligent de détection de fraude bancaire

🧠 13. Limites & pistes d’amélioration
🔸 Limites :

Variables PCA anonymisées → interprétation difficile

Dataset limité à 2 jours seulement

Pas de données comportementales (géolocalisation, device, historique client…)

🔸 Améliorations possibles :

Ajouter des explications via SHAP

Tester des auto-encodeurs (autoencoder anomaly detection)

Intégrer des données temporelles (LSTM)

Ajuster dynamiquement le seuil selon le risque

Enrichir les données avec des features métier
