# Détection de Fraude sur Cartes Bancaires

Projet de détection de fraude à la carte bancaire basé sur le dataset public **Credit Card Fraud Detection** (Kaggle – MLG ULB / Worldline).[web:2]

---

## 1. Introduction

La fraude par carte bancaire constitue un enjeu critique pour les institutions financières, les fintechs et les systèmes de paiement en ligne, en raison des pertes économiques qu’elle génère et de son impact direct sur la confiance des clients.[web:9][web:26] Avec l’augmentation continue du volume de paiements électroniques, les approches traditionnelles basées sur des règles métier statiques deviennent insuffisantes pour détecter des comportements frauduleux de plus en plus sophistiqués.[web:9]

Ce projet s’appuie sur le dataset **Credit Card Fraud Detection** publié sur Kaggle par le *Machine Learning Group (MLG)* de l’ULB en collaboration avec Worldline.[web:2][web:20] Ce jeu de données contient des transactions bancaires réelles réalisées par des porteurs de cartes européens en septembre 2013. L’objectif est de concevoir un modèle de Machine Learning capable de prédire si une transaction est frauduleuse (classe 1) ou légitime (classe 0), en tenant compte d’un fort déséquilibre entre les classes.[web:2]

---

## 2. Problématique et objectifs

La problématique centrale est la suivante : comment détecter automatiquement, en temps quasi réel, un très faible nombre de transactions frauduleuses au sein d’un volume massif de transactions normales, tout en limitant les fausses alertes pour ne pas dégrader l’expérience client.[web:9][web:26]

Objectifs opérationnels :

- Maximiser le **rappel (recall)** sur la classe fraude, afin de réduire au minimum le nombre de fraudes non détectées.[web:26][web:28]  
- Limiter les **faux positifs**, pour éviter de bloquer un trop grand nombre de transactions légitimes et conserver une bonne précision.  
- Développer un **modèle robuste et stable**, suffisamment performant pour être déployé dans un environnement réel (banque, fintech, PSP, etc.).[web:9]

---

## 3. Description du dataset

Le dataset comprend **284 807** transactions, dont **492** sont étiquetées comme frauduleuses, soit environ **0,172 %** du total, ce qui en fait un jeu de données extrêmement déséquilibré.[web:2] Il comporte **31 variables** au total : 30 variables explicatives et une variable cible binaire `Class` (0 = transaction normale, 1 = fraude).[web:2]

Principales variables :

- `Time` : temps écoulé (en secondes) depuis la première transaction du jeu de données.  
- `Amount` : montant de la transaction.  
- `V1` à `V28` : variables issues d’une transformation de type PCA appliquée aux données brutes, afin d’anonymiser les informations sensibles tout en conservant la structure statistique.[web:2]  
- `Class` : variable cible indiquant la nature frauduleuse ou non de la transaction.

Contraintes :

- Les variables PCA sont anonymisées, ce qui complique l’interprétation économique ou métier des patterns identifiés.[web:2]  
- La sévérité du déséquilibre de classes impose le recours à des techniques spécifiques de rééchantillonnage ou de pondération pour éviter qu’un modèle naïf n’apprenne à prédire uniquement la classe majoritaire.[web:28]

> Le fichier de données n’est pas versionné dans le dépôt et doit être téléchargé directement depuis Kaggle.[web:20]

---

## 4. Prétraitement des données

### 4.1 Nettoyage

Les transactions en double ont été supprimées afin d’éviter de biaiser l’apprentissage, et une vérification des valeurs manquantes a confirmé qu’elles sont absentes ou négligeables dans ce dataset.[web:2] Cette étape garantit la cohérence statistique des données utilisées pour entraîner les modèles.

### 4.2 Normalisation

Les variables `Time` et `Amount` ont été standardisées (centrage-réduction) pour les mettre sur une échelle comparable et faciliter l’optimisation des modèles sensibles à la mise à l’échelle (régression logistique, méthodes à base de distances, etc.).[web:2] Les variables `V1` à `V28`, déjà issues d’une PCA, sont par construction centrées et réduites et ne nécessitent pas de normalisation supplémentaire.[web:2]

### 4.3 Gestion du déséquilibre

Plusieurs approches sont utilisées pour traiter le déséquilibre de classes :

- **Oversampling (SMOTE)** : génération artificielle de nouvelles observations de la classe minoritaire en interpolant entre des exemples réels proches, afin de rééquilibrer le dataset.[web:7][web:28]  
- **Undersampling** : réduction du nombre d’exemples de la classe majoritaire, simple mais potentiellement destructrice d’information utile.[web:28]  
- **Pondération des classes (`class_weight`)** : affectation d’un poids plus élevé aux exemples frauduleux dans la fonction de perte, particulièrement adaptée aux modèles linéaires et aux arbres de décision.[web:28]

Dans ce projet, une combinaison **SMOTE + `class_weight="balanced"`** est recommandée afin de renforcer l’apprentissage sur la classe minoritaire tout en limitant le surapprentissage sur des données synthétiques.[web:7][web:20]

---

## 5. Analyse exploratoire (EDA)

L’analyse exploratoire montre que la variable `Amount` est fortement concentrée sur de faibles montants, tandis que certaines fraudes apparaissent associées à des fourchettes de montants spécifiques, ce qui peut motiver la création de variables dérivées (binning, log-transformations, etc.).[web:2] La variable `Time` révèle des cycles journaliers et des créneaux horaires où la densité de fraudes est plus élevée, suggérant des effets temporels exploitables pour la détection.[web:2]

La visualisation de la distribution de la variable `Class` confirme le caractère extrêmement rare des fraudes et justifie l’usage de métriques adaptées telles que la courbe **Precision–Recall**.[web:26][web:28] L’étude des distributions de certaines composantes (par exemple `V12`, `V14`, `V17`) montre des différences marquées entre transactions frauduleuses et normales, ce qui indique un fort pouvoir discriminant malgré l’absence d’interprétation métier directe.[web:1][web:3]

---

## 6. Modélisation Machine Learning

Plusieurs familles de modèles supervisés de classification binaire sont évaluées :

- **Régression logistique** : utilisée comme baseline, offrant une bonne interprétabilité globale mais des performances limitées sur des relations fortement non linéaires.  
- **Random Forest** : ensemble d’arbres de décision robustes, capable de modéliser des interactions complexes et d’intégrer des poids de classes, souvent plus performant qu’un modèle linéaire sur ce type de données.[web:5]  
- **XGBoost / LightGBM** : modèles de gradient boosting sur arbres, très efficaces pour exploiter des patterns rares dans des données déséquilibrées et fréquemment cités comme des références pour la détection de fraude.[web:8][web:20]

Dans ce contexte, **XGBoost** ou **LightGBM** offrent généralement les meilleurs compromis entre rappel, précision et F1-score, grâce à leur capacité à optimiser des fonctions de perte adaptées aux données déséquilibrées et à la flexibilité de leurs hyperparamètres.[web:8][web:28]

---

## 7. Évaluation des modèles

Dans un contexte de détection de fraude, l’**accuracy** globale est trompeuse : un modèle qui prédirait systématiquement « non fraude » obtiendrait une accuracy élevée tout en étant inutilisable.[web:26][web:28] Les métriques retenues sont donc :

- **Recall (sensibilité)** sur la classe fraude : priorité métier, car elle mesure la proportion de fraudes correctement détectées.[web:26][web:28]  
- **Precision** : proportion de transactions réellement frauduleuses parmi celles prédites comme telles.  
- **F1-score** : moyenne harmonique entre précision et rappel, utile pour évaluer l’équilibre entre détection et fausses alertes.[web:28]  
- **Matrice de confusion** : visualisation des vrais positifs, faux négatifs, vrais négatifs et faux positifs.  
- **ROC-AUC** et surtout **Precision–Recall AUC**, plus pertinentes lorsque la classe positive est rare.[web:26][web:28]

Les études empiriques montrent qu’XGBoost obtient souvent un F1-score et une AUC supérieurs, tout en maintenant un rappel élevé, tandis que la Random Forest se montre solide mais légèrement en retrait, et la régression logistique sert de baseline minimale acceptable.[web:8][web:20] Une bonne matrice de confusion se caractérise par un nombre très faible de fraudes non détectées (FN) et un niveau de faux positifs compatible avec les contraintes opérationnelles.

---

## 8. Interprétation et discussion

Atouts du modèle :

- Excellente capacité de détection des transactions frauduleuses grâce à la combinaison de techniques de rééchantillonnage et d’algorithmes d’ensemble performants.[web:8][web:28]  
- Robustesse face au déséquilibre des classes et architecture modulaire (pipeline) facilement intégrable dans un système de scoring temps réel.[web:6][web:20]

Limites :

- Le caractère anonymisé des variables PCA réduit l’explicabilité métier des décisions de classification.[web:2]  
- Le dataset couvre une période courte (deux jours de transactions), ce qui ne reflète pas nécessairement la dynamique à long terme des fraudes.[web:2]  
- L’absence de données contextuelles (adresse IP, navigateur, device, géolocalisation, historique client) limite la capacité du modèle à capter des signaux comportementaux fins.[web:6]

Pistes d’amélioration :

- Utiliser des outils d’explicabilité comme **SHAP** pour analyser la contribution de chaque variable aux décisions, en particulier pour les modèles de type XGBoost / LightGBM.[web:8]  
- Explorer des modèles de **deep learning** (autoencodeurs pour la détection d’anomalies, architectures LSTM pour la dimension temporelle), déjà étudiés sur ce dataset dans la littérature.[web:4][web:28]  
- Intégrer des données temporelles plus longues et des variables comportementales, et ajuster dynamiquement le seuil de décision en fonction du niveau de risque ou du profil client.[web:6][web:26]

---

## 9. Conclusion

La détection de fraude sur cartes bancaires illustre un problème typique de **classification déséquilibrée**, où la rareté des fraudes impose de privilégier la sensibilité et la précision sur la classe minoritaire plutôt que l’accuracy globale.[web:26][web:28] En combinant un prétraitement adapté, une analyse exploratoire ciblée et des modèles avancés tels que XGBoost ou LightGBM, il est possible de construire un système de Machine Learning capable d’identifier une proportion importante de transactions frauduleuses tout en maîtrisant les faux positifs.[web:8][web:20]

Ce projet montre le potentiel des approches data-driven pour renforcer la sécurité des paiements, réduire les pertes financières et améliorer la confiance des utilisateurs, à condition d’actualiser régulièrement les données, de surveiller les performances et de fournir des explications claires aux équipes métier et aux régulateurs.[web:6][web:9]
