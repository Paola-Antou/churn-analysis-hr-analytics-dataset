
## Description du projet

Ce projet vise à **prédire le churn (attrition)** des employés à partir du jeu de données public **IBM HR Analytics**.
L’objectif est de comprendre **les facteurs clés expliquant le départ des employés**, de **développer un modèle prédictif fiable**, et de **proposer une stratégie de rétention** fondée sur les données.

Ce travail illustre une démarche complète de **data science appliquée à la problématique de fidélisation**, depuis l’exploration des données jusqu’à la traduction des résultats en recommandations business.


##  Structure du dépôt

```
churn-analysis-hr-analytics-dataset/
│
├── data/
│   └── WA_Fn-UseC_-HR-Employee-Attrition.csv
│
├── src/
│   ├── main.ipynb          # Notebook complet (EDA, Modélisation, Interprétation)
│   ├── utils.py            # Fonctions utilitaires (prétraitement, visualisation)
│   └── config.py           # Paramètres du projet
│
├── dashboard/
│   ├── app.py              # Application Streamlit 
│   └── requirements.txt
│
├── reports/
│   ├── rapport_business.pdf
│   └── presentation.pptx   # Présentation PowerPoint 
│
└── README.md
```


## Objectifs du projet

1. **Analyser les données** pour comprendre les caractéristiques associées au départ des employés.
2. **Construire un modèle prédictif** pour estimer le risque d’attrition.
3. **Identifier les facteurs de risque** via l’interprétation des modèles (SHAP).
4. **Évaluer l’impact business** à travers la valeur vie client (CLV) et une matrice coût-bénéfice.
5. **Proposer une stratégie de rétention** ciblée et rentable.



##  Données

* **Source :** [IBM HR Analytics Employee Attrition Dataset](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset)
* **Nombre d’observations :** 1 470
* **Nombre de variables :** 35
* **Variables clés :** Age, JobSatisfaction, MonthlyIncome, DistanceFromHome, YearsAtCompany, Attrition

Le jeu de données présente un déséquilibre entre les classes (employés restants vs quittant l’entreprise), ce qui a été traité via SMOTE et ajustement de poids de classe.



## Démarche méthodologique

### 1. **Exploration des données (EDA)**

* Étude des distributions et corrélations entre variables.
* Comparaison des employés “Attrition = Yes / No”.
* Identification des variables les plus influentes (satisfaction, salaire, ancienneté…).

### 2. **Préparation et Feature Engineering**

* Encodage des variables catégorielles.
* Normalisation et gestion des valeurs manquantes.
* Création de nouvelles variables (ratios, interactions).

### 3. **Modélisation**

Plusieurs modèles ont été testés et comparés :

* Régression Logistique
* Random Forest
* XGBoost
* LightGBM

Le modèle final retenue est XGBoost, offrant la meilleure performance globale.

| Modèle              | Précision | Rappel   | F1-score | ROC AUC  |
| ------------------- | --------- | -------- | -------- | -------- |
| Logistic Regression | 0.78      | 0.65     | 0.71     | 0.80     |
| Random Forest       | 0.84      | 0.70     | 0.76     | 0.83     |
| **XGBoost (final)** | **0.85**  | **0.74** | **0.78** | **0.84** |

### 4. **Interprétabilité**

* Utilisation des **valeurs SHAP** pour expliquer les décisions du modèle.
* Variables les plus importantes :

  * Satisfaction au travail
  * Salaire mensuel
  * Distance domicile-travail
  * Ancienneté

### 5. **Impact business**

* Calcul du **Customer Lifetime Value (CLV)** ≈ 2 700 €.
* Création d’une **matrice coût-bénéfice** pour évaluer la rentabilité des actions.
* Seuil optimal d’action : risque de churn > 0.7 + CLV élevé.



## Segmentation et Stratégie de Rétention

Segmentation opérationnelle selon le risque et la valeur du client :

| Priorité | Critères                 | Type d’action                         |
| -------- | ------------------------ | ------------------------------------- |
| 1        | Haut CLV & Risque élevé  | Action immédiate, offre personnalisée |
| 2        | Moyen CLV & Risque élevé | Campagnes automatisées                |
| 3        | Bas CLV & Risque élevé   | Suivi limité                          |
| 4        | Faible risque            | Monitoring préventif                  |



## Recommandations

1. Intégrer le **scoring de churn** au sein du CRM.
2. Mettre en place des **tests A/B** pour évaluer les stratégies de rétention.
3. Prioriser les employés à **haut CLV et fort risque**.
4. Suivre la performance à **30, 60 et 90 jours** après mise en œuvre.



## Résultats clés

* **Modèle final :** XGBoost
* **ROC AUC :** 0.84
* **Rappel sur churners :** 0.74
* **Taux de précision global :** 0.75
* **CLV estimé :** 2 700 €
* **Stratégie de rétention** basée sur le risque et la valeur.


## Dashboard

Un dashboard interactif a été développé pour :

* Visualiser le taux de churn global et par segment.
* Tester la prédiction du risque sur un nouvel individu.
* Explorer les variables les plus importantes (SHAP).

**Exécution :**

```bash
cd dashboard
pip install -r requirements.txt
streamlit run app.py
```




## Technologies utilisées

* **Langage :** Python
* **Bibliothèques principales :** Pandas, NumPy, Scikit-learn, XGBoost, LightGBM, Matplotlib, Seaborn, Plotly, SHAP
* **Outils de visualisation :** Streamlit, Dash
* **Gestion de déséquilibre :** SMOTE, class weights
* **Éditeur :** Jupyter Notebook



##  Auteur

**[DAGBA Paola]** — Data Scientist

* Email : [Ton adresse email]
* Dataset : [Lien Kaggle](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset)



## Licence

Ce projet est publié à des fins éducatives et de démonstration dans le cadre d’un portfolio de data science.
Les données appartiennent à IBM et sont distribuées sous licence libre pour usage académique.
