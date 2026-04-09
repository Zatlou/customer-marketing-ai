# Rapport court

## Projet

**Titre :** Segmentation client et prédiction marketing  
**Cours :** Introduction à l'IA  
**Dataset :** *Customer Personality Analysis* (Kaggle)

## 1. Contexte et objectif

Ce mini-projet vise à répondre à une problématique marketing simple : mieux comprendre les profils clients et prédire leur probabilité de réponse à une campagne.

L'approche retenue est hybride :

- une étape **non supervisée** de segmentation client par clustering ;
- une étape **supervisée** de classification de la variable cible `Response` ;
- une comparaison entre un **modèle baseline** et un **modèle hybride** enrichi par la variable `ClusterID`.

La question centrale du projet est la suivante :

**L'ajout d'une segmentation non supervisée améliore-t-il la qualité d'une prédiction supervisée de la réponse marketing ?**

## 2. Données et préparation

Le projet s'appuie sur le dataset Kaggle **Customer Personality Analysis**, qui contient des informations socio-démographiques, des variables de comportement d'achat et un historique marketing.

Les principales étapes de préparation ont été les suivantes :

- nettoyage des noms de colonnes ;
- suppression des lignes où `Income` ou `Response` sont manquants ;
- création de variables construites :
  - `Age = 2026 - Year_Birth`
  - `Children = Kidhome + Teenhome`
  - `IsParent`
  - `TotalSpending`
  - `TotalPurchases`
- suppression de quelques âges aberrants ;
- filtrage simple des valeurs extrêmes de `Income` entre le 1er et le 99e percentile.

Après nettoyage, le dataset de travail contient **2167 observations**.

## 3. Partie non supervisée : segmentation client

La segmentation a été réalisée avec **K-Means**.

Variables utilisées pour le clustering :

- numériques : `Income`, `Age`, `Children`, `TotalSpending`, `TotalPurchases`
- catégorielles : `Education`, `Marital_Status`

Le prétraitement du clustering repose sur :

- `StandardScaler` pour les variables numériques ;
- `OneHotEncoder(handle_unknown="ignore")` pour les variables catégorielles.

Le nombre de clusters a été choisi à l'aide du **silhouette score** sur les valeurs de `k` allant de 2 à 7.  
Le meilleur résultat obtenu est :

- **k = 2**
- **silhouette score = 0.3145**

L'analyse métier des segments met en évidence deux profils principaux :

- un cluster à **revenu élevé, forte dépense, meilleure réactivité marketing** ;
- un cluster à **revenu plus modeste, dépense plus faible, réponse marketing plus limitée**.

## 4. Partie supervisée : baseline vs hybride

La partie supervisée a été implémentée avec un **MLP PyTorch** pour rester cohérente avec l'architecture proposée dans l'énoncé.

Deux expériences ont été réalisées :

- **Baseline** : sans `ClusterID`
- **Hybrid** : avec `ClusterID`

Le découpage utilisé est :

- `train_test_split`
- `test_size = 0.2`
- `stratify = y`
- `random_state = 42`

Les métriques retenues sont :

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- matrice de confusion

## 5. Résultats principaux

### Résultats sur un run principal

**Baseline**

- Accuracy : **0.7304**
- Precision : **0.3037**
- Recall : **0.6406**
- F1-score : **0.4121**
- ROC-AUC : **0.7273**

**Hybrid**

- Accuracy : **0.7235**
- Precision : **0.2879**
- Recall : **0.5938**
- F1-score : **0.3878**
- ROC-AUC : **0.7425**

### Lecture des résultats

Sur le run principal, l'ajout de `ClusterID` **améliore la ROC-AUC**, ce qui suggère que la segmentation capture une information utile pour la discrimination globale.  
En revanche, les autres métriques ne progressent pas toutes sur ce run unique.

### Vérification de stabilité

Une comparaison a également été réalisée sur plusieurs seeds (`7, 21, 42, 84, 123`) afin d'obtenir une lecture plus robuste.

Moyennes observées :

**Baseline**

- Accuracy moyenne : **0.7341**
- F1-score moyen : **0.3464**
- ROC-AUC moyen : **0.6793**

**Hybrid**

- Accuracy moyenne : **0.7406**
- F1-score moyen : **0.3473**
- ROC-AUC moyen : **0.6825**

Cette analyse montre que l'apport du clustering est **léger mais plausible**, et qu'il doit être interprété avec prudence.

## 6. Conclusion

Le projet répond à l'objectif annoncé :

- segmenter les clients de manière lisible ;
- prédire la variable `Response` ;
- comparer une baseline et un modèle hybride avec `ClusterID`.

Le principal intérêt du projet n'est pas uniquement métrique. Il est aussi :

- **méthodologique**, car il combine apprentissage supervisé et non supervisé dans un même pipeline ;
- **analytique**, car la segmentation produit une lecture marketing claire des profils clients ;
- **reproductible**, car les étapes ont été restructurées dans des scripts et un notebook final.

## 7. Limites et pistes d'amélioration

Les principales limites du projet sont les suivantes :

- performance globale encore modérée ;
- sensibilité des résultats au split et au seed ;
- K-Means reste un algorithme simple, avec des hypothèses fortes sur la structure des groupes.

Pistes d'amélioration :

- tester d'autres algorithmes de clustering (`DBSCAN`, `GMM`, clustering hiérarchique) ;
- comparer le MLP à d'autres classifieurs tabulaires ;
- réaliser un tuning d'hyperparamètres plus poussé ;
- ajouter une orchestration plus complète avec suivi d'expériences.

## 8. Rendus fournis

Les rendus associés au projet sont les suivants :

- **Notebook final** : `notebooks/marketing_pipeline_v2.ipynb`
- **Code source du pipeline** : `src/`
- **Configuration du pipeline** : `params.yaml`
- **Jeux de données traités** :
  - `data/processed/marketing_campaign_clean.csv`
  - `data/processed/marketing_campaign_clustered.csv`
- **Résultats expérimentaux** :
  - `reports/baseline_results.csv`
  - `reports/hybrid_results.csv`
  - `reports/metrics_summary.json`
  - `reports/silhouette_scores.csv`
  - `reports/cluster_summary.csv`
  - `reports/stability_summary.csv`
  - `reports/cluster_pca.png`

Ce rapport court synthétise donc la méthodologie, les résultats et les principaux livrables attendus pour le mini-projet.
