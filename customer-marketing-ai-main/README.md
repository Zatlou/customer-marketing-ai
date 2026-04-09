# customer-marketing-ai

Mini-projet de segmentation client et prédiction marketing autour du dataset Kaggle `Customer Personality Analysis`.

L'idée centrale est de comparer :

- une baseline supervisée sans information de segment ;
- un modèle hybride qui ajoute `ClusterID` appris par K-Means ;
- un pipeline reproductible avec preprocessing, clustering, entraînement et évaluation.

## Structure

- `src/ingest.py` : chargement et validation légère du CSV brut
- `src/preprocess.py` : nettoyage, feature engineering, contrôle des âges aberrants et filtrage simple des outliers sur `Income`
- `src/cluster.py` : preprocessing du clustering, recherche du meilleur `k` par silhouette score, export des clusters et d'une projection PCA
- `src/train.py` : entraînement du classifieur supervisé `MLP` en PyTorch pour baseline et hybride, sans fuite de données
- `src/evaluate.py` : calcul des métriques finales et synthèse JSON
- `src/prefect_flow.py` : version orchestrée du pipeline avec fallback séquentiel si Prefect n'est pas installé
- `notebooks/marketing_pipeline_v2.ipynb` : notebook pédagogique propre, exécutable de bout en bout, utilisé pour la soutenance et les captures
- `params.yaml` : chemins et hyperparamètres du pipeline

## Sorties principales

- `data/processed/marketing_campaign_clean.csv`
- `data/processed/marketing_campaign_clustered.csv`
- `reports/baseline_results.csv`
- `reports/hybrid_results.csv`
- `reports/metrics_summary.json`
- `reports/silhouette_scores.csv`
- `reports/cluster_summary.csv`
- `reports/stability_summary.csv`
- `reports/cluster_pca.png`

## Exécution

Depuis la racine du projet :

```bash
python3 main.py
```

Pour exécuter une étape isolée :

```bash
python3 src/preprocess.py
python3 src/cluster.py
python3 src/train.py
python3 src/evaluate.py
```

Pour lancer le flow :

```bash
python3 src/prefect_flow.py
```

Si `prefect` est installé, le flow utilise Prefect. Sinon, il exécute simplement les étapes en séquentiel.

## Remarque sur les livrables

Le notebook `marketing_pipeline_v2.ipynb` est la version la plus pédagogique et la plus présentable. Les scripts `src/` constituent la version pipeline du projet, plus proche de l'architecture annoncée dans l'énoncé.
