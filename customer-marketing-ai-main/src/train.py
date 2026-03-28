import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# --- ÉTAPE 1 : Configuration du Dataset PyTorch ---
# On transforme les données Pandas en tenseurs PyTorch pour le modèle
class MarketingDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y.values, dtype=torch.float32).view(-1, 1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# --- ÉTAPE 2 : Architecture du modèle MLP ---
# Un réseau de neurones simple avec Dropout pour éviter le surapprentissage
class ResponseMLP(nn.Module):
    def __init__(self, input_dim):
        super(ResponseMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.2), 
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1) # Sortie brute (Logits)
        )

    def forward(self, x):
        return self.net(x)

# --- ÉTAPE 3 : Préparation et Nettoyage des données ---
def prepare_data(df, use_clusters=False):
    # Variables de base (Mira)
    features = ["Income", "Age", "Children", "TotalSpending", "TotalPurchases"]
    
    # Si on teste le modèle hybride, on ajoute la segmentation de Zakia
    if use_clusters:
        features.append("ClusterID")
    
    # Remplissage des valeurs manquantes et normalisation
    X = df[features].fillna(df[features].median())
    y = df["Response"]
    
    X_scaled = StandardScaler().fit_transform(X)
    
    # Découpage 80% train / 20% test (stratifié car la cible est déséquilibrée)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, stratify=y, random_state=42
    )
    
    train_ds = MarketingDataset(X_train, y_train)
    test_ds = MarketingDataset(X_test, y_test)
    
    return DataLoader(train_ds, batch_size=64, shuffle=True), \
           DataLoader(test_ds, batch_size=256), \
           X_train.shape[1]

# --- ÉTAPE 4 : Boucle d'entraînement et Sauvegarde des résultats ---
def run_experiment(use_clusters=False, filename="baseline_results.csv"):
    # Chargement du dataset nettoyé par Zakia
    try:
        df = pd.read_csv("data/processed/marketing_campaign_clustered.csv")
    except FileNotFoundError:
        print("❌ Erreur : Le fichier de données est introuvable dans data/processed/")
        return

    train_loader, test_loader, input_dim = prepare_data(df, use_clusters)
    
    # Initialisation
    model = ResponseMLP(input_dim)
    criterion = nn.BCEWithLogitsLoss() # Adapté pour la classification binaire
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Entraînement (15 époques)
    print(f"🚀 Entraînement du modèle : {filename}...")
    for epoch in range(15):
        model.train()
        for xb, yb in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

    # --- ÉTAPE 5 : Génération des prédictions ---
    model.eval()
    all_probs = []
    all_targets = []
    
    with torch.no_grad():
        for xb, yb in test_loader:
            # On applique Sigmoid pour transformer les logits en probabilités [0, 1]
            probs = torch.sigmoid(model(xb))
            all_probs.extend(probs.numpy().flatten())
            all_targets.extend(yb.numpy().flatten())

    # Création du DataFrame de résultats pour Zakia
    results = pd.DataFrame({
        'y_true': all_targets,
        'y_prob': all_probs
    })
    results['y_pred'] = (results['y_prob'] > 0.5).astype(int)
    
    # Création du dossier reports s'il n'existe pas
    os.makedirs("reports", exist_ok=True)
    
    # Sauvegarde finale
    results.to_csv(f"reports/{filename}", index=False)
    print(f"✅ Résultats sauvegardés dans : reports/{filename}")

# --- LANCEMENT ---
if __name__ == "__main__":
    # 1. On lance la Baseline (sans clusters)
    run_experiment(use_clusters=False, filename="baseline_results.csv")
    
    # 2. On lance l'Hybride (avec clusters)
    run_experiment(use_clusters=True, filename="hybrid_results.csv")
    
    print("\n🎉 Travail de Mira terminé ! Tu peux maintenant push sur Git.")