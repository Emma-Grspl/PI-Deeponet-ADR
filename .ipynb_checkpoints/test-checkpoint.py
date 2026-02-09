import torch
import yaml
import sys
import os
import numpy as np

# Imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from src.models.adr import PI_DeepONet_ADR

# Config
with open("src/config_ADR.yaml", 'r') as f: cfg = yaml.safe_load(f)

# Modèle
device = torch.device("cpu")
model = PI_DeepONet_ADR(cfg).to(device)

# Chargement Poids
path = "run_20260208-092447/model_final.pth" # Vérifie le chemin !
ckpt = torch.load(path, map_location=device)
model.load_state_dict(ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt)
model.eval()

# Test idiot
print("\n--- TEST DE SURVIE ---")
# On crée un vecteur d'entrée bidon mais valide
# 8 paramètres physiques
p = torch.tensor([[1.0, 0.1, 0.5, 0.0, 1.0, 0.0, 0.5, 2.0]], dtype=torch.float32)
# 2 coordonnées (x=0, t=0.5)
xt = torch.tensor([[0.0, 0.5]], dtype=torch.float32)

with torch.no_grad():
    pred = model(p, xt)

print(f"Prédiction du modèle pour (x=0, t=0.5) : {pred.item():.4f}")

# Vérif des buffers de normalisation interne
print(f"Normalisation v_min dans le modèle : {model.v_min.item()}")
print(f"Normalisation v_max dans le modèle : {model.v_max.item()}")
print("Si ces valeurs ne correspondent pas à ton YAML, c'est le problème.")