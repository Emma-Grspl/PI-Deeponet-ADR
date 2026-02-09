import torch
import numpy as np
import yaml
import os
import sys

# Setup chemins
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from src.physics.solver import get_ground_truth_CN
from src.models.adr import PI_DeepONet_ADR 

def load_config(path="src/config_ADR.yaml"):
    with open(path, 'r') as f: return yaml.safe_load(f)

def autopsy():
    # 1. Config & Device
    cfg = load_config()
    device = torch.device("cpu")
    print(f"🖥️ Device: {device}")

    # 2. Chargement Modèle
    model_path = "run_20260208-092447/model_final.pth" # Vérifie ce chemin !
    if not os.path.exists(model_path):
        model_path = "run_20260208-092447/model_latest.pth"
    
    print(f"📥 Modèle: {model_path}")
    model = PI_DeepONet_ADR(cfg).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.eval()

    # 3. Génération d'un cas test (Gaussian)
    print("\n--- 🧪 GÉNÉRATION CAS TEST (Gaussian) ---")
    p = {k: v[0] for k, v in cfg['physics_ranges'].items()} # Valeurs min
    p['type'] = 0 # Gaussian
    
    # Appel Solveur
    X, T, U_true = get_ground_truth_CN(p, cfg, t_step_max=3.0)

    # 4. INSPECTION BRUTE DU SOLVEUR
    print(f"   [SOLVEUR] Type X: {type(X)} | Shape: {X.shape}")
    print(f"   [SOLVEUR] Type T: {type(T)} | Shape: {T.shape}")
    print(f"   [SOLVEUR] Type U: {type(U_true)} | Shape: {U_true.shape}")
    
    # Vérif contenu T
    t_flat = T.flatten() if isinstance(T, np.ndarray) else np.array(T)
    print(f"   [SOLVEUR] T (5 premiers) : {t_flat[:5]}")
    print(f"   [SOLVEUR] T (5 derniers) : {t_flat[-5:]}")

    # 5. PRÉDICTION DEEPONET
    # Alignement forcé pour DeepONet
    x_vec = X.flatten()
    t_vec = T.flatten()
    TX, XT = np.meshgrid(t_vec, x_vec) # (Nx, Nt)
    
    x_flat = XT.flatten()
    t_flat = TX.flatten()
    
    p_vec = np.array([p['v'], p['D'], p['mu'], p['type'], p['A'], 0.0, p['sigma'], p['k']])
    p_tensor = torch.tensor(p_vec, dtype=torch.float32).repeat(len(x_flat), 1).to(device)
    xt_tensor = torch.tensor(np.stack([x_flat, t_flat], axis=1), dtype=torch.float32).to(device)
    
    print("\n--- 🧠 PRÉDICTION NEURONALE ---")
    with torch.no_grad():
        u_pred_flat = model(p_tensor, xt_tensor).cpu().numpy().flatten()
    
    U_pred = u_pred_flat.reshape(XT.shape).T # (Nt, Nx) par défaut ici
    if U_true.shape != U_pred.shape:
        U_true = U_true.T # On essaie de matcher

    print(f"   [MODEL] U_pred Shape finale : {U_pred.shape}")
    print(f"   [MODEL] U_true Shape finale : {U_true.shape}")

    # 6. COMPARAISON DES VALEURS
    print("\n--- 🔍 COMPARAISON VALEURS ---")
    print(f"   Moyenne U_true : {np.mean(U_true):.5f} | Max: {np.max(U_true):.5f} | Min: {np.min(U_true):.5f}")
    print(f"   Moyenne U_pred : {np.mean(U_pred):.5f} | Max: {np.max(U_pred):.5f} | Min: {np.min(U_pred):.5f}")
    
    err = np.linalg.norm(U_true - U_pred) / np.linalg.norm(U_true)
    print(f"   ❌ ERREUR RELATIVE L2 : {err:.5f} ({err*100:.2f}%)")

    # 7. CHECK BUFFER NORMALISATION
    print("\n--- ⚙️ CHECK NORMALISATION ---")
    print(f"   YAML v range : {cfg['physics_ranges']['v']}")
    print(f"   Model v_min buffer : {model.v_min.item()}")
    print(f"   Model v_max buffer : {model.v_max.item()}")

if __name__ == "__main__":
    autopsy()