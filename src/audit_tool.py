import torch
import numpy as np
import sys
import os
from config import Config

# --- IMPORT DU SOLVEUR EXISTANT ---
sys.path.append(os.path.join(os.getcwd(), 'src'))
try:
    from physics.solver import crank_nicolson_adr
except ImportError:
    print("❌ Impossible d'importer crank_nicolson_adr depuis src.physics.solver")
    sys.exit(1)

# --- LE MÉDECIN (Fonction Diagnostic) ---
def diagnose_model(model, device, threshold=0.03, n_samples=100):
    """
    Audite le modèle sur les 3 familles en utilisant le solveur Crank-Nicolson existant.
    Renvoie une liste d'IDs (int) correspondant aux familles qui échouent (> threshold).
    """
    model.eval()
    
    # Paramètres de l'audit (Haute résolution pour être juge de paix)
    Nx_AUDIT = 256
    Nt_AUDIT = 200
    T_audit = 2.0
    
    # Mapping Nom -> Liste des IDs générateurs
    # 0=Tanh, 1&2=Sin-Gauss, 3&4=Gaussian
    families_map = {
        "Gaussian": [3, 4], 
        "Sin-Gauss": [1, 2], 
        "Tanh": [0]
    }
    
    failed_ids = []
    
    # Grille Spatiale fixe pour l'audit
    x_np = np.linspace(Config.x_min, Config.x_max, Nx_AUDIT)
    
    # Pour DeepONet : on veut prédire à t = T_audit
    t_np = np.full(Nx_AUDIT, T_audit)
    xt_flat = np.stack([x_np, t_np], axis=1)
    xt_tensor = torch.tensor(xt_flat, dtype=torch.float32).to(device)

    print(f"\n🩺 DIAGNOSTIC EN COURS (Seuil: {threshold:.1%})...")
    
    for fam_name, type_ids in families_map.items():
        errors = []
        # On teste n_samples cas aléatoires pour cette famille
        for _ in range(n_samples):
            # 1. Tirage paramètres aléatoires
            v = np.random.uniform(*Config.ranges['v'])
            D = np.random.uniform(0.01, Config.ranges['D'][1]) # Evite D trop petit
            mu = np.random.uniform(*Config.ranges['mu'])
            A = np.random.uniform(*Config.ranges['A'])
            x0 = 0.0 # Cohérence avec l'entraînement actuel
            sigma = np.random.uniform(0.2, Config.ranges['sigma'][1])
            k = np.random.uniform(*Config.ranges['k'])
            
            current_id = np.random.choice(type_ids)
            
            # 2. Génération Condition Initiale (u0)
            # On utilise numpy directement ici pour aller vite
            if current_id in [3, 4]: 
                u0 = A * np.exp(-(x_np - x0)**2 / (2 * sigma**2))
            elif current_id in [1, 2]: 
                u0 = A * np.exp(-(x_np - x0)**2 / (2 * sigma**2)) * np.sin(k * x_np)
            elif current_id == 0: 
                u0 = A * np.tanh((x_np - x0) / sigma)
            
            # 3. Vérité Terrain via TON Solveur Existant
            # Signature : crank_nicolson_adr(v, D, mu, xL, xR, Nx, Tmax, Nt, bc_kind, x0, u0)
            # Attention : ton solveur renvoie (x, U, t_grid) où U est shape (Nt, Nx)
            try:
                _, U_full, _ = crank_nicolson_adr(
                    v, D, mu, 
                    Config.x_min, Config.x_max, 
                    Nx_AUDIT, T_audit, Nt_AUDIT, 
                    "periodic", 
                    x0=x_np, u0=u0
                )
                # On récupère la dernière ligne (t = Tmax)
                u_true = U_full[-1, :]
                
            except Exception as e:
                print(f"⚠️ Erreur solveur : {e}. On ignore ce cas.")
                continue
            
            # 4. Prédiction DeepONet
            p_val = [v, D, mu, current_id, A, x0, sigma, k]
            p_tensor = torch.tensor([p_val] * Nx_AUDIT, dtype=torch.float32).to(device)
            
            with torch.no_grad():
                u_pred = model(p_tensor, xt_tensor).cpu().numpy().flatten()
            
            # 5. Calcul Erreur L2
            norm = np.linalg.norm(u_true) + 1e-6
            err = np.linalg.norm(u_true - u_pred) / norm
            errors.append(err)
            
        mean_err = np.mean(errors) if errors else 1.0
        status = "✅" if mean_err < threshold else "❌"
        print(f"  - {fam_name:<10} : {mean_err:.2%} {status}")
        
        if mean_err > threshold:
            failed_ids.extend(type_ids)
            
    return failed_ids