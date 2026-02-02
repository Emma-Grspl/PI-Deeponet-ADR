import torch
import numpy as np
import sys
import os
from config import Config

# --- IMPORT DU SOLVEUR ---
sys.path.append(os.path.join(os.getcwd(), 'src'))
try:
    # On importe la fonction "intelligente" qui gère les BCs (Tanh/Gauss)
    from physics.solver import get_ground_truth_CN
except ImportError:
    print("❌ Impossible d'importer get_ground_truth_CN depuis src.physics.solver")
    sys.exit(1)

# --- LE MÉDECIN (Fonction Diagnostic Corrigée) ---
def diagnose_model(model, device, threshold=0.03, t_max=None): # <--- Ajout de t_max
    """
    Audite le modèle sur la fenêtre [0, t_max].
    Renvoie une liste d'IDs (int) correspondant aux familles qui échouent.
    """
    model.eval()
    
    # Si t_max n'est pas fourni, on prend le max global (audit final)
    if t_max is None: t_max = Config.T_max

    # Paramètres de l'audit
    Nx_AUDIT = 256
    Nt_AUDIT = 200 # Suffisant pour la précision
    
    # Mapping Nom -> Liste des IDs générateurs
    families_map = {
        "Gaussian": [3, 4], 
        "Sin-Gauss": [1, 2], 
        "Tanh": [0]
    }
    
    failed_ids = []
    
    # Grille Spatiale fixe pour l'audit
    x_np = np.linspace(Config.x_min, Config.x_max, Nx_AUDIT)
    
    # On veut prédire à t = t_max (le bout du palier actuel)
    t_np = np.full(Nx_AUDIT, t_max)
    xt_flat = np.stack([x_np, t_np], axis=1)
    xt_tensor = torch.tensor(xt_flat, dtype=torch.float32).to(device)

    print(f"\n🩺 DIAGNOSTIC EN COURS (t={t_max}, Seuil: {threshold:.1%})...")
    
    for fam_name, type_ids in families_map.items():
        errors = []
        # On teste 20 cas aléatoires par famille (suffisant pour une moyenne)
        for _ in range(20): 
            # 1. Tirage paramètres aléatoires (basé sur Config)
            v = np.random.uniform(*Config.ranges['v'])
            D = np.random.uniform(0.01, Config.ranges['D'][1])
            mu = np.random.uniform(*Config.ranges['mu'])
            A = np.random.uniform(*Config.ranges['A'])
            x0 = 0.0 
            sigma = np.random.uniform(0.2, Config.ranges['sigma'][1])
            k = np.random.uniform(*Config.ranges['k'])
            
            current_id = np.random.choice(type_ids)
            
            # 2. Dictionnaire de paramètres pour le solveur intelligent
            p_dict = {
                'v': v, 'D': D, 'mu': mu, 'type': current_id,
                'A': A, 'x0': x0, 'sigma': sigma, 'k': k
            }
            
            # 3. Vérité Terrain (Avec gestion automatique des BCs !)
            try:
                # get_ground_truth_CN va choisir "tanh_pm1" ou "zero_zero" automatiquement
                _, _, U_full = get_ground_truth_CN(
                    p_dict, Config.x_min, Config.x_max, t_max, Nx_AUDIT, Nt_AUDIT
                )
                # On récupère la dernière ligne (t = t_max)
                u_true = U_full[-1, :]
                
            except Exception as e:
                print(f"⚠️ Erreur solveur : {e}. On ignore ce cas.")
                continue
            
            # 4. Prédiction DeepONet
            p_val = [v, D, mu, current_id, A, x0, sigma, k]
            # Création du tenseur de paramètres répété Nx fois
            p_tensor = torch.tensor([p_val], dtype=torch.float32).to(device).repeat(Nx_AUDIT, 1)
            
            with torch.no_grad():
                u_pred = model(p_tensor, xt_tensor).cpu().numpy().flatten()
            
            # 5. Calcul Erreur L2 Relative
            norm = np.linalg.norm(u_true) + 1e-6
            err = np.linalg.norm(u_true - u_pred) / norm
            errors.append(err)
            
        mean_err = np.mean(errors) if errors else 1.0
        status = "✅" if mean_err < threshold else "❌"
        # Affichage propre aligné
        print(f"  - {fam_name:<12} : {mean_err:.2%} {status}")
        
        if mean_err > threshold:
            failed_ids.extend(type_ids)
            
    return failed_ids