import torch
import numpy as np
import sys
import os

# Ajout du chemin src pour les imports
sys.path.append(os.path.join(os.getcwd(), 'src'))

try:
    from src.data.generators import generate_mixed_batch
    from src.physics.adr import pde_residual_adr
    from src.models.adr import PI_DeepONet_ADR
    print("✅ Imports réussis.")
except ImportError as e:
    print(f"❌ Erreur d'import. Es-tu bien à la racine ? {e}")
    sys.exit(1)

def test_manual_grad():
    print("\n--- 🛠️ TEST DE DIAGNOSTIC RAPIDE ---")
    
    # 1. Config Minimaliste (CORRIGÉE avec 'geometry')
    dummy_cfg = {
        'geometry': {
            'x_min': -5.0,
            'x_max': 5.0,
            'T_max': 1.0
        },
        'model': {
            'branch_depth': 2, 'branch_width': 20,
            'trunk_depth': 2, 'trunk_width': 20, 
            'latent_dim': 20,
            'nFourier': 10, 'sFourier': [0, 1]
        },
        'physics_ranges': {
            'v': [1.0, 1.0], 'D': [0.1, 0.1], 'mu': [0.1, 0.1],
            'A': [1.0, 1.0], 'x0': [0, 0], 'sigma': [0.5, 0.5], 'k': [1, 1]
        }
    }
    
    # Force CPU pour le test
    device = torch.device('cpu')
    print(f"📱 Device test : {device}")

    # 2. Init Modèle
    try:
        model = PI_DeepONet_ADR(dummy_cfg).to(device)
        print("✅ Modèle initialisé.")
    except Exception as e:
        print(f"❌ Erreur Init Modèle : {e}")
        return

    # 3. Génération du Batch
    print("⚡ Appel de generate_mixed_batch...")
    batch = generate_mixed_batch(
        n_samples=100, 
        bounds_phy=dummy_cfg['physics_ranges'], 
        x_min=-5.0, x_max=5.0, Tmax=1.0,
        device=device
    )
    
    # Récupération des tenseurs
    params, xt = batch[0], batch[1] 

    # 4. VERIFICATION DU DRAPEAU
    print(f"\n🧐 INSPECTION DU TENSEUR 'xt' :")
    print(f"   -> Type : {type(xt)}")
    print(f"   -> Shape : {xt.shape}")
    print(f"   -> Requires Grad ? : {xt.requires_grad}")

    if xt.requires_grad:
        print("✅ OK : Le générateur active bien le gradient !")
    else:
        print("❌ ECHEC : Le générateur renvoie requires_grad=False.")
        print("   👉 Le bug est confirmé dans generators.py (ou son cache).")

    # 5. TEST DU CRASH (Calcul PDE)
    print("\n🧮 Tentative de calcul du résidu PDE...")
    try:
        res = pde_residual_adr(model, params, xt)
        loss = torch.mean(res**2)
        
        # Tente une backward pass
        loss.backward()
        
        print("✅ SUCCÈS TOTAL : Le calcul PDE et la rétropropagation fonctionnent.")
        print("   Le problème est résolu avec ce code.")
        
    except RuntimeError as e:
        print(f"❌ CRASH DETECTÉ : {e}")
        if "requires_grad" in str(e):
            print("   👉 C'est exactement l'erreur de Jean Zay.")
    except Exception as e:
        print(f"❌ AUTRE ERREUR : {e}")

if __name__ == "__main__":
    test_manual_grad()