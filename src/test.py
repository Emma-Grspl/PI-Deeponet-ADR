import torch
import sys
import os

# Ajout du chemin src
sys.path.append(os.path.join(os.getcwd(), 'src'))

try:
    # On importe les fonctions suspectes et le modèle
    from src.training.smart_trainer import monitor_gradients, compute_ntk_weights
    from src.data.generators import generate_mixed_batch
    from src.models.adr import PI_DeepONet_ADR
    print("✅ Imports réussis.")
except ImportError as e:
    print(f"❌ Erreur d'import : {e}")
    sys.exit(1)

def torture_test():
    print("\n--- 🛠️ TEST DE TORTURE (Monitor & NTK) ---")
    
    # 1. Config minimale
    dummy_cfg = {
        'geometry': {'x_min': -5.0, 'x_max': 5.0, 'T_max': 1.0},
        'model': {
            'branch_depth': 2, 'branch_width': 20,
            'trunk_depth': 2, 'trunk_width': 20, 
            'latent_dim': 20, 'nFourier': 10, 'sFourier': [0, 1]
        },
        'physics_ranges': {
            'v': [1.0, 1.0], 'D': [0.1, 0.1], 'mu': [0.1, 0.1],
            'A': [1.0, 1.0], 'x0': [0, 0], 'sigma': [0.5, 0.5], 'k': [1, 1]
        }
    }
    
    # Device CPU
    device = torch.device('cpu')
    model = PI_DeepONet_ADR(dummy_cfg).to(device)
    print("✅ Modèle initialisé.")

    # 2. Génération d'un Batch
    batch = generate_mixed_batch(50, dummy_cfg['physics_ranges'], -5, 5, 1.0, device=device)
    
    # --- LA TORTURE ---
    # On extrait le tenseur xt
    params, xt, xt_ic, u_true_ic, bc_l, bc_r, u_bc_l, u_bc_r = batch
    
    # 😈 SABOTAGE : On force le gradient à FALSE (comme sur le cluster)
    xt.requires_grad_(False)
    print(f"😈 SABOTAGE : xt.requires_grad forcé à {xt.requires_grad}")
    
    # On recrée un tuple "empoisonné"
    bad_batch = (params, xt, xt_ic, u_true_ic, bc_l, bc_r, u_bc_l, u_bc_r)

    # 3. Test de monitor_gradients
    print("\n🧪 Test 1 : monitor_gradients...")
    try:
        # Si la fonction est patchée, elle va réactiver le gradient toute seule
        ratio, cos = monitor_gradients(model, bad_batch)
        print(f"✅ SUCCÈS monitor_gradients ! (Ratio={ratio:.2f})")
    except RuntimeError as e:
        print(f"❌ ÉCHEC monitor_gradients : {e}")
        print("👉 Il faut modifier src/training/smart_trainer.py !")
        return

    # 4. Test de compute_ntk_weights
    print("\n🧪 Test 2 : compute_ntk_weights...")
    # Re-sabotage
    xt.requires_grad_(False)
    bad_batch = (params, xt, xt_ic, u_true_ic, bc_l, bc_r, u_bc_l, u_bc_r)
    
    try:
        w = compute_ntk_weights(model, bad_batch, w_ic_ref=100.0)
        print(f"✅ SUCCÈS compute_ntk_weights ! (Weight={w:.2f})")
    except RuntimeError as e:
        print(f"❌ ÉCHEC compute_ntk_weights : {e}")
        return

    print("\n🎉 TOUT EST BON. LE CODE EST BLINDÉ.")

if __name__ == "__main__":
    torture_test()