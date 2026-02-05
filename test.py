import torch
import os
import sys
import yaml

# --- 1. GESTION DES CHEMINS ---
project_root = os.getcwd()
src_path = os.path.join(project_root, 'src')
if src_path not in sys.path:
    sys.path.append(src_path)

print("🔍 DÉBUT DU TEST D'INTÉGRATION (Mode YAML)")

# --- 2. TEST DES IMPORTS ---
try:
    import src.physics.solver as test_solver
    print(f"Contenu de solver.py : {dir(test_solver)}")
    from src.training.smart_trainer import load_config
    from src.models.adr import PI_DeepONet_ADR
    from src.data.generators import generate_mixed_batch
    from src.physics.adr import pde_residual_adr
    from src.physics.solver import get_ground_truth_CN
    print("✅ Imports : OK")
except ImportError as e:
    print(f"❌ Erreur d'import : {e}")
    sys.exit(1)

def run_test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 3. TEST CHARGEMENT YAML
    config_path = os.path.join(src_path, "config_ADR.yaml")
    if not os.path.exists(config_path):
        print(f"❌ Fichier config introuvable : {config_path}")
        return
    
    try:
        cfg = load_config(config_path)
        print("✅ Chargement config_ADR.yaml : OK")
    except Exception as e:
        print(f"❌ Erreur lecture YAML : {e}")
        return

    # 4. TEST INITIALISATION MODÈLE (Lien avec Config)
    try:
        model = PI_DeepONet_ADR(cfg).to(device)
        print(f"✅ Initialisation modèle (Trunk Depth: {cfg['model']['trunk_depth']}) : OK")
    except Exception as e:
        print(f"❌ Erreur init modèle (Vérifie les clés du YAML) : {e}")
        return

    # 5. TEST GÉNÉRATION DE DONNÉES (Lien avec Config)
    try:
        batch = generate_mixed_batch(
            n_samples=128, 
            bounds_phy=cfg['physics_ranges'],
            x_min=cfg['geometry']['x_min'],
            x_max=cfg['geometry']['x_max'],
            Tmax=0.3,
            device=device
        )
        print("✅ Génération Batch (generate_mixed_batch) : OK")
    except Exception as e:
        print(f"❌ Erreur génération batch (Signature ou clés YAML) : {e}")
        return

    # 6. TEST CALCUL PHYSIQUE (Autograd & Résidu)
    try:
        params, xt, _, _, _, _, _, _ = batch
        res = pde_residual_adr(model, params, xt)
        loss = torch.mean(res**2)
        loss.backward()
        print("✅ Calcul Résidu & Backprop : OK")
    except Exception as e:
        print(f"❌ Erreur physique/autograd : {e}")
        return

    # 7. TEST DU SOLVEUR (Audit)
    try:
        # On simule un dictionnaire de paramètres pour un type Tanh (0)
        p_test = {'v': 1.0, 'D': 0.1, 'mu': 0.5, 'type': 0, 'A': 1.0, 'x0': 0.0, 'sigma': 0.5, 'k': 2.0}
        X, T, U = get_ground_truth_CN(p_test, cfg, t_step_max=0.3)
        print(f"✅ Solveur Crank-Nicolson (Audit) : OK (Shape: {U.shape})")
    except Exception as e:
        print(f"❌ Erreur Solveur/Audit : {e}")
        return

    print("\n🚀 TOUS LES VOYANTS SONT AU VERT !")
    print("Le système est prêt pour le lancement multi-zones.")

if __name__ == "__main__":
    run_test()