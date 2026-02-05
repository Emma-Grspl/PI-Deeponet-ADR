import sys
import os
import torch
import yaml
from datetime import datetime

# --- GESTION DES CHEMINS ---
project_root = os.getcwd()
src_path = os.path.join(project_root, 'src')
if src_path not in sys.path:
    sys.path.append(src_path)

# --- IMPORTS DÉDIÉS ---
try:
    # On importe le chargeur de config depuis le nouveau smart_trainer
    from src.training.smart_trainer import load_config, train_smart_time_marching
    from src.models.adr import PI_DeepONet_ADR
    print("✅ Imports réussis.")
except ImportError as e:
    print(f"❌ Erreur d'import : {e}")
    sys.exit(1)

def main():
    # 1. CHARGEMENT DE LA CONFIGURATION YAML
    # On charge le fichier YAML (assure-toi que le chemin est correct)
    config_path = os.path.join(project_root, "src", "config_ADR.yaml")
    cfg = load_config(config_path)

    # 2. SETUP DOSSIER DE SAUVEGARDE
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_name = f"run_{timestamp}"
    run_dir = os.path.join(project_root, "results", run_name)
    os.makedirs(run_dir, exist_ok=True)
    
    # On met à jour le dossier de sauvegarde dans la config chargée
    cfg['audit']['save_dir'] = run_dir
    
    print(f"🚀 Lancement Entraînement ADR (Jean Zay)")
    print(f"📁 Dossier de sortie : {run_dir}")
    print(f"⚙️  Paramètres clés :")
    print(f"   - Tmax: {cfg['geometry']['T_max']}")
    print(f"   - Zones: {cfg['time_stepping']['zones']}")
    print(f"   - Points (n_sample): {cfg['training']['n_sample']}")
    print(f"   - Macro Loops: {cfg['training']['nb_loop']}")

    # 3. DEVICE
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"📱 Device : {device}")
    
    # 4. INITIALISATION MODÈLE
    model = PI_DeepONet_ADR().to(device)

    # 5. ENTRAÎNEMENT (Smart Time Marching)
    # Note : Le trainer utilise 'cfg' interne, on passe juste le modèle et les ranges
    try:
        model = train_smart_time_marching(
            model,
            bounds=cfg['physics_ranges']
        )
    except Exception as e:
        print(f"❌ Erreur critique : {e}")
        emergency_path = os.path.join(run_dir, "model_CRASHED.pth")
        torch.save(model.state_dict(), emergency_path)
        raise e

    # 6. SAUVEGARDE FINALE
    final_model_path = os.path.join(run_dir, "model_final.pth")
    torch.save(model.state_dict(), final_model_path)
    print(f"✅ Modèle final sauvegardé : {final_model_path}")

if __name__ == "__main__":
    main()