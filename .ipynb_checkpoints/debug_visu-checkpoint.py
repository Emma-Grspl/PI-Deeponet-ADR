import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# --- GESTION DES CHEMINS (La partie critique) ---
project_root = os.getcwd()
src_path = os.path.join(project_root, 'src')

# 1. On ajoute la racine pour pouvoir faire 'from src...'
if project_root not in sys.path:
    sys.path.append(project_root)

# 2. On ajoute AUSSI 'src' pour que les fichiers internes trouvent 'config.py' directement
if src_path not in sys.path:
    sys.path.append(src_path)

# --- IMPORTS ---
try:
    # Maintenant, Python cherche dans 'src' donc il trouve 'config' tout court
    from config import Config
    from src.models.adr import PI_DeepONet_ADR
    from src.physics.solver import get_ground_truth_CN
    print("✅ Imports réussis.")
except ImportError as e:
    print(f"❌ Erreur d'import : {e}")
    print("👉 Vérifie que tu es bien à la racine du projet (là où il y a le dossier 'src')")
    sys.exit(1)

def main():
    # ... (Le reste du code reste identique) ...
    # Je te remets la suite pour être sûr que tu as tout
    device = torch.device("cpu") # Force CPU pour éviter les bugs MPS/Cuda en debug
    model_path = "model_debug.pth"

    print(f"📂 Dossier courant : {os.getcwd()}")
    print(f"🔎 Recherche du modèle : {model_path}")

    model = PI_DeepONet_ADR().to(device)
    
    if os.path.exists(model_path):
        try:
            # map_location='cpu' est indispensable ici
            checkpoint = torch.load(model_path, map_location=device)
            # Si le checkpoint contient tout l'état (avec optimizer, etc.), on prend juste les poids
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            print(f"✅ Modèle chargé !")
        except Exception as e:
            print(f"⚠️ Erreur chargement ({e}). On utilise un modèle vierge.")
    else:
        print(f"⚠️ Fichier introuvable. On utilise un modèle vierge.")

    model.eval()

    # Paramètres Test (Vitesse POSITIVE v=1.0 -> Vers la droite)
    p_dict = {
        'v': 1.0, 'D': 0.05, 'mu': 1.0, 'type': 2, 
        'A': 1.0, 'x0': 0.0, 'sigma': 0.5, 'k': 1.0
    }
    
    # Vérité
    Nx = 200;
# --- Imports du projet ---
# Assure-toi d'être à la racine du projet
sys.path.append(os.getcwd())

try:
    from src.config import Config
    from src.models.adr import PI_DeepONet_ADR
    from src.physics.solver import get_ground_truth_CN
    print("✅ Imports réussis.")
except ImportError as e:
    print(f"❌ Erreur d'import : {e}")
    print("⚠️ Lance ce script depuis la racine du projet (là où il y a le dossier 'src')")
    sys.exit(1)

def main():
    # 1. Setup
    device = torch.device("cpu") # On force CPU pour le debug local, c'est plus sûr
    model_path = "model_debug.pth" # Le fichier que tu as téléchargé via SCP

    # 2. Charger le modèle
    model = PI_DeepONet_ADR().to(device)
    
    if os.path.exists(model_path):
        try:
            # map_location est crucial pour charger un modèle CUDA sur un CPU
            checkpoint = torch.load(model_path, map_location=device)
            model.load_state_dict(checkpoint)
            print(f"✅ Modèle '{model_path}' chargé avec succès.")
        except Exception as e:
            print(f"⚠️ Erreur chargement modèle : {e}")
            print("👉 On utilise un modèle vierge (juste pour vérifier le code).")
    else:
        print(f"⚠️ Fichier '{model_path}' introuvable. On utilise un modèle vierge.")

    model.eval()

    # 3. Paramètres du Cas Test (C'est là qu'on vérifie la physique !)
    # On prend une Gaussienne simple qui avance
    p_dict = {
        'v': 1.0,       # Vitesse positive -> Doit aller à DROITE (si l'axe est x>0)
        'D': 0.05,      # Diffusion faible -> La bosse ne doit pas s'écraser trop vite
        'mu': 1.0,      # Pas de réaction
        'type': 4,      # Gaussienne
        'A': 0.7,
        'x0': 0.0,      # Centré en 0
        'sigma': 0.5,
        'k': 1.0
    }
    print(f"⚙️  Paramètres physiques : {p_dict}")

    # 4. Génération Vérité Terrain (Le Juge)
    Nx = 200
    Nt = 100
    t_max_visu = 0.17 # On regarde à t=1.0
    
    # Appel au solveur (Crank-Nicolson)
    try:
        X_grid, T_grid, U_true = get_ground_truth_CN(
            p_dict, Config.x_min, Config.x_max, t_max_visu, Nx, Nt
        )
    except Exception as e:
        print(f"❌ Erreur du solveur : {e}")
        return

    # 5. Prédiction du Modèle (L'Élève)
    X_flat = X_grid.flatten()
    T_flat = T_grid.flatten()
    
    # Création des tenseurs
    xt_in = np.stack([X_flat, T_flat], axis=1)
    xt_tensor = torch.tensor(xt_in, dtype=torch.float32).to(device)
    
    # Vecteur paramètres [v, D, mu, type, A, x0, sigma, k]
    p_vec = np.array([
        p_dict['v'], p_dict['D'], p_dict['mu'], p_dict['type'], 
        p_dict['A'], p_dict['x0'], p_dict['sigma'], p_dict['k']
    ])
    # Répétition pour chaque point de la grille
    p_tensor = torch.tensor(p_vec, dtype=torch.float32).unsqueeze(0).repeat(len(X_flat), 1).to(device)

    # Forward pass
    with torch.no_grad():
        u_pred_flat = model(p_tensor, xt_tensor).cpu().numpy().flatten()
    
    # Reshape (Nt, Nx) ou (Nx, Nt) selon le solveur -> On force (Nx, Nt) pour l'affichage
    # Le solveur renvoie souvent (Nx, Nt) pour X_grid. Vérifions.
    U_pred = u_pred_flat.reshape(X_grid.shape)

    # 6. Affichage
    plt.figure(figsize=(10, 8))
    
    # Récupération de l'axe X (première colonne de X_grid)
    x = X_grid[:, 0]
    
    # --- Plot t=0 (Condition Initiale) ---
    plt.subplot(2, 1, 1)
    # t=0 correspond à l'indice 0 en temps
    plt.plot(x, U_true[:, 0], 'k--', linewidth=2, label="Vérité (Solver)")
    plt.plot(x, U_pred[:, 0], 'b-', linewidth=1.5, label="Modèle (DeepONet)")
    plt.title("t = 0.0 (Condition Initiale)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # --- Plot t=t_max (Dynamique) ---
    plt.subplot(2, 1, 2)
    # t=end correspond au dernier indice
    plt.plot(x, U_true[:, -1], 'k--', linewidth=2, label=f"Vérité t={t_max_visu}")
    plt.plot(x, U_pred[:, -1], 'r-', linewidth=1.5, label=f"Modèle t={t_max_visu}")
    plt.title(f"t = {t_max_visu} (Advection-Diffusion)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
