import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Ajout du chemin src pour les imports
# On suppose que ce script est à la racine du projet
sys.path.append(os.path.join(os.getcwd(), 'src'))

try:
    from models.adr import PI_DeepONet_ADR
    from config import Config
except ImportError as e:
    print(f"❌ Erreur d'import : {e}")
    print("👉 Assurez-vous d'être bien à la racine du projet 'These_DeepONet_ADR'")
    sys.exit(1)

# --- CONFIGURATION ---
# Le nom exact de ton dossier de résultats
RUN_FOLDER = "results/run_20260130-143855" 
# Le fichier checkpoint (on prend t=2.0)
CHECKPOINT = "model_checkpoint_t2.0.pth"     

DEVICE = torch.device("cpu") # On analyse sur CPU

def analytical_solution(x, t, v, D, A, x0, sigma):
    """ Solution analytique exacte pour une Gaussienne """
    if t == 0:
        return A * np.exp(- (x - x0)**2 / (2 * sigma**2))
    
    sigma_t = np.sqrt(sigma**2 + 2 * D * t)
    x_center = x0 + v * t
    A_t = A * (sigma / sigma_t)
    
    return A_t * np.exp(- (x - x_center)**2 / (2 * sigma_t**2))

def main():
    # 1. Chargement du modèle
    model_path = os.path.join(RUN_FOLDER, CHECKPOINT)
    if not os.path.exists(model_path):
        print(f"❌ Fichier introuvable : {model_path}")
        print(f"   Vérifiez que vous êtes bien à la racine du projet.")
        return

    print(f"📥 Chargement du modèle : {CHECKPOINT}")
    # On initialise le modèle (assure-toi que l'architecture correspond à celle entraînée)
    model = PI_DeepONet_ADR().to(DEVICE)
    
    # Chargement des poids (map_location='cpu' est important si entraîné sur GPU)
    checkpoint = torch.load(model_path, map_location=DEVICE,weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    err = checkpoint.get('final_error', 0.0)
    step = checkpoint.get('step', '?')
    print(f"✅ Modèle chargé (Step: {step}, Erreur enregistrée: {err:.2%})")

    # 2. Paramètres du Cas Test (Gaussienne qui avance)
    # Ces valeurs doivent être dans les ranges de ta Config
    v_test = 1.0
    D_test = 0.05
    mu_test = 0.0  # Sans réaction pour comparer facilement
    
    # Paramètres de la condition initiale (Gaussienne)
    A_test = 1.0
    x0_test = 0.0
    sigma_test = 0.2
    k_test = 0.0
    type_test = 3  # Type 3 = Gaussienne simple

    # Vecteur paramètres [v, D, mu, type, A, x0, sigma, k]
    params_values = [v_test, D_test, mu_test, type_test, A_test, x0_test, sigma_test, k_test]
    
    # 3. Préparation de la visualisation
    x_np = np.linspace(Config.x_min, Config.x_max, 200)
    times_to_plot = [0.0, 0.5, 1.0, 1.5, 2.0]
    
    plt.figure(figsize=(14, 7))
    
    for t_val in times_to_plot:
        # --- Préparation Batch ---
        N = len(x_np)
        # On répète les paramètres N fois
        params_tensor = torch.tensor([params_values] * N, dtype=torch.float32).to(DEVICE)
        
        # Coordonnées (x, t)
        t_np = np.full_like(x_np, t_val)
        xt_tensor = torch.tensor(np.stack([x_np, t_np], axis=1), dtype=torch.float32).to(DEVICE)
        
        # --- Prédiction ---
        with torch.no_grad():
            u_pred = model(params_tensor, xt_tensor).cpu().numpy().flatten()
            
        # --- Solution Exacte ---
        u_true = analytical_solution(x_np, t_val, v_test, D_test, A_test, x0_test, sigma_test)
        
        # --- Plot ---
        # Couleur dégradée en fonction du temps
        color = plt.cm.viridis(t_val / 2.5) 
        
        # Trait plein = Prédiction DeepONet
        plt.plot(x_np, u_pred, '-', color=color, linewidth=2, label=f'NN t={t_val}')
        # Trait pointillé = Solution Exacte
        plt.plot(x_np, u_true, '--', color=color, alpha=0.5)

    plt.title(f"Advection-Diffusion (v={v_test}, D={D_test})\nComparaison Modèle vs Analytique (Pointillés)")
    plt.xlabel("x")
    plt.ylabel("u(x,t)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    output_file = "resultat_analyse_local.png"
    plt.savefig(output_file, dpi=150)
    print(f"📊 Graphique sauvegardé : {output_file}")
    plt.show() # Affiche la fenêtre si possible

if __name__ == "__main__":
    main()
