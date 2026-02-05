import numpy as np
import matplotlib.pyplot as plt
import os
import yaml
from src.physics.solver import get_ground_truth_CN

# 1. Chargement de la config pour la géométrie
def load_config(path="src/config_ADR.yaml"):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

cfg = load_config()

def plot_snapshots():
    # --- Configuration des paramètres fixés ---
    V, D, mu =1.0, 0.2, 1.0
    t_snapshots = [0.0, 0.5, 1.0, 1.5, 2.0]
    
    # --- Définition des 3 Conditions Initiales (Types) ---
    # Type 0: Tanh (Front)
    # Type 1: Gaussian Sinus (Ondelette)
    # Type 3: Gaussian (Pic)
    test_cases = [
        {'type': 0, 'name': 'Front Tanh', 'A': 0.7, 'sigma': 0.5, 'k': 3.0},
        {'type': 1, 'name': 'Sinus-Gauss', 'A': 0.7, 'sigma': 0.6, 'k': 1.0},
        {'type': 3, 'name': 'Pure Gaussian', 'A': 0.7, 'sigma': 0.4, 'k': 3.0}
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    
    for ax, case in zip(axes, test_cases):
        # Préparation du dictionnaire de paramètres pour le solveur
        p_dict = {
            'v': V, 'D': D, 'mu': mu,
            'type': case['type'], 
            'A': case['A'], 
            'x0': 0.0, 
            'sigma': case['sigma'], 
            'k': case['k']
        }
        
        print(f"🌡️ Résolution pour : {case['name']}...")
        
        # Appel du solveur via ton wrapper agnostique
        # On demande une résolution temporelle fine (Nt=200) pour la précision
        X, T, U = get_ground_truth_CN(p_dict, cfg)
        
        # X et T sont des meshgrids de shape (Nx, Nt)
        # U est de shape (Nx, Nt)
        
        x_coords = X[:, 0]
        t_coords = T[0, :]
        
        # --- Tracé des Snapshots ---
        for t_val in t_snapshots:
            # On trouve l'indice temporel le plus proche de t_val
            idx = np.argmin(np.abs(t_coords - t_val))
            actual_t = t_coords[idx]
            
            ax.plot(x_coords, U[:, idx], label=f"t={actual_t:.1f}")

        ax.set_title(f"Physique: {case['name']}\n(V={V}, D={D}, $\mu$={mu})")
        ax.set_xlabel("Espace x")
        ax.grid(True, alpha=0.3)
        if case['type'] == 0:
            ax.set_ylabel("Amplitude u(x,t)")
        ax.legend()

    plt.tight_layout()
    output_path = "solver_snapshots.png"
    plt.savefig(output_path, dpi=200)
    print(f"✅ Visualisation sauvegardée sous : {output_path}")
    plt.show()

if __name__ == "__main__":
    plot_snapshots()