import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import yaml
from tqdm import tqdm

# Imports locaux
from src.physics.solver import get_ground_truth_CN

def load_config(path="src/config_ADR.yaml"):
    if not os.path.exists(path): path = "src/config_ADR.yaml"
    with open(path, 'r') as f: return yaml.safe_load(f)

def visualize_classical_solver():
    # 1. Config & Dossiers
    cfg = load_config()
    output_dir = "outputs/solver_classique"
    os.makedirs(output_dir, exist_ok=True)
    
    p_base = {
        'v': 0.8, 'D': 0.05, 'mu': 0.5, 'A': 0.7, 'sigma': 0.6, 'k': 2.0
    }
    
    t_max = cfg['geometry']['T_max']
    x_min, x_max = cfg['geometry']['x_min'], cfg['geometry']['x_max']
    
    ic_types = {0: "Tanh", 1: "Sin-Gauss", 4: "Gaussian"}
    colors = plt.cm.viridis(np.linspace(0, 1, 5))
    
    print(f"🎨 Génération des visuels dans {output_dir}...")

    for type_id, type_name in ic_types.items():
        print(f"   👉 Traitement : {type_name}...")
        
        p_dict = p_base.copy()
        p_dict['type'] = type_id
        
        # Résolution
        cfg_visu = cfg.copy()
        cfg_visu['audit']['Nx_solver'] = 500
        
        X, T, U = get_ground_truth_CN(p_dict, cfg_visu, t_step_max=t_max)
        
        # --- NORMALISATION DES FORMES ---
        # On veut U de forme (Nt, Nx)
        # On veut t_vec de forme (Nt,)
        # On veut x_vec de forme (Nx,)
        
        # 1. Gestion du Temps T
        if T.ndim == 2:
            # On cherche la dimension qui varie (le temps)
            if T[0,0] != T[0,1]: t_vec = T[0, :] # Varie sur les colonnes
            else: t_vec = T[:, 0] # Varie sur les lignes
        else:
            t_vec = T

        # 2. Gestion de U pour qu'il matche (Len(t_vec), Nx)
        if U.shape[0] != len(t_vec):
            U = U.T
            
        # 3. Génération propre de X (Sûreté maximale)
        # U est maintenant (Nt, Nx), donc Nx est la 2ème dimension
        Nx = U.shape[1]
        x_vec = np.linspace(x_min, x_max, Nx)
            
        # --- 1. SNAPSHOTS ---
        plt.figure(figsize=(10, 6))
        
        times_to_plot = [0.0, t_max/6, t_max/4, t_max/2, t_max]
        indices = []
        for val in times_to_plot:
            idx = (np.abs(t_vec - val)).argmin()
            indices.append(idx)
        
        for i, idx in enumerate(indices):
            idx = min(idx, U.shape[0] - 1)
            t_val = t_vec[idx]
            
            label = f"t = {t_val:.2f}s"
            if i == 0: label += " (Init)"
            elif i == len(indices)-1: label += " (Final)"
            
            plt.plot(x_vec, U[idx, :], label=label, color=colors[i], linewidth=2)
            
        plt.title(f"ADR Dynamics : {type_name}\n(v={p_base['v']}, D={p_base['D']}, R={p_base['mu']})")
        plt.xlabel("Position x")
        plt.ylabel("u(x,t)")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.xlim(x_min, x_max)
        
        y_min, y_max = np.min(U), np.max(U)
        margin = (y_max - y_min) * 0.1
        plt.ylim(y_min - margin, y_max + margin)
        
        plt.savefig(f"{output_dir}/snapshot_{type_name}.png", dpi=150)
        plt.close()
        
        # --- 2. ANIMATION ---
        fig, ax = plt.subplots(figsize=(10, 6))
        line, = ax.plot([], [], 'b-', linewidth=2)
        time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)
        
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min - margin, y_max + margin)
        ax.set_xlabel('x')
        ax.set_ylabel('u(x,t)')
        ax.set_title(f'Animation {type_name}')
        ax.grid(True, alpha=0.3)
        
        skip = max(1, len(t_vec) // 100)
        
        def init():
            line.set_data([], [])
            time_text.set_text('')
            return line, time_text
        
        def animate(i):
            idx = i * skip
            if idx >= len(t_vec): idx = len(t_vec) - 1
            line.set_data(x_vec, U[idx, :])
            time_text.set_text(f't = {t_vec[idx]:.2f}s')
            return line, time_text
        
        ani = animation.FuncAnimation(fig, animate, init_func=init,
                                      frames=len(t_vec)//skip, interval=50, blit=True)
        
        ani.save(f"{output_dir}/anim_{type_name}.gif", writer='pillow', fps=20)
        plt.close()

    print(f"✅ Terminé ! Tout est dans {output_dir}")

if __name__ == "__main__":
    visualize_classical_solver()