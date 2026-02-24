import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.cm as cm 
import yaml
import os
import sys
from tqdm import tqdm

file_path = os.path.abspath(__file__) 
project_root = os.path.dirname(os.path.dirname(os.path.dirname(file_path)))

if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.utils.CN_ADR import crank_nicolson_adr
from src.models.PI_DeepOnet_ADR import PI_DeepONet_ADR 
from src.data.generators import get_ic_value

# ==========================================
# 1. UTILITAIRES DE CHARGEMENT
# ==========================================

def load_config(path=None):
    if path is None: 
        path = os.path.join(project_root, "configs", "config_ADR.yaml")
    with open(path, 'r') as f: 
        return yaml.safe_load(f)

def load_model(model_path, cfg, device):
    print(f"📥 Loading model : {model_path}")
    model = PI_DeepONet_ADR(cfg).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    if 'model_state_dict' in checkpoint: 
        model.load_state_dict(checkpoint['model_state_dict'])
    else: 
        model.load_state_dict(checkpoint)
    model.eval()
    return model

# ==========================================
# 2. MOTEUR DE PRÉDICTION
# ==========================================

def predict_all(model, p_dict, cfg, t_max, device, calc_analytical=False):
    """Génère U_cn, U_don et U_ana (si demandé) pour un set de paramètres donné."""
    
    # 🛡️ SÉCURISATION : Utilisation de valeurs par défaut si les clés sont absentes du YAML
    Nx = cfg['geometry'].get('Nx', 400)
    Nt = cfg['geometry'].get('Nt', 200)
    x_min = cfg['geometry'].get('x_min', -5.0)
    x_max = cfg['geometry'].get('x_max', 8.0)
    
    x = np.linspace(x_min, x_max, Nx)
    t = np.linspace(0, t_max, Nt)
    
    u0 = get_ic_value(x, "mixed", p_dict)
    bc_kind = "zero_zero" if p_dict['type'] in [1, 3] else "tanh_pm1"
    
    # 1. Crank-Nicolson
    _, U_cn, _ = crank_nicolson_adr(
        v=p_dict['v'], D=p_dict['D'], mu=p_dict['mu'],
        xL=x_min, xR=x_max, Nx=Nx, Tmax=t_max, Nt=Nt,
        bc_kind=bc_kind, x0=x, u0=u0
    )
    # On force STRICTEMENT la shape (Nx, Nt)
    if U_cn.shape == (Nt, Nx): 
        U_cn = U_cn.T
        
    # 2. DeepONet
    X_grid, T_grid = np.meshgrid(x, t, indexing='ij')
    x_flat, t_flat = X_grid.flatten(), T_grid.flatten()
    p_vec = np.array([p_dict['v'], p_dict['D'], p_dict['mu'], p_dict['type'], 
                      p_dict['A'], 0.0, p_dict['sigma'], p_dict['k']])
    p_tensor = torch.tensor(p_vec, dtype=torch.float32).repeat(len(x_flat), 1).to(device)
    xt_tensor = torch.tensor(np.stack([x_flat, t_flat], axis=1), dtype=torch.float32).to(device)
    
    with torch.no_grad():
        u_pred_flat = model(p_tensor, xt_tensor).cpu().numpy().flatten()
    U_don = u_pred_flat.reshape(Nx, Nt) # Déjà en (Nx, Nt)
    
    # 3. Analytique (Advection pure uniquement)
    U_ana = None
    if calc_analytical:
        U_ana = np.zeros((Nx, Nt)) # On garantit la même shape (Nx, Nt)
        for i, t_val in enumerate(t):
            x_shifted = x - p_dict['v'] * t_val
            U_ana[:, i] = get_ic_value(x_shifted, "mixed", p_dict)
            
    return x, t, U_cn, U_don, U_ana

# ==========================================
# 3. MOTEUR DE PLOTS (Générique pour les 2 comparaisons)
# ==========================================

def generate_5_plots(x, t, results_dict, title_prefix, out_dir, types_names):
    """Génère les 5 graphiques (L2 Moyenne, L2 Temporelle, Heatmaps, Snapshots, Animation)"""
    os.makedirs(out_dir, exist_ok=True)
    target_types = list(results_dict.keys())
    
    # 1. L2 Moyenne (Barres)
    plt.figure(figsize=(10, 6))
    means = [np.mean(results_dict[tid]['global_l2']) for tid in target_types]
    stds = [np.std(results_dict[tid]['global_l2']) for tid in target_types]
    names = [types_names[tid] for tid in target_types]
    colors = cm.RdPu(np.linspace(0.4, 0.8, len(target_types)))
    
    bars = plt.bar(names, means, yerr=stds, capsize=10, alpha=0.8, color=colors)
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height, f'{height:.2%}', ha='center', va='bottom', fontweight='bold')
    plt.title(f"Mean L2 Relative Error : {title_prefix}")
    plt.ylabel("Relative Error L2")
    plt.grid(axis='y', alpha=0.3)
    plt.savefig(f"{out_dir}/1_Global_Mean_L2.png", dpi=300)
    plt.close()

    # 2. L2 Temporelle (Courbes)
    plt.figure(figsize=(10, 6))
    for idx, tid in enumerate(target_types):
        mean_temp_l2 = np.mean(results_dict[tid]['temporal_l2'], axis=0)
        plt.plot(t, mean_temp_l2, label=types_names[tid], color=colors[idx], lw=2)
    plt.title(f"L2 Error Over Time : {title_prefix}")
    plt.xlabel("Time (s)")
    plt.ylabel("Relative Error L2(t)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{out_dir}/2_Temporal_L2.png", dpi=300)
    plt.close()

    # 3. Heatmaps d'Erreur Absolue Moyenne
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)
    for idx, tid in enumerate(target_types):
        ax = axes[idx]
        mean_error_grid = np.mean(results_dict[tid]['error_grids'], axis=0)
        im = ax.pcolormesh(t, x, mean_error_grid, shading='gouraud', cmap='RdPu')
        ax.set_title(f"Absolute Error: {types_names[tid]}")
        ax.set_xlabel("Time (t)")
        if idx == 0: ax.set_ylabel("Space (x)")
        plt.colorbar(im, ax=ax)
    plt.savefig(f"{out_dir}/3_Error_Heatmaps.png", dpi=300)
    plt.close()

    # 4. Snapshots Comparatifs (sur le dernier run enregistré)
    t_indices = [0, len(t)//4, len(t)//2, -1]
    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    for idx, tid in enumerate(target_types):
        ax = axes[idx]
        u_ref, u_pred = results_dict[tid]['last_u_ref'], results_dict[tid]['last_u_pred']
        ax.set_title(f"Snapshots : {types_names[tid]}", fontweight='bold')
        for i, t_idx in enumerate(t_indices):
            c = cm.RdPu(np.linspace(0.4, 0.9, len(t_indices)))[i]
            ax.plot(x, u_ref[:, t_idx], 'k:', alpha=0.6, lw=1)
            ax.plot(x, u_pred[:, t_idx], color=c, label=f"t={t[t_idx]:.2f}s" if idx==0 else "", lw=2)
        ax.set_ylim(-1.5, 1.5)
        ax.grid(True, alpha=0.2)
        if idx == 0: ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    axes[-1].set_xlabel("Space (x)")
    plt.tight_layout()
    plt.savefig(f"{out_dir}/4_Snapshots.png", dpi=300)
    plt.close()

    # 5. Animation
    fig, axes = plt.subplots(3, 1, figsize=(8, 10), sharex=True)
    lines = []
    for idx, tid in enumerate(target_types):
        ax = axes[idx]
        l1, = ax.plot([], [], 'k--', label='Reference')
        l2, = ax.plot([], [], color='fuchsia', label='DeepONet', lw=2)
        ax.set_ylim(-1.5, 1.5)
        ax.set_xlim(x[0], x[-1])
        ax.set_title(types_names[tid])
        if idx == 0: ax.legend()
        lines.append((l1, l2, results_dict[tid]['last_u_ref'], results_dict[tid]['last_u_pred']))

    def update(frame):
        artists = []
        for (l1, l2, u_r, u_p) in lines:
            if frame < len(t):
                l1.set_data(x, u_r[:, frame])
                l2.set_data(x, u_p[:, frame])
                artists.extend([l1, l2])
        return artists

    ani = animation.FuncAnimation(fig, update, frames=range(0, len(t), 2), blit=True)
    ani.save(f"{out_dir}/5_Animation.gif", writer='pillow', fps=20)
    plt.close()

# ==========================================
# 4. BOUCLE D'ANALYSE PRINCIPALE
# ==========================================

def run_analysis(model, cfg, device):
    target_types = [0, 1, 3]
    types_names = {0: "Tanh", 1: "Sin-Gauss", 3: "Gaussian"}
    Tmax = cfg['geometry']['T_max']
    
    base_out_dir = "outputs/PI_DeepOnet_analyse"
    dir_cn = f"{base_out_dir}/DeepONet_vs_CN"
    dir_ana = f"{base_out_dir}/DeepONet_vs_Analytique"
    dir_showdown = f"{base_out_dir}/Showdown"
    os.makedirs(dir_showdown, exist_ok=True)

    # --- PARTIE 1 : DeepONet vs Crank-Nicolson (Généralisation) ---
    print("\n🚀 [1/3] Analyse : DeepONet vs Crank-Nicolson (Paramètres Aléatoires)")
    res_cn = {i: {'global_l2': [], 'temporal_l2': [], 'error_grids': [], 'last_u_ref': None, 'last_u_pred': None} for i in target_types}
    
    for tid in target_types:
        for _ in tqdm(range(1000), desc=f"IC {types_names[tid]}"): # 200 samples pour l'exemple
            p = {k: np.random.uniform(v[0], v[1]) for k, v in cfg['physics_ranges'].items()}
            p['type'] = tid
            x, t, u_cn, u_don, _ = predict_all(model, p, cfg, Tmax, device)
            
            num = np.linalg.norm(u_cn - u_don)
            den = np.linalg.norm(u_cn) + 1e-8
            res_cn[tid]['global_l2'].append(num / den)
            
            res_cn[tid]['temporal_l2'].append(np.linalg.norm(u_cn - u_don, axis=0) / (np.linalg.norm(u_cn, axis=0) + 1e-8))
            res_cn[tid]['error_grids'].append(np.abs(u_cn - u_don))
            res_cn[tid]['last_u_ref'] = u_cn
            res_cn[tid]['last_u_pred'] = u_don
            
    generate_5_plots(x, t, res_cn, "DeepONet vs CN", dir_cn, types_names)

    # --- PARTIE 2 : DeepONet vs Analytique (Advection Pure) ---
    print("\n🚀 [2/3] Analyse : DeepONet vs Analytique (Pure Advection)")
    res_ana = {i: {'global_l2': [], 'temporal_l2': [], 'error_grids': [], 'last_u_ref': None, 'last_u_pred': None} for i in target_types}
    
    # Stockage spécifique pour le Showdown
    showdown_cn_l2 = []
    showdown_don_l2 = []

    p_pure_adv = {'A': 1.0, 'D': 0.0, 'mu': 0.0, 'v': 1.0, 'k': 2.0, 'sigma': 0.5, 'x0': 0.0}
    
    for tid in target_types:
        p = p_pure_adv.copy(); p['type'] = tid
        x, t, u_cn, u_don, u_ana = predict_all(model, p, cfg, Tmax, device, calc_analytical=True)
        
        # Métriques DeepONet vs Analytique
        num = np.linalg.norm(u_ana - u_don)
        den = np.linalg.norm(u_ana) + 1e-8
        res_ana[tid]['global_l2'].append(num / den)
        res_ana[tid]['temporal_l2'].append(np.linalg.norm(u_ana - u_don, axis=0) / (np.linalg.norm(u_ana, axis=0) + 1e-8))
        res_ana[tid]['error_grids'].append(np.abs(u_ana - u_don))
        res_ana[tid]['last_u_ref'] = u_ana
        res_ana[tid]['last_u_pred'] = u_don
        
        # Calcul de la L2 de CN pour le Showdown
        cn_err = np.linalg.norm(u_ana - u_cn) / (np.linalg.norm(u_ana) + 1e-8)
        showdown_cn_l2.append(cn_err)
        showdown_don_l2.append(num / den) # L2 DeepONet
        
    # Génération des 5 plots pour Analytique
    generate_5_plots(x, t, res_ana, "DeepONet vs Analytical", dir_ana, types_names)

    # --- PARTIE 3 : L'Histogramme Showdown ---
    print("\n🏆 [3/3] Génération du Showdown (CN vs Analytique) vs (DeepONet vs Analytique)")
    fig, ax = plt.subplots(figsize=(10, 6))
    x_pos = np.arange(len(target_types))
    width = 0.35
    
    rects1 = ax.bar(x_pos - width/2, showdown_cn_l2, width, label='Crank-Nicolson vs Ana', color='blue', alpha=0.7)
    rects2 = ax.bar(x_pos + width/2, showdown_don_l2, width, label='DeepONet vs Ana', color='fuchsia', alpha=0.8)
    
    ax.set_ylabel('Relative L2 Error vs Analytical')
    ax.set_title('Accuracy Showdown : Classical Numerical vs DeepONet')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([types_names[i] for i in target_types])
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    for rects in [rects1, rects2]:
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.2%}', xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontweight='bold')
            
    plt.savefig(f"{dir_showdown}/11_Accuracy_Showdown.png", dpi=300)
    plt.close()

if __name__ == "__main__":
    MODEL_PATH = "models_saved/model_best_validation.pth"
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Erreur chemin: {MODEL_PATH}")
        sys.exit(1)

    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"🖥️ Device : {device}")

    cfg = load_config()
    model = load_model(MODEL_PATH, cfg, device)
    
    run_analysis(model, cfg, device)
    print("\n✅ Terminé ! Tous les résultats (11 fichiers) sont dans : outputs/PI_DeepOnet_analyse/")