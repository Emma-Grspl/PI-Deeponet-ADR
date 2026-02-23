import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.cm as cm 
from matplotlib.gridspec import GridSpec
import yaml
import os
import sys
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from src.utils.CN_ADR import get_ground_truth_CN
from src.models.PI_DeepOnet_ADR import PI_DeepONet_ADR 

def load_config(path="src/configs/config_ADR.yaml"):
    with open(path, 'r') as f: return yaml.safe_load(f)

def load_model(model_path, cfg, device):
    print(f"loading model : {model_path}")
    model = PI_DeepONet_ADR(cfg).to(device)
    
    checkpoint = torch.load(model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
        
    model.eval()
    return model

#classical solver CN
def predict_case(model, p_dict, cfg, t_max, device):
    X, T, U_true = get_ground_truth_CN(p_dict, cfg, t_step_max=t_max)
    x_flat = X.flatten()
    t_flat = T.flatten()
    
    p_vec = np.array([p_dict['v'], p_dict['D'], p_dict['mu'], p_dict['type'], 
                      p_dict['A'], 0.0, p_dict['sigma'], p_dict['k']])
    
    p_tensor = torch.tensor(p_vec, dtype=torch.float32).repeat(len(x_flat), 1).to(device)
    xt_tensor = torch.tensor(np.stack([x_flat, t_flat], axis=1), dtype=torch.float32).to(device)
    
    with torch.no_grad():
        u_pred_flat = model(p_tensor, xt_tensor).cpu().numpy().flatten()
        
    U_pred = u_pred_flat.reshape(X.shape)
    t_vec = T[0, :] 
    x_vec = X[:, 0]
    return x_vec, t_vec, U_true, U_pred

#Analysis
def run_analysis(model, cfg, device):
    target_types = [0, 1, 3]
    types_names = {0: "Tanh", 1: "Sin-Gauss", 3: "Gaussian"}
    Tmax = cfg['geometry']['T_max']
    
    os.makedirs("outputs/PI_DeepOnet_analyse/analyse", exist_ok=True)
    
    # --- 1. Statistiques & Préparation des données ---
    print(f"Calculation of statistics (1000 samples per type)")
    temporal_errs = {i: [] for i in target_types}
    global_l2 = {i: [] for i in target_types}
    # Pour les Heatmaps d'erreur (Espace x Temps)
    error_grids = {i: [] for i in target_types} 
    
    ref_time_grid = None
    ref_x_grid = None

    for tid in target_types:
        print(f"IC Type: {types_names[tid]}")
        for _ in tqdm(range(1000)):
            p = {k: np.random.uniform(v[0], v[1]) for k, v in cfg['physics_ranges'].items()}
            p['type'] = tid
            
            try:
                x, t, u_true, u_pred = predict_case(model, p, cfg, Tmax, device)
                if ref_time_grid is None: 
                    ref_time_grid = t
                    ref_x_grid = x

                num = np.linalg.norm(u_true - u_pred)
                den = np.linalg.norm(u_true) + 1e-8
                global_l2[tid].append(num / den)
                
                norm_diff_t = np.linalg.norm(u_true - u_pred, axis=0)
                norm_true_t = np.linalg.norm(u_true, axis=0) + 1e-8
                temporal_errs[tid].append(norm_diff_t / norm_true_t)

                local_err = np.abs(u_true - u_pred) / (np.max(np.abs(u_true)) + 1e-8)
                error_grids[tid].append(local_err)
                
            except Exception: continue

    #Statistics
    plt.figure(figsize=(10, 6))
    means = [np.mean(global_l2[i]) if global_l2[i] else 0 for i in target_types]
    stds = [np.std(global_l2[i]) if global_l2[i] else 0 for i in target_types]
    names_list = [types_names[i] for i in target_types]
    colors = cm.RdPu(np.linspace(0.4, 0.8, len(target_types)))
    bars = plt.bar(names_list, means, yerr=stds, capsize=10, alpha=0.8, color=colors)
    
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height, f'{height:.2%}', ha='center', va='bottom', fontweight='bold')
    
    plt.title(f"Mean L2 Relative Error (Tmax={Tmax})")
    plt.ylabel("Relative Error L2")
    plt.grid(axis='y', alpha=0.3)
    plt.savefig("outputs/PI_DeepOnet_analyse/analyse/1_global_statistics.png", dpi=300)
    plt.close()

    #Heatmaps L2
    print("Error Heatmaps")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)
    for idx, tid in enumerate(target_types):
        ax = axes[idx]
        if error_grids[tid]:
            mean_error_grid = np.mean(np.array(error_grids[tid]), axis=0)
            im = ax.pcolormesh(ref_time_grid, ref_x_grid, mean_error_grid, shading='gouraud', cmap='RdPu')
            ax.set_title(f"Mean Error: {types_names[tid]}")
            ax.set_xlabel("Time (t)")
            if idx == 0: ax.set_ylabel("Space (x)")
            plt.colorbar(im, ax=ax)
    plt.savefig("outputs/PI_DeepOnet_analyse/analyse/2_Error_Heatmaps.png", dpi=300)
    plt.close()

    # Animation Comparison 
    print("Animation generation")
    p_fixed = {'A': 0.7, 'D': 0.05, 'mu': 0.8, 'v': 1.0, 'k': 2.0, 'sigma': 0.5}
    fig, axes = plt.subplots(3, 1, figsize=(8, 10), sharex=True)
    data_lines = [] 
    
    for idx, tid in enumerate(target_types):
        p = p_fixed.copy(); p['type'] = tid
        x, t, u_true, u_pred = predict_case(model, p, cfg, Tmax, device)
        ax = axes[idx]
        l1, = ax.plot([], [], 'k--', label='Ref')
        l2, = ax.plot([], [], color='fuchsia', linestyle='-', label='DeepONet', lw=2)
        ax.set_ylim(-1.5, 1.5)
        ax.set_xlim(x[0], x[-1])
        ax.set_title(types_names[tid], fontsize=11, fontweight='bold')
        data_lines.append((l1, l2, (x, t, u_true, u_pred)))
        if idx==0: ax.legend(loc='upper right')

    def update(frame):
        artists = []
        for (l1, l2, (x, t, ut, up)) in data_lines:
            if frame < len(t):
                l1.set_data(x, ut[:, frame])
                l2.set_data(x, up[:, frame])
                artists.extend([l1, l2])
        return artists

    frames = range(0, len(data_lines[0][2][1]), 2)
    ani = animation.FuncAnimation(fig, update, frames=frames, blit=True)
    ani.save("outputs/PI_DeepOnet_analyse/analyse/3_animation_comparison.gif", writer='pillow', fps=20)
    plt.close()

    # Temporal Snapshots
    print("Temporal Snapshots")
    target_times = [0.0, Tmax/6, Tmax/4, Tmax/2, Tmax]
    
    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    colors_time = cm.RdPu(np.linspace(0.3, 0.95, len(target_times)))

    for idx, tid in enumerate(target_types):
        p = p_fixed.copy(); p['type'] = tid
        x, t_grid, u_true, u_pred = predict_case(model, p, cfg, Tmax, device)
        ax = axes[idx]
        ax.set_title(f"{types_names[tid]} (Snapshots)", fontsize=12, fontweight='bold')
        
        for j, t_target in enumerate(target_times):
            idx_t = np.argmin(np.abs(t_grid - t_target))
            ax.plot(x, u_true[:, idx_t], color='gray', linestyle=':', alpha=0.5, linewidth=1)
            label = f"t={t_target:.2f}s" if idx == 0 else ""
            ax.plot(x, u_pred[:, idx_t], color=colors_time[j], linestyle='-', linewidth=2, label=label)

        ax.set_ylim(-1.5, 1.5)
        ax.set_ylabel("u(x, t)")
        ax.grid(True, alpha=0.2)
        if idx == 0: ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', title="Temps")

    axes[-1].set_xlabel("Space (x)")
    plt.tight_layout()
    plt.savefig("outputs/PI_DeepOnet_analyse/analyse/4_snapshots.png", dpi=300)
    plt.close()

if __name__ == "__main__":
    MODEL_PATH = "outputs/models/model_final.pth"
    if not os.path.exists(MODEL_PATH):
        MODEL_PATH = "models/model_latest.pth"
        if not os.path.exists(MODEL_PATH):
            print("Err path.")
            sys.exit(1)

    device = torch.device("cpu")
    print(f"🖥️ Device : {device}")

    cfg = load_config()
    model = load_model(MODEL_PATH, cfg, device)
    
    run_analysis(model, cfg, device)
    print("Results in 'outputs/PI_DeepOnet_analyse/analyse'")