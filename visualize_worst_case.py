import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from tqdm import tqdm
from scipy.sparse import diags, linalg

# --- IMPORTS PROJET ---
sys.path.append(os.path.join(os.getcwd(), 'src'))
try:
    from models.adr import PI_DeepONet_ADR
    from config import Config
except ImportError:
    print("❌ Erreur d'import. Assurez-vous d'être à la racine du projet.")
    sys.exit(1)

# --- CONFIGURATION ---
RUN_FOLDER = "results/run_20260130-143855" 
CHECKPOINT = "model_checkpoint_t2.0.pth"     
DEVICE = torch.device("cpu")

N_SEARCH_SAMPLES = 500  # Nombre de cas à scanner pour trouver le pire
Nx_AUDIT = 256
Nt_AUDIT = 200

# --- SOLVEUR CRANK-NICOLSON (JUGE) ---
def crank_nicolson_adr(v, D, mu, xL, xR, Nx, Tmax, Nt, u0):
    x = np.linspace(xL, xR, Nx)
    dx = x[1] - x[0]
    dt = Tmax / Nt
    u = np.asarray(u0).flatten()
    U = np.zeros((Nt, Nx))
    U[0, :] = u
    
    # Matrices
    L_main = -2.0 * np.ones(Nx)
    L_off  =  1.0 * np.ones(Nx - 1)
    L = diags([L_off, L_main, L_off], offsets=[-1, 0, 1], format="csc") / (dx**2 + 1e-12)
    
    Dx_off = 0.5 * np.ones(Nx - 1)
    Dx = diags([-Dx_off, Dx_off], offsets=[-1, 1], format="csc") / (dx + 1e-12)
    I = diags([np.ones(Nx)], [0], format="csc")
    
    # BC Periodic
    L = L.tolil(); Dx = Dx.tolil()
    L[0, -1] = 1.0/dx**2; L[-1, 0] = 1.0/dx**2
    Dx[0, -1] = -0.5/dx; Dx[-1, 0] = 0.5/dx
    L = L.tocsc(); Dx = Dx.tocsc()
    
    A = (I - 0.5 * dt * (-v * Dx + D * L)).tocsc()
    B = (I + 0.5 * dt * (-v * Dx + D * L)).tocsc()
    
    for n in range(1, Nt):
        R = mu * (u - u**3)
        rhs = B @ u + dt * R
        u = linalg.spsolve(A, rhs)
        U[n, :] = u
    return x, U

# --- GÉNÉRATEUR (Focus Sin-Gauss, x0=0) ---
def get_random_singauss_params():
    v = np.random.uniform(*Config.ranges['v'])
    D = np.random.uniform(0.01, Config.ranges['D'][1])
    mu = np.random.uniform(*Config.ranges['mu'])
    A = np.random.uniform(*Config.ranges['A'])
    x0 = 0.0 # Force 0
    sigma = np.random.uniform(0.2, Config.ranges['sigma'][1])
    k = np.random.uniform(*Config.ranges['k'])
    
    # Type 2 = Sin-Gauss
    return [v, D, mu, 2, A, x0, sigma, k]

def get_initial_condition(x, params):
    # Sin-Gauss formula: A * exp(...) * sin(kx)
    A, x0, sigma, k = params[4], params[5], params[6], params[7]
    return A * np.exp(-(x-x0)**2/(2*sigma**2)) * np.sin(k*x)

# --- MAIN ---
def main():
    # 1. Chargement Modèle
    path = os.path.join(RUN_FOLDER, CHECKPOINT)
    model = PI_DeepONet_ADR().to(DEVICE)
    try: ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
    except: ckpt = torch.load(path, map_location=DEVICE)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    
    # 2. Grilles
    T_max = 2.0
    x_np = np.linspace(Config.x_min, Config.x_max, Nx_AUDIT)
    t_np = np.linspace(0, T_max, Nt_AUDIT)
    X_grid, T_grid = np.meshgrid(x_np, t_np, indexing='ij')
    xt_flat = np.hstack((X_grid.reshape(-1, 1), T_grid.reshape(-1, 1)))
    xt_tensor = torch.tensor(xt_flat, dtype=torch.float32).to(DEVICE)
    
    print(f"🕵️‍♂️ Chasse au 'Pire Cas' sur {N_SEARCH_SAMPLES} exemples Sin-Gauss...")
    
    worst_error = -1.0
    worst_params = None
    worst_U_true = None
    worst_U_pred = None
    
    # --- PHASE 1 : SCAN ---
    for _ in tqdm(range(N_SEARCH_SAMPLES)):
        # Generate
        p_val = get_random_singauss_params()
        v, D, mu = p_val[0], p_val[1], p_val[2]
        
        # Ground Truth
        u0 = get_initial_condition(x_np, p_val)
        _, U_cn = crank_nicolson_adr(v, D, mu, Config.x_min, Config.x_max, Nx_AUDIT, T_max, Nt_AUDIT, u0)
        U_true = U_cn.T.flatten()
        
        # Prediction
        p_tensor = torch.tensor([p_val] * len(xt_flat), dtype=torch.float32).to(DEVICE)
        with torch.no_grad():
            U_pred = model(p_tensor, xt_tensor).cpu().numpy().flatten()
            
        # Error
        norm_true = np.linalg.norm(U_true) + 1e-6
        err = np.linalg.norm(U_true - U_pred) / norm_true
        
        if err > worst_error:
            worst_error = err
            worst_params = p_val
            worst_U_true = U_true.reshape(Nx_AUDIT, Nt_AUDIT)
            worst_U_pred = U_pred.reshape(Nx_AUDIT, Nt_AUDIT)

    # --- PHASE 2 : RÉSULTAT ---
    print("\n" + "🚨"*20)
    print(f"PIRE CAS TROUVÉ (Erreur L2: {worst_error:.2%})")
    v, D, mu, _, A, x0, sigma, k = worst_params
    print(f"Paramètres :")
    print(f"  v     = {v:.3f}")
    print(f"  D     = {D:.4f}")
    print(f"  mu    = {mu:.3f}")
    print(f"  k     = {k:.3f} (Fréquence)")
    print(f"  sigma = {sigma:.3f} (Largeur)")
    print("🚨"*20 + "\n")

    # --- PLOT ---
    times_to_plot = [0.0, 0.5, 1.0, 1.5, 2.0]
    plt.figure(figsize=(15, 8))
    
    # On récupère les indices temporels correspondants
    t_indices = [int(t * (Nt_AUDIT-1) / T_max) for t in times_to_plot]
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(times_to_plot)))
    
    for i, t_idx in enumerate(t_indices):
        t_val = times_to_plot[i]
        col = colors[i]
        
        # Ground Truth (Trait Pointillé)
        plt.plot(x_np, worst_U_true[:, t_idx], '--', color=col, alpha=0.6, linewidth=2, 
                 label=f'CN (Vrai) t={t_val}' if i==0 else "")
        
        # DeepONet (Trait Plein)
        plt.plot(x_np, worst_U_pred[:, t_idx], '-', color=col, linewidth=2, 
                 label=f'DeepONet t={t_val}')

    plt.title(f"Analyse du PIRE CAS (Err: {worst_error:.2%})\nv={v:.2f}, D={D:.3f}, k={k:.2f}, sigma={sigma:.2f}")
    plt.xlabel("Position x")
    plt.ylabel("u(x,t)")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    output_file = "analyse_worst_case.png"
    plt.savefig(output_file, dpi=150)
    print(f"📸 Graphique sauvegardé : {output_file}")
    plt.show()

if __name__ == "__main__":
    main()
