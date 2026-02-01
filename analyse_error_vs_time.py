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

N_SAMPLES_PER_FAMILY = 100  # Nombre de courbes à moyenner
Nx_AUDIT = 128              # Résolution spatiale suffisante
Nt_AUDIT = 100              # Résolution temporelle pour la courbe d'erreur

# --- SOLVEUR CRANK-NICOLSON (Le Juge) ---
def crank_nicolson_adr(v, D, mu, xL, xR, Nx, Tmax, Nt, u0):
    x = np.linspace(xL, xR, Nx)
    dx = x[1] - x[0]
    dt = Tmax / Nt
    u = np.asarray(u0).flatten()
    U = np.zeros((Nt, Nx))
    U[0, :] = u
    
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

# --- GÉNÉRATEUR (Force x0=0) ---
def get_random_params(family):
    v = np.random.uniform(*Config.ranges['v'])
    D = np.random.uniform(0.01, Config.ranges['D'][1])
    mu = np.random.uniform(*Config.ranges['mu'])
    A = np.random.uniform(*Config.ranges['A'])
    x0 = 0.0 # Force 0
    sigma = np.random.uniform(0.2, Config.ranges['sigma'][1])
    k = np.random.uniform(*Config.ranges['k'])
    
    if family == "Gaussian": type_id = 3
    elif family == "Sin-Gauss": type_id = 2
    elif family == "Tanh": type_id = 0
    
    return [v, D, mu, type_id, A, x0, sigma, k]

def get_ic(x, p):
    t, A, x0, s, k = p[3], p[4], p[5], p[6], p[7]
    if t==3: return A*np.exp(-(x-x0)**2/(2*s**2))
    if t==2: return A*np.exp(-(x-x0)**2/(2*s**2))*np.sin(k*x)
    if t==0: return A*np.tanh((x-x0)/s)
    return np.zeros_like(x)

# --- MAIN ---
def main():
    # Load Model
    path = os.path.join(RUN_FOLDER, CHECKPOINT)
    model = PI_DeepONet_ADR().to(DEVICE)
    try: ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
    except: ckpt = torch.load(path, map_location=DEVICE)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    
    # Grilles
    T_max = 2.0
    x_np = np.linspace(Config.x_min, Config.x_max, Nx_AUDIT)
    t_np = np.linspace(0, T_max, Nt_AUDIT)
    
    # Prépare input DeepONet
    X_grid, T_grid = np.meshgrid(x_np, t_np, indexing='ij') # (Nx, Nt)
    xt_flat = np.hstack((X_grid.reshape(-1, 1), T_grid.reshape(-1, 1)))
    xt_tensor = torch.tensor(xt_flat, dtype=torch.float32).to(DEVICE)
    
    families = ["Gaussian", "Sin-Gauss", "Tanh"]
    results = {}
    
    print(f"📉 Calcul de l'évolution temporelle de l'erreur ({N_SAMPLES_PER_FAMILY} cas/famille)...")
    
    for fam in families:
        # On va stocker l'erreur à chaque pas de temps pour tous les samples
        # Shape: (N_SAMPLES, Nt_AUDIT)
        error_history = np.zeros((N_SAMPLES_PER_FAMILY, Nt_AUDIT))
        
        for i in tqdm(range(N_SAMPLES_PER_FAMILY), desc=f"Analyzing {fam}"):
            # 1. Params & Solveur
            p_val = get_random_params(fam)
            u0 = get_ic(x_np, p_val)
            _, U_true_mat = crank_nicolson_adr(p_val[0], p_val[1], p_val[2], 
                                               Config.x_min, Config.x_max, Nx_AUDIT, T_max, Nt_AUDIT, u0)
            
            # U_true_mat est (Nt, Nx), on veut calculer l'erreur spatiale à chaque t
            
            # 2. DeepONet
            p_tensor = torch.tensor([p_val] * len(xt_flat), dtype=torch.float32).to(DEVICE)
            with torch.no_grad():
                U_pred_flat = model(p_tensor, xt_tensor).cpu().numpy().flatten()
            
            # Reshape en (Nx, Nt) puis Transpose en (Nt, Nx) pour matcher le solveur
            U_pred_mat = U_pred_flat.reshape(Nx_AUDIT, Nt_AUDIT).T
            
            # 3. Calcul erreur par pas de temps
            for t_idx in range(Nt_AUDIT):
                u_t_true = U_true_mat[t_idx, :]
                u_t_pred = U_pred_mat[t_idx, :]
                
                norm_true = np.linalg.norm(u_t_true)
                if norm_true < 1e-6: norm_true = 1e-6 # Sécurité
                
                rel_err = np.linalg.norm(u_t_true - u_t_pred) / norm_true
                error_history[i, t_idx] = rel_err

        # Moyenne sur les samples
        results[fam] = np.mean(error_history, axis=0)

    # --- PLOTTING ---
    plt.figure(figsize=(10, 6))
    
    colors = {'Gaussian': 'blue', 'Sin-Gauss': 'red', 'Tanh': 'green'}
    styles = {'Gaussian': '-', 'Sin-Gauss': '--', 'Tanh': '-.'}
    
    for fam in families:
        plt.plot(t_np, results[fam] * 100, label=f"{fam}", color=colors[fam], linestyle=styles[fam], linewidth=2)
        
    plt.title("Évolution de l'Erreur L2 Relative en fonction du Temps")
    plt.xlabel("Temps t")
    plt.ylabel("Erreur Relative Moyenne (%)")
    plt.legend()
    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    plt.xlim(0, T_max)
    
    # Ajout d'une ligne rouge pointillée pour le seuil de 5%
    plt.axhline(y=5, color='gray', linestyle=':', alpha=0.5, label='Seuil 5%')
    
    output_file = "erreur_vs_temps.png"
    plt.savefig(output_file, dpi=150)
    print(f"📊 Graphique sauvegardé : {output_file}")
    plt.show()

if __name__ == "__main__":
    main()
