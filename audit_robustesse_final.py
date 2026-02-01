import torch
import numpy as np
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

# --- CONFIGURATION AUDIT ---
RUN_FOLDER = "results/run_20260130-143855" 
CHECKPOINT = "model_checkpoint_t2.0.pth"     
DEVICE = torch.device("cpu")

N_SAMPLES = 500   # Nombre de cas par famille
Nx_AUDIT = 256    # Résolution Spatiale fine pour le juge
Nt_AUDIT = 200    # Résolution Temporelle fine pour le juge

# --- TON SOLVEUR CRANK-NICOLSON ---
def crank_nicolson_adr(v, D, mu, xL, xR, Nx, Tmax, Nt, bc_kind, x0=None, u0=None):
    if Nt == 0: Nt = 1
    
    x = np.linspace(xL, xR, Nx)
    dx = x[1] - x[0]
    dt = Tmax / Nt

    if bc_kind == "periodic":
        uL, uR = 0.0, 0.0
    else:
        raise ValueError(f"Audit only supports periodic BC for now.")

    if u0 is None: raise ValueError("Provide u0")

    # On s'assure que u0 est plat
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
    if bc_kind == "periodic":
        L = L.tolil(); Dx = Dx.tolil()
        L[0, -1] = 1.0/dx**2; L[-1, 0] = 1.0/dx**2
        Dx[0, -1] = -0.5/dx; Dx[-1, 0] = 0.5/dx
        L = L.tocsc(); Dx = Dx.tocsc()

    # Crank-Nicolson System
    A = (I - 0.5 * dt * (-v * Dx + D * L)).tocsc()
    B = (I + 0.5 * dt * (-v * Dx + D * L)).tocsc()

    # Time Loop
    for n in range(1, Nt):
        # Terme de réaction explicite (ou semi-implicite, ici simple)
        R = mu * (u - u**3) 
        rhs = B @ u + dt * R
        u = linalg.spsolve(A, rhs)  
        U[n, :] = u

    return x, U

# --- GÉNÉRATEUR (Avec x0=0 forcé) ---
def get_random_params(ic_family):
    """ Tire des paramètres avec x0=0 """
    v = np.random.uniform(*Config.ranges['v'])
    # On évite les D trop petits qui font des chocs numériques
    D = np.random.uniform(0.01, Config.ranges['D'][1]) 
    mu = np.random.uniform(*Config.ranges['mu'])
    
    A = np.random.uniform(*Config.ranges['A'])
    x0 = 0.0  # FORCE ZÉRO
    sigma = np.random.uniform(0.2, Config.ranges['sigma'][1]) # Pas trop fin
    k = np.random.uniform(*Config.ranges['k'])
    
    if ic_family == "Gaussian": ic_type = 3
    elif ic_family == "Sin-Gauss": ic_type = 2
    elif ic_family == "Tanh": ic_type = 0
    
    return [v, D, mu, ic_type, A, x0, sigma, k]

def get_initial_condition(x, params):
    ic_type, A, x0, sigma, k = params[3], params[4], params[5], params[6], params[7]
    if ic_type == 3: return A * np.exp(-(x-x0)**2/(2*sigma**2))
    elif ic_type == 2: return A * np.exp(-(x-x0)**2/(2*sigma**2)) * np.sin(k*x)
    elif ic_type == 0: return A * np.tanh((x-x0)/sigma)
    return np.zeros_like(x)

# --- MAIN ---
def main():
    model_path = os.path.join(RUN_FOLDER, CHECKPOINT)
    print(f"📥 Chargement : {model_path}")
    
    model = PI_DeepONet_ADR().to(DEVICE)
    try:
        ckpt = torch.load(model_path, map_location=DEVICE, weights_only=False)
    except:
        ckpt = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    
    families = ["Gaussian", "Sin-Gauss", "Tanh"]
    results = {}

    print(f"🚀 Audit via Crank-Nicolson ({Nx_AUDIT}x{Nt_AUDIT}) sur {N_SAMPLES*len(families)} cas...")
    
    # Grille commune (Audit Time = Tmax = 2.0)
    T_audit = 2.0
    x_np = np.linspace(Config.x_min, Config.x_max, Nx_AUDIT)
    t_np = np.linspace(0, T_audit, Nt_AUDIT)
    
    # Meshgrid pour DeepONet (Nx * Nt)
    X_grid, T_grid = np.meshgrid(x_np, t_np, indexing='ij') 
    # indexing='ij' -> X_grid shape (Nx, Nt)
    
    xt_flat = np.hstack((X_grid.reshape(-1, 1), T_grid.reshape(-1, 1)))
    xt_tensor = torch.tensor(xt_flat, dtype=torch.float32).to(DEVICE)

    for family in families:
        errors = []
        for _ in tqdm(range(N_SAMPLES), desc=f"Audit {family}"):
            # 1. Params
            p_val = get_random_params(family)
            v, D, mu = p_val[0], p_val[1], p_val[2]
            
            # 2. Vérité Terrain (Crank-Nicolson)
            # Génération u0 sur la grille fine
            u0 = get_initial_condition(x_np, p_val)
            
            # Solve
            _, U_cn = crank_nicolson_adr(
                v, D, mu, Config.x_min, Config.x_max, Nx_AUDIT, T_audit, Nt_AUDIT, 
                "periodic", x0=None, u0=u0
            )
            # U_cn est (Nt, Nx). Transposons pour avoir (Nx, Nt)
            U_true = U_cn.T.flatten() # (Nx*Nt,)
            
            # 3. Prédiction DeepONet
            p_tensor = torch.tensor([p_val] * len(xt_flat), dtype=torch.float32).to(DEVICE)
            with torch.no_grad():
                u_pred = model(p_tensor, xt_tensor).cpu().numpy().flatten()
            
            # 4. Erreur L2
            norm_true = np.linalg.norm(U_true)
            if norm_true < 1e-6: norm_true = 1e-6
            err = np.linalg.norm(U_true - u_pred) / norm_true
            errors.append(err)
            
        results[family] = np.array(errors)

    # --- RAPPORT ---
    print("\n" + "="*45)
    print(f"📊 RÉSULTATS CRANK-NICOLSON (x0=0)")
    print("="*45)
    print(f"{'Famille':<12} | {'Moyenne':<10} | {'Médiane':<10} | {'Max':<10}")
    print("-" * 55)
    
    global_errs = []
    for fam in families:
        errs = results[fam]
        global_errs.extend(errs)
        mean_e = np.mean(errs)
        med_e = np.median(errs)
        max_e = np.max(errs)
        status = "✅" if mean_e < 0.10 else "⚠️"
        print(f"{status} {fam:<10} | {mean_e:.2%}   | {med_e:.2%}   | {max_e:.2%}")

    print("-" * 55)
    print(f"🌍 GLOBAL : {np.mean(global_errs):.2%}")
    print("="*45)

if __name__ == "__main__":
    main()
