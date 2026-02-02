import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from scipy.integrate import odeint

# Imports
sys.path.append(os.path.join(os.getcwd(), 'src'))
from models.adr import PI_DeepONet_ADR
from config import Config

# --- CONFIG ---
RUN_FOLDER = "results/run_20260130-143855" # Vérifie le dossier
CHECKPOINT = "model_checkpoint_t2.0.pth"
DEVICE = torch.device("cpu")

# --- SOLVEUR (Copie de ton code) ---
def discretized_adr_periodic(u, t, x, v, D, mu):
    dx = x[1] - x[0]
    N = len(u)
    # Indices périodiques
    idx = np.arange(N)
    idx_left = np.roll(idx, 1)
    idx_right = np.roll(idx, -1)
    u_left = u[idx_left]
    u_right = u[idx_right]
    
    d2u_dx2 = (u_right - 2*u + u_left) / (dx**2)
    du_dx = (u_right - u_left) / (2*dx)
    return D * d2u_dx2 - v * du_dx - mu * u

def main():
    # 1. Chargement
    path = os.path.join(RUN_FOLDER, CHECKPOINT)
    model = PI_DeepONet_ADR().to(DEVICE)
    try:
        ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
    except:
        ckpt = torch.load(path, map_location=DEVICE)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    # 2. On définit un CAS TEST MANUEL (Pas de random)
    # On prend des valeurs "gentilles" pour tester
    v = 1.0
    D = 0.1
    mu = 0.0
    ic_type = 3     # Gaussienne
    A = 1.0
    x0 = 0.0        # ON FORCE ZERO ICI
    sigma = 0.3     # Assez large pour être vu par le solveur
    k = 0.0
    
    # Paramètres pour le modèle
    p_val = [v, D, mu, ic_type, A, x0, sigma, k]
    print(f"🧐 Test avec params: x0={x0}, sigma={sigma}, D={D}")

    # 3. Grille (Augmentons la résolution pour aider le solveur)
    Nx = 200 
    x_np = np.linspace(Config.x_min, Config.x_max, Nx)
    t_test = 2.0  # On regarde à la fin
    
    # 4. SOLVEUR NUMÉRIQUE (Vérité Terrain Audit)
    u0 = A * np.exp(- (x_np - x0)**2 / (2 * sigma**2))
    t_grid = np.linspace(0, t_test, 50)
    sol = odeint(discretized_adr_periodic, u0, t_grid, args=(x_np, v, D, mu))
    u_solver = sol[-1, :] # Dernière étape

    # 5. MODÈLE (DeepONet)
    # On prépare l'input (Nx points, même t)
    t_tensor = np.full_like(x_np, t_test)
    xt_flat = np.stack([x_np, t_tensor], axis=1)
    xt_tensor = torch.tensor(xt_flat, dtype=torch.float32).to(DEVICE)
    
    p_tensor = torch.tensor([p_val] * Nx, dtype=torch.float32).to(DEVICE)
    
    with torch.no_grad():
        u_nn = model(p_tensor, xt_tensor).cpu().numpy().flatten()

    # 6. ANALYTIQUE (Pour arbitrer)
    # Formule exacte Gaussienne
    sigma_t = np.sqrt(sigma**2 + 2 * D * t_test)
    x_center = x0 + v * t_test
    # Périodicité simple pour l'analytique (approximation si loin des bords)
    u_exact = (A * sigma / sigma_t) * np.exp(- (x_np - x_center)**2 / (2 * sigma_t**2))

    # 7. Plot
    plt.figure(figsize=(10, 6))
    plt.plot(x_np, u_solver, 'k-', lw=3, alpha=0.5, label='Solveur Audit (ODE)')
    plt.plot(x_np, u_nn, 'r--', lw=2, label='DeepONet')
    plt.plot(x_np, u_exact, 'g:', lw=2, label='Analytique Exacte')
    
    err = np.linalg.norm(u_solver - u_nn) / np.linalg.norm(u_solver)
    plt.title(f"Diagnostic Audit (t={t_test})\nErreur Relative Solver vs NN: {err:.2%}")
    plt.legend()
    plt.grid(True)
    plt.savefig("debug_audit.png")
    plt.show()

if __name__ == "__main__":
    main()
