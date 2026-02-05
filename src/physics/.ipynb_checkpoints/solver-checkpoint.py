import numpy as np
from scipy.sparse import diags, linalg
from src.data.generators import get_validation_data_adr

# -------------------------------------------------------------------------
# Core Solver : Crank-Nicolson (Maths pures, inchangé)
# -------------------------------------------------------------------------
def crank_nicolson_adr(v, D, mu, xL, xR, Nx, Tmax, Nt, bc_kind, x0=None, u0=None):
    if Nt == 0: Nt = 1
    x = np.linspace(xL, xR, Nx)
    dx = x[1] - x[0]
    dt = Tmax / Nt

    if x0 is None or u0 is None: raise ValueError("Provide x0, u0")

    # Interpolation initiale
    u = np.interp(x, np.asarray(x0).flatten(), np.asarray(u0).flatten())
    U = np.zeros((Nt, Nx))
    U[0, :] = u
    
    # Valeurs aux bords selon le type
    uL, uR = (-1.0, 1.0) if bc_kind == "tanh_pm1" else (0.0, 0.0)

    # --- Matrices Opérateurs ---
    L_main = -2.0 * np.ones(Nx)
    L_off  =  1.0 * np.ones(Nx - 1)
    L = diags([L_off, L_main, L_off], offsets=[-1, 0, 1], format="lil") / (dx**2)

    Dx_off = 0.5 * np.ones(Nx - 1)
    Dx = diags([-Dx_off, Dx_off], offsets=[-1, 1], format="lil") / dx
    
    if bc_kind == "neumann_zero":
        L[0, 0:2] = [-1.0/dx**2, 1.0/dx**2]
        L[-1, -2:] = [1.0/dx**2, -1.0/dx**2]
        Dx[0, :] = 0.0
        Dx[-1, :] = 0.0

    L = L.tocsc()
    Dx = Dx.tocsc()
    I = diags([np.ones(Nx)], [0], format="csc")

    A = (I - 0.5 * dt * (-v * Dx + D * L)).tolil()
    B = (I + 0.5 * dt * (-v * Dx + D * L)).tolil()

    if bc_kind in ["tanh_pm1", "zero_zero"]:
        for M in (A, B):
            M[0, :] = 0.0; M[-1, :] = 0.0
            M[0, 0] = 1.0; M[-1, -1] = 1.0

    A, B = A.tocsc(), B.tocsc()

    for n in range(1, Nt):
        R = mu * (u - u**3)
        rhs = B @ u + dt * R
        
        if bc_kind in ["tanh_pm1", "zero_zero"]:
            rhs[0], rhs[-1] = uL, uR
        
        u = linalg.spsolve(A, rhs)
        U[n, :] = u

    return x, U, np.linspace(0, Tmax, Nt)


# -------------------------------------------------------------------------
# Audit Wrapper (Agnostique, reçoit la config en argument)
# -------------------------------------------------------------------------
def get_ground_truth_CN(params_dict, full_cfg, t_step_max=None):
    """
    Interface pour l'audit. 
    full_cfg : dictionnaire chargé depuis config_ADR.yaml
    t_step_max : permet d'écraser le T_max pour un audit en cours de palier.
    """
    # 1. Extraction des paramètres
    g_cfg = full_cfg['geometry']
    a_cfg = full_cfg['audit']
    t_cfg = full_cfg['training']
    
    x_min, x_max = g_cfg['x_min'], g_cfg['x_max']
    T_max = t_step_max if t_step_max is not None else g_cfg['T_max']
    Nx = a_cfg['Nx_solver']
    
    # Nt calculé pour respecter le dt d'échantillonnage
    # On utilise ici le dt du premier pas pour rester cohérent avec la structure temporelle
    dt_ref = full_cfg['time_stepping']['zones'][0]['dt']
    Nt = int(np.ceil(T_max / dt_ref))
    if Nt < 2: Nt = 2

    # 2. Détermination des BC
    equation_type = int(params_dict.get('type', 0))
    selected_bc = "tanh_pm1" if equation_type == 0 else "zero_zero"

    # 3. IC (get_validation_data_adr doit aussi être nettoyé de Config)
    ic_kwargs = params_dict.copy()
    val_data = get_validation_data_adr(
        N0=Nx, Nb=Nt, 
        ic_kind="mixed", bc_kind="periodic", 
        ic_kwargs=ic_kwargs, 
        xL=x_min, xR=x_max, Tmax=T_max
    )

    # 4. Solver
    _, U_true_matrix, _ = crank_nicolson_adr(
        v=params_dict['v'], 
        D=params_dict['D'], 
        mu=params_dict['mu'], 
        xL=x_min, xR=x_max, 
        Nx=Nx, Tmax=T_max, Nt=Nt, 
        bc_kind=selected_bc, 
        x0=val_data["x0"], 
        u0=val_data["u0"]
    )

    # 5. Formatage [Nx, Nt]
    if U_true_matrix.shape == (Nt, Nx):
        U_true_matrix = U_true_matrix.T

    x = np.linspace(x_min, x_max, Nx)
    t = np.linspace(0, T_max, Nt)
    X_grid, T_grid = np.meshgrid(x, t, indexing='ij')

    return X_grid, T_grid, U_true_matrix