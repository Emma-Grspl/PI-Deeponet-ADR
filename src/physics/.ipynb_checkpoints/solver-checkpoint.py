import numpy as np
from scipy.sparse import diags, linalg
from src.data.generators import get_validation_data_adr
from config import Config  # <--- Ajout de l'import Config

# -------------------------------------------------------------------------
# Core Solver : Crank-Nicolson (Maths pures, pas de dépendance Config directe)
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
    if bc_kind == "tanh_pm1":
        uL, uR = -1.0, 1.0
    else:
        uL, uR = 0.0, 0.0

    # --- Matrices Opérateurs de base ---
    L_main = -2.0 * np.ones(Nx)
    L_off  =  1.0 * np.ones(Nx - 1)
    L = diags([L_off, L_main, L_off], offsets=[-1, 0, 1], format="lil") / (dx**2)

    Dx_off = 0.5 * np.ones(Nx - 1)
    Dx = diags([-Dx_off, Dx_off], offsets=[-1, 1], format="lil") / dx
    
    # --- Application des Conditions aux Limites ---
    if bc_kind in ["tanh_pm1", "zero_zero"]:
        # Dirichlet : On force les lignes des bords à l'identité dans les matrices
        # On le fera après avoir construit A et B pour plus de simplicité
        pass 
    elif bc_kind == "neumann_zero":
        # Neumann : du/dx = 0 aux bords
        L[0, 0:2] = [-1.0/dx**2, 1.0/dx**2]
        L[-1, -2:] = [1.0/dx**2, -1.0/dx**2]
        Dx[0, :] = 0.0
        Dx[-1, :] = 0.0

    L = L.tocsc()
    Dx = Dx.tocsc()
    I = diags([np.ones(Nx)], [0], format="csc")

    # Construction des matrices du système
    A = (I - 0.5 * dt * (-v * Dx + D * L)).tolil()
    B = (I + 0.5 * dt * (-v * Dx + D * L)).tolil()

    # Application stricte de Dirichlet dans les matrices si nécessaire
    if bc_kind in ["tanh_pm1", "zero_zero"]:
        for M in (A, B):
            M[0, :] = 0.0; M[-1, :] = 0.0
            M[0, 0] = 1.0; M[-1, -1] = 1.0

    A = A.tocsc(); B = B.tocsc()

    # Boucle temporelle
    for n in range(1, Nt):
        R = mu * (u - u**3)
        rhs = B @ u + dt * R
        
        # Application des valeurs aux bords dans le second membre (Dirichlet)
        if bc_kind in ["tanh_pm1", "zero_zero"]:
            rhs[0] = uL
            rhs[-1] = uR
        
        u = linalg.spsolve(A, rhs)
        U[n, :] = u

    return x, U, np.linspace(0, Tmax, Nt)


# -------------------------------------------------------------------------
# Audit Wrapper (Connecté à Config)
# -------------------------------------------------------------------------
def get_ground_truth_CN(params_dict, x_min=None, x_max=None, T_max=None, Nx=None, Nt=None):
    """
    Standardized interface for auditing and validation.
    Utilise Config pour les valeurs par défaut.
    """
    # 1. Valeurs par défaut depuis Config
    if x_min is None: x_min = Config.x_min
    if x_max is None: x_max = Config.x_max
    if T_max is None: T_max = Config.T_max
    if Nx is None: Nx = Config.Nx_solver  # Grille fine pour le solveur
    
    # Calcul automatique de Nt si non fourni, basé sur dt du config
    if Nt is None: 
        Nt = int(np.ceil(T_max / Config.dt))
        if Nt < 2: Nt = 2

    # 2. Détermination Intelligente des BC (Conditions aux Limites)
    # Type 0, 1, 2 : Gaussiennes (décroissent vers 0 aux bords) -> "zero_zero" ou "periodic" (si x assez grand)
    # Type 3, 4 : Fronts (Tanh) -> -1 à gauche, +1 à droite -> "tanh_pm1"
    equation_type = int(params_dict.get('type', 0))
    
    if equation_type == 0:  # Fronts Tanh
        selected_bc = "tanh_pm1"
    else:                        # Gaussiennes (Sin-Gauss, etc.)
        selected_bc = "zero_zero" # Plus sûr que périodique pour éviter les réentrées

    # 3. Préparation des données initiales (IC)
    ic_kwargs = params_dict.copy()
    
    val_data = get_validation_data_adr(
        N0=Nx, Nb=Nt, 
        ic_kind="mixed", bc_kind="periodic", # Ici peu importe, c'est juste pour générer x0/u0
        ic_kwargs=ic_kwargs, 
        xL=x_min, xR=x_max, Tmax=T_max
    )

    # 4. Exécution du Solver avec les BONS BC
    raw_result = crank_nicolson_adr(
        v=params_dict['v'], 
        D=params_dict['D'], 
        mu=params_dict['mu'], 
        xL=x_min, xR=x_max, 
        Nx=Nx, Tmax=T_max, Nt=Nt, 
        bc_kind=selected_bc,  # <--- C'EST CORRIGÉ ICI
        x0=val_data["x0"], 
        u0=val_data["u0"]
    )

    if isinstance(raw_result, tuple):
        U_true_matrix = raw_result[1] 
    else:
        U_true_matrix = raw_result

    # 5. Formatage de sortie (Shape [Nx, Nt])
    if U_true_matrix.shape == (Nt, Nx):
        U_true_matrix = U_true_matrix.T

    # Création des grilles pour l'interpolation ou le plot
    x = np.linspace(x_min, x_max, Nx)
    t = np.linspace(0, T_max, Nt)
    X_grid, T_grid = np.meshgrid(x, t, indexing='ij')

    return X_grid, T_grid, U_true_matrix