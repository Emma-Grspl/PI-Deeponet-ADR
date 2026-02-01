import torch
import numpy as np
from config import Config  # <--- Ajout crucial

# Device detection
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps") # Mac
else:
    DEVICE = torch.device("cpu")

# --- IC Generation Utils ---
def get_ic_value(x, ic_kind, ic_params):
    """
    Génère la condition initiale (IC). Inchangé dans la logique, 
    mais utilise les params passés dynamiquement.
    """
    if ic_params is None: ic_params = {}
    
    # torch or numpy 
    is_torch = isinstance(x, torch.Tensor)
    if not is_torch and not isinstance(x, np.ndarray):
        x = np.array(x)

    # Tool configuration
    if is_torch:
        exp, sin, tanh = torch.exp, torch.sin, torch.tanh
        zeros_like = torch.zeros_like
        cast = lambda m: m.float()
        check_any = lambda m: m.sum() > 0 
    else:
        exp, sin, tanh = np.exp, np.sin, np.tanh
        zeros_like = np.zeros_like
        cast = lambda m: np.asarray(m).astype(float)
        check_any = lambda m: np.any(m)

    # Logic
    if ic_kind == "mixed":
        types = ic_params.get("type") 
        A     = ic_params.get("A", 1.0)
        x0    = ic_params.get("x0", 0.0)
        sigma = ic_params.get("sigma", 0.5)
        k     = ic_params.get("k", 2.0)

        u0 = zeros_like(x)

        # 0 = Tanh
        mask_0 = cast(types == 0)
        if check_any(mask_0): 
            u0 += mask_0 * A * tanh((x - x0) / (sigma + 1e-8))

        # 1 & 2 = Gaussian sinus
        mask_gs = cast((types == 1) | (types == 2))
        if check_any(mask_gs):
            u0 += mask_gs * A * exp( - (x - x0)**2 / (2 * sigma**2 + 1e-8) ) * sin(k * x)

        # 3 & 4 = Gaussian
        mask_gauss = cast((types == 3) | (types == 4))
        if check_any(mask_gauss):
            u0 += mask_gauss * A * exp( - (x - x0)**2 / (2 * sigma**2 + 1e-8) )

        return u0

    else:
        # Fallback pour single types (legacy)
        kind_clean = ic_kind.replace(" ", "_")
        def get_scalar(key, default):
            val = ic_params.get(key, default)
            if hasattr(val, 'item'): return val.item()
            return val

        A = get_scalar("A", 1.0)
        x0 = get_scalar("x0", 0.0)
        sigma = get_scalar("sigma", 0.5)
        k = get_scalar("k", 2.0)

        if kind_clean == "tanh": return A * tanh((x - x0)/sigma)
        elif kind_clean == "gaussian": return A * exp(-(x-x0)**2/(2*sigma**2))
        elif kind_clean == "gaussian_sinus": return A * exp(-(x-x0)**2/(2*sigma**2)) * sin(k*x)
        else: return zeros_like(x)


# --- Validation Data for Solver ---
def get_validation_data_adr(N0, Nb, ic_kind="mixed", bc_kind="periodic", 
                            ic_kwargs=None, xL=None, xR=None, Tmax=None):
    """
    Prépare les grilles pour le solveur classique (Validation/Audit).
    Utilise Config pour les valeurs par défaut.
    """
    if ic_kwargs is None: ic_kwargs = {}
    
    # Valeurs par défaut Config
    if xL is None: xL = Config.x_min
    if xR is None: xR = Config.x_max
    if Tmax is None: Tmax = Config.T_max

    # Grille spatiale
    x0_np = np.linspace(xL, xR, N0)
    
    # Grille temporelle
    t_b_np = np.linspace(0, Tmax, Nb)
    
    # Condition initiale u0
    u0_np = get_ic_value(x0_np, ic_kind, ic_kwargs)

    data = {
        "x0": x0_np,      # Shape (N0,)
        "u0": u0_np,      # Shape (N0,)
        "t_b": t_b_np,    # Shape (Nb,)
        "xL": xL, "xR": xR,
        "ic_kind": ic_kind,
        "bc_kind": bc_kind
    }
    return data
