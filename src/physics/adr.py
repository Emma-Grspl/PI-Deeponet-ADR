import torch

def pde_residual_adr(model, params, xt):
    """
    Calcule le résidu de l'équation ADR : u_t + v*u_x - D*u_xx - mu*(u - u^3) = 0
    Avec sécurité (clamping) sur le terme cubique.
    """
    xt = xt.clone().detach().requires_grad_(True)
    u = model(params, xt)

    # Calcul des dérivées (Autograd)
    grads = torch.autograd.grad(u, xt, torch.ones_like(u), create_graph=True)[0]
    u_x = grads[:, 0:1]
    u_t = grads[:, 1:2]
    u_xx = torch.autograd.grad(u_x, xt, torch.ones_like(u_x), create_graph=True)[0][:, 0:1]

    # Extraction des paramètres physiques
    v_in  = params[:, 0:1]
    D_in  = params[:, 1:2]
    mu_in = params[:, 2:3]

    # --- SÉCURITÉ CRITIQUE ---
    # On limite u juste pour le calcul de la réaction non-linéaire.
    # Cela empêche u^3 d'exploser si le modèle prédit des valeurs folles au début.
    u_safe = torch.clamp(u, min=-1.2, max=1.2)

    # Terme de réaction : mu * (u - u^3)
    reaction = mu_in * (u_safe - u_safe**3)

    # Résidu PDE (Attention au signe de v_in, doit être cohérent avec le solver)
    res = u_t + v_in * u_x - D_in * u_xx - reaction

    return res