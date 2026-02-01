import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm

# Import de la configuration centralisée
from config import Config

# Imports du projet
from src.physics.adr import pde_residual_adr
from src.data.generators import generate_mixed_batch
from src.physics.solver import get_ground_truth_CN
from src.visualization.plots import evaluate_global_accuracy

# --- AJOUT ICI ---
from src.audit_tool import diagnose_model

def audit_time_window(model, current_t_max, bounds):
    """ 
    Audit rapide utilisant les paramètres de Config.
    Génère des paramètres physiques aléatoires via Config.get_p_dict(),
    compare avec le solveur CN et retourne si l'erreur est sous le seuil.
    """
    device = next(model.parameters()).device
    model.eval()
    
    # Récupération des paramètres depuis Config
    Nx = Config.Nx_audit
    Nt = Config.Nt_audit
    x_min, x_max = Config.x_min, Config.x_max
    threshold = Config.threshold
    # On fait un audit rapide (10% du n_sample habituel) pour ne pas ralentir le train
    n_samples = max(10, Config.n_sample // 10) 
    
    errors = []
    
    for _ in range(n_samples):
        # 1. Génération via la méthode centralisée du Config
        p_dict = Config.get_p_dict()

        # 2. Appel au solveur (utilise les params générés)
        # On passe x_min/max explicitement, le reste est géré par p_dict et Config
        X_grid, T_grid, U_true_np = get_ground_truth_CN(p_dict, x_min, x_max, current_t_max, Nx, Nt)

        # 3. Préparation des tenseurs pour le modèle
        X_flat = X_grid.flatten()
        T_flat = T_grid.flatten()
        xt_in = np.stack([X_flat, T_flat], axis=1)
        xt_tensor = torch.tensor(xt_in, dtype=torch.float32).to(device)
        
        # Construction du vecteur de paramètres pour le réseau
        # Ordre attendu : [v, D, mu, type, A, x0, sigma, k]
        p_vec = np.array([
            p_dict['v'], p_dict['D'], p_dict['mu'], p_dict['type'], 
            p_dict['A'], p_dict['x0'], p_dict['sigma'], p_dict['k']
        ])
        p_tensor = torch.tensor(p_vec, dtype=torch.float32).unsqueeze(0).repeat(len(X_flat), 1).to(device)

        # 4. Prédiction
        with torch.no_grad():
            u_pred_flat = model(p_tensor, xt_tensor).cpu().numpy().flatten()
        
        U_pred_np = u_pred_flat.reshape(Nx, Nt)
        
        # 5. Calcul erreur relative
        err_norm = np.linalg.norm(U_true_np - U_pred_np)
        true_norm = np.linalg.norm(U_true_np)
        
        # Sécurité division par zéro
        if true_norm < 1e-6: true_norm = 1e-6
        
        errors.append(err_norm / true_norm)

    mean_error = np.mean(errors)
    return mean_error < threshold, mean_error

def train_step_time_window(model, bounds, t_max, n_iters_main, use_lbfgs=True):
    """ 
    Entraîne sur [0, t_max] avec diagnostic et correction ciblée (Smart Training).
    """
    device = next(model.parameters()).device
    
    # Config
    target_error = Config.threshold
    batch_size = Config.batch_size
    max_retries = Config.max_retry # Utilisé pour les boucles de correction
    
    w_res = Config.weight_res
    w_ic = Config.weight_ic
    w_bc = Config.weight_bc
    
    lr_base = Config.learning_rate 
    
    # ====================================================
    # 1. MAIN TRAINING (Entraînement Global)
    # ====================================================
    optimizer = optim.Adam(model.parameters(), lr=lr_base)
    model.train()
    
    pbar = tqdm(range(n_iters_main), desc=f"Train T={t_max:.1f}", leave=False)
    for i in pbar:
        optimizer.zero_grad()
        
        # Génération STANDARD (tous les types)
        params, xt, xt_ic, u_true_ic, xt_bc_left, xt_bc_right = generate_mixed_batch(
            batch_size, bounds, Config.x_min, Config.x_max, t_max, allowed_types=None
        )
        
        # Calcul des Loss (Code identique)
        loss_pde = torch.mean(pde_residual_adr(model, params, xt)**2)
        loss_ic = torch.mean((model(params, xt_ic) - u_true_ic)**2)
        u_pred_left = model(params, xt_bc_left)
        u_pred_right = model(params, xt_bc_right)
        loss_bc = torch.mean((u_pred_left - u_pred_right)**2)
        
        loss = (w_res * loss_pde) + (w_ic * loss_ic) + (w_bc * loss_bc)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if (i + 1) % 500 == 0:
            pbar.set_postfix({'loss': f"{loss.item():.2e}"})

    # ====================================================
    # 2. DIAGNOSTIC & CORRECTION CIBLÉE (SMART LOOP)
    # ====================================================
    # On tente de corriger jusqu'à max_retries fois
    for attempt in range(max_retries + 1):
        
        # A. DIAGNOSTIC : Qui est malade ?
        # Note: diagnose_model utilise son propre seuil (ex: 3% ou 5%)
        failed_ids = diagnose_model(model, device, threshold=target_error)
        
        # Si tout le monde va bien, on vérifie l'audit global et on sort
        if not failed_ids:
            success, err = audit_time_window(model, t_max, bounds)
            if success:
                return True, err
            else:
                # Cas rare : Diagnostic OK mais Audit Global KO -> On force un tour général
                print(f"    ⚠️ Diagnostic OK mais Audit Global KO ({err:.2%}). On continue...")
        else:
            print(f"    ⚠️ PROBLÈMES SUR LES TYPES : {failed_ids}")

        # Si on est au dernier essai, on tente le LBFGS (l'arme ultime)
        if attempt == max_retries:
            print(f"    🚑 DERNIER RECOURS : LBFGS...")
            lbfgs = optim.LBFGS(model.parameters(), lr=1.0, max_iter=50, line_search_fn="strong_wolfe")
            
            def closure():
                lbfgs.zero_grad()
                # On utilise failed_ids si dispo, sinon tout le monde pour le LBFGS
                types_for_lbfgs = failed_ids if failed_ids else None
                params, xt, xt_ic, u_true_ic, xt_bc_left, xt_bc_right = generate_mixed_batch(
                    batch_size, bounds, Config.x_min, Config.x_max, t_max, allowed_types=types_for_lbfgs
                )
                loss_pde = torch.mean(pde_residual_adr(model, params, xt)**2)
                loss_ic = torch.mean((model(params, xt_ic) - u_true_ic)**2)
                loss_bc = torch.mean((model(params, xt_bc_left) - model(params, xt_bc_right))**2)
                loss = w_res * loss_pde + w_ic * loss_ic + w_bc * loss_bc
                loss.backward()
                return loss
            
            try: lbfgs.step(closure)
            except: pass
            
            # Audit final après LBFGS
            success, err = audit_time_window(model, t_max, bounds)
            return success, err

        # B. FOCUS TRAINING : On ne s'entraîne QUE sur les malades
        # On réduit un peu le LR pour le fine-tuning
        lr_focus = lr_base / (2 ** (attempt + 1))
        optimizer_focus = optim.Adam(model.parameters(), lr=lr_focus)
        
        # Nombre d'itérations de correction (ex: 2000 iters)
        n_focus_iters = n_iters_main // 2 
        
        pbar_focus = tqdm(range(n_focus_iters), desc=f"Correction #{attempt+1} {failed_ids}", leave=False)
        for i in pbar_focus:
            optimizer_focus.zero_grad()
            
            # 🔥 MAGIE : On force allowed_types=failed_ids
            params, xt, xt_ic, u_true_ic, xt_bc_left, xt_bc_right = generate_mixed_batch(
                batch_size, bounds, Config.x_min, Config.x_max, t_max, allowed_types=failed_ids
            )
            
            loss_pde = torch.mean(pde_residual_adr(model, params, xt)**2)
            loss_ic = torch.mean((model(params, xt_ic) - u_true_ic)**2)
            loss_bc = torch.mean((model(params, xt_bc_left) - model(params, xt_bc_right))**2)
            
            loss = (w_res * loss_pde) + (w_ic * loss_ic) + (w_bc * loss_bc)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer_focus.step()
            
            if (i+1) % 500 == 0:
                pbar_focus.set_postfix({'loss': f"{loss.item():.2e}"})

    # Si on arrive ici, c'est qu'on a épuisé les retries
    success, err = audit_time_window(model, t_max, bounds)
    return success, err
    
def train_smart_time_marching(model, bounds, n_warmup, n_iters_per_step):
    """
    Orchestrateur principal avec sauvegardes intermédiaires à chaque dt.
    """
    save_dir = Config.save_dir
    batch_size = Config.batch_size
    os.makedirs(save_dir, exist_ok=True) 
    
    print(f"⚡ DÉMARRAGE TRAINING (Config Driven)")
    print(f"    -> Warmup (t=0): {n_warmup} iters")
    print(f"    -> Time Step: {Config.dt}, Max T: {Config.T_max}")
    print(f"    -> Batch Size: {batch_size}")

    # --- PHASE 0 : WARMUP STRICT (t=0) ---
    if n_warmup > 0:
        print("\n🧊 PHASE 0 : Fixation Condition Initiale (t=0)...")
        optimizer = optim.Adam(model.parameters(), lr=Config.learning_rate)
        model.train()
        
        pbar_warmup = tqdm(range(n_warmup), desc="Warmup IC")
        for i in pbar_warmup:
            optimizer.zero_grad()
            
            # 🔥 MODIFICATION CRITIQUE : Le générateur renvoie maintenant 6 valeurs
            # Même si on n'utilise pas les BCs pour le warmup (t=0), il faut les déballer avec `_`
            params, xt, xt_ic, u_true_ic, _, _ = generate_mixed_batch(
                batch_size, bounds, Config.x_min, Config.x_max, 0.0
            )
            
            u_pred_ic = model(params, xt_ic)
            loss = torch.mean((u_pred_ic - u_true_ic)**2)
            
            loss.backward()
            optimizer.step()
            
            pbar_warmup.set_postfix({'loss_ic': f"{loss.item():.2e}"})
            
            if (i + 1) % 500 == 0:
                 pbar_warmup.write(f"    [Warmup] Iter {i+1}/{n_warmup} | Loss IC: {loss.item():.6e}")
        
        torch.save(model.state_dict(), os.path.join(save_dir, "model_post_warmup.pth"))
        print("    ✅ Warmup terminé et sauvegardé.")

    # --- PHASE 1 : TIME MARCHING ---
    T_end = Config.T_max
    if T_end < Config.dt: T_end = Config.dt
    
    time_steps = [round(t, 2) for t in np.arange(Config.dt, T_end + Config.dt/1000.0, Config.dt)]
    
    for i, t_step in enumerate(time_steps):
        print(f"\n⏳ --- PALIER TEMPOREL : [0, {t_step}] ---")
        
        success, final_err = train_step_time_window(
            model, 
            bounds, 
            t_max=t_step, 
            n_iters_main=n_iters_per_step, 
            use_lbfgs=True
        )
        
        if success:
            print(f"    ✅ PALIER VALIDÉ (Err: {final_err:.2%})")
        else:
            print(f"    ❌ PALIER NON VALIDÉ (Err: {final_err:.2%}). Expansion forcée.")
        
        checkpoint_name = f"model_checkpoint_t{t_step}.pth"
        checkpoint_path = os.path.join(save_dir, checkpoint_name)
        torch.save({
            'step': i,
            't_max': t_step,
            'model_state_dict': model.state_dict(),
            'final_error': final_err,
        }, checkpoint_path)
        print(f"    💾 Checkpoint sauvegardé : {checkpoint_name}")

    print("\n🎉 Entraînement terminé.")
    
    # Evaluation finale
    evaluate_global_accuracy(
        model, 
        Config.Nx_audit, 
        bounds, 
        Config.x_min, 
        Config.x_max, 
        Config.T_max, 
        save_dir
    )
    
    return model