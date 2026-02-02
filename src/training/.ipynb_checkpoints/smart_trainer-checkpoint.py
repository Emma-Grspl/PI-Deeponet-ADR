import torch
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from config import Config
from src.physics.adr import pde_residual_adr
from src.data.generators import generate_mixed_batch
from src.audit_tool import diagnose_model
from src.physics.solver import get_ground_truth_CN

# --- AUDIT RAPIDE ---
def audit_global_fast(model, current_t_max):
    device = next(model.parameters()).device
    model.eval()
    Nx = Config.Nx_audit
    Nt = Config.Nt_audit
    errors = []
    
    for _ in range(20): # 20 samples aléatoires
        p_dict = Config.get_p_dict()
        try:
            X_grid, T_grid, U_true_np = get_ground_truth_CN(
                p_dict, Config.x_min, Config.x_max, current_t_max, Nx, Nt
            )
        except: continue

        X_flat, T_flat = X_grid.flatten(), T_grid.flatten()
        xt_tensor = torch.tensor(np.stack([X_flat, T_flat], axis=1), dtype=torch.float32).to(device)
        
        p_vec = np.array([p_dict['v'], p_dict['D'], p_dict['mu'], p_dict['type'], 
                          p_dict['A'], p_dict['x0'], p_dict['sigma'], p_dict['k']])
        p_tensor = torch.tensor(p_vec, dtype=torch.float32).unsqueeze(0).repeat(len(X_flat), 1).to(device)

        with torch.no_grad():
            u_pred = model(p_tensor, xt_tensor).cpu().numpy().flatten()
        
        U_true = U_true_np.flatten()
        # Erreur Relative pondérée pour éviter division par zéro
        err = np.linalg.norm(U_true - u_pred) / (np.linalg.norm(U_true) + 1e-7)
        errors.append(err)

    if not errors: return False, 1.0
    mean_err = np.mean(errors)
    return mean_err < Config.threshold, mean_err

# --- STEP D'ENTRAINEMENT ---
def train_step_time_window(model, bounds, t_max, n_iters_main):
    device = next(model.parameters()).device
    lr_current = Config.learning_rate
    
    print(f"\n🔵 DÉBUT PALIER t=[0, {t_max}]")

    # --- 1. DEFINITION DE LA LOSS (DIRICHLET STABLE) ---
    # On accepte maintenant les cibles u_true_bc_l et u_true_bc_r venant du générateur
    def compute_total_loss(params, xt, xt_ic, u_true_ic, xt_bc_l, xt_bc_r, u_true_bc_l, u_true_bc_r):
        # A. Loss Physique (PDE)
        # Note: le clamping u^3 est géré dans pde_residual_adr
        loss_pde = torch.mean(pde_residual_adr(model, params, xt)**2)
        
        # B. Loss Condition Initiale (IC)
        loss_ic = torch.mean((model(params, xt_ic) - u_true_ic)**2)
        
        # C. Loss Bords (Dirichlet)
        # On force le modèle à valoir exactement la cible (-1/+1 ou 0)
        u_pred_l = model(params, xt_bc_l)
        u_pred_r = model(params, xt_bc_r)
        
        loss_bc = torch.mean((u_pred_l - u_true_bc_l)**2) + \
                  torch.mean((u_pred_r - u_true_bc_r)**2)
        
        return Config.weight_res * loss_pde + Config.weight_ic * loss_ic + Config.weight_bc * loss_bc

    # =========================================================================
    # ÉTAPE 1 : ENTRAINEMENT GLOBAL (Tout le monde)
    # =========================================================================
    global_success = False
    
    for attempt in range(4): # 3 tentatives Adam, 1 tentative LBFGS
        
        # --- CAS LBFGS (Tentative 4) ---
        if attempt == 3: 
            print(f"  👉 Tentative Globale {attempt+1}/4 : LBFGS")
            lbfgs = optim.LBFGS(model.parameters(), lr=1.0, max_iter=50, line_search_fn="strong_wolfe")
            def closure():
                lbfgs.zero_grad()
                # Le générateur renvoie maintenant 8 valeurs
                batch = generate_mixed_batch(Config.batch_size, bounds, Config.x_min, Config.x_max, t_max)
                loss = compute_total_loss(*batch)
                loss.backward()
                return loss
            try: lbfgs.step(closure)
            except: pass
            
        # --- CAS ADAM (Tentatives 1, 2, 3) ---
        else: 
            print(f"  👉 Tentative Globale {attempt+1}/4 : Adam (LR={lr_current:.1e})")
            optimizer = optim.Adam(model.parameters(), lr=lr_current)
            model.train()
            
            for i in range(n_iters_main):
                optimizer.zero_grad()
                batch = generate_mixed_batch(Config.batch_size, bounds, Config.x_min, Config.x_max, t_max)
                loss = compute_total_loss(*batch)
                loss.backward()
                optimizer.step()
                
                if (i + 1) % 1000 == 0:
                    print(f"    [Global Adam] Iter {i+1}/{n_iters_main} | Loss: {loss.item():.2e}")

        # --- AUDIT GLOBAL ---
        success, err = audit_global_fast(model, t_max)
        print(f"     📊 Audit Global: {err:.2%} (Seuil: {Config.threshold:.0%}) -> {'✅ OK' if success else '❌ KO'}")
        
        if success:
            global_success = True
            break
        
        # Si échec, on baisse le LR pour la prochaine tentative Adam
        if attempt < 2: lr_current *= 0.5

    if global_success: return True, err

    # =========================================================================
    # ÉTAPE 2 : CORRECTION CIBLÉE (SMART MIXING)
    # =========================================================================
    print(f"\n🩺 Audit Spécifique par Famille...")
    failed_ids = diagnose_model(model, device, threshold=Config.threshold)
    
    if not failed_ids: return True, err

    # --- Stratégie Anti-Oubli ---
    # On identifie les cas sains pour les mélanger aux malades
    all_types = [0, 1, 2, 3, 4]
    success_ids = [t for t in all_types if t not in failed_ids]

    print(f"🚑 Correction sur {failed_ids} (Rappel mémoire sur {success_ids})...")

    # Création de la liste pondérée pour le générateur
    # Ratio approx: 80% Malades / 20% Sains
    weighted_types = []
    
    # On booste les malades (x4)
    for f_id in failed_ids:
        weighted_types.extend([f_id] * 4)
        
    # On garde une trace des sains (x1)
    for s_id in success_ids:
        weighted_types.extend([s_id] * 1)

    # --- Boucle de Correction ---
    optimizer_focus = optim.Adam(model.parameters(), lr=lr_current)
    
    # On entraîne un peu plus longtemps pour bien ancrer la correction (ex: 3000 iters)
    for i in range(3000):
        optimizer_focus.zero_grad()
        
        # Le générateur va piocher dans weighted_types
        batch = generate_mixed_batch(
            Config.batch_size, bounds, Config.x_min, Config.x_max, t_max, 
            allowed_types=weighted_types 
        )
        
        loss = compute_total_loss(*batch)
        loss.backward()
        optimizer_focus.step()
        
        if (i + 1) % 1000 == 0:
            print(f"    [Focus Adam (Mix 80/20)] Iter {i+1} | Loss: {loss.item():.2e}")

    # --- VERDICT FINAL ---
    failed_ids_final = diagnose_model(model, device, threshold=Config.threshold)
    if not failed_ids_final:
        print("✅ Correction réussie ! Tout est rentré dans l'ordre.")
        return True, 0.0
    else:
        print(f"❌ ÉCHEC DÉFINITIF sur t={t_max}. Types résistants: {failed_ids_final}")
        return False, 1.0

def train_smart_time_marching(model, bounds, n_warmup, n_iters_per_step):
    save_dir = Config.save_dir
    print(f"⚡ DÉMARRAGE TRAINING (Protocole Strict & Verbose)")
    print(f"    -> Warmup (t=0): {n_warmup} iters")
    print(f"    -> Time Step: {Config.dt}, Max T: {Config.T_max}")

    # --- WARMUP (t=0) ---
    if n_warmup > 0:
        print("\n🧊 PHASE 0 : WARMUP (Condition Initiale)...")
        optimizer = optim.Adam(model.parameters(), lr=Config.learning_rate)
        model.train()
        pbar = tqdm(range(n_warmup))
        for i in pbar:
            optimizer.zero_grad()
            params, xt, xt_ic, u_true_ic, _, _ = generate_mixed_batch(
                Config.batch_size, bounds, Config.x_min, Config.x_max, 0.0
            )
            loss = torch.mean((model(params, xt_ic) - u_true_ic)**2)
            loss.backward()
            optimizer.step()
            
            # --- MODIFICATION ICI : Print tous les 500 itérations ---
            if (i + 1) % 500 == 0: 
                msg = f"    [Warmup] Iter {i+1} | Loss IC: {loss.item():.2e}"
                tqdm.write(msg)
        
        torch.save(model.state_dict(), f"{save_dir}/model_post_warmup.pth")
        print("✅ Warmup OK.")

    # --- BOUCLE TEMPORELLE ---
    T_end = Config.T_max
    time_steps = [round(t, 2) for t in np.arange(Config.dt, T_end + Config.dt/1000.0, Config.dt)]
    
    for t_step in time_steps:
        success, _ = train_step_time_window(model, bounds, t_max=t_step, n_iters_main=n_iters_per_step)
        
        if success:
            torch.save({
                't_max': t_step,
                'model_state_dict': model.state_dict()
            }, f"{save_dir}/model_checkpoint_t{t_step}.pth")
        else:
            print("🛑 Arrêt d'urgence : Le modèle ne valide pas le palier.")
            break

    print("\n🏁 Fin du programme.")
    return model