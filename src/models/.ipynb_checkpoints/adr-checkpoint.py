import torch
import torch.nn as nn
import numpy as np

class MultiScaleFourierFeatureEncoding(nn.Module):
    """
    Module de projection Fourier multi-échelles.
    """
    def __init__(self, in_features, num_frequencies, scales):
        super().__init__()
        self.scales = scales
        n_scales = len(scales)
        
        if n_scales == 0: 
            self.register_parameter("B", None)
            return

        freqs_per_scale = num_frequencies // n_scales
        all_B = []
        for s in scales:
            B_s = torch.randn(in_features, freqs_per_scale) * s
            all_B.append(B_s)

        remainder = num_frequencies - (freqs_per_scale * n_scales)
        if remainder > 0:
            med_scale = np.median(scales)
            all_B.append(torch.randn(in_features, remainder) * med_scale)

        self.B = nn.Parameter(torch.cat(all_B, dim=1), requires_grad=False)

    def forward(self, x):
        # x [N, in] @ B [in, num_freq] -> [N, num_freq]
        x_proj = 2 * np.pi * x @ self.B
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class PI_DeepONet_ADR(nn.Module):
    """
    Deep Operator Network (DeepONet) piloté par dictionnaire (YAML).
    """
    def __init__(self, full_cfg):          
        super().__init__()
        
        # --- 1. Extraction des sous-configs ---
        m_cfg = full_cfg['model']
        g_cfg = full_cfg['geometry']
        p_cfg = full_cfg['physics_ranges']

        # Dimensions
        self.input_branch_dim = 8  # v, D, mu, type, A, x0, sigma, k
        self.input_trunk_dim = 2   # x, t
        
        latent_dim = m_cfg['latent_dim']
        num_fourier = m_cfg['nFourier']
        scales_fourier = m_cfg['sFourier']

        # --- 2. Buffers de Normalisation ---
        self.register_buffer('lb_geom', torch.tensor([g_cfg['x_min'], 0.0], dtype=torch.float32))
        self.register_buffer('ub_geom', torch.tensor([g_cfg['x_max'], g_cfg['T_max']], dtype=torch.float32))

        # Physique (Standardization)
        self.register_buffer('v_min', torch.tensor(p_cfg['v'][0]))
        self.register_buffer('v_max', torch.tensor(p_cfg['v'][1]))
        self.register_buffer('D_min', torch.tensor(p_cfg['D'][0]))
        self.register_buffer('D_max', torch.tensor(p_cfg['D'][1]))
        self.register_buffer('mu_min', torch.tensor(p_cfg['mu'][0]))
        self.register_buffer('mu_max', torch.tensor(p_cfg['mu'][1]))

        # Scaling IC
        self.register_buffer('A_scale', torch.tensor(max(abs(p_cfg['A'][0]), abs(p_cfg['A'][1]))))      
        self.register_buffer('sigma_scale', torch.tensor(max(abs(p_cfg['sigma'][0]), abs(p_cfg['sigma'][1])))) 
        self.register_buffer('k_scale', torch.tensor(max(abs(p_cfg['k'][0]), abs(p_cfg['k'][1]))))      

        # --- 3. Construction Trunk ---
        if num_fourier > 0:
            self.trunk_encoder = MultiScaleFourierFeatureEncoding(self.input_trunk_dim, num_fourier, scales_fourier)
            trunk_in_dim_actual = num_fourier * 2
        else:
            self.trunk_encoder = None
            trunk_in_dim_actual = self.input_trunk_dim 

        self.activation = nn.SiLU() 
        self.trunk_input_map = nn.Linear(trunk_in_dim_actual, latent_dim)

        # --- 4. Construction Branch & Trunk Layers ---
        branch_hidden = [m_cfg['branch_width']] * m_cfg['branch_depth']
        self.branch_net = self._build_branch_net(self.input_branch_dim, latent_dim, branch_hidden)
        
        self.trunk_layers_list = nn.ModuleList()
        self.branch_transform = nn.ModuleList()

        # Construction dynamique selon trunk_depth
        for _ in range(m_cfg['trunk_depth']):
            self.trunk_layers_list.append(nn.Linear(latent_dim, latent_dim))
            self.branch_transform.append(nn.Linear(latent_dim, 2 * latent_dim))

        self.final_layer = nn.Linear(latent_dim, 1)
        
        self.apply(self._init_weights)
        with torch.no_grad(): 
            self.final_layer.weight.mul_(0.01)

    def _build_branch_net(self, in_dim, out_dim, hidden_list):
        layers = []
        dims = [in_dim] + hidden_list
        for i in range(len(dims)-1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            layers.append(self.activation)
        layers.append(nn.Linear(dims[-1], out_dim))
        return nn.Sequential(*layers)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            nn.init.zeros_(m.bias)

    def normalize_tensor(self, x, min_val, max_val):
        return 2.0 * (x - min_val) / (max_val - min_val + 1e-8) - 1.0

    def forward(self, params, xt):
        # Normalisation
        p_v  = self.normalize_tensor(params[:, 0:1], self.v_min, self.v_max)
        p_D  = self.normalize_tensor(params[:, 1:2], self.D_min, self.D_max)
        p_mu = self.normalize_tensor(params[:, 2:3], self.mu_min, self.mu_max)
        p_type  = params[:, 3:4]
        p_A     = params[:, 4:5] / self.A_scale 
        p_x0    = torch.zeros_like(params[:, 5:6]) 
        p_sigma = params[:, 6:7] / self.sigma_scale
        p_k     = params[:, 7:8] / self.k_scale
        
        params_norm = torch.cat([p_v, p_D, p_mu, p_type, p_A, p_x0, p_sigma, p_k], dim=1)
        xt_norm = self.normalize_tensor(xt, self.lb_geom, self.ub_geom)

        # Processing
        xt_embed = self.trunk_encoder(xt_norm) if self.trunk_encoder is not None else xt_norm
        context_B = self.branch_net(params_norm) 
        Z = self.activation(self.trunk_input_map(xt_embed)) 

        for layer_T, layer_B in zip(self.trunk_layers_list, self.branch_transform):
            Z_trunk = layer_T(Z)
            UV = layer_B(context_B)
            U, V = torch.split(UV, Z.shape[1], dim=1)
            Z = self.activation((1 - Z_trunk) * U + Z_trunk * V)

        return self.final_layer(Z)