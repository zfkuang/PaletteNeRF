import torch
import torch.nn as nn
import torch.nn.functional as F

from encoding import get_encoder
from activation import trunc_exp
from .renderer import PaletteRenderer
from .utils import normalize

class PaletteNetwork(PaletteRenderer):
    def __init__(self,
                 opt,
                 encoding="hashgrid",
                 encoding_dir="sphere_harmonics",
                 encoding_bg="hashgrid",
                 num_layers=2,
                 hidden_dim=64,
                 geo_feat_dim=15,
                 num_layers_color=3,
                 hidden_dim_color=64,
                 num_layers_bg=2,
                 hidden_dim_bg=64,
                 bound=1,
                 **kwargs,
                 ):
        super().__init__(opt, bound, **kwargs)

        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.geo_feat_dim = geo_feat_dim
        self.encoder, self.in_dim = get_encoder(encoding, desired_resolution=2048 * bound)
        self.encoder_palette, self.in_dim_palette = get_encoder(encoding, desired_resolution=2048 * bound)
        self.encoder_clip, self.in_dim_clip = get_encoder(encoding, desired_resolution=2048 * bound)

        self.num_basis = opt.num_basis

        # sigma network
        sigma_net = []
        for l in range(num_layers):
            if l == 0:
                in_dim = self.in_dim
            else:
                in_dim = hidden_dim
            
            if l == num_layers - 1:
                out_dim = 1 + self.geo_feat_dim # 1 sigma + 15 SH features for color
            else:
                out_dim = hidden_dim
            
            sigma_net.append(nn.Linear(in_dim, out_dim, bias=False))

        self.sigma_net = nn.ModuleList(sigma_net)

        self.num_layers_color = num_layers_color        
        self.hidden_dim_color = hidden_dim_color
        self.encoder_dir, self.in_dim_dir = get_encoder(encoding_dir)

        # view-dependent color network
        # note: this network is named as "color_net" in order to inherit the weights from the vanilla NeRF's color network.
        color_net = []
        for l in range(num_layers_color):
            if l == 0:
                in_dim = self.in_dim_dir + self.geo_feat_dim
            else:
                in_dim = hidden_dim
            
            if l == num_layers_color - 1:
                out_dim = 3 # 3 rgb
            else:
                out_dim = hidden_dim
            
            color_net.append(nn.Linear(in_dim, out_dim, bias=False))

        self.color_net = nn.ModuleList(color_net)

        # diffuse color network
        diff_net = []
        for l in range(num_layers_color):
            if l == 0:
                in_dim = self.geo_feat_dim
            else:
                in_dim = hidden_dim
            
            if l == num_layers_color - 1:
                out_dim = 3 # 3 rgb
            else:
                out_dim = hidden_dim
            
            diff_net.append(nn.Linear(in_dim, out_dim, bias=False))

        self.diff_net = nn.ModuleList(diff_net)

        # palette basis network
        basis_net = []
        for l in range(num_layers):
            if l == 0:
                in_dim = self.in_dim_palette + 3
            else:
                in_dim = hidden_dim
            
            if l == num_layers - 1:
                out_dim = self.geo_feat_dim # 1 sigma + 15 SH features for color
            else:
                out_dim = hidden_dim
            
            basis_net.append(nn.Linear(in_dim, out_dim, bias=False))

        self.basis_net = nn.ModuleList(basis_net)

        # color offset, radiance and color weights
        self.offsets_radiance_net = nn.Linear(self.geo_feat_dim, self.num_basis*3+1)
        self.omega_net = nn.Sequential(nn.Linear(self.geo_feat_dim, self.num_basis, bias=False), nn.Softplus())

        # clip feature network
        if opt.pred_clip:
            clip_net = []
            for l in range(num_layers):
                if l == 0:
                    in_dim = self.in_dim_clip
                else:
                    in_dim = hidden_dim
                
                if l == num_layers - 1:
                    out_dim = opt.clip_dim # 1 sigma + 15 SH features for color
                else:
                    out_dim = hidden_dim
                
                clip_net.append(nn.Linear(in_dim, out_dim, bias=False))
            self.clip_net = nn.ModuleList(clip_net)

        # background network
        if self.bg_radius > 0:
            self.num_layers_bg = num_layers_bg        
            self.hidden_dim_bg = hidden_dim_bg
            self.encoder_bg, self.in_dim_bg = get_encoder(encoding_bg, input_dim=2, num_levels=4, log2_hashmap_size=19, desired_resolution=2048) # much smaller hashgrid 
            
            bg_net = []
            for l in range(num_layers_bg):
                if l == 0:
                    in_dim = self.in_dim_bg + self.in_dim_dir
                else:
                    in_dim = hidden_dim_bg
                
                if l == num_layers_bg - 1:
                    out_dim = 3 # 3 rgb
                else:
                    out_dim = hidden_dim_bg
                
                bg_net.append(nn.Linear(in_dim, out_dim, bias=False))

            self.bg_net = nn.ModuleList(bg_net)
        else:
            self.bg_net = None


    def forward(self, x, d):
        # x: [N, 3], in [-bound, bound]
        # d: [N, 3], nomalized in [-1, 1]

        # Predict sigma
        h = self.encoder(x, bound=self.bound)
        for l in range(self.num_layers):
            h = self.sigma_net[l](h)
            if l != self.num_layers - 1:
                h = F.relu(h, inplace=True)

        sigma = trunc_exp(h[..., 0])
        geo_feat = h[..., 1:].detach()

        # Predict clip feature
        if self.opt.pred_clip:
            h = self.encoder_clip(x, bound=self.bound)
            for l in range(self.num_layers):
                h = self.clip_net[l](h)
                if l != self.num_layers - 1:
                    h = F.relu(h, inplace=True)
            clip_feat = h
        else:
            clip_feat = torch.zeros_like(sigma[...,None].repeat(1, self.opt.clip_dim))
        #sigma = F.relu(h[..., 0])

        # Predict palette basis
        omega, offsets_radiance, view_dep, diffuse = self.color(x, d, geo_feat=geo_feat)

        return sigma, clip_feat, omega, offsets_radiance, view_dep, diffuse

    def density(self, x):
        # x: [N, 3], in [-bound, bound]
        x = self.encoder(x, bound=self.bound)
        h = x
        for l in range(self.num_layers):
            h = self.sigma_net[l](h)
            if l != self.num_layers - 1:
                h = F.relu(h, inplace=True)

        #sigma = F.relu(h[..., 0])
        sigma = trunc_exp(h[..., 0])
        geo_feat = h[..., 1:]

        return {
            'sigma': sigma,
            'geo_feat': geo_feat,
        }

    def background(self, x, d):
        # x: [N, 2], in [-1, 1]

        h = self.encoder_bg(x) # [N, C]
        d = self.encoder_dir(d)

        h = torch.cat([d, h], dim=-1)
        for l in range(self.num_layers_bg):
            h = self.bg_net[l](h)
            if l != self.num_layers_bg - 1:
                h = F.relu(h, inplace=True)
        
        # sigmoid activation for rgb
        rgbs = torch.sigmoid(h)

        return rgbs

    # Allow masked inference
    def color(self, x, d, mask=None, geo_feat=None, **kwargs):
        # x: [N, 3] in [-bound, bound]
        # mask: [N,], bool, indicates where we actually needs to compute rgb.
        
        if mask is not None:
            omega = torch.zeros(x.shape[0], self.num_basis, dtype=x.dtype, device=x.device) # [N, NB]
            omega = F.softmax(omega, dim=-1) # B, N_B
            offsets_radiance = torch.zeros(x.shape[0], 3*self.num_basis, dtype=x.dtype, device=x.device) # [N, 3]
            view_dep = torch.zeros(x.shape[0], 3, dtype=x.dtype, device=x.device) # [N, NB]
            diffuse = torch.zeros(x.shape[0], 3, dtype=x.dtype, device=x.device) # [N, NB]        
            if not mask.any():
                return omega, offsets_radiance, view_dep, diffuse
            x = x[mask]        
            d = d[mask]        
            geo_feat = geo_feat[mask]       
  
        # Diffuse color
        h = geo_feat.detach()
        for l in range(self.num_layers_color):
            h = self.diff_net[l](h)
            if l != self.num_layers_color - 1:
                h = F.relu(h, inplace=True)
        h_diffuse = F.sigmoid(h)

        # View-dependent color
        d = self.encoder_dir(d)
        h = torch.cat([d, geo_feat.detach()], dim=-1)
        for l in range(self.num_layers_color):
            h = self.color_net[l](h)
            if l != self.num_layers_color - 1:
                h = F.relu(h, inplace=True)
        h_view_dep = F.sigmoid(h)
        
        # Palette basis
        h = self.encoder_palette(x, bound=self.bound)
        h = torch.cat([h, h_diffuse.detach()], dim=-1)
        for l in range(self.num_layers):
            h = self.basis_net[l](h)
            if l != self.num_layers - 1:
                h = F.elu(h, inplace=True)
        h_palette_geo_feat = h
        
        h_offsets_radiance = self.offsets_radiance_net(h_palette_geo_feat) # B, N_B*3
        h_omega = self.omega_net(h_palette_geo_feat)+0.05 # B, N_B
        h_omega = h_omega / (h_omega.sum(dim=-1, keepdim=True)) # B, N_B
        
        if mask is not None:
            offsets_radiance[mask] = h_offsets_radiance.to(offsets_radiance.dtype) # fp16 --> fp32
            omega[mask] = h_omega.to(omega.dtype) # fp16 --> fp32
            view_dep[mask] = h_view_dep.to(view_dep.dtype) # fp16 --> fp32
            diffuse[mask] = h_diffuse.to(diffuse.dtype) # fp16 --> fp32
        else:
            offsets_radiance = h_offsets_radiance
            omega = h_omega
            view_dep = h_view_dep 
            diffuse = h_diffuse 
        
        return omega, offsets_radiance, view_dep, diffuse

    # Optimizer utils
    def get_params(self, lr):

        params = [
            {'params': self.encoder.parameters(), 'lr': lr},
            {'params': self.encoder_palette.parameters(), 'lr': lr},
            {'params': self.encoder_clip.parameters(), 'lr': lr},
            {'params': self.sigma_net.parameters(), 'lr': lr},
            {'params': self.encoder_dir.parameters(), 'lr': lr},
            {'params': self.color_net.parameters(), 'lr': lr}, 
            {'params': self.diff_net.parameters(), 'lr': lr}, 
            {'params': self.offsets_radiance_net.parameters(), 'lr': lr}, 
            {'params': self.omega_net.parameters(), 'lr': lr}, 
            {'params': self.basis_color, 'lr': lr}, 
            # {'params': self.basis_roughness, 'lr': lr}, 
        ]

        if self.opt.use_initialization_from_rgbxy:
            params.append({'params': self.hist_weights, 'lr': lr})
        if self.bg_radius > 0:
            params.append({'params': self.encoder_bg.parameters(), 'lr': lr})
            params.append({'params': self.bg_net.parameters(), 'lr': lr})
        
        if self.opt.pred_clip:
            params.append({'params': self.clip_net.parameters(), 'lr': lr})
            
        return params
