import torch
import torch.nn as nn
import torch.nn.functional as F

from encoding import get_encoder
from activation import trunc_exp
from .renderer import NBDRenderer
from .utils import normalize

class NBDNetwork(NBDRenderer):
    def __init__(self,
                 opt,
                 encoding="frequency",
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

        # sigma network
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.geo_feat_dim = geo_feat_dim
        self.encoder, self.in_dim = get_encoder(encoding, desired_resolution=2048 * bound)

        self.num_basis = opt.num_basis

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

        # color network
        self.num_layers_color = num_layers_color        
        self.hidden_dim_color = hidden_dim_color
        self.encoder_dir, self.in_dim_dir = get_encoder(encoding_dir)
        
        dir_embed_net = []
        for l in range(num_layers_color):
            if l == 0:
                in_dim = self.in_dim_dir + self.geo_feat_dim
            else:
                in_dim = hidden_dim
            dir_embed_net.append(nn.Sequential(nn.Linear(in_dim, hidden_dim, bias=False), 
                                nn.ReLU(inplace=True)))

        self.dir_embed_net = nn.ModuleList(dir_embed_net)
        # self.color_net = nn.Sequential(nn.Linear(hidden_dim, 3, bias=False), nn.relu())

        self.mu_net = nn.Linear(self.geo_feat_dim, self.num_basis*3)
        self.omega_net = nn.Linear(self.geo_feat_dim, self.num_basis)

        if self.opt.hybrid:
            if self.opt.single_radiance:                
                self.amplitude_net = nn.Sequential(nn.Linear(self.geo_feat_dim, 1), nn.Softplus())
                self.radiance_net = nn.Sequential(nn.Linear(hidden_dim, self.num_basis-self.opt.sg_num_basis), nn.Softplus())
            elif self.opt.no_omega:           
                self.amplitude_net = nn.Sequential(nn.Linear(self.geo_feat_dim, self.opt.sg_num_basis), nn.Softplus())
                self.radiance_net = nn.Sequential(nn.Linear(hidden_dim, self.num_basis-self.opt.sg_num_basis), nn.Softplus())
            else:
                self.amplitude_net = nn.Sequential(nn.Linear(self.geo_feat_dim, self.opt.sg_num_basis), nn.Softplus())
                self.radiance_net = nn.Sequential(nn.Linear(hidden_dim, self.num_basis-self.opt.sg_num_basis), nn.Softplus())
            if self.opt.guidance:
                self.color_net = nn.Sequential(nn.Linear(hidden_dim, 3), nn.Sigmoid())
            self.roughness_net = nn.Sequential(nn.Linear(1, 1), nn.Sigmoid())
        elif not self.opt.viewdir_roughness:
            self.roughness_net = nn.Sequential(nn.Linear(self.geo_feat_dim, self.num_basis), nn.Sigmoid())
            self.radiance_net = nn.Sequential(nn.Linear(self.geo_feat_dim, self.num_basis), nn.Softplus())
        else:
            self.roughness_net = nn.Sequential(nn.Linear(hidden_dim, self.num_basis), nn.Sigmoid())
            self.radiance_net = nn.Sequential(nn.Linear(hidden_dim, self.num_basis), nn.Softplus())

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


    # def forward(self, x, d):
    #     # x: [N, 3], in [-bound, bound]
    #     # d: [N, 3], nomalized in [-1, 1]

    #     # sigma
    #     x = self.encoder(x, bound=self.bound)

    #     h = x
    #     for l in range(self.num_layers):
    #         h = self.sigma_net[l](h)
    #         if l != self.num_layers - 1:
    #             h = F.relu(h, inplace=True)

    #     #sigma = F.relu(h[..., 0])
    #     sigma = trunc_exp(h[..., 0])
    #     geo_feat = h[..., 1:]

    #     # color
        
    #     d = self.encoder_dir(d)
    #     h = torch.cat([d, geo_feat], dim=-1)
    #     for l in range(self.num_layers_color):
    #         h = self.color_net[l](h)
    #         if l != self.num_layers_color - 1:
    #             h = F.relu(h, inplace=True)
        
    #     # sigmoid activation for rgb
    #     color = torch.sigmoid(h)

    #     return sigma, color

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

    # allow masked inference
    def color(self, x, d, mask=None, geo_feat=None, **kwargs):
        # x: [N, 3] in [-bound, bound]
        # mask: [N,], bool, indicates where we actually needs to compute rgb.

        if mask is not None:
            mu = torch.ones(mask.shape[0], 3*self.num_basis, dtype=x.dtype, device=x.device) # [N, 3]
            mu = normalize(mu.reshape(mu.shape[0], -1, 3)).reshape(mu.shape[0], -1)
            omega = torch.zeros(mask.shape[0], self.num_basis, dtype=x.dtype, device=x.device) # [N, NB]
            omega = F.softmax(omega, dim=-1) # B, N_B
            roughness = torch.ones(mask.shape[0], self.num_basis, dtype=x.dtype, device=x.device) # [N, NB]
            radiance = torch.zeros(mask.shape[0], self.num_basis, dtype=x.dtype, device=x.device) # [N, NB]
            color = torch.zeros(mask.shape[0], 3, dtype=x.dtype, device=x.device) # [N, NB]
            # in case of empty mask
            if not mask.any():
                return mu, omega, roughness, radiance, color
            x = x[mask]
            d = d[mask]
            geo_feat = geo_feat[mask]
        h_mu = self.mu_net(geo_feat) # B, N_B*3
        h_mu = normalize(h_mu.reshape(h_mu.shape[0], -1, 3)).reshape(h_mu.shape[0], -1)
        h_omega = self.omega_net(geo_feat) # B, N_B
        if self.opt.hybrid:
            h_omega_sg = F.softmax(h_omega[:,:self.opt.sg_num_basis], dim=-1) # B, N_B
            h_omega_dir = F.softmax(h_omega[:,self.opt.sg_num_basis:], dim=-1) # B, N_B
            h_omega = torch.cat([h_omega_sg, h_omega_dir], dim=-1)
        else:
            h_omega = F.softmax(h_omega, dim=-1) # B, N_B

        d = self.encoder_dir(d)
        h = torch.cat([d, geo_feat], dim=-1)
        for l in range(self.num_layers_color):
            h = self.dir_embed_net[l](h)

        if self.opt.guidance:
            h_color = self.color_net(h)
        else:
            h_color = torch.zeros_like(h_mu)

        if self.opt.hybrid:
            if self.opt.single_radiance:
                h_radiance_sg = self.amplitude_net(geo_feat).repeat(1, self.opt.sg_num_basis)
                h_radiance_dir = self.radiance_net(h)
            elif self.opt.no_omega:
                h_radiance_sg = self.amplitude_net(geo_feat)
                h_radiance_dir = self.radiance_net(h)
            else:
                h_radiance_sg = self.amplitude_net(geo_feat)
                h_radiance_dir = self.radiance_net(h)
            h_radiance = torch.cat([h_radiance_sg, h_radiance_dir], dim=-1)
            h_roughness = torch.zeros_like(h_radiance)

        elif not self.opt.viewdir_roughness:
            h_roughness = self.roughness_net(geo_feat) # B, N_b*ch_rad
            h_radiance = self.radiance_net(geo_feat) # B, N_B
        else:
            h_roughness = self.roughness_net(h) # B, N_b*ch_rad
            h_radiance = self.radiance_net(h) # B, N_B

        if mask is not None:
            mu[mask] = h_mu.to(mu.dtype) # fp16 --> fp32
            omega[mask] = h_omega.to(omega.dtype) # fp16 --> fp32
            roughness[mask] = h_roughness.to(roughness.dtype) # fp16 --> fp32
            radiance[mask] = h_radiance.to(radiance.dtype) # fp16 --> fp32
            color[mask] = h_color.to(radiance.dtype) # fp16 --> fp32
        else:
            mu = h_mu
            omega = h_omega
            roughness = h_roughness
            radiance = h_radiance
            color = h_color 

        return mu, omega, roughness, radiance, color

    # optimizer utils
    def get_params(self, lr):

        params = [
            {'params': self.encoder.parameters(), 'lr': lr},
            {'params': self.sigma_net.parameters(), 'lr': lr},
            {'params': self.encoder_dir.parameters(), 'lr': lr},
            {'params': self.dir_embed_net.parameters(), 'lr': lr}, 
            {'params': self.mu_net.parameters(), 'lr': lr}, 
            {'params': self.omega_net.parameters(), 'lr': lr}, 
            {'params': self.roughness_net.parameters(), 'lr': lr}, 
            {'params': self.radiance_net.parameters(), 'lr': lr}, 
            {'params': self.basis_roughness, 'lr': lr}, 
            {'params': self.basis_color, 'lr': lr}, 
        ]
        if self.bg_radius > 0:
            params.append({'params': self.encoder_bg.parameters(), 'lr': lr})
            params.append({'params': self.bg_net.parameters(), 'lr': lr})
        
        return params
