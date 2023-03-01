from importlib.metadata import requires
import math
import trimesh
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import raymarching
from nerf.utils import custom_meshgrid
from .utils import normalize, rgb_to_hsv, hsv_to_rgb
from nerf.utils import srgb_to_linear

# Utility functions 

def cos_distance(x, y, safemode=False):
    result = 1-(normalize(x) * normalize(y)).sum(dim=-1)
    if safemode:
        result = result * torch.minimum(x.norm(dim=-1), y.norm(dim=-1)).clip(max=0.2)*5
    return result

def isclose(x, val, threshold = 1e-6):
    return torch.abs(x - val) <= threshold

def safe_pow(x, p):
    sqrt_in = torch.relu(torch.where(isclose(x, 0.0), torch.ones_like(x) * 1e-6, x))
    return torch.pow(sqrt_in, p)

def safe_linear_to_srgb(x):
    sqrt_in = torch.relu(torch.where(isclose(x, 0.0), torch.ones_like(x) * 1e-6, x))
    return 1.055 * sqrt_in ** 0.41666 - 0.055

def sample_pdf(bins, weights, n_samples, det=False):
    # This implementation is from NeRF
    # bins: [B, T], old_z_vals
    # weights: [B, T - 1], bin weights.
    # return: [B, n_samples], new_z_vals

    # Get pdf
    weights = weights + 1e-5  # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)
    # Take uniform samples
    if det:
        u = torch.linspace(0. + 0.5 / n_samples, 1. - 0.5 / n_samples, steps=n_samples).to(weights.device)
        u = u.expand(list(cdf.shape[:-1]) + [n_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [n_samples]).to(weights.device)

    # Invert CDF
    u = u.contiguous()
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.max(torch.zeros_like(inds - 1), inds - 1)
    above = torch.min((cdf.shape[-1] - 1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # (B, n_samples, 2)

    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = (cdf_g[..., 1] - cdf_g[..., 0])
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u - cdf_g[..., 0]) / denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

    return samples

def plot_pointcloud(pc, color=None):
    # pc: [N, 3]
    # color: [N, 3/4]
    print('[visualize points]', pc.shape, pc.dtype, pc.min(0), pc.max(0))
    pc = trimesh.PointCloud(pc, color)
    # axis
    axes = trimesh.creation.axis(axis_length=4)
    # sphere
    sphere = trimesh.creation.icosphere(radius=1)
    trimesh.Scene([pc, axes, sphere]).show()


# Controller for regional appearance editing
class RegionEdit(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.mean_xyz = None
        self.mean_clip = None
        self.std_xyz = 1
        self.std_clip = 1
        self.weight_mode = False
        self.delta_hsv = torch.zeros(self.opt.num_basis, 3)
        self.delta_hsv[...,1:3] = 1

    def update_cent(self, mean_xyz=None, mean_clip=None):     
        self.mean_xyz = None if mean_xyz is None else mean_xyz[None,...]
        self.mean_clip = None if mean_clip is None else mean_clip[None,...]
    
    def update_std(self, std_xyz=None, std_clip=None):
        if std_xyz is not None:
            self.std_xyz = std_xyz
        if std_clip is not None:
            self.std_clip = std_clip

    def update_delta(self, rgb_orig, rgb_new):
        '''
            Given the original palettes' rgb and modified palettes' rgb, calculating the change in HSV Space.
            More specifically, difference in H channel and scales in S,V channel
        '''
        if rgb_orig.device != self.delta_hsv.device:
            self.delta_hsv = self.delta_hsv.type_as(rgb_orig)
        rgb_all = torch.cat([rgb_orig, rgb_new], dim=0)
        hsv_all = rgb_to_hsv(rgb_all)
        hsv_orig = hsv_all[:self.opt.num_basis]
        hsv_new = hsv_all[self.opt.num_basis:]

        self.delta_hsv[:, 0] = torch.fmod((hsv_new[:,0]-hsv_orig[:,0]+360), 360)
        self.delta_hsv[:, 1] = (hsv_new[:,1]/hsv_orig[:,1]+1e-9)
        self.delta_hsv[:, 2] = (hsv_new[:,2]/hsv_orig[:,2]+1e-9)

    def forward(self, rgbs, xyz=None, clip_feat=None):
        hsv = rgb_to_hsv(rgbs)
        if rgbs.device != self.delta_hsv.device:
            self.delta_hsv = self.delta_hsv.type_as(rgbs)
        weight = torch.ones_like(rgbs[...,0:1,0])
        
        # Euclidean distance based filtering
        if xyz is not None and self.mean_xyz is not None:
            weight *= torch.exp(-((xyz-self.mean_xyz)**2.).sum(dim=-1, keepdim=True)/self.std_xyz)
            
        # Semantic map based filtering
        if clip_feat is not None and self.mean_clip is not None:
            #temp = ((clip_feat-self.mean_clip)**2).sum(dim=-1, keepdim=True)
            weight *= torch.exp(-((clip_feat-self.mean_clip)**2.).sum(dim=-1, keepdim=True)/self.std_clip)
            # temp /= (self.mean_clip**2+1e-6).sum(dim=-1, keepdim=True)
            #weight *= (temp < self.std_clip).float() # vtorch.exp(-(temp/self.std_clip))

        hsv_new = hsv.clone()
        hsv_new[...,0] = torch.fmod((hsv[...,0]+self.delta_hsv[...,0]+360), 360)
        hsv_new[...,1] = torch.clip((hsv[...,1]*self.delta_hsv[...,1]), 0)
        hsv_new[...,2] = torch.clip((hsv[...,2]*self.delta_hsv[...,2]), 0)

        rgb_new = hsv_to_rgb(hsv_new)
        if self.weight_mode:
            return weight[...,None].repeat(1, self.opt.num_basis, 3) 
        else:
            return torch.lerp(rgbs, rgb_new, weight[...,None])
        
        
# Solver of User-guided Photorealistic style transfer 
class Stylizer(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        dI = torch.zeros(opt.num_basis, dtype=torch.float32)
        self.dI = torch.nn.Parameter(dI, requires_grad=True)
        dP = torch.zeros(1, opt.num_basis, 3, dtype=torch.float32)
        self.dP = torch.nn.Parameter(dP, requires_grad=True)
        ddelta = torch.eye(3, dtype=torch.float32)[None,:,:].repeat(opt.num_basis, 1, 1) # N_p x 3 x 3
        self.ddelta = torch.nn.Parameter(ddelta, requires_grad=True)
        
    def ARAP_loss(self):
        I = torch.eye(3, dtype=torch.float32, device=self.ddelta.device)[None,:,:]
        return ((torch.bmm(self.ddelta, self.ddelta.transpose(1, 2)) - I)**2).sum()
    
    def forward(self, radiance, omega, palette, delta, dir=None):
        assert(not self.opt.multiply_delta and self.opt.separate_radiance)
        
        prefix = delta.shape[:-2]
        radiance = radiance.reshape(-1, 1, 1)
        omega = omega.reshape(-1, self.opt.num_basis, 1)
        palette = palette.reshape(-1, self.opt.num_basis, 3)
        delta = delta.reshape(-1, self.opt.num_basis, 3)
        
        palette = (palette+self.dP) # N x N-p x 3
        delta = torch.einsum("npi, pij->npj", delta, self.ddelta) # N x N_p x 3
    
        basis_rgb = ((F.softplus(radiance).repeat(1, self.opt.num_basis, 1)+self.dI[None,:,None]).clamp(0)*(palette+delta)).clamp(0, 1) # N x N_p x 3
        basis_rgb = omega*basis_rgb # N, N_p, 3
        rgbs = basis_rgb.sum(dim=-2) # N, 3
        
        if dir is not None:
            rgbs += dir.detach() # N, 3
        return rgbs.reshape(*prefix, 3)
    
    
class PaletteRenderer(nn.Module):
    def __init__(self,
                 opt,
                 bound=1,
                 cuda_ray=False,
                 density_scale=1, # scale up deltas (or sigmas), to make the density grid more sharp. larger value than 1 usually improves performance.
                 min_near=0.2,
                 density_thresh=0.01,
                 bg_radius=-1,
                 ):
        super().__init__()

        self.bound = bound
        self.cascade = 1 + math.ceil(math.log2(bound))
        self.grid_size = 128
        self.density_scale = density_scale
        self.min_near = min_near
        self.density_thresh = density_thresh
        self.bg_radius = bg_radius # radius of the background sphere.
        self.num_basis = opt.num_basis
        self.freeze_basis_color = opt.use_initialization_from_rgbxy
        self.require_smooth_loss = False
        self.color_weight = 0
        self.opt = opt
        self.edit = None
        self.stylizer = None
        
        self.view_dep_weight = 1
        self.offsets_weight = 1
        
        # prepare aabb with a 6D tensor (xmin, ymin, zmin, xmax, ymax, zmax)
        # NOTE: aabb (can be rectangular) is only used to generate points, we still rely on bound (always cubic) to calculate density grid and hashing.
        aabb_train = torch.FloatTensor([-bound, -bound, -bound, bound, bound, bound])
        aabb_infer = aabb_train.clone()
        self.register_buffer('aabb_train', aabb_train)
        self.register_buffer('aabb_infer', aabb_infer)

        if opt.test or not opt.use_initialization_from_rgbxy:
            # No initialization from the rgbxy palette extraction method.
            self.basis_color = torch.zeros([self.num_basis, 3])+0.5
            self.basis_color = nn.Parameter(self.basis_color, requires_grad=True)
        else:
            # Palette initialized from extraction method.
            # Parameter reserved to be initialized later.
            self.basis_color = None

        # extra state for cuda raymarching
        self.cuda_ray = cuda_ray
        if cuda_ray:
            # density grid
            density_grid = torch.zeros([self.cascade, self.grid_size ** 3]) # [CAS, H * H * H]
            density_bitfield = torch.zeros(self.cascade * self.grid_size ** 3 // 8, dtype=torch.uint8) # [CAS * H * H * H // 8]
            self.register_buffer('density_grid', density_grid)
            self.register_buffer('density_bitfield', density_bitfield)
            self.mean_density = 0
            self.iter_density = 0
            # step counter
            step_counter = torch.zeros(16, 2, dtype=torch.int32) # 16 is hardcoded for averaging...
            self.register_buffer('step_counter', step_counter)
            self.mean_count = 0
            self.local_step = 0

    def initialize_palette(self, color_list=None, hist_weights=None):
        if color_list is None:
            if self.basis_color is None:
                self.basis_color = torch.zeros([self.num_basis, 3]) + 0.5
                self.basis_color = nn.Parameter(self.basis_color, requires_grad=True)
        else:
            self.basis_color = torch.zeros([self.num_basis, 3])
            for i, color in enumerate(color_list):
                if self.opt.color_space == "linear":
                    self.basis_color[i] = srgb_to_linear(torch.FloatTensor(color))
                else:
                    self.basis_color[i] = torch.FloatTensor(color)
            self.basis_color = nn.Parameter(self.basis_color, requires_grad=True)

        self.basis_color_origin = nn.Parameter(self.basis_color.data, requires_grad=False)
        
        # initialize hist weights (to generate blending weight supervision)
        if hist_weights is not None:
            self.hist_weights = torch.from_numpy(hist_weights).float()
            self.hist_weights = self.hist_weights.permute(3, 0, 1, 2).unsqueeze(0)
            self.hist_weights = nn.Parameter(self.hist_weights, requires_grad=False)

    def forward(self, x, d):
        raise NotImplementedError()

    # separated density and color query (can accelerate non-cuda-ray mode.)
    def density(self, x):
        raise NotImplementedError()

    def color(self, x, d, mask=None, **kwargs):
        raise NotImplementedError()

    def reset_extra_state(self):
        if not self.cuda_ray:
            return 
        # density grid
        self.density_grid.zero_()
        self.mean_density = 0
        self.iter_density = 0
        # step counter
        self.step_counter.zero_()
        self.mean_count = 0
        self.local_step = 0
        
    def run(self, rays_o, rays_d, num_steps=128, upsample_steps=128, bg_color=None, perturb=False, gui_mode=False, test_mode=False, **kwargs):
        # TODO: add pure pytorch version
        raise ValueError("Pure pytorch version is not available for now.")

    def run_cuda(self, rays_o, rays_d, dt_gamma=0, bg_color=None, perturb=False, force_all_rays=False, 
                 max_steps=1024, T_thresh=1e-4, gui_mode=False, **kwargs):
        # rays_o, rays_d: [B, N, 3], assumes B == 1
        # return: image: [B, N, 3], depth: [B, N]

        prefix = rays_o.shape[:-1]
        rays_o = rays_o.contiguous().view(-1, 3)
        rays_d = rays_d.contiguous().view(-1, 3)

        N = rays_o.shape[0] # N = B * N, in fact
        device = rays_o.device

        # pre-calculate near far
        nears, fars = raymarching.near_far_from_aabb(rays_o, rays_d, self.aabb_train if self.training else self.aabb_infer, self.min_near)

        # mix background color
        if self.bg_radius > 0:
            # use the bg model to calculate bg_color
            sph = raymarching.sph_from_ray(rays_o, rays_d, self.bg_radius) # [N, 2] in [-1, 1]
            bg_color = self.background(sph, rays_d) # [N, 3]
        elif bg_color is None:
            bg_color = 1

        results = {}

        if self.training:
            # setup counter
            counter = self.step_counter[self.local_step % 16]
            counter.zero_() # set to 0
            self.local_step += 1

            xyzs, dirs, deltas, rays = raymarching.march_rays_train(rays_o, rays_d, self.bound, self.density_bitfield, self.cascade, self.grid_size, nears, fars, counter, self.mean_count, perturb, 128, force_all_rays, dt_gamma, max_steps)
            M = xyzs.shape[0]

            # Predict sigma, feature and palette basis from network
            sigmas, clip_feat, omega, offsets_radiance, view_dep, diffuse = self(xyzs, dirs)
            
            offsets, radiance = offsets_radiance[...,:-1], offsets_radiance[...,-1:]
            sigmas = self.density_scale * sigmas
            sigmas = sigmas.detach()

            radiance = radiance.reshape(M, 1, 1)
            offsets = offsets.reshape(M, self.num_basis, 3)
            omega = omega.reshape(M, self.num_basis, 1)
            view_dep = view_dep.reshape(M, 3)
            diffuse = diffuse.reshape(M, 3)
            clip_feat = clip_feat.reshape(M, self.opt.clip_dim)

            if self.opt.no_delta:
                d_color = d_color*0
            if self.opt.no_dir:
                dir_color = dir_color*0

            basis_color = self.basis_color[None,:,:].clamp(0, 1)
            if self.freeze_basis_color:
                basis_color = basis_color.detach()
                
            # Compositing palette basis
            final_color = (F.softplus(radiance)*(basis_color+offsets))
            basis_rgb = omega*final_color # (N_rays, N_sample, N_basis, 3)
            rgbs = basis_rgb.sum(dim=-2) + view_dep.detach() # (N_rays, N_samples_, 3)
            direct_rgb = diffuse+view_dep # (N_rays, N_samples_, 3)

            weights_sum, depth, image = raymarching.composite_rays_train(sigmas, rgbs, deltas, rays, T_thresh)

            # calculating norms as losses
            omega_sparsity = omega[...,0].sum(dim=-1, keepdim=True) / ((omega[...,0]**2).sum(dim=-1, keepdim=True)+1e-6)-1 # N_rays, N_sample, 1
            offsets_norm = (offsets**2).sum(dim=-1).sum(dim=-1, keepdim=True) # (N_rays, N_samples_, 1)
            view_dep_norm = (view_dep**2).sum(dim=-1, keepdim=True) # (N_rays, N_samples_, 1)
            if self.require_smooth_loss:
                # add perturbations on points
                xyzs_diff = (xyzs + torch.rand_like(xyzs) * self.bound * 0.03).clamp(-self.bound, self.bound)
                xyzs_weight = (xyzs-xyzs_diff).norm(dim=-1, keepdim=True)**2 / (self.bound * 0.07)**2 
                _, clip_feat_diff, omega_diff, _, _, diffuse_diff =  self(xyzs_diff, dirs)
                omega_diff = omega_diff.reshape(M, self.num_basis, 1)
                diffuse_diff = diffuse_diff.reshape(M, 3)
                
                if self.opt.pred_clip and self.opt.sigma_clip > 0:
                    clip_weight = (clip_feat-clip_feat_diff).norm(dim=-1, keepdim=True) / self.opt.sigma_clip
                else:
                    clip_weight = 0

                rgb_weight = (diffuse-diffuse_diff).norm(dim=-1, keepdim=True)**2 / self.opt.sigma_color # (N_rays, N_samples_, 1)
                
                smooth_weight = torch.exp(- xyzs_weight - rgb_weight - clip_weight).detach()
                smooth_norm = ((omega_diff-omega)[...,0]**2).sum(dim=-1, keepdim=True) * smooth_weight
                if self.opt.pred_clip:
                    smooth_norm += ((clip_feat_diff-clip_feat)**2).sum(dim=-1, keepdim=True) * smooth_weight
                    
            else: 
                smooth_norm = torch.zeros_like(omega_sparsity)

            # Concatenate all buffers and composite them in a single pass 
            all_buffer = torch.cat([omega_sparsity, view_dep_norm, offsets_norm, smooth_norm, 
                                    view_dep, direct_rgb, diffuse, clip_feat, omega[...,0]], axis=-1)
            all_map = raymarching.composite_rays_flex_train(sigmas, all_buffer, deltas, rays, T_thresh)
            
            # Dispatch the composed pixel maps
            omega_sparsity_map = all_map[...,0:1]
            view_dep_norm_map = all_map[...,1:2]
            offsets_norm_map = all_map[...,2:3]
            smooth_norm_map = all_map[...,3:4]
            view_dep_rgb_map = all_map[...,4:7]
            direct_rgb_map = all_map[...,7:10]
            diffuse_rgb_map = all_map[...,10:13]
            clip_feat_map = all_map[...,13:13+self.opt.clip_dim]
            basis_acc_map = all_map[...,13+self.opt.clip_dim:13+self.opt.clip_dim+self.opt.num_basis]

            image = image + (1 - weights_sum).unsqueeze(-1) * bg_color
            depth = torch.clamp(depth - nears, min=0) / (fars - nears)
            image = image.view(*prefix, 3)
            depth = depth.view(*prefix)
            direct_rgb_map = direct_rgb_map + (1 - weights_sum).unsqueeze(-1) * bg_color
            
            omega_sparsity_map = omega_sparsity_map.view(*prefix)
            view_dep_norm_map = view_dep_norm_map.view(*prefix)
            offsets_norm_map = offsets_norm_map.view(*prefix)
            smooth_norm_map = smooth_norm_map.view(*prefix)
            view_dep_rgb_map = view_dep_rgb_map.view(*prefix, 3)
            direct_rgb_map = direct_rgb_map.view(*prefix, 3)
            diffuse_rgb_map = diffuse_rgb_map.view(*prefix, 3)
            clip_feat_map = clip_feat_map.view(*prefix, self.opt.clip_dim)
            basis_acc_map = basis_acc_map.view(*prefix, self.num_basis)

            results['depth'] = depth
            results['image'] = image
            results['weights_sum'] = weights_sum
            
            results['smooth_norm'] = smooth_norm_map
            results['omega_sparsity'] = omega_sparsity_map
            results['view_dep_norm'] = view_dep_norm_map
            results['offsets_norm'] = offsets_norm_map
            
            results['direct_rgb'] = direct_rgb_map
            results['view_dep_rgb'] = view_dep_rgb_map
            results['diffuse_rgb'] = diffuse_rgb_map
            
            results['clip_feat'] = clip_feat_map
            results['basis_acc'] = basis_acc_map
        else:
            # allocate outputs 
            # if use autocast, must init as half so it won't be autocasted and lose reference.
            #dtype = torch.half if torch.is_autocast_enabled() else torch.float32
            # output should always be float32! only network inference uses half.
            dtype = torch.float32

            weights_sum = torch.zeros(N, dtype=dtype, device=device)
            depth = torch.zeros(N, dtype=dtype, device=device)
            image = torch.zeros(N, 3, dtype=dtype, device=device)
            
            view_dep_rgb_map = torch.zeros(N, 3, dtype=dtype, device=device)
            direct_rgb_map = torch.zeros(N, 3, dtype=dtype, device=device)
            basis_rgb_map = torch.zeros(N, 3*self.opt.num_basis, dtype=dtype, device=device)
            unscaled_basis_rgb_map = torch.zeros(N, 3*self.opt.num_basis, dtype=dtype, device=device)
            basis_acc_map = torch.zeros(N, self.opt.num_basis, dtype=dtype, device=device)
            clip_feat_map = torch.zeros(N, self.opt.clip_dim, dtype=dtype, device=device)

            n_alive = N
            rays_alive = torch.arange(n_alive, dtype=torch.int32, device=device) # [N]
            rays_t = nears.clone() # [N]

            step = 0
            
            while step < max_steps:

                # count alive rays 
                n_alive = rays_alive.shape[0]
                
                # exit loop
                if n_alive <= 0:
                    break

                # decide compact_steps
                n_step = max(min(N // n_alive, 8), 1)

                xyzs, dirs, deltas = raymarching.march_rays(n_alive, n_step, rays_alive, rays_t, rays_o, rays_d, self.bound, self.density_bitfield, self.cascade, self.grid_size, nears, fars, 128, perturb if step == 0 else False, dt_gamma, max_steps)
                M = xyzs.shape[0]

                sigmas, clip_feat, omega, offsets_radiance, view_dep, diffuse = self(xyzs, dirs)
                offsets, radiance = offsets_radiance[...,:-1], offsets_radiance[...,-1:]
                
                radiance = radiance.reshape(M, 1, 1)
                offsets = offsets.reshape(M, self.num_basis, 3)
                omega = omega.reshape(M, self.num_basis, 1)
                view_dep = view_dep.reshape(M, 3)
                diffuse = diffuse.reshape(M, 3)
                clip_feat = clip_feat.reshape(M, self.opt.clip_dim)

                if self.opt.no_delta:
                    d_color = d_color*0
                if self.opt.no_dir:
                    dir_color = dir_color*0
                        

                basis_color = self.basis_color[None,:,:].clamp(0, 1)
                
                if self.stylizer is not None: # Photorealistic style transfer
                    rgbs = self.stylizer(radiance, omega, basis_color, offsets, view_dep)
                else:
                    final_color = (F.softplus(radiance)*(basis_color+self.offsets_weight * offsets))# .clamp(0, 1)
                    unscaled_final_color = (basis_color+offsets)# .clamp(0, 1)

                    if self.edit is not None: # filtered appearence editing
                        final_color = self.edit(final_color, xyzs, clip_feat)
                    
                    basis_rgb = omega*final_color # .clamp(0, 1) # N_rays, N_sample, N_basis, 3
                    unscaled_basis_rgb = unscaled_final_color

                    # adding view dependent color with editable scale
                    rgbs = basis_rgb.sum(dim=-2) + self.view_dep_weight * view_dep # (N_rays, N_samples_, 3)

                # density_outputs = self.density(xyzs) # [M,], use a dict since it may include extra things, like geo_feat for rgb.
                # sigmas = density_outputs['sigma']
                # rgbs = self.color(xyzs, dirs, **density_outputs)
                sigmas = self.density_scale * sigmas

                if not gui_mode:
                    ### auxiliary informations of palettes for debugging

                    direct_rgb = diffuse+view_dep
                    basis_rgb = basis_rgb.reshape(M, self.opt.num_basis*3) # (N_rays, N_samples_, N_basis*3)
                    unscaled_basis_rgb = unscaled_basis_rgb.reshape(M, self.opt.num_basis*3) # (N_rays, N_samples_, N_basis*3)
                    
                    raymarching.composite_rays_flex(n_alive, n_step, 3, rays_alive, rays_t, sigmas, direct_rgb, deltas, weights_sum, direct_rgb_map, T_thresh)
                    raymarching.composite_rays_flex(n_alive, n_step, 3, rays_alive, rays_t, sigmas, view_dep, deltas, weights_sum, view_dep_rgb_map, T_thresh)
                    raymarching.composite_rays_flex(n_alive, n_step, self.opt.num_basis, rays_alive, rays_t, sigmas, omega, deltas, weights_sum, basis_acc_map, T_thresh)
                    raymarching.composite_rays_flex(n_alive, n_step, self.opt.num_basis*3, rays_alive, rays_t, sigmas, basis_rgb, deltas, weights_sum, basis_rgb_map, T_thresh)
                    raymarching.composite_rays_flex(n_alive, n_step, self.opt.num_basis*3, rays_alive, rays_t, sigmas, unscaled_basis_rgb, deltas, weights_sum, unscaled_basis_rgb_map, T_thresh)
                 
                # clip feature map
                raymarching.composite_rays_flex(n_alive, n_step, self.opt.clip_dim, rays_alive, rays_t, sigmas, clip_feat, deltas, weights_sum, clip_feat_map, T_thresh)

                ### !!! IMPORTANT !!! make sure this composite rays function is executed after all composite rays flex operations
                # Since this step will modify rays_alive
                raymarching.composite_rays(n_alive, n_step, rays_alive, rays_t, sigmas, rgbs, deltas, weights_sum, depth, image, T_thresh)

                rays_alive = rays_alive[rays_alive >= 0]

                step += n_step

            image = image + (1 - weights_sum).unsqueeze(-1) * bg_color
            depth_origin = depth.clone()
            depth = torch.clamp(depth - nears, min=0) / (fars - nears)
            image = image.view(*prefix, 3)
            depth = depth.view(*prefix)
            depth_origin = depth_origin.view(*prefix)
            results['depth'] = depth
            results['depth_origin'] = depth_origin
            results['image'] = image
            results['weights_sum'] = weights_sum

            clip_feat_map = clip_feat_map.view(*prefix, self.opt.clip_dim)
            results['clip_feat'] = clip_feat_map

            if not gui_mode:
                direct_rgb_map = direct_rgb_map + (1 - weights_sum).unsqueeze(-1) * bg_color
                view_dep_rgb_map = view_dep_rgb_map.view(*prefix, 3)
                direct_rgb_map = direct_rgb_map.view(*prefix, 3)
                basis_acc_map = basis_acc_map.view(*prefix, self.num_basis)
                basis_rgb_map = basis_rgb_map.view(*prefix, self.num_basis*3)
                unscaled_basis_rgb_map = unscaled_basis_rgb_map.view(*prefix, self.num_basis*3)
                results['direct_rgb'] = direct_rgb_map
                results['view_dep_rgb'] = view_dep_rgb_map
                results['basis_rgb'] = basis_rgb_map
                results['unscaled_basis_rgb'] = unscaled_basis_rgb_map
                results['basis_acc'] = basis_acc_map

        return results

    def render(self, rays_o, rays_d, staged=False, max_ray_batch=4096, test_mode=False, gui_mode=False, **kwargs):
        # rays_o, rays_d: [B, N, 3], assumes B == 1
        # return: pred_rgb: [B, N, 3]

        if self.cuda_ray:
            _run = self.run_cuda
        else:
            _run = self.run

        B, N = rays_o.shape[:2]
        device = rays_o.device

        # never stage when cuda_ray
        if staged and not self.cuda_ray:
            pass 
            # TODO: add pure pytorch version
            # results = {}
            # for b in range(B):
            #     head = 0
            #     while head < N:
            #         tail = min(head + max_ray_batch, N)
            #         results_ = _run(rays_o[b:b+1, head:tail], rays_d[b:b+1, head:tail], test_mode=test_mode, gui_mode=gui_mode, **kwargs)
            #         # results_.pop("weights_sum")
            #         for k, v in results_.items():
            #             if v is None:
            #                 continue
            #             if k == 'weights_sum':
            #                 v = v[None,...]
            #             if k not in results.keys():
            #                 if v.ndim == 2:
            #                     results[k] = torch.empty((B, N), device=device)
            #                 else:
            #                     results[k] = torch.empty((B, N, v.shape[-1]), device=device)
            #             results[k][b:b+1, head:tail] = v
            #         head += max_ray_batch
        else:
            results = _run(rays_o, rays_d, gui_mode=gui_mode, **kwargs)

        return results