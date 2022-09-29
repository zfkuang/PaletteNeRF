import math
import trimesh
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import raymarching
from nerf.utils import custom_meshgrid
from .utils import normalize
from nerf.utils import srgb_to_linear

def cos_distance(x, y):
    return 1-(normalize(x) * normalize(y)).sum(dim=-1)

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
        self.color_weight = 0
        self.opt = opt

        # prepare aabb with a 6D tensor (xmin, ymin, zmin, xmax, ymax, zmax)
        # NOTE: aabb (can be rectangular) is only used to generate points, we still rely on bound (always cubic) to calculate density grid and hashing.
        aabb_train = torch.FloatTensor([-bound, -bound, -bound, bound, bound, bound])
        aabb_infer = aabb_train.clone()
        self.register_buffer('aabb_train', aabb_train)
        self.register_buffer('aabb_infer', aabb_infer)

        if not opt.use_initialization_from_rgbxy:
            self.basis_color = torch.zeros([self.num_basis, 3])+0.5
            self.basis_color = nn.Parameter(self.basis_color, requires_grad=True)
        else:
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

    def initialize_color(self, color_list=None, hist_weights=None):
        if color_list is None:
            if self.basis_color is None:
                self.basis_color = torch.zeros([self.num_basis, 3]) + 0.5
                self.basis_color = nn.Parameter(self.basis_color, requires_grad=True)
        else:
            # assert(self.num_basis == color_list.shape[0]*len(self.opt.roughness_list))
            self.basis_color = torch.zeros([self.num_basis, 3])
            for i, color in enumerate(color_list):
                self.basis_color[i::color_list.shape[0]] = srgb_to_linear(torch.FloatTensor(color))
            self.basis_color = nn.Parameter(self.basis_color, requires_grad=True)

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

    def run(self, rays_o, rays_d, num_steps=128, upsample_steps=128, bg_color=None, perturb=False, **kwargs):
        # rays_o, rays_d: [B, N, 3], assumes B == 1
        # bg_color: [3] in range [0, 1]
        # return: image: [B, N, 3], depth: [B, N]

        prefix = rays_o.shape[:-1]
        rays_o = rays_o.contiguous().view(-1, 3)
        rays_d = rays_d.contiguous().view(-1, 3)

        N = rays_o.shape[0] # N = B * N, in fact
        device = rays_o.device

        # choose aabb
        aabb = self.aabb_train if self.training else self.aabb_infer

        # sample steps
        nears, fars = raymarching.near_far_from_aabb(rays_o, rays_d, aabb, self.min_near)
        nears.unsqueeze_(-1)
        fars.unsqueeze_(-1)

        #print(f'nears = {nears.min().item()} ~ {nears.max().item()}, fars = {fars.min().item()} ~ {fars.max().item()}')

        z_vals = torch.linspace(0.0, 1.0, num_steps, device=device).unsqueeze(0) # [1, T]
        z_vals = z_vals.expand((N, num_steps)) # [N, T]
        z_vals = nears + (fars - nears) * z_vals # [N, T], in [nears, fars]

        # perturb z_vals
        sample_dist = (fars - nears) / num_steps
        if perturb:
            z_vals = z_vals + (torch.rand(z_vals.shape, device=device) - 0.5) * sample_dist
            #z_vals = z_vals.clamp(nears, fars) # avoid out of bounds xyzs.

        # generate xyzs
        xyzs = rays_o.unsqueeze(-2) + rays_d.unsqueeze(-2) * z_vals.unsqueeze(-1) # [N, 1, 3] * [N, T, 1] -> [N, T, 3]
        xyzs = torch.min(torch.max(xyzs, aabb[:3]), aabb[3:]) # a manual clip.

        #plot_pointcloud(xyzs.reshape(-1, 3).detach().cpu().numpy())

        # query SDF and RGB
        density_outputs = self.density(xyzs.reshape(-1, 3))

        #sigmas = density_outputs['sigma'].view(N, num_steps) # [N, T]
        for k, v in density_outputs.items():
            density_outputs[k] = v.view(N, num_steps, -1)

        # upsample z_vals (nerf-like)
        if upsample_steps > 0:
            with torch.no_grad():

                deltas = z_vals[..., 1:] - z_vals[..., :-1] # [N, T-1]
                deltas = torch.cat([deltas, sample_dist * torch.ones_like(deltas[..., :1])], dim=-1)

                alphas = 1 - torch.exp(-deltas * self.density_scale * density_outputs['sigma'].squeeze(-1)) # [N, T]
                alphas_shifted = torch.cat([torch.ones_like(alphas[..., :1]), 1 - alphas + 1e-15], dim=-1) # [N, T+1]
                weights = alphas * torch.cumprod(alphas_shifted, dim=-1)[..., :-1] # [N, T]

                # sample new z_vals
                z_vals_mid = (z_vals[..., :-1] + 0.5 * deltas[..., :-1]) # [N, T-1]
                new_z_vals = sample_pdf(z_vals_mid, weights[:, 1:-1], upsample_steps, det=not self.training).detach() # [N, t]

                new_xyzs = rays_o.unsqueeze(-2) + rays_d.unsqueeze(-2) * new_z_vals.unsqueeze(-1) # [N, 1, 3] * [N, t, 1] -> [N, t, 3]
                new_xyzs = torch.min(torch.max(new_xyzs, aabb[:3]), aabb[3:]) # a manual clip.

            # only forward new points to save computation
            new_density_outputs = self.density(new_xyzs.reshape(-1, 3))
            #new_sigmas = new_density_outputs['sigma'].view(N, upsample_steps) # [N, t]
            for k, v in new_density_outputs.items():
                new_density_outputs[k] = v.view(N, upsample_steps, -1)

            # re-order
            z_vals = torch.cat([z_vals, new_z_vals], dim=1) # [N, T+t]
            z_vals, z_index = torch.sort(z_vals, dim=1)

            xyzs = torch.cat([xyzs, new_xyzs], dim=1) # [N, T+t, 3]
            xyzs = torch.gather(xyzs, dim=1, index=z_index.unsqueeze(-1).expand_as(xyzs))

            for k in density_outputs:
                tmp_output = torch.cat([density_outputs[k], new_density_outputs[k]], dim=1)
                density_outputs[k] = torch.gather(tmp_output, dim=1, index=z_index.unsqueeze(-1).expand_as(tmp_output))

        deltas = z_vals[..., 1:] - z_vals[..., :-1] # [N, T+t-1]
        deltas = torch.cat([deltas, sample_dist * torch.ones_like(deltas[..., :1])], dim=-1)
        alphas = 1 - torch.exp(-deltas * self.density_scale * density_outputs['sigma'].squeeze(-1)) # [N, T+t]
        alphas_shifted = torch.cat([torch.ones_like(alphas[..., :1]), 1 - alphas + 1e-15], dim=-1) # [N, T+t+1]
        weights = alphas * torch.cumprod(alphas_shifted, dim=-1)[..., :-1] # [N, T+t]

        # detach density output from original nerf
        weights = weights.detach()

        dirs = rays_d.view(-1, 1, 3).expand_as(xyzs)
        for k, v in density_outputs.items():
            density_outputs[k] = v.view(-1, v.shape[-1]).detach()

        mask = weights > 1e-4 # hard coded
        d_color, omega, color = self.color(xyzs.reshape(-1, 3), dirs.reshape(-1, 3), mask=mask.reshape(-1), **density_outputs)
        d_color = d_color.reshape(N, -1, self.num_basis, 3)
        omega = omega.reshape(N, -1, self.num_basis, 1)
        color = color.reshape(N, -1, 3)

        basis_color = self.basis_color[None,None,:,:].clamp(0, 1)
        #basis_color = normalize(basis_color)
        # basis_color = utils.lab_to_rgb(torch.concatenate([self.basis_color[...,:1]*0+75, 
        #                                                     self.basis_color[...,1:]], dim=-1))
        # basis_color = basis_color[None,None,:,:]

        if self.freeze_basis_color:
            basis_color = basis_color.detach()

        if self.opt.multiply_delta:
            final_color = (basis_color*d_color).clamp(0, 1)
        else:
            final_color = (basis_color+d_color).clamp(0, 1)
        basis_rgb = omega*final_color # N_rays, N_sample, N_basis, 3

        omega_norm = omega[...,0].sum(dim=-1, keepdim=True)/((omega[...,0]**2).sum(dim=-1, keepdim=True)+1e-6)-1 # N_rays, N_sample, 1
        omega_norm_map = (weights[:,:,None].detach()*omega_norm).sum(dim=1) # N_rays, 1
        omega_norm_map = omega_norm_map.view(*prefix, 1)

        rgb = basis_rgb.sum(dim=-2) + color # (N_rays, N_samples_, 3)
        rgb_map = (weights[:,:,None]*rgb).sum(dim=1) # N_rays, 3
        if self.opt.color_space != "linear":
            rgb_map = safe_pow(rgb_map, 1/2.4)
        
        basis_rgb = basis_rgb.reshape(N, basis_rgb.shape[1], -1) # (N_rays, N_samples_, N_basis*3)
        basis_rgb_map = (weights[:,:,None]*basis_rgb).sum(dim=1) # N_rays, N_basis*3
        basis_rgb_map = safe_pow(basis_rgb_map, 1/2.4).clamp(0, 1)
        basis_acc_map = (weights[:,:,None].detach()*omega[...,0]).sum(dim=1)

        # delta_rgb_norm = d_color.norm(dim=-1).mean(dim=-1, keepdim=True) # (N_rays, N_samples_, 1)
        delta_rgb_norm = cos_distance(final_color, basis_color).mean(dim=-1, keepdim=True) # (N_rays, N_samples_, 1)
        delta_rgb_norm_map = (weights[:,:,None]*delta_rgb_norm).sum(dim=1) # N_rays, 1
        
        dir_rgb_map = (weights[:,:,None]*color).sum(dim=1) # N_rays, 3
        dir_rgb_map = safe_pow(dir_rgb_map, 1/2.4).clamp(0, 1)
        dir_rgb_norm = color.norm(dim=-1, keepdim=True) # (N_rays, N_samples_, 1)
        dir_rgb_norm_map = (weights[:,:,None]*dir_rgb_norm).sum(dim=1) # N_rays, 1

        # calculate weight_sum (mask)
        weights_sum = weights.sum(dim=-1) # [N]
        
        # calculate depth 
        ori_z_vals = ((z_vals - nears) / (fars - nears)).clamp(0, 1)
        depth = torch.sum(weights * ori_z_vals, dim=-1)

        # mix background color
        if self.bg_radius > 0:
            # use the bg model to calculate bg_color
            sph = raymarching.sph_from_ray(rays_o, rays_d, self.bg_radius) # [N, 2] in [-1, 1]
            bg_color = self.background(sph, rays_d.reshape(-1, 3)) # [N, 3]
        elif bg_color is None:
            bg_color = 1
            
        rgb_map = rgb_map + (1 - weights_sum).unsqueeze(-1) * bg_color

        rgb_map = rgb_map.view(*prefix, 3)
        depth = depth.view(*prefix)
        basis_rgb_map = basis_rgb_map.view(*prefix, self.num_basis*3)
        basis_acc_map = basis_acc_map.view(*prefix, self.num_basis)
        omega_norm_map = omega_norm_map.view(*prefix)
        dir_rgb_norm_map = dir_rgb_norm_map.view(*prefix)
        delta_rgb_norm_map = delta_rgb_norm_map.view(*prefix)
        dir_rgb_map = dir_rgb_map.view(*prefix, 3)
        basis_acc_map = basis_acc_map.view(*prefix, self.num_basis)

        # tmp: reg loss in mip-nerf 360
        # z_vals_shifted = torch.cat([z_vals[..., 1:], sample_dist * torch.ones_like(z_vals[..., :1])], dim=-1)
        # mid_zs = (z_vals + z_vals_shifted) / 2 # [N, T]
        # loss_dist = (torch.abs(mid_zs.unsqueeze(1) - mid_zs.unsqueeze(2)) * (weights.unsqueeze(1) * weights.unsqueeze(2))).sum() + 1/3 * ((z_vals_shifted - z_vals_shifted) * (weights ** 2)).sum()
        return {
            'depth': depth,
            'image': rgb_map,
            'omega_norm': omega_norm_map,
            'dir_norm': dir_rgb_norm_map,
            'delta_norm': delta_rgb_norm_map,
            'dir_rgb': dir_rgb_map,
            'basis_rgb': basis_rgb_map,
            'basis_acc': basis_acc_map,
            'weights_sum': weights_sum,
        }


    # def run_cuda(self, rays_o, rays_d, dt_gamma=0, bg_color=None, perturb=False, force_all_rays=False, max_steps=1024, T_thresh=1e-4, **kwargs):
    #     # rays_o, rays_d: [B, N, 3], assumes B == 1
    #     # return: image: [B, N, 3], depth: [B, N]

    #     prefix = rays_o.shape[:-1]
    #     rays_o = rays_o.contiguous().view(-1, 3)
    #     rays_d = rays_d.contiguous().view(-1, 3)

    #     N = rays_o.shape[0] # N = B * N, in fact
    #     device = rays_o.device

    #     # pre-calculate near far
    #     nears, fars = raymarching.near_far_from_aabb(rays_o, rays_d, self.aabb_train if self.training else self.aabb_infer, self.min_near)

    #     # mix background color
    #     if self.bg_radius > 0:
    #         # use the bg model to calculate bg_color
    #         sph = raymarching.sph_from_ray(rays_o, rays_d, self.bg_radius) # [N, 2] in [-1, 1]
    #         bg_color = self.background(sph, rays_d) # [N, 3]
    #     elif bg_color is None:
    #         bg_color = 1

    #     results = {}

    #     if self.training:
    #         # setup counter
    #         counter = self.step_counter[self.local_step % 16]
    #         counter.zero_() # set to 0
    #         self.local_step += 1

    #         xyzs, dirs, deltas, rays = raymarching.march_rays_train(rays_o, rays_d, self.bound, self.density_bitfield, self.cascade, self.grid_size, nears, fars, counter, self.mean_count, perturb, 128, force_all_rays, dt_gamma, max_steps)

    #         #plot_pointcloud(xyzs.reshape(-1, 3).detach().cpu().numpy())
            
    #         sigmas, rgbs = self(xyzs, dirs)
    #         # density_outputs = self.density(xyzs) # [M,], use a dict since it may include extra things, like geo_feat for rgb.
    #         # sigmas = density_outputs['sigma']
    #         # rgbs = self.color(xyzs, dirs, **density_outputs)
    #         sigmas = self.density_scale * sigmas

    #         #print(f'valid RGB query ratio: {mask.sum().item() / mask.shape[0]} (total = {mask.sum().item()})')

    #         # special case for CCNeRF's residual learning
    #         if len(sigmas.shape) == 2:
    #             K = sigmas.shape[0]
    #             depths = []
    #             images = []
    #             for k in range(K):
    #                 weights_sum, depth, image = raymarching.composite_rays_train(sigmas[k], rgbs[k], deltas, rays, T_thresh)
    #                 image = image + (1 - weights_sum).unsqueeze(-1) * bg_color
    #                 depth = torch.clamp(depth - nears, min=0) / (fars - nears)
    #                 images.append(image.view(*prefix, 3))
    #                 depths.append(depth.view(*prefix))
            
    #             depth = torch.stack(depths, axis=0) # [K, B, N]
    #             image = torch.stack(images, axis=0) # [K, B, N, 3]

    #         else:

    #             weights_sum, depth, image = raymarching.composite_rays_train(sigmas, rgbs, deltas, rays, T_thresh)
    #             image = image + (1 - weights_sum).unsqueeze(-1) * bg_color
    #             depth = torch.clamp(depth - nears, min=0) / (fars - nears)
    #             image = image.view(*prefix, 3)
    #             depth = depth.view(*prefix)
            
    #         results['weights_sum'] = weights_sum

    #     else:
           
    #         # allocate outputs 
    #         # if use autocast, must init as half so it won't be autocasted and lose reference.
    #         #dtype = torch.half if torch.is_autocast_enabled() else torch.float32
    #         # output should always be float32! only network inference uses half.
    #         dtype = torch.float32
            
    #         weights_sum = torch.zeros(N, dtype=dtype, device=device)
    #         depth = torch.zeros(N, dtype=dtype, device=device)
    #         image = torch.zeros(N, 3, dtype=dtype, device=device)
            
    #         n_alive = N
    #         rays_alive = torch.arange(n_alive, dtype=torch.int32, device=device) # [N]
    #         rays_t = nears.clone() # [N]

    #         step = 0
            
    #         while step < max_steps:

    #             # count alive rays 
    #             n_alive = rays_alive.shape[0]
                
    #             # exit loop
    #             if n_alive <= 0:
    #                 break

    #             # decide compact_steps
    #             n_step = max(min(N // n_alive, 8), 1)

    #             xyzs, dirs, deltas = raymarching.march_rays(n_alive, n_step, rays_alive, rays_t, rays_o, rays_d, self.bound, self.density_bitfield, self.cascade, self.grid_size, nears, fars, 128, perturb if step == 0 else False, dt_gamma, max_steps)

    #             sigmas, rgbs = self(xyzs, dirs)
    #             # density_outputs = self.density(xyzs) # [M,], use a dict since it may include extra things, like geo_feat for rgb.
    #             # sigmas = density_outputs['sigma']
    #             # rgbs = self.color(xyzs, dirs, **density_outputs)
    #             sigmas = self.density_scale * sigmas

    #             raymarching.composite_rays(n_alive, n_step, rays_alive, rays_t, sigmas, rgbs, deltas, weights_sum, depth, image, T_thresh)

    #             rays_alive = rays_alive[rays_alive >= 0]

    #             #print(f'step = {step}, n_step = {n_step}, n_alive = {n_alive}, xyzs: {xyzs.shape}')

    #             step += n_step

    #         image = image + (1 - weights_sum).unsqueeze(-1) * bg_color
    #         depth = torch.clamp(depth - nears, min=0) / (fars - nears)
    #         image = image.view(*prefix, 3)
    #         depth = depth.view(*prefix)
        
    #     results['depth'] = depth
    #     results['image'] = image

    #     return results

    @torch.no_grad()
    def mark_untrained_grid(self, poses, intrinsic, S=64):
        # poses: [B, 4, 4]
        # intrinsic: [3, 3]

        if not self.cuda_ray:
            return
        
        if isinstance(poses, np.ndarray):
            poses = torch.from_numpy(poses)

        B = poses.shape[0]
        
        fx, fy, cx, cy = intrinsic
        
        X = torch.arange(self.grid_size, dtype=torch.int32, device=self.density_bitfield.device).split(S)
        Y = torch.arange(self.grid_size, dtype=torch.int32, device=self.density_bitfield.device).split(S)
        Z = torch.arange(self.grid_size, dtype=torch.int32, device=self.density_bitfield.device).split(S)

        count = torch.zeros_like(self.density_grid)
        poses = poses.to(count.device)

        # 5-level loop, forgive me...

        for xs in X:
            for ys in Y:
                for zs in Z:
                    
                    # construct points
                    xx, yy, zz = custom_meshgrid(xs, ys, zs)
                    coords = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)], dim=-1) # [N, 3], in [0, 128)
                    indices = raymarching.morton3D(coords).long() # [N]
                    world_xyzs = (2 * coords.float() / (self.grid_size - 1) - 1).unsqueeze(0) # [1, N, 3] in [-1, 1]

                    # cascading
                    for cas in range(self.cascade):
                        bound = min(2 ** cas, self.bound)
                        half_grid_size = bound / self.grid_size
                        # scale to current cascade's resolution
                        cas_world_xyzs = world_xyzs * (bound - half_grid_size)

                        # split batch to avoid OOM
                        head = 0
                        while head < B:
                            tail = min(head + S, B)

                            # world2cam transform (poses is c2w, so we need to transpose it. Another transpose is needed for batched matmul, so the final form is without transpose.)
                            cam_xyzs = cas_world_xyzs - poses[head:tail, :3, 3].unsqueeze(1)
                            cam_xyzs = cam_xyzs @ poses[head:tail, :3, :3] # [S, N, 3]
                            
                            # query if point is covered by any camera
                            mask_z = cam_xyzs[:, :, 2] > 0 # [S, N]
                            mask_x = torch.abs(cam_xyzs[:, :, 0]) < cx / fx * cam_xyzs[:, :, 2] + half_grid_size * 2
                            mask_y = torch.abs(cam_xyzs[:, :, 1]) < cy / fy * cam_xyzs[:, :, 2] + half_grid_size * 2
                            mask = (mask_z & mask_x & mask_y).sum(0).reshape(-1) # [N]

                            # update count 
                            count[cas, indices] += mask
                            head += S
    
        # mark untrained grid as -1
        self.density_grid[count == 0] = -1

        #print(f'[mark untrained grid] {(count == 0).sum()} from {resolution ** 3 * self.cascade}')

    @torch.no_grad()
    def update_extra_state(self, decay=0.95, S=128):
        # call before each epoch to update extra states.

        if not self.cuda_ray:
            return 
        
        ### update density grid

        tmp_grid = - torch.ones_like(self.density_grid)
        
        # full update.
        if self.iter_density < 16:
        #if True:
            X = torch.arange(self.grid_size, dtype=torch.int32, device=self.density_bitfield.device).split(S)
            Y = torch.arange(self.grid_size, dtype=torch.int32, device=self.density_bitfield.device).split(S)
            Z = torch.arange(self.grid_size, dtype=torch.int32, device=self.density_bitfield.device).split(S)

            for xs in X:
                for ys in Y:
                    for zs in Z:
                        
                        # construct points
                        xx, yy, zz = custom_meshgrid(xs, ys, zs)
                        coords = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)], dim=-1) # [N, 3], in [0, 128)
                        indices = raymarching.morton3D(coords).long() # [N]
                        xyzs = 2 * coords.float() / (self.grid_size - 1) - 1 # [N, 3] in [-1, 1]

                        # cascading
                        for cas in range(self.cascade):
                            bound = min(2 ** cas, self.bound)
                            half_grid_size = bound / self.grid_size
                            # scale to current cascade's resolution
                            cas_xyzs = xyzs * (bound - half_grid_size)
                            # add noise in [-hgs, hgs]
                            cas_xyzs += (torch.rand_like(cas_xyzs) * 2 - 1) * half_grid_size
                            # query density
                            sigmas = self.density(cas_xyzs)['sigma'].reshape(-1).detach()
                            sigmas *= self.density_scale
                            # assign 
                            tmp_grid[cas, indices] = sigmas

        # partial update (half the computation)
        # TODO: why no need of maxpool ?
        else:
            N = self.grid_size ** 3 // 4 # H * H * H / 4
            for cas in range(self.cascade):
                # random sample some positions
                coords = torch.randint(0, self.grid_size, (N, 3), device=self.density_bitfield.device) # [N, 3], in [0, 128)
                indices = raymarching.morton3D(coords).long() # [N]
                # random sample occupied positions
                occ_indices = torch.nonzero(self.density_grid[cas] > 0).squeeze(-1) # [Nz]
                rand_mask = torch.randint(0, occ_indices.shape[0], [N], dtype=torch.long, device=self.density_bitfield.device)
                occ_indices = occ_indices[rand_mask] # [Nz] --> [N], allow for duplication
                occ_coords = raymarching.morton3D_invert(occ_indices) # [N, 3]
                # concat
                indices = torch.cat([indices, occ_indices], dim=0)
                coords = torch.cat([coords, occ_coords], dim=0)
                # same below
                xyzs = 2 * coords.float() / (self.grid_size - 1) - 1 # [N, 3] in [-1, 1]
                bound = min(2 ** cas, self.bound)
                half_grid_size = bound / self.grid_size
                # scale to current cascade's resolution
                cas_xyzs = xyzs * (bound - half_grid_size)
                # add noise in [-hgs, hgs]
                cas_xyzs += (torch.rand_like(cas_xyzs) * 2 - 1) * half_grid_size
                # query density
                sigmas = self.density(cas_xyzs)['sigma'].reshape(-1).detach()
                sigmas *= self.density_scale
                # assign 
                tmp_grid[cas, indices] = sigmas

        ## max-pool on tmp_grid for less aggressive culling [No significant improvement...]
        # invalid_mask = tmp_grid < 0
        # tmp_grid = F.max_pool3d(tmp_grid.view(self.cascade, 1, self.grid_size, self.grid_size, self.grid_size), kernel_size=3, stride=1, padding=1).view(self.cascade, -1)
        # tmp_grid[invalid_mask] = -1

        # ema update
        valid_mask = (self.density_grid >= 0) & (tmp_grid >= 0)
        self.density_grid[valid_mask] = torch.maximum(self.density_grid[valid_mask] * decay, tmp_grid[valid_mask])
        self.mean_density = torch.mean(self.density_grid.clamp(min=0)).item() # -1 regions are viewed as 0 density.
        #self.mean_density = torch.mean(self.density_grid[self.density_grid > 0]).item() # do not count -1 regions
        self.iter_density += 1

        # convert to bitfield
        density_thresh = min(self.mean_density, self.density_thresh)
        self.density_bitfield = raymarching.packbits(self.density_grid, density_thresh, self.density_bitfield)

        ### update step counter
        total_step = min(16, self.local_step)
        if total_step > 0:
            self.mean_count = int(self.step_counter[:total_step, 0].sum().item() / total_step)
        self.local_step = 0

        #print(f'[density grid] min={self.density_grid.min().item():.4f}, max={self.density_grid.max().item():.4f}, mean={self.mean_density:.4f}, occ_rate={(self.density_grid > 0.01).sum() / (128**3 * self.cascade):.3f} | [step counter] mean={self.mean_count}')


    def render(self, rays_o, rays_d, staged=False, max_ray_batch=4096, **kwargs):
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
            results = {}
            for b in range(B):
                head = 0
                while head < N:
                    tail = min(head + max_ray_batch, N)
                    results_ = _run(rays_o[b:b+1, head:tail], rays_d[b:b+1, head:tail], **kwargs)
                    # results_.pop("weights_sum")
                    for k, v in results_.items():
                        if k == 'weights_sum':
                            v = v[None,...]
                        if k not in results.keys():
                            if v.ndim == 2:
                                results[k] = torch.empty((B, N), device=device)
                            else:
                                results[k] = torch.empty((B, N, v.shape[-1]), device=device)
                        results[k][b:b+1, head:tail] = v
                    head += max_ray_batch
            
        else:
            results = _run(rays_o, rays_d, **kwargs)

        return results