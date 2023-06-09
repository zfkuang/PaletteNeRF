import os
import glob
import tqdm
import math
import imageio
import random
import warnings
import tensorboardX

import numpy as np
import pandas as pd

import time
from datetime import datetime

import cv2
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.cluster import KMeans
from skimage import io, color

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributed as dist
from torch.autograd import Function
from torch.cuda.amp import custom_bwd, custom_fwd
from torch.utils.data import Dataset, DataLoader

import trimesh
import mcubes
from rich.console import Console
from torch_ema import ExponentialMovingAverage

from packaging import version as pver
import palette
from nerf.utils import *
import lpips

from typing import Tuple
from .rgbsg import *

from sklearn.decomposition import PCA

try:
    import _palette_func as _backend
except ImportError:
    print("Loading module palette_func...")
    from .backend import _backend

class SparsityMeter:
    def __init__(self, opt, device=None):
        self.V = 0
        self.N = 0
        self.num_basis = opt.num_basis
        self.device=device
        self.basis_metric=True

    def clear(self):
        self.V = 0
        self.N = 0
        
    def update(self, omega):
        omega = omega.to(self.device) # [B, H, W, N_p], range[0, 1]
        
        # simplified since max_pixel_value is 1 here.
        # psnr = -10 * np.log10(np.mean((preds - truths) ** 2))
        omega_sparsity = omega.sum(dim=-1, keepdim=True)/((omega**2).sum(dim=-1, keepdim=True)+1e-6)-1 # N_rays, N_sample, 1
        
        self.V += omega_sparsity.mean()
        self.N += 1

    def measure(self):
        return self.V / self.N

    def write(self, writer, global_step, prefix=""):
        writer.add_scalar(os.path.join(prefix, "Sparsity"), self.measure(), global_step)

    def report(self):
        return f'Sparsity = {self.measure():.6f}'

class TVMeter:
    def __init__(self, opt, device=None):
        self.V = 0
        self.N = 0
        self.num_basis = opt.num_basis
        self.device=device
        self.basis_metric=True

    def clear(self):
        self.V = 0
        self.N = 0
        
    def update(self, omega):
        omega = omega.to(self.device) # [B, H, W, N_p], range[0, 1]
        
        # simplified since max_pixel_value is 1 here.
        # psnr = -10 * np.log10(np.mean((preds - truths) ** 2))

        w_variance = torch.mean(torch.pow(omega[:,:,:-1,:] - omega[:,:,1:,:], 2))        
        h_variance = torch.mean(torch.pow(omega[:,:-1,:,:] - omega[:,1:,:,:], 2))
        tv = w_variance + h_variance
        self.V += tv*100
        self.N += 1

    def measure(self):
        return self.V / self.N

    def write(self, writer, global_step, prefix=""):
        writer.add_scalar(os.path.join(prefix, "TV"), self.measure(), global_step)

    def report(self):
        return f'TV = {self.measure():.6f}'

    
def get_palette_weight_with_hist(rgb, hist_weights):
    assert(hist_weights.ndim == 5)
    rgb_shape = rgb.shape
    rgb = rgb.reshape(-1, 3)
    rgb = rgb[None,None,None,:,[2,1,0]]*2-1
    weight = torch.nn.functional.grid_sample(hist_weights, rgb, mode='bilinear', padding_mode='zeros', align_corners=True)
    weight = weight.squeeze().permute(1, 0)
    return weight.reshape(rgb_shape[:-1] + (-1,))

def normalize(tensor):
    return tensor / (tensor.norm(dim=-1, keepdim=True)+1e-9)
   
def compute_RGB_histogram(
    colors_rgb: np.ndarray,
    weights: np.ndarray,
    bits_per_channel: int
) -> Tuple[np.ndarray, np.ndarray]:
    assert colors_rgb.ndim == 2 and colors_rgb.shape[1] == 3
    assert weights.ndim == 1
    assert len(colors_rgb) == len(weights)
    assert 1 <= bits_per_channel and bits_per_channel <=8

    try:
        bin_weights, bin_centers_rgb = _backend.compute_RGB_histogram(
            colors_rgb.flatten(), weights.flatten(), bits_per_channel)
    except RuntimeError as err:
        print(err)
        assert False

    return bin_weights, bin_centers_rgb

def run_kmeans(
    n_clusters: int,
    points: np.ndarray,
    init: np.ndarray,
    sample_weight: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    print(f'running kmeans with K = {n_clusters}')
    kmeans = KMeans(n_clusters=n_clusters, init=init).fit(X=points, sample_weight=sample_weight)
    centers = kmeans.cluster_centers_
    labels = kmeans.labels_

    center_weights = np.zeros(n_clusters)
    for i in range(n_clusters):
        center_weights[i] = np.sum(sample_weight[labels==i])

    idcs = np.argsort(center_weights * -1)

    return centers[idcs], center_weights[idcs]

def palette_extraction(
    inputs: dict,
    output_dir: str,
    tau: float = 8e-3,
    palette_size = None,
    normalize_input = False,
    error_thres = 5.0 / 255.0
):
    '''
        Extract palettes with the RGBXY method
    '''
    assert palette_size is None or palette_size >= 4 ## convex hull should have at least 4 vertices
    print(f'extracting palette with {palette_size} colors')

    if not os.path.exists(output_dir):
        print(f'create output directory {output_dir}')
        os.makedirs(output_dir)
    
    output_prefix = "%s/extract"%output_dir
    ## radiance sampling
    start = time.time()

    colors = inputs['colors']
    weights = np.ones_like(colors[...,0])
    colors = colors.reshape(-1,3)
    weights = weights.flatten()

    assert len(weights[weights < 0]) == 0, 'negative weight indicates the failure of radiance sampling'

    ## save radiance samples (outside timing analysis)
    res = 800
    n_total = res**2
    random.seed(0)
    idcs = random.sample(range(len(colors)), n_total)
    assert len(idcs) == len(set(idcs)), 'each element of idcs should be unique'
    img = colors[idcs].reshape(res,res,3)
    Image.fromarray((img*255).round().clip(0,255).astype(np.uint8)).save(output_prefix+"-radiance-raw.png")

    ## radiance sample filtering
    start = time.time()

    ## coarse histogram (2^3 = 8 bins)
    bin_weights_coarse, bin_centers_coarse = compute_RGB_histogram(colors, weights, bits_per_channel=3)
    sum_weights = np.sum(bin_weights_coarse)
    bin_weights_coarse /= sum_weights

    idcs = bin_weights_coarse > tau
    bin_weights_coarse = bin_weights_coarse[idcs]
    bin_centers_coarse = bin_centers_coarse[idcs]

    ## fine histogram (2^5 = 32 bins)
    bin_weights_fine, bin_centers_fine = compute_RGB_histogram(colors, weights, bits_per_channel=5)
    idcs = bin_weights_fine > 0
    bin_weights_fine = bin_weights_fine[idcs]
    bin_weights_fine /= sum_weights
    bin_centers_fine = bin_centers_fine[idcs]

    centers, center_weights = run_kmeans(
        n_clusters=len(bin_weights_coarse), points=bin_centers_fine,
        init=bin_centers_coarse, sample_weight=bin_weights_fine)

    ## convex hull simplification
    start = time.time()
    palette_rgb = Hull_Simplification_posternerf(
        centers.astype(np.double), output_prefix,
        pixel_counts=center_weights,
        error_thres=error_thres,
        target_size=palette_size)
    _, hist_rgb = compute_RGB_histogram(colors, weights, bits_per_channel=5)
    
    if normalize_input:
        hist_rgb = hist_rgb + 0.05
        hist_rgb_norm = np.linalg.norm(hist_rgb, axis=-1, keepdims=True) #.clip(min=0.1)
        hist_rgb = hist_rgb / hist_rgb_norm

    # Generate weight
    hist_weights = Tan18.Get_ASAP_weights_using_Tan_2016_triangulation_and_then_barycentric_coordinates(hist_rgb.astype(np.double).reshape((-1,1,3)), 
                        palette_rgb, None, order=0) # N_bin_center x 1 x num_palette
    hist_weights = hist_weights.reshape([32,32,32,palette_rgb.shape[0]])
    

    ## save palette
    palette_img = get_bigger_palette_to_show(palette_rgb)
    Image.fromarray((palette_img*255).round().clip(0,255).astype(np.uint8)).save(output_prefix+"-palette.png")

    write_palette_txt(palette_rgb, output_prefix+'-palette.txt')
    np.savez(os.path.join(output_dir, 'palette.npz'), palette=palette_rgb)
    np.savez(os.path.join(output_dir, 'hist_weights.npz'), hist_weights=hist_weights)

# CUDA-based rgb,hsv conversion for speed-up
class _rgb_to_hsv(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, input):

        if not input.is_cuda: input = input.cuda()
        
        prefix = input.shape[:-1]
        input = input.contiguous().view(-1, 3)

        n_rays = input.shape[0]
        output = torch.empty(n_rays, 3, device=input.device, dtype=input.dtype)

        _backend.rgb_to_hsv(n_rays, input, output)
        output = output.reshape(*prefix, 3)

        return output

rgb_to_hsv = _rgb_to_hsv.apply

class _hsv_to_rgb(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, input):

        if not input.is_cuda: input = input.cuda()
        
        prefix = input.shape[:-1]
        input = input.contiguous().view(-1, 3)

        n_rays = input.shape[0]
        output = torch.empty(n_rays, 3, device=input.device, dtype=input.dtype)

        _backend.hsv_to_rgb(n_rays, input, output)
        output = output.reshape(*prefix, 3)

        return output

hsv_to_rgb = _hsv_to_rgb.apply

class PaletteTrainer(object):
    def __init__(self, 
                 name, # name of this experiment
                 opt, # extra conf
                 model, # network 
                 criterion=None, # loss function, if None, assume inline implementation in train_step
                 optimizer=None, # optimizer
                 ema_decay=None, # if use EMA, set the decay
                 lr_scheduler=None, # scheduler
                 metrics=[], # metrics for evaluation, if None, use val_loss to measure performance, else use the first metric.
                 local_rank=0, # which GPU am I
                 world_size=1, # total num of GPUs
                 device=None, # device to use, usually setting to None is OK. (auto choose device)
                 mute=False, # whether to mute all print
                 fp16=False, # amp optimize level
                 eval_interval=1, # eval once every $ epoch
                 max_keep_ckpt=2, # max num of saved ckpts in disk
                 workspace='workspace', # workspace to save logs & ckpts
                 best_mode='min', # the smaller/larger result, the better
                 use_loss_as_metric=True, # use loss as the first metric
                 report_metric_at_train=False, # also report metrics at training
                 use_checkpoint="latest", # which ckpt to use at init time
                 nerf_path=None,
                 use_tensorboardX=True, # whether to use tensorboard for logging
                 scheduler_update_every_step=False, # whether to call scheduler.step() after every train step
                 ):
        
        self.name = name
        self.opt = opt
        self.mute = mute
        self.metrics = metrics
        self.local_rank = local_rank
        self.world_size = world_size
        self.workspace = workspace
        self.ema_decay = ema_decay
        self.fp16 = fp16
        self.best_mode = best_mode
        self.use_loss_as_metric = use_loss_as_metric
        self.report_metric_at_train = report_metric_at_train
        self.max_keep_ckpt = max_keep_ckpt
        self.eval_interval = eval_interval
        self.use_checkpoint = use_checkpoint
        self.use_tensorboardX = use_tensorboardX
        self.time_stamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        self.scheduler_update_every_step = scheduler_update_every_step
        self.device = device if device is not None else torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')
        self.console = Console()
        self.val_len = 10
        self.require_smooth_loss = False

        model.to(self.device)
        if self.world_size > 1:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
        self.model = model

        if isinstance(criterion, nn.Module):
            criterion.to(self.device)
        self.criterion = criterion

        # optionally use LPIPS loss for patch-based training
        if self.opt.patch_size > 1:
            import lpips
            self.criterion_lpips = lpips.LPIPS(net='alex').to(self.device)

        if optimizer is None:
            self.optimizer = optim.Adam(self.model.parameters(), lr=0.001, weight_decay=5e-4) # naive adam
        else:
            self.optimizer = optimizer(self.model)

        if lr_scheduler is None:
            self.lr_scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda epoch: 1) # fake scheduler
        else:
            self.lr_scheduler = lr_scheduler(self.optimizer)

        if ema_decay is not None:
            self.ema = ExponentialMovingAverage(self.model.parameters(), decay=ema_decay)
        else:
            self.ema = None

        self.scaler = torch.cuda.amp.GradScaler(enabled=self.fp16)

        # variable init
        self.epoch = 0
        self.global_step = 0
        self.local_step = 0
        self.stats = {
            "loss": [],
            "valid_loss": [],
            "results": [], # metrics[0], or valid_loss
            "checkpoints": [], # record path of saved ckpt, to automatically remove old ckpt
            "best_result": None,
            }

        # auto fix
        if len(metrics) == 0 or self.use_loss_as_metric:
            self.best_mode = 'min'

        # workspace prepare
        self.log_ptr = None
        if self.workspace is not None:
            os.makedirs(self.workspace, exist_ok=True)        
            self.log_path = os.path.join(workspace, f"log_{self.name}.txt")
            self.log_ptr = open(self.log_path, "a+")

            self.ckpt_path = os.path.join(self.workspace, 'checkpoints')
            self.best_path = f"{self.ckpt_path}/{self.name}.pth"
            os.makedirs(self.ckpt_path, exist_ok=True)
            
        self.log(f'[INFO] Trainer: {self.name} | {self.time_stamp} | {self.device} | {"fp16" if self.fp16 else "fp32"} | {self.workspace}')
        self.log(f'[INFO] #parameters: {sum([p.numel() for p in model.parameters() if p.requires_grad])}')

        if self.workspace is not None:
            if self.use_checkpoint == "scratch":
                self.log("[INFO] Training from scratch ...")
            elif self.use_checkpoint == "latest":
                self.log("[INFO] Loading latest checkpoint ...")
                self.load_checkpoint()
            elif self.use_checkpoint == "latest_model":
                self.log("[INFO] Loading latest checkpoint (model only)...")
                self.load_checkpoint(model_only=True)
            elif self.use_checkpoint == "best":
                if os.path.exists(self.best_path):
                    self.log("[INFO] Loading best checkpoint ...")
                    self.load_checkpoint(self.best_path)
                else:
                    self.log(f"[INFO] {self.best_path} not found, loading latest ...")
                    self.load_checkpoint()
            else: # path to ckpt
                self.log(f"[INFO] Loading {self.use_checkpoint} ...")
                self.load_checkpoint(self.use_checkpoint)
        
        self.nerf_path = nerf_path
        if self.nerf_path is not None:
            self.log(f"[INFO] Loading NeRF at {self.nerf_path} ...")
            self.load_nerf_checkpoint(self.nerf_path)
             
    def __del__(self):
        if self.log_ptr: 
            self.log_ptr.close()

    def log(self, *args, **kwargs):
        if self.local_rank == 0:
            if not self.mute: 
                #print(*args)
                self.console.print(*args, **kwargs)
            if self.log_ptr: 
                print(*args, file=self.log_ptr)
                self.log_ptr.flush() # write immediately to file

    ### ------------------------------	
    def train_step(self, data):

        rays_o = data['rays_o'] # [B, N, 3]
        rays_d = data['rays_d'] # [B, N, 3]

        images = data['images'] # [B, N, 3/4]
        if self.opt.pred_clip:     
            gt_clip_feat = data['feat_images']

        B, N, C = images.shape

        if self.opt.color_space == 'linear':
            images[..., :3] = srgb_to_linear(images[..., :3])

        if C == 3 or self.model.bg_radius > 0:
            bg_color = 1
        # train with random background color if not using a bg model and has alpha channel.
        else:
            #bg_color = torch.ones(3, device=self.device) # [3], fixed white background
            #bg_color = torch.rand(3, device=self.device) # [3], frame-wise random.
            bg_color = torch.rand_like(images[..., :3]) # [N, 3], pixel-wise random.

        if C == 4:
            gt_rgb = images[..., :3] * images[..., 3:] + bg_color * (1 - images[..., 3:])
        else:
            gt_rgb = images
            
        if hasattr(self.model, "hist_weights"):
            gt_weights = get_palette_weight_with_hist(gt_rgb, self.model.hist_weights).detach()
        else:
            gt_weights = None
        
        # outputs = self.model.render(rays_o, rays_d, staged=False, bg_color=bg_color, perturb=True, force_all_rays=False if self.opt.patch_size == 1 else True, **vars(self.opt))
        outputs = self.model.render(rays_o, rays_d, staged=False, bg_color=bg_color, perturb=True, force_all_rays=True, **vars(self.opt))
        
        pred_rgb = outputs['image']
        
        # MSE loss
        loss = self.criterion(pred_rgb, gt_rgb).mean(-1) # [B, N, 3] --> [B, N]
        loss_direct = self.criterion(outputs['direct_rgb'], gt_rgb).mean() # [B, N, 3] --> [B, N]

        # pred clip feature
        if self.opt.pred_clip:
            pred_clip_feat = outputs['clip_feat']
            loss_clip_feat = self.criterion(pred_clip_feat, gt_clip_feat).mean()
          
        # patch-based smooth loss (optional) 
        smooth_patch_loss = 0
        if self.opt.random_size > 0 and self.require_smooth_loss == True and self.opt.lambda_patchsmooth > 0 and 'inds' in data.keys():
            diffuse = outputs['diffuse_rgb'].reshape(-1, 3) # N x 3
            omega = outputs['basis_acc'].reshape(-1, self.opt.num_basis, 1) # N x N_p 
            idx_origin = data['inds']
            idx = torch.stack([idx_origin//data['W'], idx_origin%data['W']], dim=-1).reshape(-1, 2) # N x 2
            
            half_N = N // 2
            diffuse_diff = diffuse[half_N:]
            diffuse = diffuse[:half_N]
            omega_diff = omega[half_N:]
            omega = omega[:half_N]
            idx_diff = idx[half_N:].float()
            idx = idx[:half_N].float()
            
            xyzs_weight_patch = (idx-idx_diff).norm(dim=-1, keepdim=True)**2 / 100
            rgb_weight_patch = (diffuse-diffuse_diff).norm(dim=-1, keepdim=True)**2 / self.opt.sigma_color # (N_rays, N_samples_, 1)
            
            smooth_weight_patch = (xyzs_weight_patch + rgb_weight_patch).detach()
            smooth_weight_patch = torch.exp(-smooth_weight_patch)
            smooth_patch_norm = ((omega-omega_diff)[...,0]**2).sum(dim=-1, keepdim=True) * smooth_weight_patch
            smooth_patch_loss = smooth_patch_norm.mean()        
                
        loss_dict = {}
        
        # Regularizations
        pred_omega_sparsity = outputs['omega_sparsity']
        sparsity_loss = pred_omega_sparsity.mean()

        pred_offsets_norm = outputs['offsets_norm']
        offsets_loss = pred_offsets_norm.mean()

        pred_view_dep_norm = outputs['view_dep_norm']
        view_dep_loss = pred_view_dep_norm.mean()

        # Blending weights supervision
        pred_weights = outputs['basis_acc']
        if gt_weights is not None:
            weights_guide_loss = ((gt_weights-pred_weights)**2).mean()
        else:
            weights_guide_loss = 0
        
        if self.opt.lambda_smooth > 0 and self.model.require_smooth_loss:
            pred_smooth_norm = outputs['smooth_norm']
            smooth_loss = pred_smooth_norm.mean()
        else:
            smooth_loss = 0
            
        palette_loss = ((self.model.basis_color - self.model.basis_color_origin)**2).sum(dim=-1).mean()

        loss_dict['loss_sparsity'] = self.opt.lambda_sparsity * sparsity_loss
        loss += loss_dict['loss_sparsity']

        loss_dict['loss_offsets'] =  self.opt.lambda_offsets * offsets_loss
        loss += loss_dict['loss_offsets']

        loss_dict['loss_view_dep'] =  self.opt.lambda_view_dep * view_dep_loss
        loss += loss_dict['loss_view_dep']

        loss_dict['loss_smooth'] =  self.opt.lambda_smooth * smooth_loss
        loss += loss_dict['loss_smooth']
        
        loss_dict['loss_patchsmooth'] =  self.opt.lambda_patchsmooth * smooth_patch_loss
        loss += loss_dict['loss_patchsmooth']

        loss_dict['loss_palette'] = self.lambda_palette * palette_loss
        loss += loss_dict['loss_palette']

        loss_dict['loss_weight'] =  self.lambda_weight * weights_guide_loss
        loss += loss_dict['loss_weight']

        loss_dict['loss_direct'] = loss_direct
        loss += loss_dict['loss_direct']

        if self.opt.pred_clip:
            loss_dict['loss_clip_feat'] = loss_clip_feat
            loss += loss_dict['loss_clip_feat']

        # special case for CCNeRF's rank-residual training
        if len(loss.shape) == 3: # [K, B, N]
            loss = loss.mean(0)

        # update error_map
        if self.error_map is not None:
            index = data['index'] # [B]
            inds = data['inds_coarse'] # [B, N]

            # take out, this is an advanced indexing and the copy is unavoidable.
            error_map = self.error_map[index] # [B, H * W]

            # [debug] uncomment to save and visualize error map
            # if self.global_step % 1001 == 0:
            #     tmp = error_map[0].view(128, 128).cpu().numpy()
            #     print(f'[write error map] {tmp.shape} {tmp.min()} ~ {tmp.max()}')
            #     tmp = (tmp - tmp.min()) / (tmp.max() - tmp.min())
            #     cv2.imwrite(os.path.join(self.workspace, f'{self.global_step}.jpg'), (tmp * 255).astype(np.uint8))

            error = loss.detach().to(error_map.device) # [B, N], already in [0, 1]
            
            # ema update
            ema_error = 0.1 * error_map.gather(1, inds) + 0.9 * error
            error_map.scatter_(1, inds, ema_error)

            # put back
            self.error_map[index] = error_map

        loss = loss.mean()

        # extra loss
        # pred_weights_sum = outputs['weights_sum'] + 1e-8
        # loss_ws = - 1e-1 * pred_weights_sum * torch.log(pred_weights_sum) # entropy to encourage weights_sum to be 0 or 1.
        # loss = loss + loss_ws.mean()

        return pred_rgb, gt_rgb, loss, loss_dict

    def eval_step(self, data):

        rays_o = data['rays_o'] # [B, N, 3]
        rays_d = data['rays_d'] # [B, N, 3]
        images = data['images'] # [B, H, W, 3/4]
        B, H, W, C = images.shape

        if self.opt.color_space == 'linear':
            images[..., :3] = srgb_to_linear(images[..., :3])

        # eval with fixed background color
        bg_color = 1
        if C == 4:
            gt_rgb = images[..., :3] * images[..., 3:] + bg_color * (1 - images[..., 3:])
        else:
            gt_rgb = images
        if hasattr(self.model, "hist_weights"):
            gt_weights = get_palette_weight_with_hist(gt_rgb, self.model.hist_weights).detach()
        else:
            gt_weights = None
        
        outputs = self.model.render(rays_o, rays_d, staged=True, bg_color=bg_color, perturb=False, test_mode=True, **vars(self.opt))

        pred_rgb = outputs['image'].reshape(B, H, W, 3)
        pred_depth = outputs['depth'].reshape(B, H, W)

        loss = self.criterion(pred_rgb, gt_rgb).mean()

        return pred_rgb, pred_depth, gt_rgb, gt_weights, loss, outputs
    
    def train(self, train_loader, valid_loader, max_epochs):
        if self.use_tensorboardX and self.local_rank == 0:
            self.writer = tensorboardX.SummaryWriter(os.path.join(self.workspace, "run", self.name))

        # mark untrained region (i.e., not covered by any camera from the training dataset)
        # if self.model.cuda_ray:
        #     self.model.mark_untrained_grid(train_loader._data.poses, train_loader._data.intrinsics)

        # get a ref to error_map
        self.error_map = train_loader._data.error_map
        
        self.lambda_palette = 0
        for epoch in range(self.epoch + 1, max_epochs + 1):
            self.epoch = epoch

            self.lambda_weight = self.opt.lambda_weight * max(0, (1 - epoch/self.opt.lweight_decay_epoch))

            self.train_one_epoch(train_loader)
                
            if self.workspace is not None and self.local_rank == 0:
                self.save_checkpoint(full=True, best=False)

            if self.epoch % self.eval_interval == 0:
                self.evaluate_one_epoch(valid_loader)
                self.save_checkpoint(full=False, best=True)
            
            # Release the palettes from the initialzed value
            if epoch >= self.opt.max_freeze_palette_epoch or not self.opt.use_initialization_from_rgbxy:
                self.model.freeze_basis_color = False
                self.lambda_palette = self.opt.lambda_palette
                
            # Add smooth loss after a few epochs
            if epoch >= self.opt.smooth_loss_start_epoch:
                self.require_smooth_loss = True
                self.model.require_smooth_loss = True

        if self.use_tensorboardX and self.local_rank == 0:
            self.writer.close()

    def evaluate(self, loader, name=None, save_images=True):
        self.use_tensorboardX, use_tensorboardX = False, self.use_tensorboardX
        self.evaluate_one_epoch(loader, name, save_images=save_images)
        self.use_tensorboardX = use_tensorboardX

    def train_one_epoch(self, loader):
        self.log(f"==> Start Training Epoch {self.epoch}, lr={self.optimizer.param_groups[0]['lr']:.6f} ...")

        total_loss = 0
        if self.local_rank == 0 and self.report_metric_at_train:
            for metric in self.metrics:
                metric.clear()

        self.model.train()
    
        # distributedSampler: must call set_epoch() to shuffle indices across multiple epochs
        # ref: https://pytorch.org/docs/stable/data.html
        if self.world_size > 1:
            loader.sampler.set_epoch(self.epoch)
        
        if self.local_rank == 0:
            pbar = tqdm.tqdm(total=len(loader) * loader.batch_size, bar_format='{desc}: {percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')

        self.local_step = 0

        for data in loader:
            
            # update grid every 16 steps
            # if self.model.cuda_ray and self.global_step % self.opt.update_extra_interval == 0:
            #     with torch.cuda.amp.autocast(enabled=self.fp16):
            #         self.model.update_extra_state()
                    
            self.local_step += 1
            self.global_step += 1

            self.optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=self.fp16):
                preds, truths, loss, loss_dict = self.train_step(data)
         
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            if self.scheduler_update_every_step:
                self.lr_scheduler.step()

            loss_val = loss.item()
            total_loss += loss_val

            if self.local_rank == 0:
                if self.report_metric_at_train:
                    for metric in self.metrics:
                        metric.update(preds, truths)
                        
                if self.use_tensorboardX:
                    self.writer.add_scalar("train/loss", loss_val, self.global_step)
                    for k, v in loss_dict.items():
                        self.writer.add_scalar("train/%s"%k, v, self.global_step)
                    self.writer.add_scalar("train/lr", self.optimizer.param_groups[0]['lr'], self.global_step)

                if self.scheduler_update_every_step:
                    pbar.set_description(f"loss={loss_val:.4f} ({total_loss/self.local_step:.4f}), lr={self.optimizer.param_groups[0]['lr']:.6f}")
                else:
                    pbar.set_description(f"loss={loss_val:.4f} ({total_loss/self.local_step:.4f})")
                pbar.update(loader.batch_size)

        if self.ema is not None:
            self.ema.update()

        average_loss = total_loss / self.local_step
        self.stats["loss"].append(average_loss)

        if self.local_rank == 0:
            pbar.close()
            if self.report_metric_at_train:
                for metric in self.metrics:
                    self.log(metric.report(), style="red")
                    if self.use_tensorboardX:
                        metric.write(self.writer, self.epoch, prefix="train")
                    metric.clear()

        if not self.scheduler_update_every_step:
            if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.lr_scheduler.step(average_loss)
            else:
                self.lr_scheduler.step()

        self.log(f"==> Finished Epoch {self.epoch}.")

    def evaluate_one_epoch(self, loader, name=None, save_images=True):
        self.log(f"++> Evaluate at epoch {self.epoch} ...")

        if name is None:
            name = f'{self.name}_ep{self.epoch:04d}'

        total_loss = 0
        if self.local_rank == 0:
            for metric in self.metrics:
                metric.clear()

        self.model.eval()

        if self.ema is not None:
            self.ema.store()
            self.ema.copy_to()

        data_len = min(len(loader), self.val_len)

        if self.local_rank == 0:
            pbar = tqdm.tqdm(total=data_len * loader.batch_size, bar_format='{desc}: {percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')

        with torch.no_grad():
            self.local_step = 0

            for data_idx, data in enumerate(loader):    
                self.local_step += 1
                if self.local_step == data_len+1:
                    break

                with torch.cuda.amp.autocast(enabled=self.fp16):
                    preds, preds_depth, truths, gt_weights, loss, outputs = self.eval_step(data)

                # all_gather/reduce the statistics (NCCL only support all_*)
                if self.world_size > 1:
                    dist.all_reduce(loss, op=dist.ReduceOp.SUM)
                    loss = loss / self.world_size
                    
                    preds_list = [torch.zeros_like(preds).to(self.device) for _ in range(self.world_size)] # [[B, ...], [B, ...], ...]
                    dist.all_gather(preds_list, preds)
                    preds = torch.cat(preds_list, dim=0)

                    preds_depth_list = [torch.zeros_like(preds_depth).to(self.device) for _ in range(self.world_size)] # [[B, ...], [B, ...], ...]
                    dist.all_gather(preds_depth_list, preds_depth)
                    preds_depth = torch.cat(preds_depth_list, dim=0)

                    truths_list = [torch.zeros_like(truths).to(self.device) for _ in range(self.world_size)] # [[B, ...], [B, ...], ...]
                    dist.all_gather(truths_list, truths)
                    truths = torch.cat(truths_list, dim=0)
                
                loss_val = loss.item()
                total_loss += loss_val

                # only rank = 0 will perform evaluation.
                if self.local_rank == 0:

                    for metric in self.metrics:
                        if hasattr(metric, "basis_metric"):
                            basis_acc = outputs['basis_acc'][0].reshape(1, preds[0].shape[0], preds[0].shape[1], self.opt.num_basis)
                            metric.update(basis_acc)
                        else:
                            metric.update(preds, truths)
                        
                    # Render evaluation images/passes
                    if save_images:
                        # save image
                        save_path = os.path.join(self.workspace, 'validation', f'{name}_{self.local_step:04d}_rgb.png')
                        save_path_depth = os.path.join(self.workspace, 'validation', f'{name}_{self.local_step:04d}_depth.png')
                        save_path_gt_weights = os.path.join(self.workspace, 'validation', f'{name}_{self.local_step:04d}_basis_acc_guide.png')
                        
                        save_path_basis_img = os.path.join(self.workspace, 'validation', f'{name}_{self.local_step:04d}_basis_img.png')
                        save_path_basis_acc = os.path.join(self.workspace, 'validation', f'{name}_{self.local_step:04d}_basis_acc.png')
                        save_path_basis_color = os.path.join(self.workspace, 'validation', f'{name}_{self.local_step:04d}_basis_color.png')
                        save_path_unscaled_basis_color = os.path.join(self.workspace, 'validation', f'{name}_{self.local_step:04d}_unscaled_basis_color.png')
                        save_path_view_dep_color = os.path.join(self.workspace, 'validation', f'{name}_{self.local_step:04d}_view_dep_color.png')
                        save_path_clip_feat = os.path.join(self.workspace, 'validation', f'{name}_{self.local_step:04d}_clip_feat.png')

                        #self.log(f"==> Saving validation image to {save_path}")
                        os.makedirs(os.path.dirname(save_path), exist_ok=True)

                        if self.opt.color_space == 'linear':
                            preds = linear_to_srgb(preds)

                        if gt_weights is not None:
                            gt_weights = gt_weights[0].detach().cpu().numpy()
                            gt_weights = (gt_weights.clip(0,1) * 255).astype(np.uint8)
                            gt_weights = gt_weights.transpose(0, 2, 1).reshape(gt_weights.shape[0], -1)

                        pred = preds[0].detach().cpu().numpy()
                        pred = (pred.clip(0,1) * 255).astype(np.uint8)
                        pred_depth = preds_depth[0].detach().cpu().numpy()
                        pred_depth = (pred_depth * 255).astype(np.uint8)
                        pred_view_dep_color = outputs['view_dep_rgb'][0].detach().cpu().numpy()
                        pred_view_dep_color = pred_view_dep_color.reshape(pred.shape[0], pred.shape[1], 3)
                        pred_view_dep_color = (pred_view_dep_color.clip(0,1) * 255).astype(np.uint8)
                        pred_direct_color = outputs['direct_rgb'][0].detach().cpu().numpy()
                        pred_direct_color = pred_direct_color.reshape(pred.shape[0], pred.shape[1], 3)
                        pred_direct_color = (pred_direct_color.clip(0,1) * 255).astype(np.uint8)
                        
                        pred_clip_feat = outputs['clip_feat'][0].detach().cpu().numpy()
                        pred_clip_feat_flat = pred_clip_feat.reshape(-1, self.opt.clip_dim)
                        pca2 = PCA(n_components=3)
                        # pca.fit(outputs_flat)
                        pred_clip_feat_flat = pca2.fit_transform(pred_clip_feat_flat)
                        pred_clip_feat = pred_clip_feat_flat.reshape(pred.shape[0], pred.shape[1], -1)
                        pred_clip_feat = ((pred_clip_feat+1)/2*255).clip(0, 255).astype(np.uint8)
                        cv2.imwrite(save_path_clip_feat, cv2.cvtColor(pred_clip_feat, cv2.COLOR_RGB2BGR))

                        pred_basis_img = []
                        pred_basis_acc = []
                        pred_basis_color = []
                        pred_unscaled_basis_color = []
                        for i in range(self.opt.num_basis):
                            basis_img = outputs['basis_rgb'][0,:,i*3:(i+1)*3].reshape(pred.shape[0], pred.shape[1], 3)
                            pred_basis_img.append(basis_img.detach().cpu().numpy())  

                            unscaled_basis_img = outputs['unscaled_basis_rgb'][0,:,i*3:(i+1)*3].reshape(pred.shape[0], pred.shape[1], 3)
                            pred_unscaled_basis_color.append(unscaled_basis_img.detach().cpu().numpy())  

                            basis_acc = outputs['basis_acc'][0,:,i:(i+1)].reshape(pred.shape[0], pred.shape[1])
                            pred_basis_acc.append(basis_acc.detach().cpu().numpy())

                            basis_color = self.model.basis_color[i,None,None,:].repeat(100, 100, 1)
                            basis_color = basis_color.clamp(0, 1)
                            pred_basis_color.append(basis_color.detach().cpu().numpy())

                        pred_basis_img = (np.concatenate(pred_basis_img, axis=1).clip(0,1)*255).astype(np.uint8)
                        pred_basis_acc = (np.concatenate(pred_basis_acc, axis=1).clip(0,1)*255).astype(np.uint8)
                        pred_basis_color = (np.concatenate(pred_basis_color, axis=1).clip(0,1)*255).astype(np.uint8)
                        pred_unscaled_basis_color = (np.concatenate(pred_unscaled_basis_color, axis=1).clip(0,1)*255).astype(np.uint8)

                        cv2.imwrite(save_path_basis_img, cv2.cvtColor(pred_basis_img, cv2.COLOR_RGB2BGR))
                        cv2.imwrite(save_path_basis_acc, pred_basis_acc)
                        cv2.imwrite(save_path_basis_color, cv2.cvtColor(pred_basis_color, cv2.COLOR_RGB2BGR))
                        cv2.imwrite(save_path_unscaled_basis_color, cv2.cvtColor(pred_unscaled_basis_color, cv2.COLOR_RGB2BGR))
                        cv2.imwrite(save_path_view_dep_color, cv2.cvtColor(pred_view_dep_color, cv2.COLOR_RGB2BGR))
                        
                        cv2.imwrite(save_path, cv2.cvtColor(pred, cv2.COLOR_RGB2BGR))
                        if gt_weights is not None:
                            cv2.imwrite(save_path_gt_weights, gt_weights)
                        cv2.imwrite(save_path_depth, pred_depth)

                    pbar.set_description(f"loss={loss_val:.4f} ({total_loss/self.local_step:.4f})")
                    pbar.update(loader.batch_size)


        average_loss = total_loss / self.local_step
        self.stats["valid_loss"].append(average_loss)

        if self.local_rank == 0:
            pbar.close()
            if not self.use_loss_as_metric and len(self.metrics) > 0:
                result = self.metrics[0].measure()
                self.stats["results"].append(result if self.best_mode == 'min' else - result) # if max mode, use -result
            else:
                self.stats["results"].append(average_loss) # if no metric, choose best by min loss

            for metric in self.metrics:
                self.log(metric.report(), style="blue")
                if self.use_tensorboardX:
                    metric.write(self.writer, self.epoch, prefix="evaluate")
                metric.clear()

        if self.ema is not None:
            self.ema.restore()

        self.log(f"++> Evaluate epoch {self.epoch} Finished.")

    # moved out bg_color and perturb for more flexible control...
    def test_step(self, data, bg_color=None, perturb=False, gui_mode=False):  

        rays_o = data['rays_o'] # [B, N, 3]
        rays_d = data['rays_d'] # [B, N, 3]

        if bg_color is not None:
            bg_color = bg_color.to(self.device)

        outputs = self.model.render(rays_o, rays_d, staged=True, bg_color=bg_color, perturb=perturb, gui_mode=gui_mode, test_mode=True, **vars(self.opt))

        outputs['preds'] = outputs['image']
        outputs['preds_depth'] = outputs['depth']
        if 'depth_origin' not in outputs.keys():
            outputs['depth_origin'] = outputs['depth']
        outputs['preds_xyz'] = data['rays_o'] + data['rays_d'] * outputs['depth_origin'][...,None]
        if 'weights_sum' in outputs.keys():
            outputs['preds_weight'] = outputs['weights_sum']

        return outputs 

    def test(self, loader, save_path=None, name=None, write_video=True, selected_idx=None, gui_mode=False):

        if save_path is None:
            save_path = os.path.join(self.workspace, 'results')

        if name is None:
            name = f'{self.name}_ep{self.epoch:04d}'

        os.makedirs(save_path, exist_ok=True)
        
        self.log(f"==> Start Test, save results to {save_path}")

        pbar = tqdm.tqdm(total=len(loader) * loader.batch_size, bar_format='{percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
        self.model.eval()

        if write_video:
            all_preds = []
            all_preds_depth = []
            all_preds_basis_img = []
            all_preds_basis_acc = []
            all_preds_basis_color = []
            all_preds_view_dep_color = []

        with torch.no_grad():

            import time
            tik = time.time()
            for t, data in enumerate(loader):
                if selected_idx is not None and t != selected_idx:
                    continue
                H, W = data['H'], data['W']
                with torch.cuda.amp.autocast(enabled=self.fp16):
                    outputs = self.test_step(data, gui_mode=gui_mode)


                pred = outputs['preds'][0].reshape(H, W, 3)
                
                if self.opt.color_space == 'linear':
                    pred = linear_to_srgb(pred)
                    
                pred = pred.detach().cpu().numpy()
                pred = (pred.clip(0, 1) * 255).astype(np.uint8)

                pred_depth = outputs['preds_depth'][0].reshape(H, W).detach().cpu().numpy()
                pred_depth = (pred_depth * 255).astype(np.uint8)
                
                # prepare results of all testing frames
                if not self.opt.gui:
                    pred_view_dep_color = outputs['view_dep_rgb'][0].reshape(H, W, 3)
                    pred_view_dep_color = pred_view_dep_color.detach().cpu().numpy()
                    pred_view_dep_color = (pred_view_dep_color.clip(0, 1) * 255).astype(np.uint8)

                    pred_basis_img = []
                    pred_basis_acc = []
                    pred_basis_color = []

                    for i in range(self.opt.num_basis):
                        basis_img = outputs['basis_rgb'][0,:,i*3:(i+1)*3].reshape(H, W, 3)
                        pred_basis_img.append(basis_img.detach().cpu().numpy())                        
                        basis_acc = outputs['basis_acc'][0,:,i:(i+1)].reshape(H, W)
                        pred_basis_acc.append(basis_acc.detach().cpu().numpy())
                        basis_color = self.model.basis_color[i,None,None,:].repeat(100, 100, 1)
                        basis_color = basis_color.clamp(0, 1)
                        # basis_color = lab_to_rgb(torch.concatenate([basis_color[...,:1]*0+75, basis_color[...,1:]], dim=-1))
                        pred_basis_color.append(basis_color.detach().cpu().numpy())

                    pred_basis_img = (np.concatenate(pred_basis_img, axis=1).clip(0,1)*255).astype(np.uint8)
                    pred_basis_acc = (np.concatenate(pred_basis_acc, axis=1).clip(0,1)*255).astype(np.uint8)
                    pred_basis_color = (np.concatenate(pred_basis_color, axis=1).clip(0,1)*255).astype(np.uint8)

                    if write_video:
                        all_preds_basis_img.append(pred_basis_img)
                        all_preds_basis_acc.append(pred_basis_acc)
                        all_preds_basis_color.append(pred_basis_color)
                        all_preds_view_dep_color.append(pred_view_dep_color)

                    else:
                        imageio.imwrite(os.path.join(save_path, f'{name}_{t:04d}_basis_img.png'), pred_basis_img)
                        imageio.imwrite(os.path.join(save_path, f'{name}_{t:04d}_basis_acc.png'), pred_basis_acc)
                        imageio.imwrite(os.path.join(save_path, f'{name}_{t:04d}_basis_color.png'), pred_basis_color)
                        imageio.imwrite(os.path.join(save_path, f'{name}_{t:04d}_view_dep_color.png'), pred_view_dep_color)

                if write_video:
                    all_preds.append(pred)
                    all_preds_depth.append(pred_depth)
                else:
                    imageio.imwrite(os.path.join(save_path, f'{name}_{t:04d}_rgb.png'), pred)
                    imageio.imwrite(os.path.join(save_path, f'{name}_{t:04d}_depth.png'), pred_depth)

                pbar.update(loader.batch_size)
            tok = time.time()
            print("Used time: %.4f, average time: %.4f"%(tok-tik, (tok-tik)/len(loader)))
        if write_video:
            all_preds = np.stack(all_preds, axis=0)
            all_preds_depth = np.stack(all_preds_depth, axis=0)

            def mwrite(filename, frames):
                frames = frames[:, :frames.shape[1]//2*2, :frames.shape[2]//2*2]
                imageio.mimwrite(filename, frames, fps=25, quality=8, macro_block_size=1)
            mwrite(os.path.join(save_path, f'{name}_rgb.mp4'), all_preds)
            mwrite(os.path.join(save_path, f'{name}_depth.mp4'), all_preds_depth)

            if not self.opt.gui:
                all_preds_basis_img = np.stack(all_preds_basis_img, axis=0)
                all_preds_basis_acc = np.stack(all_preds_basis_acc, axis=0)
                all_preds_basis_color = np.stack(all_preds_basis_color, axis=0)
                all_preds_view_dep_color = np.stack(all_preds_view_dep_color, axis=0)

                _, W_img = all_preds_basis_img.shape[1:3]
                W_img = W_img//self.opt.num_basis
                _, W_p = all_preds_basis_color.shape[1:3]
                W_p = W_p//self.opt.num_basis

                mwrite(os.path.join(save_path, '%s_basis_img.mp4'%(name)), all_preds_basis_img)
                mwrite(os.path.join(save_path, '%s_basis_acc.mp4'%(name)), all_preds_basis_acc)
                mwrite(os.path.join(save_path, '%s_basis_color.mp4'%(name)), all_preds_basis_color)
                mwrite(os.path.join(save_path, '%s_view_dep_color.mp4'%(name)), all_preds_view_dep_color)
                for i in range(self.opt.num_basis):
                    mwrite(os.path.join(save_path, '%s_basis_%02d_img.mp4'%(name, i)), all_preds_basis_img[:, :, W_img*i:W_img*(i+1)])
                    mwrite(os.path.join(save_path, '%s_basis_%02d_acc.mp4'%(name, i)), all_preds_basis_acc[:, :, W_img*i:W_img*(i+1)])
                    mwrite(os.path.join(save_path, '%s_basis_%02d_color.mp4'%(name, i)), all_preds_basis_color[:, :, W_p*i:W_p*(i+1)])

        self.log(f"==> Finished Test.")

    # [GUI] test on a single image
    def test_gui(self, pose, intrinsics, W, H, bg_color=None, spp=1, downscale=1, gui_mode=True):
        
        # render resolution (may need downscale to for better frame rate)
        rH = int(H * downscale)
        rW = int(W * downscale)
        intrinsics = intrinsics * downscale

        pose = torch.from_numpy(pose).unsqueeze(0).to(self.device)

        rays = get_rays(pose, intrinsics, rH, rW, -1)
        data = {
            'rays_o': rays['rays_o'],
            'rays_d': rays['rays_d'],
            'H': rH,
            'W': rW,
        }
        
        self.model.eval()

        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=self.fp16):
                # here spp is used as perturb random seed! (but not perturb the first sample)
                output_dict = self.test_step(data, bg_color=bg_color, perturb=False if spp == 1 else spp, gui_mode=gui_mode)
        preds = output_dict['preds'].reshape(-1, rH, rW, 3).clamp(0, 1)
        preds_depth = output_dict['preds_depth'].reshape(-1, rH, rW)
        pred_xyz = output_dict['preds_xyz'].reshape(-1, rH, rW, 3)
        pred_clip_feat = output_dict['clip_feat'].reshape(-1, rH, rW, self.opt.clip_dim)
        # interpolation to the original resolution
        if downscale != 1:
            # TODO: have to permute twice with torch...
            preds = F.interpolate(preds.permute(0, 3, 1, 2), size=(H, W), mode='nearest').permute(0, 2, 3, 1).contiguous()
            preds_depth = F.interpolate(preds_depth.unsqueeze(1), size=(H, W), mode='nearest').squeeze(1)
            pred_xyz = F.interpolate(pred_xyz.permute(0, 3, 1, 2), size=(H, W), mode='nearest').permute(0, 2, 3, 1).contiguous()
            pred_clip_feat = F.interpolate(pred_clip_feat.permute(0, 3, 1, 2), size=(H, W), mode='nearest').permute(0, 2, 3, 1).contiguous()

        if self.opt.color_space == 'linear':
            preds = linear_to_srgb(preds)

        pred = preds[0].detach().cpu().numpy()
        pred_depth = preds_depth[0].detach().cpu().numpy()
        pred_xyz = pred_xyz[0].detach().cpu().numpy()
        pred_clip_feat = pred_clip_feat[0].detach().cpu().numpy()

        outputs = {
            'image': pred,
            'depth': pred_depth,
            'xyz': pred_xyz,
            'clip_feat': pred_clip_feat,
        }

        return outputs
 
    def extract_palette(self, loader, normalize_input=False, save_path=None, name=None):
        '''
            Extract palette using RGBXY palette extraction method with the pretrained vanilla NeRF.
        '''
        if save_path is None:
            save_path = self.workspace

        if name is None:
            name = f'{self.name}_ep{self.epoch:04d}'

        os.makedirs(save_path, exist_ok=True)
        
        self.log(f"==> Start Test, save results to {save_path}")

        pbar = tqdm.tqdm(total=len(loader) * loader.batch_size, bar_format='{percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
        self.model.eval()

        all_preds = []
        all_preds_norm = []
        all_preds_xyz = []
        all_preds_depth = []

        with torch.no_grad():
            for i, data in enumerate(loader):
                with torch.cuda.amp.autocast(enabled=self.fp16):
                    outputs = self.test_step(data)
                    
                preds = data['images'][...,:3].reshape(-1, 3)
                
                def test_img(img_name, img, H, W):
                    img = img.reshape([H, W, 3]).detach().cpu().numpy()
                    img = (img.clip(0,1)*255).astype(np.uint8)
                    cv2.imwrite(img_name, img[...,[2,1,0]])

                if self.opt.color_space == 'linear':
                    preds = srgb_to_linear(preds)
                    
                preds_norm = preds + 0.05
                preds_norm = preds_norm / preds_norm.norm(dim=-1, p=2, keepdim=True)

                preds_depth = outputs['preds_depth'][0]
                preds_xyz = outputs['preds_xyz'][0]
                preds_weight = outputs['preds_weight']
                if preds_weight.shape[0] == 1:
                    preds_weight = preds_weight[0]
                                
                # if i == 0:
                #     test_img("test_norm.png", preds_norm, data['images'].shape[1], data['images'].shape[2])
                #     test_img("test.png", preds, data['images'].shape[1], data['images'].shape[2])
                    
                valid = (preds_weight>5e-1)
                preds = preds[valid]
                preds_norm = preds_norm[valid]
                preds_depth = preds_depth[valid]
                preds_xyz = preds_xyz[valid]
                all_preds.append(preds)
                all_preds_xyz.append(preds_xyz)
                all_preds_depth.append(preds_depth)
                all_preds_norm.append(preds_norm)
                pbar.update(loader.batch_size)

            colors_norm = torch.cat(all_preds_norm, dim=0).detach().cpu().numpy()
            xyzs = torch.cat(all_preds_xyz, dim=0).detach().cpu().numpy()
            input_dict = {"colors":colors_norm, "xyzs":xyzs}
            
            palette_extraction(input_dict, save_path, normalize_input=normalize_input, error_thres=self.opt.error_thres)
            
    def save_checkpoint(self, name=None, full=False, best=False, remove_old=True):

        if name is None:
            name = f'{self.name}_ep{self.epoch:04d}'

        state = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'stats': self.stats,
        }

        if self.model.cuda_ray:
            state['mean_count'] = self.model.mean_count
            state['mean_density'] = self.model.mean_density

        if full:
            state['optimizer'] = self.optimizer.state_dict()
            state['lr_scheduler'] = self.lr_scheduler.state_dict()
            state['scaler'] = self.scaler.state_dict()
            if self.ema is not None:
                state['ema'] = self.ema.state_dict()

        if not best:

            state['model'] = self.model.state_dict()

            file_path = f"{self.ckpt_path}/{name}.pth"

            if remove_old:
                self.stats["checkpoints"].append(file_path)

                if len(self.stats["checkpoints"]) > self.max_keep_ckpt:
                    old_ckpt = self.stats["checkpoints"].pop(0)
                    if os.path.exists(old_ckpt):
                        os.remove(old_ckpt)

            torch.save(state, file_path)

        else:    
            if len(self.stats["results"]) > 0:
                if self.stats["best_result"] is None or self.stats["results"][-1] < self.stats["best_result"]:
                    self.log(f"[INFO] New best result: {self.stats['best_result']} --> {self.stats['results'][-1]}")
                    self.stats["best_result"] = self.stats["results"][-1]

                    # save ema results 
                    if self.ema is not None:
                        self.ema.store()
                        self.ema.copy_to()

                    state['model'] = self.model.state_dict()

                    # we don't consider continued training from the best ckpt, so we discard the unneeded density_grid to save some storage (especially important for dnerf)
                    if 'density_grid' in state['model']:
                        del state['model']['density_grid']

                    if self.ema is not None:
                        self.ema.restore()
                    
                    torch.save(state, self.best_path)
            else:
                self.log(f"[WARN] no evaluated results found, skip saving best checkpoint.")
            
    def load_checkpoint(self, checkpoint=None, model_only=False):
        if checkpoint is None:
            checkpoint_list = sorted(glob.glob(f'{self.ckpt_path}/{self.name}_ep*.pth'))
            if checkpoint_list:
                checkpoint = checkpoint_list[-1]
                self.log(f"[INFO] Latest checkpoint is {checkpoint}")
            else:
                self.log("[WARN] No checkpoint found, model randomly initialized.")
                return

        checkpoint_dict = torch.load(checkpoint, map_location=self.device)
        
        if 'model' not in checkpoint_dict:
            self.model.load_state_dict(checkpoint_dict)
            self.log("[INFO] loaded model.")
            return

        missing_keys, unexpected_keys = self.model.load_state_dict(checkpoint_dict['model'], strict=False)
        self.log("[INFO] loaded model.")
        if len(missing_keys) > 0:
            self.log(f"[WARN] missing keys: {missing_keys}")
        if len(unexpected_keys) > 0:
            self.log(f"[WARN] unexpected keys: {unexpected_keys}")   

        if self.model.cuda_ray:
            if 'mean_count' in checkpoint_dict:
                self.model.mean_count = checkpoint_dict['mean_count']
            if 'mean_density' in checkpoint_dict:
                self.model.mean_density = checkpoint_dict['mean_density']
            else:
                self.log(f"[WARN] mean_density not found, generating new density grid...")   
                with torch.cuda.amp.autocast(enabled=self.fp16):
                    self.model.update_extra_state()
        
        if model_only:
            return

        self.stats = checkpoint_dict['stats']
        self.epoch = checkpoint_dict['epoch']
        self.global_step = checkpoint_dict['global_step']
        self.log(f"[INFO] load at epoch {self.epoch}, global step {self.global_step}")

    def load_nerf_checkpoint(self, ckpt_path):

        checkpoint_list = sorted(glob.glob(f'{ckpt_path}/checkpoints/ngp_ep*.pth'))
        assert(checkpoint_list)

        checkpoint = checkpoint_list[-1]
        self.log(f"[INFO] Latest checkpoint is {checkpoint}")

        checkpoint_dict = torch.load(checkpoint, map_location=self.device)
        missing_keys, unexpected_keys = self.model.load_state_dict(checkpoint_dict['model'], strict=False)
        
        self.log("[INFO] unexpected_keys:", unexpected_keys)
        assert(len(unexpected_keys)==0)

        self.log("[INFO] loaded nerf model.")
        if len(missing_keys) > 0:
            self.log(f"Missing keys should be fine.")        
            
        if self.model.cuda_ray:
            if 'mean_count' in checkpoint_dict:
                self.model.mean_count = checkpoint_dict['mean_count']
            if 'mean_density' in checkpoint_dict:
                self.model.mean_density = checkpoint_dict['mean_density']
            # with torch.cuda.amp.autocast(enabled=self.fp16):
            #     self.model.update_extra_state()

