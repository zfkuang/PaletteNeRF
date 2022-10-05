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

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributed as dist
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

try:
    import _nbd_palette as _backend
except ImportError:
    print("Loading module nbd_palette...")
    from .backend import _backend

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
    use_normalize = False
):
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
        error_thres=5.0/255.0,
        target_size=palette_size)

    _, hist_rgb = compute_RGB_histogram(colors, weights, bits_per_channel=5)
    if use_normalize:
        hist_rgb = hist_rgb+0.05
        hist_rgb_norm = np.linalg.norm(hist_rgb, axis=-1, keepdims=True)
        hist_rgb = hist_rgb / hist_rgb_norm

    hist_weights = Tan18.Get_ASAP_weights_using_Tan_2016_triangulation_and_then_barycentric_coordinates(hist_rgb.astype(np.double).reshape((-1,1,3)), 
                        palette_rgb, None, order=0) # N_bin_center x 1 x num_palette
    hist_weights = hist_weights.reshape([32,32,32,palette_rgb.shape[0]])
    if use_normalize:
        hist_weights = hist_weights * hist_rgb_norm.reshape([32, 32, 32, 1])
    ## Generate weight

    ## save palette
    palette_img = get_bigger_palette_to_show(palette_rgb)
    Image.fromarray((palette_img*255).round().clip(0,255).astype(np.uint8)).save(output_prefix+"-palette.png")

    write_palette_txt(palette_rgb, output_prefix+'-palette.txt')
    np.savez(os.path.join(output_dir, 'palette.npz'), palette=palette_rgb)
    np.savez(os.path.join(output_dir, 'hist_weights.npz'), hist_weights=hist_weights)



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
             
        # clip loss prepare
        if opt.rand_pose >= 0: # =0 means only using CLIP loss, >0 means a hybrid mode.
            from nerf.clip_utils import CLIPLoss
            self.clip_loss = CLIPLoss(self.device)
            self.clip_loss.prepare_text([self.opt.clip_text]) # only support one text prompt now...

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
        gt_weights = get_palette_weight_with_hist(gt_rgb, self.model.hist_weights).detach()
        
        # outputs = self.model.render(rays_o, rays_d, staged=False, bg_color=bg_color, perturb=True, force_all_rays=False if self.opt.patch_size == 1 else True, **vars(self.opt))
        outputs = self.model.render(rays_o, rays_d, staged=False, bg_color=bg_color, perturb=True, force_all_rays=True, **vars(self.opt))
        
        pred_rgb = outputs['image']
        
        # MSE loss
        loss = self.criterion(pred_rgb, gt_rgb).mean(-1) # [B, N, 3] --> [B, N]
        loss_direct = self.criterion(outputs['direct_rgb'], gt_rgb).mean(-1) # [B, N, 3] --> [B, N]
        loss += loss_direct
        # patch-based rendering
        if self.opt.patch_size > 1:
            gt_rgb = gt_rgb.view(-1, self.opt.patch_size, self.opt.patch_size, 3).permute(0, 3, 1, 2).contiguous()
            pred_rgb = pred_rgb.view(-1, self.opt.patch_size, self.opt.patch_size, 3).permute(0, 3, 1, 2).contiguous()

            # torch_vis_2d(gt_rgb[0])
            # torch_vis_2d(pred_rgb[0])

            # LPIPS loss [not useful...]
            loss = loss + 1e-3 * self.criterion_lpips(pred_rgb, gt_rgb)

        loss_dict = {}
        pred_omega = outputs['omega_norm']
        sparsity_loss = pred_omega.mean()

        pred_delta_color = outputs['delta_norm']
        delta_loss = pred_delta_color.mean()

        pred_dir_color = outputs['dir_norm']
        dir_loss = pred_dir_color.mean()

        pred_weights = outputs['basis_acc']
        weights_guide_loss = ((gt_weights-pred_weights)**2).mean()
        
        if self.opt.lambda_smooth > 0 and self.model.require_smooth_loss:
            pred_smooth_norm = outputs['smooth_norm']
            smooth_loss = pred_smooth_norm.mean()
        else:
            smooth_loss = 0

        loss = loss + self.opt.lambda_sparsity * sparsity_loss + self.opt.lambda_delta * delta_loss + self.opt.lambda_dir * dir_loss
        loss = loss + smooth_loss * self.opt.lambda_smooth
        loss = loss + weights_guide_loss * self.lambda_weight

        loss_dict['loss_sparsity'] =  self.opt.lambda_sparsity * sparsity_loss
        loss_dict['loss_delta'] =  self.opt.lambda_delta * delta_loss
        loss_dict['loss_dir'] =  self.opt.lambda_dir * dir_loss
        loss_dict['loss_smooth'] =  self.opt.lambda_smooth * smooth_loss
        loss_dict['loss_weight'] =  self.lambda_weight * weights_guide_loss
        loss_dict['loss_weight_norm'] =  self.opt.lambda_weight * weights_guide_loss
        loss_dict['loss_direct'] = loss_direct.mean()

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
        gt_weights = get_palette_weight_with_hist(gt_rgb, self.model.hist_weights).detach()
        
        outputs = self.model.render(rays_o, rays_d, staged=True, bg_color=bg_color, perturb=False, test_mode=True, **vars(self.opt))

        pred_rgb = outputs['image'].reshape(B, H, W, 3)
        pred_depth = outputs['depth'].reshape(B, H, W)

        loss = self.criterion(pred_rgb, gt_rgb).mean()

        return pred_rgb, pred_depth, gt_rgb, gt_weights, loss, outputs
    
    def train(self, train_loader, valid_loader, max_epochs):
        if self.use_tensorboardX and self.local_rank == 0:
            self.writer = tensorboardX.SummaryWriter(os.path.join(self.workspace, "run", self.name))

        # mark untrained region (i.e., not covered by any camera from the training dataset)
        if self.model.cuda_ray:
            self.model.mark_untrained_grid(train_loader._data.poses, train_loader._data.intrinsics)

        # get a ref to error_map
        self.error_map = train_loader._data.error_map
        
        for epoch in range(self.epoch + 1, max_epochs + 1):
            self.epoch = epoch

            self.lambda_weight = self.opt.lambda_weight * max(0, (1 - epoch/self.opt.lweight_decay_epoch))
            self.train_one_epoch(train_loader)

            if self.workspace is not None and self.local_rank == 0:
                self.save_checkpoint(full=True, best=False)

            if self.epoch % self.eval_interval == 0:
                self.evaluate_one_epoch(valid_loader)
                self.save_checkpoint(full=False, best=True)
            
            if epoch >= self.opt.max_freeze_palette_epoch or not self.opt.use_initialization_from_rgbxy:
                self.model.freeze_basis_color = False
            if epoch >= self.opt.smooth_epoch:
                self.model.require_smooth_loss = True

        if self.use_tensorboardX and self.local_rank == 0:
            self.writer.close()

    def evaluate(self, loader, name=None):
        self.use_tensorboardX, use_tensorboardX = False, self.use_tensorboardX
        self.evaluate_one_epoch(loader, name)
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
            if self.model.cuda_ray and self.global_step % self.opt.update_extra_interval == 0:
                with torch.cuda.amp.autocast(enabled=self.fp16):
                    self.model.update_extra_state()
                    
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

    def evaluate_one_epoch(self, loader, name=None):
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

            for data in loader:    
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
                        metric.update(preds, truths)
                    # TODO: add evaluation here
                    # save image
                    save_path = os.path.join(self.workspace, 'validation', f'{name}_{self.local_step:04d}_rgb.png')
                    save_path_depth = os.path.join(self.workspace, 'validation', f'{name}_{self.local_step:04d}_depth.png')
                    save_path_gt_weights = os.path.join(self.workspace, 'validation', f'{name}_{self.local_step:04d}_basis_acc_guide.png')
                    
                    save_path_basis_img = os.path.join(self.workspace, 'validation', f'{name}_{self.local_step:04d}_basis_img.png')
                    save_path_basis_acc = os.path.join(self.workspace, 'validation', f'{name}_{self.local_step:04d}_basis_acc.png')
                    save_path_basis_color = os.path.join(self.workspace, 'validation', f'{name}_{self.local_step:04d}_basis_color.png')
                    save_path_dir_color = os.path.join(self.workspace, 'validation', f'{name}_{self.local_step:04d}_dir_color.png')

                    #self.log(f"==> Saving validation image to {save_path}")
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)

                    if self.opt.color_space == 'linear':
                        preds = linear_to_srgb(preds)

                    gt_weights = gt_weights[0].detach().cpu().numpy()
                    gt_weights = (gt_weights.clip(0,1) * 255).astype(np.uint8)
                    gt_weights = gt_weights.transpose(0, 2, 1).reshape(gt_weights.shape[0], -1)

                    pred = preds[0].detach().cpu().numpy()
                    pred = (pred.clip(0,1) * 255).astype(np.uint8)
                    pred_depth = preds_depth[0].detach().cpu().numpy()
                    pred_depth = (pred_depth * 255).astype(np.uint8)
                    pred_dir_color = outputs['dir_rgb'][0].detach().cpu().numpy()
                    pred_dir_color = pred_dir_color.reshape(pred.shape[0], pred.shape[1], 3)
                    pred_dir_color = (pred_dir_color.clip(0,1) * 255).astype(np.uint8)
                    pred_direct_color = outputs['direct_rgb'][0].detach().cpu().numpy()
                    pred_direct_color = pred_direct_color.reshape(pred.shape[0], pred.shape[1], 3)
                    pred_direct_color = (pred_direct_color.clip(0,1) * 255).astype(np.uint8)

                    pred_basis_img = []
                    pred_basis_acc = []
                    pred_basis_color = []
                    for i in range(self.opt.num_basis):
                        basis_img = outputs['basis_rgb'][0,:,i*3:(i+1)*3].reshape(pred.shape[0], pred.shape[1], 3)
                        pred_basis_img.append(basis_img.detach().cpu().numpy())                        
                        basis_acc = outputs['basis_acc'][0,:,i:(i+1)].reshape(pred.shape[0], pred.shape[1])
                        pred_basis_acc.append(basis_acc.detach().cpu().numpy())
                        basis_color = self.model.basis_color[i,None,None,:].repeat(100, 100, 1)
                        basis_color = basis_color.clamp(0, 1)
                        # basis_color = basis_color/(basis_color.norm(dim=-1, keepdim=True)+1e-6).detach()
                        # basis_color = lab_to_rgb(torch.concatenate([basis_color[...,:1]*0+75, basis_color[...,1:]], dim=-1))
                        pred_basis_color.append(basis_color.detach().cpu().numpy())

                    pred_basis_img = (np.concatenate(pred_basis_img, axis=1).clip(0,1)*255).astype(np.uint8)
                    pred_basis_acc = (np.concatenate(pred_basis_acc, axis=1).clip(0,1)*255).astype(np.uint8)
                    pred_basis_color = (np.concatenate(pred_basis_color, axis=1).clip(0,1)*255).astype(np.uint8)

                    cv2.imwrite(save_path_basis_img, cv2.cvtColor(pred_basis_img, cv2.COLOR_RGB2BGR))
                    cv2.imwrite(save_path_basis_acc, pred_basis_acc)
                    cv2.imwrite(save_path_basis_color, cv2.cvtColor(pred_basis_color, cv2.COLOR_RGB2BGR))
                    cv2.imwrite(save_path_dir_color, cv2.cvtColor(pred_dir_color, cv2.COLOR_RGB2BGR))
                    
                    cv2.imwrite(save_path, cv2.cvtColor(pred, cv2.COLOR_RGB2BGR))
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
        outputs['preds_xyz'] = data['rays_o'] + data['rays_d'] * outputs['depth'][...,None]
        outputs['preds_weight'] = outputs['weights_sum']

        return outputs 

    def test(self, loader, save_path=None, name=None, write_video=True):

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
            all_preds_dir_color = []

        with torch.no_grad():

            for i, data in enumerate(loader):
                
                H, W = data['H'], data['W']
                with torch.cuda.amp.autocast(enabled=self.fp16):
                    outputs = self.test_step(data)


                pred = outputs['preds'][0].reshape(H, W, 3)
                
                if self.opt.color_space == 'linear':
                    pred = linear_to_srgb(pred)
                    
                pred = pred.detach().cpu().numpy()
                pred = (pred.clip(0, 1) * 255).astype(np.uint8)

                pred_depth = outputs['preds_depth'][0].reshape(H, W).detach().cpu().numpy()
                pred_depth = (pred_depth * 255).astype(np.uint8)
                
                pred_basis_dir_color = outputs['dir_rgb'][0].reshape(H, W, 3)
                pred_basis_dir_color = pred_basis_dir_color.detach().cpu().numpy()
                pred_basis_dir_color = (pred_basis_dir_color.clip(0, 1) * 255).astype(np.uint8)

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
                    all_preds.append(pred)
                    all_preds_depth.append(pred_depth)
                    all_preds_basis_img.append(pred_basis_img)
                    all_preds_basis_acc.append(pred_basis_acc)
                    all_preds_basis_color.append(pred_basis_color)
                    all_preds_dir_color.append(pred_basis_dir_color)

                else:
                    cv2.imwrite(os.path.join(save_path, f'{name}_{i:04d}_rgb.png'), cv2.cvtColor(pred, cv2.COLOR_RGB2BGR))
                    cv2.imwrite(os.path.join(save_path, f'{name}_{i:04d}_depth.png'), pred_depth)
                    cv2.imwrite(os.path.join(save_path, f'{name}_{i:04d}_basis_img.png'), pred_basis_img)
                    cv2.imwrite(os.path.join(save_path, f'{name}_{i:04d}_basis_acc.png'), pred_basis_acc)
                    cv2.imwrite(os.path.join(save_path, f'{name}_{i:04d}_basis_color.png'), pred_basis_color)
                    cv2.imwrite(os.path.join(save_path, f'{name}_{i:04d}_dir_color.png'), all_preds_dir_color)

                pbar.update(loader.batch_size)
        
        if write_video:
            all_preds = np.stack(all_preds, axis=0)
            all_preds_depth = np.stack(all_preds_depth, axis=0)
            all_preds_basis_img = np.stack(all_preds_basis_img, axis=0)
            all_preds_basis_acc = np.stack(all_preds_basis_acc, axis=0)
            all_preds_basis_color = np.stack(all_preds_basis_color, axis=0)
            all_preds_dir_color = np.stack(all_preds_dir_color, axis=0)

            _, W_img = all_preds_basis_img.shape[1:3]
            W_img = W_img//self.opt.num_basis
            _, W_p = all_preds_basis_color.shape[1:3]
            W_p = W_p//self.opt.num_basis
            imageio.mimwrite(os.path.join(save_path, f'{name}_rgb.mp4'), all_preds, fps=25, quality=8, macro_block_size=1)
            imageio.mimwrite(os.path.join(save_path, f'{name}_depth.mp4'), all_preds_depth, fps=25, quality=8, macro_block_size=1)
            imageio.mimwrite(os.path.join(save_path, '%s_basis_img.mp4'%(name)), all_preds_basis_img, fps=25, quality=8, macro_block_size=1)
            imageio.mimwrite(os.path.join(save_path, '%s_basis_acc.mp4'%(name)), all_preds_basis_acc, fps=25, quality=8, macro_block_size=1)
            imageio.mimwrite(os.path.join(save_path, '%s_basis_color.mp4'%(name)), all_preds_basis_color, fps=25, quality=8, macro_block_size=1)
            imageio.mimwrite(os.path.join(save_path, '%s_dir_color.mp4'%(name)), all_preds_dir_color, fps=25, quality=8, macro_block_size=1)
            for i in range(self.opt.num_basis):
                imageio.mimwrite(os.path.join(save_path, '%s_basis_%02d_img.mp4'%(name, i)), all_preds_basis_img[:, :, W_img*i:W_img*(i+1)], fps=25, quality=8, macro_block_size=1)
                imageio.mimwrite(os.path.join(save_path, '%s_basis_%02d_acc.mp4'%(name, i)), all_preds_basis_acc[:, :, W_img*i:W_img*(i+1)], fps=25, quality=8, macro_block_size=1)
                imageio.mimwrite(os.path.join(save_path, '%s_basis_%02d_color.mp4'%(name, i)), all_preds_basis_color[:, :, W_p*i:W_p*(i+1)], fps=25, quality=8, macro_block_size=1)

        self.log(f"==> Finished Test.")

    # [GUI] test on a single image
    def test_gui(self, pose, intrinsics, W, H, bg_color=None, spp=1, downscale=1):
        
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
                output_dict = self.test_step(data, bg_color=bg_color, perturb=False if spp == 1 else spp, gui_mode=True)
        preds = output_dict['preds'].reshape(-1, rH, rW, 3).clamp(0, 1)
        preds_depth = output_dict['preds_depth'].reshape(-1, rH, rW)

        # interpolation to the original resolution
        if downscale != 1:
            # TODO: have to permute twice with torch...
            preds = F.interpolate(preds.permute(0, 3, 1, 2), size=(H, W), mode='nearest').permute(0, 2, 3, 1).contiguous()
            preds_depth = F.interpolate(preds_depth.unsqueeze(1), size=(H, W), mode='nearest').squeeze(1)

        if self.opt.color_space == 'linear':
            preds = linear_to_srgb(preds)

        pred = preds[0].detach().cpu().numpy()
        pred_depth = preds_depth[0].detach().cpu().numpy()

        outputs = {
            'image': pred,
            'depth': pred_depth,
        }

        return outputs
 
    def sample_rays(self, loader, save_path=None, name=None):

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
                
                def test_img(img):
                    img = img.reshape([800, 800, 3]).detach().cpu().numpy()
                    img = (img.clip(0,1)*255).astype(np.uint8)
                    cv2.imwrite("test.png", img[...,[2,1,0]])

                if self.opt.color_space == 'linear':
                    preds = srgb_to_linear(preds)

                preds_norm = preds+0.05
                preds_norm = preds_norm / preds_norm.norm(dim=-1, keepdim=True)
                preds_depth = outputs['preds_depth'][0]
                preds_xyz = outputs['preds_xyz'][0]
                preds_weight = outputs['preds_weight'][0]
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
            
            colors = torch.cat(all_preds, dim=0).detach().cpu().numpy()
            xyzs = torch.cat(all_preds_xyz, dim=0).detach().cpu().numpy()
            input_dict = {"colors":colors, "xyzs":xyzs}
            palette_extraction(input_dict, save_path)
            colors_norm = torch.cat(all_preds_norm, dim=0).detach().cpu().numpy()
            xyzs = torch.cat(all_preds_xyz, dim=0).detach().cpu().numpy()
            input_dict = {"colors":colors_norm, "xyzs":xyzs}
            palette_extraction(input_dict, save_path.replace("version", "normalized_version"), use_normalize=True)

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

    def load_nerf_checkpoint(self, ckpt_path=None):
        checkpoint_dict = torch.load(ckpt_path, map_location=self.device)
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
            with torch.cuda.amp.autocast(enabled=self.fp16):
                self.model.update_extra_state()
