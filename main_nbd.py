import torch
import argparse

from nbd.provider import NeRFDataset
from nbd.gui import NeRFGUI
from nbd.utils import *

from functools import partial
from loss import huber_loss

#torch.autograd.set_detect_anomaly(True)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str)
    parser.add_argument('-O', action='store_true', help="equals --fp16 --cuda_ray --preload")
    parser.add_argument('--test', action='store_true', help="test mode")
    parser.add_argument('--workspace', type=str, default='workspace')
    parser.add_argument('--seed', type=int, default=0)

    ### training options
    parser.add_argument('--iters', type=int, default=30000, help="training iters")
    parser.add_argument('--lr', type=float, default=1e-2, help="initial learning rate")
    parser.add_argument('--ckpt', type=str, default='scratch')
    parser.add_argument('--num_rays', type=int, default=4096, help="num rays sampled per image for each training step")
    parser.add_argument('--cuda_ray', action='store_true', help="use CUDA raymarching instead of pytorch")
    parser.add_argument('--max_steps', type=int, default=1024, help="max num steps sampled per ray (only valid when using --cuda_ray)")
    parser.add_argument('--num_steps', type=int, default=512, help="num steps sampled per ray (only valid when NOT using --cuda_ray)")
    parser.add_argument('--upsample_steps', type=int, default=0, help="num steps up-sampled per ray (only valid when NOT using --cuda_ray)")
    parser.add_argument('--update_extra_interval', type=int, default=16, help="iter interval to update extra status (only valid when using --cuda_ray)")
    parser.add_argument('--max_ray_batch', type=int, default=4096, help="batch size of rays at inference to avoid OOM (only valid when NOT using --cuda_ray)")
    parser.add_argument('--patch_size', type=int, default=1, help="[experimental] render patches in training, so as to apply LPIPS loss. 1 means disabled, use [64, 32, 16] to enable")

    ### network backbone options
    parser.add_argument('--fp16', action='store_true', help="use amp mixed precision training")
    parser.add_argument('--ff', action='store_true', help="use fully-fused MLP")
    parser.add_argument('--tcnn', action='store_true', help="use TCNN backend")

    ### dataset options
    parser.add_argument('--color_space', type=str, default='srgb', help="Color space, supports (linear, srgb)")
    parser.add_argument('--preload', action='store_true', help="preload all data into GPU, accelerate training but use more GPU memory")
    # (the default value is for the fox dataset)
    parser.add_argument('--bound', type=float, default=2, help="assume the scene is bounded in box[-bound, bound]^3, if > 1, will invoke adaptive ray marching.")
    parser.add_argument('--scale', type=float, default=0.33, help="scale camera location into box[-bound, bound]^3")
    parser.add_argument('--offset', type=float, nargs='*', default=[0, 0, 0], help="offset of camera location")
    parser.add_argument('--dt_gamma', type=float, default=1/128, help="dt_gamma (>=0) for adaptive ray marching. set to 0 to disable, >0 to accelerate rendering (but usually with worse quality)")
    parser.add_argument('--min_near', type=float, default=0.2, help="minimum near distance for camera")
    parser.add_argument('--density_thresh', type=float, default=10, help="threshold for density grid to be occupied")
    parser.add_argument('--bg_radius', type=float, default=-1, help="if positive, use a background model at sphere(bg_radius)")

    ### GUI options
    parser.add_argument('--gui', action='store_true', help="start a GUI")
    parser.add_argument('--W', type=int, default=1920, help="GUI width")
    parser.add_argument('--H', type=int, default=1080, help="GUI height")
    parser.add_argument('--radius', type=float, default=5, help="default GUI camera radius from center")
    parser.add_argument('--fovy', type=float, default=50, help="default GUI camera fovy")
    parser.add_argument('--max_spp', type=int, default=64, help="GUI rendering max sample per pixel")

    ### experimental
    parser.add_argument('--error_map', action='store_true', help="use error map to sample rays")
    parser.add_argument('--clip_text', type=str, default='', help="text input for CLIP guidance")
    parser.add_argument('--rand_pose', type=int, default=-1, help="<0 uses no rand pose, =0 only uses rand pose, >0 sample one rand pose every $ known poses")

    ## Adobe options
    parser.add_argument("--num_basis", type=int, default=12, help='number of basis')
    parser.add_argument("--no_reflect", action='store_true',  help='if specified, use viewing direction instead of reflection')
    
    parser.add_argument('--lambda_roughness', type=float, default=0, help='weight of roughness regularization loss')
    parser.add_argument("--viewdir_roughness", action='store_true', help='if specified, use viewing direction for roughness prediction')
    parser.add_argument("--add_point_roughness", action='store_true', help='if specified, use roughness prediction from network')
    parser.add_argument("--roughness_list",type=float, nargs="+", default=0, help='preset roughness list')
    parser.add_argument("--optimize_roughness", action='store_true', help='if specified, optimize roughness')
    
    parser.add_argument("--use_initialization_from_rays", action='store_true', help='if specified, use initialization from machine learning algorithms')
    parser.add_argument("--color_cluster_num", type=int, help='number of color clusters')
    parser.add_argument("--no_sg", action='store_true', help='if specified, do not use spherical gaussians')
    
    # Adobe Hybrid options   
    parser.add_argument("--no_omega", action='store_true', help='if specified, do not use omega')
    parser.add_argument("--hybrid", action='store_true', help='if specified, use hybrid mode')
    parser.add_argument("--sg_num_basis", type=int, default=0, help='number of sg basis')
    parser.add_argument("--lambda_sparsity", type=float, default=0.01, help='weight of sparsity loss')
    parser.add_argument("--lambda_dir", type=float, default=0.01, help='weight of dir loss')
    parser.add_argument("--guidance", action='store_true', help='if specified, blende color from nerf with color from basis')
    parser.add_argument("--single_radiance", action='store_true', help='if specified, use single radiance instead of multiple')
    parser.add_argument("--max_guide_epoch", type=int, default=20, help='number of maximum epoch for guidance')
    
    opt = parser.parse_args()

    if opt.O:
        opt.fp16 = True
        opt.cuda_ray = True
        opt.preload = True
    
    if opt.patch_size > 1:
        opt.error_map = False # do not use error_map if use patch-based training
        # assert opt.patch_size > 16, "patch_size should > 16 to run LPIPS loss."
        assert opt.num_rays % (opt.patch_size ** 2) == 0, "patch_size ** 2 should be dividable by num_rays."


    if opt.ff:
        raise ValueError("--ff is Not supported.")
        # opt.fp16 = True
        # assert opt.bg_radius <= 0, "background model is not implemented for --ff"
        # from nerf.network_ff import NeRFNetwork
    elif opt.tcnn:
        raise ValueError("--tcnn is Not supported.")
        # opt.fp16 = True
        # assert opt.bg_radius <= 0, "background model is not implemented for --tcnn"
        # from nerf.network_tcnn import NeRFNetwork
    else:
        from nbd.network import NBDNetwork

    print(opt)
    
    seed_everything(opt.seed)

    model = NBDNetwork(
        opt=opt,
        encoding="hashgrid",
        bound=opt.bound,
        cuda_ray=opt.cuda_ray,
        density_scale=1,
        min_near=opt.min_near,
        density_thresh=opt.density_thresh,
        bg_radius=opt.bg_radius,
    )
    
    print(model)

    criterion = torch.nn.MSELoss(reduction='none')
    #criterion = partial(huber_loss, reduction='none')
    #criterion = torch.nn.HuberLoss(reduction='none', beta=0.1) # only available after torch 1.10 ?

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # if opt.ckpt == 'scratch':
    workspace_dir = os.path.join("results", opt.workspace)
    workspace_list = glob.glob("%s/version*"%workspace_dir)
    workspace_list = max([int(x.split("_")[-1]) for x in workspace_list])
    opt.workspace = "%s/version_%d"%(workspace_dir, (1-opt.test)+workspace_list)
    # else:
    #     opt.workspace = os.path.join("results", opt.workspace)

    if opt.test:

        model.initialize_color()

        metrics = [PSNRMeter(), LPIPSMeter(device=device)]
        trainer = Trainer('ngp', opt, model, device=device, workspace=opt.workspace, criterion=criterion, fp16=opt.fp16, metrics=metrics, use_checkpoint=opt.ckpt)
        
        if opt.gui:
            gui = NBDGUI(opt, trainer)
            gui.render()
        
        else:
            test_loader = NeRFDataset(opt, device=device, type='test', n_test=1).dataloader()

            if test_loader.has_gt:
                trainer.evaluate(test_loader) # blender has gt, so evaluate it.
    
            trainer.test(test_loader, write_video=True) # test and save video
            
            trainer.save_mesh(resolution=256, threshold=10)
    
    else:

        optimizer = lambda model: torch.optim.Adam(model.get_params(opt.lr), betas=(0.9, 0.99), eps=1e-15)

        train_dataset = NeRFDataset(opt, device=device, type='train')
        train_loader = train_dataset.dataloader()        
        if opt.use_initialization_from_rays:
            model.initialize_color(train_dataset.cluster_color)

        # decay to 0.1 * init_lr at last iter step
        scheduler = lambda optimizer: optim.lr_scheduler.LambdaLR(optimizer, lambda iter: 0.1 ** min(iter / opt.iters, 1))

        metrics = [PSNRMeter(), LPIPSMeter(device=device)]
        trainer = Trainer('ngp', opt, model, device=device, workspace=opt.workspace, optimizer=optimizer, criterion=criterion, ema_decay=0.95, fp16=opt.fp16, lr_scheduler=scheduler, scheduler_update_every_step=True, metrics=metrics, use_checkpoint=opt.ckpt, eval_interval=120)


        if opt.gui:
            gui = NBDGUI(opt, trainer, train_loader)
            gui.render()
        
        else:
            valid_loader = NeRFDataset(opt, device=device, type='val', downscale=1).dataloader()

            max_epoch = np.ceil(opt.iters / len(train_loader)).astype(np.int32)
            trainer.train(train_loader, valid_loader, max_epoch)

            # also test
            test_loader = NeRFDataset(opt, device=device, type='test', n_test=10).dataloader()
            
            if test_loader.has_gt:
                trainer.evaluate(test_loader) # blender has gt, so evaluate it.
            
            trainer.test(test_loader, write_video=True) # test and save video
            
            trainer.save_mesh(resolution=256, threshold=10)