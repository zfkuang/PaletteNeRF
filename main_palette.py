import torch
import argparse

from palette.provider import NeRFDataset
from palette.gui import PaletteGUI
from palette.utils import PaletteTrainer
from palette.utils import *

from functools import partial
from loss import huber_loss

#torch.autograd.set_detect_anomaly(True)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str)
    parser.add_argument('nerf_path', type=str)
    parser.add_argument('-O', action='store_true', help="equals --fp16 --cuda_ray --preload")
    parser.add_argument('--test', action='store_true', help="test mode")
    parser.add_argument('--seed', type=int, default=0)

    ### training options
    parser.add_argument('--iters', type=int, default=30000, help="training iters")
    parser.add_argument('--ckpt', type=str, default='latest')
    parser.add_argument('--lr', type=float, default=1e-2, help="initial learning rate")
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

    ### dataset options
    parser.add_argument('--color_space', type=str, default='srgb', choices=['srgb', 'linear'], help="Color space, supports (linear, srgb)")
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
    parser.add_argument('--W', type=int, default=960, help="GUI width")
    parser.add_argument('--H', type=int, default=540, help="GUI height")
    parser.add_argument('--radius', type=float, default=5, help="default GUI camera radius from center")
    parser.add_argument('--fovy', type=float, default=50, help="default GUI camera fovy")
    parser.add_argument('--max_spp', type=int, default=64, help="GUI rendering max sample per pixel")

    ### experimental
    parser.add_argument('--error_map', action='store_true', help="use error map to sample rays")
    parser.add_argument('--clip_text', type=str, default='', help="text input for CLIP guidance")
    parser.add_argument('--rand_pose', type=int, default=-1, help="<0 uses no rand pose, =0 only uses rand pose, >0 sample one rand pose every $ known poses")

    ### Palette 
    parser.add_argument('--extract_palette', action='store_true', help="extract palette")
    parser.add_argument('--update_grid', action='store_true', help="update density grid")
    parser.add_argument("--num_basis", type=int, default=12, help='number of basis')
    parser.add_argument("--use_initialization_from_rgbxy", action='store_true', help='if specified, use initialization from rgbxy')

    # Adobe Hybrid options   
    parser.add_argument('--use_normalized_palette', action='store_true', help="use_normalized palette")
    parser.add_argument('--continue_training', action='store_true', help="continue training")
    parser.add_argument('--multiply_delta', action='store_true', help="multiply basis color with delta color")
    parser.add_argument("--lambda_sparsity", type=float, default=0.002, help='weight of sparsity loss')
    parser.add_argument("--lambda_smooth", type=float, default=0.002, help='weight of smooth loss')
    parser.add_argument("--lambda_dir", type=float, default=0.02, help='weight of dir loss')
    parser.add_argument("--lambda_delta", type=float, default=0.1, help='weight of delta color loss')
    parser.add_argument("--lambda_weight", type=float, default=0.2, help='weight of weight loss')
    parser.add_argument("--lweight_decay_epoch", type=int, default=100, help='epoch number when lambda weight drops to 0')
    parser.add_argument("--max_freeze_palette_epoch", type=int, default=100, help='number of maximum epoch to freeze palette color')
    parser.add_argument("--smooth_epoch", type=int, default=100, help='number of maximum epoch before add smooth loss')
    parser.add_argument("--model_mode", type=str, choices=["nerf", "palette"], default="nerf", help='type of model')
    # parser.add_argument("--max_freeze_geometry_epoch", type=int, default=20, help='number of maximum epoch to freeze geometry')
    
    # CLIP feat options
    parser.add_argument('--pred_clip', action='store_true', help="predict clip featuer")
    parser.add_argument("--clip_dim", type=int, default=16, help='dimension of clip feature')

    opt = parser.parse_args()


    if opt.O:
        opt.fp16 = True
        opt.cuda_ray = True
        opt.preload = True
    
    if opt.patch_size > 1:
        opt.error_map = False # do not use error_map if use patch-based training
        # assert opt.patch_size > 16, "patch_size should > 16 to run LPIPS loss."
        assert opt.num_rays % (opt.patch_size ** 2) == 0, "patch_size ** 2 should be dividable by num_rays."

    palette_workspace=os.path.dirname(os.path.dirname(opt.nerf_path)).replace("results", "results_palette")
    if opt.use_normalized_palette:
        palette_workspace = palette_workspace.replace("version", "normalized_version")
    os.makedirs(palette_workspace, exist_ok=True)
    
    workspace = os.path.dirname(palette_workspace)
    workspace_list = glob.glob("%s/version*"%workspace)
    workspace_list = max([0] + [int(x.split("_")[-1]) for x in workspace_list])
    workspace = "%s/version_%d"%(workspace, (1-max(opt.test, opt.continue_training))+workspace_list) 

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if opt.use_initialization_from_rgbxy:
        palette = np.load(os.path.join(palette_workspace, 'palette.npz'))['palette']
        opt.num_basis = palette.shape[0]

    print(opt)
    seed_everything(opt.seed)
    
    if opt.model_mode == "nerf":        
        from nerf.network import NeRFNetwork
        model = NeRFNetwork(
            encoding="hashgrid",
            bound=opt.bound,
            cuda_ray=opt.cuda_ray,
            density_scale=1,
            min_near=opt.min_near,
            density_thresh=opt.density_thresh,
            bg_radius=opt.bg_radius,
        )
    elif opt.model_mode == "palette":
        from palette.network import PaletteNetwork
        model = PaletteNetwork(
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

    if opt.extract_palette:    
        trainer = PaletteTrainer('palette', opt, model, device=device, 
                fp16=opt.fp16, workspace=palette_workspace, nerf_path=opt.nerf_path)
        train_loader = NeRFDataset(opt, device=device, type='traintest').dataloader()
        trainer.sample_rays(train_loader) # test and save video
    elif opt.update_grid:
        trainer = PaletteTrainer('palette', opt, model, device=device, workspace=workspace, fp16=opt.fp16,
                                use_checkpoint=opt.ckpt, nerf_path=None)
        train_loader = NeRFDataset(opt, device=device, type='train').dataloader()
        trainer.model.mark_untrained_grid(train_loader._data.poses, train_loader._data.intrinsics)
        trainer.model.update_extra_state()
        trainer.save_checkpoint(full=True, best=False)
    elif opt.test:
        trainer = PaletteTrainer('palette', opt, model, device=device, workspace=workspace, fp16=opt.fp16,
                                use_checkpoint=opt.ckpt, nerf_path=None)
        if opt.gui:
            assert(os.path.exists(os.path.join(palette_workspace, 'palette.npz')))
            palette = np.load(os.path.join(palette_workspace, 'palette.npz'))['palette']
            hist_weights = np.load(os.path.join(palette_workspace, 'hist_weights.npz'))['hist_weights']
            gui = PaletteGUI(opt, trainer, palette, hist_weights)
            gui.render()
        else:
            test_loader = NeRFDataset(opt, device=device, type='test', n_test=30).dataloader()

            if test_loader.has_gt:
                trainer.evaluate(test_loader) # blender has gt, so evaluate it.

            trainer.test(test_loader, write_video=True) # test and save video
    else:      
        assert(os.path.exists(os.path.join(palette_workspace, 'palette.npz')))
        palette = np.load(os.path.join(palette_workspace, 'palette.npz'))['palette']
        hist_weights = np.load(os.path.join(palette_workspace, 'hist_weights.npz'))['hist_weights']
        if opt.use_initialization_from_rgbxy:
            model.initialize_color(palette, hist_weights)
            assert(palette.shape[0] == opt.num_basis)

        criterion = torch.nn.MSELoss(reduction='none')
        optimizer = lambda model: torch.optim.Adam(model.get_params(opt.lr), betas=(0.9, 0.99), eps=1e-15)
        # decay to 0.1 * init_lr at last iter step
        scheduler = lambda optimizer: optim.lr_scheduler.LambdaLR(optimizer, lambda iter: 0.1 ** min(iter / opt.iters, 1))
        metrics = [PSNRMeter(), LPIPSMeter(device=device)]

        trainer = PaletteTrainer('palette', opt, model, device=device, workspace=workspace, optimizer=optimizer, criterion=criterion, 
            ema_decay=0.95, fp16=opt.fp16, lr_scheduler=scheduler, scheduler_update_every_step=True, metrics=metrics, 
            use_checkpoint=opt.ckpt, nerf_path=opt.nerf_path, eval_interval=50)

        train_loader = NeRFDataset(opt, device=device, type='train').dataloader()
        # palette = np.concatenate([palette[:-2], palette[-2:]], axis=0)


        valid_loader = NeRFDataset(opt, device=device, type='val', downscale=1).dataloader()

        max_epoch = np.ceil(opt.iters / len(train_loader)).astype(np.int32)
        trainer.train(train_loader, valid_loader, max_epoch)

        # also test
        test_loader = NeRFDataset(opt, device=device, type='test', n_test=30).dataloader()
        
        if test_loader.has_gt:
            trainer.evaluate(test_loader) # blender has gt, so evaluate it.
        
        trainer.test(test_loader, write_video=True) # test and save vide