from configparser import Interpolation
import math
import torch
import torch.optim as optim
import numpy as np
import pickle
import dearpygui.dearpygui as dpg
from scipy.spatial.transform import Rotation as R

from nerf.utils import *
from .renderer import RegionEdit, Stylizer


class OrbitCamera:
    def __init__(self, W, H, r=2, fovy=60):
        self.W = W
        self.H = H
        self.radius = r # camera distance from center
        self.fovy = fovy # in degree
        self.center = np.array([0, 0, 0], dtype=np.float32) # look at this point
        self.rot = R.from_quat([1, 0, 0, 0]) # init camera matrix: [[1, 0, 0], [0, -1, 0], [0, 0, 1]] (to suit ngp convention)
        self.up = np.array([0, 1, 0], dtype=np.float32) # need to be normalized!

    # pose
    @property
    def pose(self):
        # first move camera to radius
        res = np.eye(4, dtype=np.float32)
        res[2, 3] -= self.radius
        # rotate
        rot = np.eye(4, dtype=np.float32)
        rot[:3, :3] = self.rot.as_matrix()
        res = rot @ res
        # translate
        res[:3, 3] -= self.center
        return res
    
    # intrinsics
    @property
    def intrinsics(self):
        focal = self.H / (2 * np.tan(np.radians(self.fovy) / 2))
        return np.array([focal, focal, self.W // 2, self.H // 2])
    
    def orbit(self, dx, dy):
        # rotate along camera up/side axis!
        side = self.rot.as_matrix()[:3, 0] # why this is side --> ? # already normalized.
        rotvec_x = self.up * np.radians(-0.1 * dx)
        rotvec_y = side * np.radians(-0.1 * dy)
        self.rot = R.from_rotvec(rotvec_x) * R.from_rotvec(rotvec_y) * self.rot

    def scale(self, delta):
        self.radius *= 1.1 ** (-delta)

    def pan(self, dx, dy, dz=0):
        # pan in camera coordinate system (careful on the sensitivity!)
        self.center += 0.0005 * self.rot.as_matrix()[:3, :3] @ np.array([dx, dy, dz])
    

class PaletteGUI:
    def __init__(self, opt, trainer, palette, hist_weights, train_loader=None, video_loader=None, debug=True):
        self.opt = opt # shared with the trainer's opt to support in-place modification of rendering parameters.
        self.W = opt.W
        self.H = opt.H
        self.cam = OrbitCamera(opt.W, opt.H, r=opt.radius, fovy=opt.fovy)
        self.debug = debug
        self.bg_color = torch.ones(3, dtype=torch.float32) # default white bg
        self.training = False
        self.step = 0 # training step 

        self.trainer = trainer
        self.train_loader = train_loader
        self.video_loader = video_loader
        if train_loader is not None:
            self.trainer.error_map = train_loader._data.error_map

        self.render_buffer = np.zeros((self.W, self.H, 3), dtype=np.float32)
        self.selected_point = None
        self.selected_pixel = None 
        self.need_update = True # camera moved, should reset accumulation
        self.spp = 1 # sample per pixel
        self.mode = 'image' # choose from ['image', 'depth']

        self.dynamic_resolution = True
        self.downscale = 1
        self.train_steps = 16
        
        self.stylize = False
        self.need_optimize_stylize = False
        self.cached_stylizer = None
        self.lambda_ARAP = 0.1
        self.style_point_list = []
        self.style_color_list = []
        self.style_pixel = None
        self.style_image = np.zeros((400, 400, 3), dtype=np.float32)
        self.drawed_style_image = self.style_image
        self.style_W = 0
        self.style_H = 0

        self.trainer.model.edit = RegionEdit(opt)
        self.load_palette()

        dpg.create_context()
        self.register_dpg()
        self.test_step()
        
    def load_palette(self):
        self.weight_mode = False
        self.palette = self.trainer.model.basis_color.clone()
        self.origin_palette = self.palette.clone()
        # self.hist_weights = torch.from_numpy(hist_weights).float()
        # self.hist_weights = self.hist_weights.permute(3, 0, 1, 2).unsqueeze(0)
        self.highlight_palette_id = 0
        
    def __del__(self):
        dpg.destroy_context()


    def train_step(self):

        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        starter.record()

        outputs = self.trainer.train_gui(self.train_loader, step=self.train_steps)

        ender.record()
        torch.cuda.synchronize()
        t = starter.elapsed_time(ender)

        self.step += self.train_steps
        self.need_update = True

        dpg.set_value("_log_train_time", f'{t:.4f}ms ({int(1000/t)} FPS)')
        dpg.set_value("_log_train_log", f'step = {self.step: 5d} (+{self.train_steps: 2d}), loss = {outputs["loss"]:.4f}, lr = {outputs["lr"]:.5f}')

        # dynamic train steps
        # max allowed train time per-frame is 500 ms
        full_t = t / self.train_steps * 16
        train_steps = min(16, max(4, int(16 * 500 / full_t)))
        if train_steps > self.train_steps * 1.2 or train_steps < self.train_steps * 0.8:
            self.train_steps = train_steps

    def prepare_buffer(self, outputs):
        if self.mode == 'image':
            return outputs['image']
        else:
            return np.expand_dims(outputs['depth'], -1).repeat(3, -1)

    def test_step(self):
        # TODO: seems we have to move data from GPU --> CPU --> GPU?
        
        # Optimize for stylization
        if self.need_optimize_stylize and len(self.style_point_list) > 0:
            xyzs = torch.stack(self.style_point_list, dim=0)
            gt_rgbs = torch.from_numpy(np.stack(self.style_color_list, axis=0)).type_as(xyzs)
            dirs = torch.zeros_like(xyzs)
            M = xyzs.shape[0]
            
            with torch.no_grad():
                _, _, d_color, omega, _, diffuse = self.trainer.model(xyzs, dirs)
                radiance = d_color[...,-1:]
                d_color = d_color[...,:-1]
                radiance = radiance.reshape(M, 1, 1)
                d_color = d_color.reshape(M, self.opt.num_basis, 3)
                omega = omega.reshape(M, self.opt.num_basis, 1)
                diffuse = diffuse.reshape(M, 3)
                basis_color = self.trainer.model.basis_color[None,:,:].clamp(0, 1)
            
            total_iters = 1000
            stylizer = Stylizer(self.opt).to(xyzs.device)
            style_optimizer = optim.SGD(stylizer.parameters(), lr=0.01) 
            style_scheduler = optim.lr_scheduler.LambdaLR(style_optimizer, lambda iter: 0.1 ** min(iter /total_iters, 1))
            loss = 0 
            
            pbar = tqdm.trange(total_iters)
            for i in pbar:
                style_optimizer.zero_grad()
                rgbs = stylizer(radiance, omega, basis_color, d_color)
                loss = ((rgbs-gt_rgbs)**2).sum()
                loss_ARAP = stylizer.ARAP_loss() * self.lambda_ARAP
                loss += loss_ARAP 
                loss.backward()
                style_optimizer.step()
                style_scheduler.step()
                pbar.set_description(f"Optimizing Stlization, Loss={loss:.3f}")
            
            self.cached_stylizer = stylizer
            if self.stylize:
                self.trainer.model.stylizer = self.cached_stylizer
            self.need_optimize_stylize = False
            self.need_update = True
        
        max_spp = self.opt.max_spp if self.dynamic_resolution else 1
        if self.need_update or self.spp < max_spp:
        
            starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            starter.record()
            outputs = self.trainer.test_gui(self.cam.pose, self.cam.intrinsics, self.W, self.H, self.bg_color, self.spp, self.downscale, gui_mode=True)

            ender.record()
            torch.cuda.synchronize()
            t = starter.elapsed_time(ender)

            # update dynamic resolution
            if self.dynamic_resolution:
                # max allowed infer time per-frame is 200 ms
                full_t = t / (self.downscale ** 2)
                downscale = min(1, max(1/4, math.sqrt(100 / full_t)))
                if downscale > self.downscale * 1.2 or downscale < self.downscale * 0.8:
                    self.downscale = downscale

            output_buffer = self.prepare_buffer(outputs)
            if self.selected_pixel is not None:
                y, x = self.selected_pixel
                self.xyz = torch.from_numpy(outputs['xyz'][x, y]).type_as(self.palette)
                self.clip_feat = torch.from_numpy(outputs['clip_feat'][x, y]).type_as(self.palette)
                if self.xyz.isnan().sum() == 0: # valid point
                    self.trainer.model.edit.update_cent(self.xyz, self.clip_feat)
                    self.selected_point = self.xyz
                self.selected_pixel = None
            # depth = outputs['depth']
            # clip_feature = outputs['clip_feature']
            # import pdb
            # pdb.set_trace()
            # recolor render buffer
            # if self.palette_mode:
            # with torch.no_grad():
            #     render_img = torch.from_numpy(output_buffer)[None,None,:,:,[2,1,0]]*2-1
            #     weight = torch.nn.functional.grid_sample(self.hist_weights, render_img, mode='nearest', padding_mode='zeros', align_corners=True)
            #     weight = weight.squeeze().permute(1, 2, 0)
            #     render_img = torch.matmul(weight, self.palette)
            #     output_buffer = render_img.reshape(output_buffer.shape).detach().numpy()

            if self.need_update:
                self.render_buffer = output_buffer
                self.spp = 1
                self.need_update = False
            else:
                self.render_buffer = (self.render_buffer * self.spp + output_buffer) / (self.spp + 1)
                self.spp += 1
            
            dpg.set_value("_log_infer_time", f'{t:.4f}ms ({int(1000/t)} FPS)')
            dpg.set_value("_log_resolution", f'{int(self.downscale * self.W)}x{int(self.downscale * self.H)}')
            dpg.set_value("_log_spp", self.spp)
            dpg.set_value("_texture", self.render_buffer)
            
        dpg.set_value("_style_texture", self.drawed_style_image)
            
        selected_point = self.selected_point
        if selected_point is not None:
            selected_point = selected_point.detach().cpu().numpy()
        dpg.set_value("_img_point", "Image Point: " + str(selected_point))
        dpg.set_value("_style_pixel", "Style Pixel: " + str(self.style_pixel))
            
    def register_dpg(self):

        ### register texture 

        with dpg.texture_registry(show=False):
            dpg.add_raw_texture(self.W, self.H, self.render_buffer, format=dpg.mvFormat_Float_rgb, tag="_texture")
            
        with dpg.texture_registry(show=False):
            dpg.add_raw_texture(400, 400, self.drawed_style_image, format=dpg.mvFormat_Float_rgb, tag="_style_texture")

        ### register window

        # the rendered image, as the primary window
        with dpg.window(tag="_primary_window", width=self.W, height=self.H):

            # add the texture
            dpg.add_image("_texture")

        dpg.set_primary_window("_primary_window", True)

        # control window
        with dpg.window(label="Control", tag="_control_window", width=400, height=self.H, pos=(self.W, 0)):

            # button theme
            with dpg.theme() as theme_button:
                with dpg.theme_component(dpg.mvButton):
                    dpg.add_theme_color(dpg.mvThemeCol_Button, (23, 3, 18))
                    dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (51, 3, 47))
                    dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (83, 18, 83))
                    dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 5)
                    dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 3, 3)

            # time
            if not self.opt.test:
                with dpg.group(horizontal=True):
                    dpg.add_text("Train time: ")
                    dpg.add_text("no data", tag="_log_train_time")                    

            with dpg.group(horizontal=True):
                dpg.add_text("Infer time: ")
                dpg.add_text("no data", tag="_log_infer_time")
            
            with dpg.group(horizontal=True):
                dpg.add_text("SPP: ")
                dpg.add_text("1", tag="_log_spp")

            # train button
            if not self.opt.test:
                with dpg.collapsing_header(label="Train", default_open=True):

                    # train / stop
                    with dpg.group(horizontal=True):
                        dpg.add_text("Train: ")

                        def callback_train(sender, app_data):
                            if self.training:
                                self.training = False
                                dpg.configure_item("_button_train", label="start")
                            else:
                                self.training = True
                                dpg.configure_item("_button_train", label="stop")

                        dpg.add_button(label="start", tag="_button_train", callback=callback_train)
                        dpg.bind_item_theme("_button_train", theme_button)

                        def callback_reset(sender, app_data):
                            @torch.no_grad()
                            def weight_reset(m: nn.Module):
                                reset_parameters = getattr(m, "reset_parameters", None)
                                if callable(reset_parameters):
                                    m.reset_parameters()
                            self.trainer.model.apply(fn=weight_reset)
                            self.trainer.model.reset_extra_state() # for cuda_ray density_grid and step_counter
                            self.need_update = True

                        dpg.add_button(label="reset", tag="_button_reset", callback=callback_reset)
                        dpg.bind_item_theme("_button_reset", theme_button)

                    # save ckpt
                    with dpg.group(horizontal=True):
                        dpg.add_text("Checkpoint: ")

                        def callback_save(sender, app_data):
                            self.trainer.save_checkpoint(full=True, best=False)
                            dpg.set_value("_log_ckpt", "saved " + os.path.basename(self.trainer.stats["checkpoints"][-1]))
                            self.trainer.epoch += 1 # use epoch to indicate different calls.

                        dpg.add_button(label="save", tag="_button_save", callback=callback_save)
                        dpg.bind_item_theme("_button_save", theme_button)

                        dpg.add_text("", tag="_log_ckpt")
                    
                    # save mesh
                    with dpg.group(horizontal=True):
                        dpg.add_text("Marching Cubes: ")

                        def callback_mesh(sender, app_data):
                            self.trainer.save_mesh(resolution=256, threshold=10)
                            dpg.set_value("_log_mesh", "saved " + f'{self.trainer.name}_{self.trainer.epoch}.ply')
                            self.trainer.epoch += 1 # use epoch to indicate different calls.

                        dpg.add_button(label="mesh", tag="_button_mesh", callback=callback_mesh)
                        dpg.bind_item_theme("_button_mesh", theme_button)

                        dpg.add_text("", tag="_log_mesh")

                    with dpg.group(horizontal=True):
                        dpg.add_text("", tag="_log_train_log")

            
            # rendering options
            with dpg.collapsing_header(label="Options", default_open=True):

                # dynamic rendering resolution
                with dpg.group(horizontal=True):

                    def callback_set_dynamic_resolution(sender, app_data):
                        if self.dynamic_resolution:
                            self.dynamic_resolution = False
                            self.downscale = 1
                        else:
                            self.dynamic_resolution = True
                        self.need_update = True

                    dpg.add_checkbox(label="dynamic resolution", default_value=self.dynamic_resolution, callback=callback_set_dynamic_resolution)
                    dpg.add_text(f"{self.W}x{self.H}", tag="_log_resolution")

                # mode combo
                def callback_change_mode(sender, app_data):
                    self.mode = app_data
                    self.need_update = True
                
                dpg.add_combo(('image', 'depth'), label='mode', default_value=self.mode, callback=callback_change_mode)

                # bg_color picker
                def callback_change_bg(sender, app_data):
                    self.bg_color = torch.tensor(app_data[:3], dtype=torch.float32) # only need RGB in [0, 1]
                    self.need_update = True

                dpg.add_color_edit((255, 255, 255), label="Background Color", width=200, tag="_color_editor", no_alpha=True, callback=callback_change_bg)
                
                with dpg.group(horizontal=True):
                    def callback_renderview(sender, app_data):
                        self.trainer.test(self.train_loader, save_path="./results_gui", write_video=False, selected_idx=self.test_cam_id, gui_mode=True) # test and save video

                    dpg.add_button(label="render view", tag="_button_render_view", callback=callback_renderview)
                    dpg.bind_item_theme("_button_render_view", theme_button)

                    if self.video_loader is not None:
                        def callback_rendervideo(sender, app_data):
                            self.trainer.test(self.video_loader, save_path="./results_gui", write_video=True, gui_mode=True) # test and save video

                        dpg.add_button(label="render video", tag="_button_render_video", callback=callback_rendervideo)
                        dpg.bind_item_theme("_button_render_video", theme_button)
                    
                def callback_set_testcam(sender, app_data):
                    self.test_cam_id = app_data-1
                    test_pose = self.train_loader._data.poses[app_data-1].detach().cpu().numpy()
                    self.cam.rot = R.from_matrix(test_pose[:3, :3])
                    self.cam.radius = 2
                    center = test_pose[:3, :3] @ np.array([0, 0, -self.cam.radius])[...,np.newaxis]
                    self.cam.center = center[:,0] - test_pose[:3, 3]
                    #     @property
                    # def pose(self):
                    #     # first move camera to radius
                    #     res = np.eye(4, dtype=np.float32)
                    #     res[2, 3] -= self.radius
                    #     # rotate
                    #     rot = np.eye(4, dtype=np.float32)
                    #     rot[:3, :3] = self.rot.as_matrix()
                    #     res = rot @ res
                    #     # translate
                    #     res[:3, 3] -= self.center
                    #     return res
                    def intrinsics(self):
                        focal = self.H / (2 * np.tan(np.radians(self.fovy) / 2))
                        return np.array([focal, focal, self.W // 2, self.H // 2])
                    fovy = self.train_loader._data.intrinsics[1] 
                    self.cam.fovy = np.degrees(np.arctan(self.train_loader._data.H / fovy / 2) * 2)
                    self.cam.pose
                    self.need_update = True
                dpg.add_slider_int(label="test_pose", min_value=1, max_value=len(self.train_loader._data.poses), format="%d", default_value=0, callback=callback_set_testcam)

                # fov slider
                def callback_set_fovy(sender, app_data):
                    self.cam.fovy = app_data
                    self.need_update = True

                dpg.add_slider_int(label="FoV (vertical)", min_value=1, max_value=120, format="%d deg", default_value=self.cam.fovy, callback=callback_set_fovy)

                # dt_gamma slider
                def callback_set_dt_gamma(sender, app_data):
                    self.opt.dt_gamma = app_data
                    self.need_update = True

                dpg.add_slider_float(label="dt_gamma", min_value=0, max_value=0.1, format="%.5f", default_value=self.opt.dt_gamma, callback=callback_set_dt_gamma)

                # max_steps slider
                def callback_set_max_steps(sender, app_data):
                    self.opt.max_steps = app_data
                    self.need_update = True

                dpg.add_slider_int(label="max steps", min_value=1, max_value=1024, format="%d", default_value=self.opt.max_steps, callback=callback_set_max_steps)

                # aabb slider
                def callback_set_aabb(sender, app_data, user_data):
                    # user_data is the dimension for aabb (xmin, ymin, zmin, xmax, ymax, zmax)
                    self.trainer.model.aabb_infer[user_data] = app_data

                    # also change train aabb ? [better not...]
                    #self.trainer.model.aabb_train[user_data] = app_data

                    self.need_update = True

                dpg.add_separator()
                dpg.add_text("Axis-aligned bounding box:")

                with dpg.group(horizontal=True):
                    dpg.add_slider_float(label="x", width=150, min_value=-self.opt.bound, max_value=0, format="%.2f", default_value=-self.opt.bound, callback=callback_set_aabb, user_data=0)
                    dpg.add_slider_float(label="", width=150, min_value=0, max_value=self.opt.bound, format="%.2f", default_value=self.opt.bound, callback=callback_set_aabb, user_data=3)

                with dpg.group(horizontal=True):
                    dpg.add_slider_float(label="y", width=150, min_value=-self.opt.bound, max_value=0, format="%.2f", default_value=-self.opt.bound, callback=callback_set_aabb, user_data=1)
                    dpg.add_slider_float(label="", width=150, min_value=0, max_value=self.opt.bound, format="%.2f", default_value=self.opt.bound, callback=callback_set_aabb, user_data=4)

                with dpg.group(horizontal=True):
                    dpg.add_slider_float(label="z", width=150, min_value=-self.opt.bound, max_value=0, format="%.2f", default_value=-self.opt.bound, callback=callback_set_aabb, user_data=2)
                    dpg.add_slider_float(label="", width=150, min_value=0, max_value=self.opt.bound, format="%.2f", default_value=self.opt.bound, callback=callback_set_aabb, user_data=5)
                
            with dpg.collapsing_header(label="Palette", default_open=True):

                def refresh_palette_color():
                    highlight_color = (self.palette[self.highlight_palette_id].detach().cpu().numpy()*255).clip(0, 255).astype(np.uint8)
                    self.trainer.model.edit.update_delta(self.origin_palette, self.palette)
                    #self.trainer.model.basis_color.data = self.palette.type_as(self.trainer.model.basis_color.data)
                    dpg.set_value("_palette_color_editor", tuple(highlight_color))

                def callback_set_weight_mode(sender, app_data):
                    if self.weight_mode:
                        self.weight_mode = False
                    else:
                        self.weight_mode = True
                    self.trainer.model.edit.weight_mode = self.weight_mode
                    self.need_update = True

                dpg.add_checkbox(label="weight mode", default_value=self.weight_mode, callback=callback_set_weight_mode)
                
                def call_back_set_std_xyz(sender, app_data):
                    self.trainer.model.edit.update_std(std_xyz=app_data)
                    self.need_update = True
                dpg.add_slider_float(label="std_xyz", min_value=0, max_value=20, format="%f", 
                                    default_value=1, callback=call_back_set_std_xyz)
                def call_back_set_std_clip(sender, app_data):
                    self.trainer.model.edit.update_std(std_clip=app_data)
                    self.need_update = True
                dpg.add_slider_float(label="std_clip", min_value=0, max_value=20, format="%f", 
                                    default_value=1, callback=call_back_set_std_clip)

                def callback_reset_palette(sender, app_data):
                    self.palette = self.origin_palette.clone()
                    self.trainer.model.dir_weight = 1
                    refresh_palette_color()
                    self.need_update = True
                    
                dpg.add_button(label="reset", tag="_button_reset_palette", callback=callback_reset_palette)
                dpg.bind_item_theme("_button_reset_palette", theme_button)

                def call_back_set_dir_weight(sender, app_data):
                    self.trainer.model.dir_weight = app_data
                    self.need_update = True
                dpg.add_slider_float(label="dir_weight", min_value=0, max_value=20, format="%f", 
                                    default_value=1, callback=call_back_set_dir_weight)
                
                def callback_set_palette_id(sender, app_data):
                    self.highlight_palette_id = app_data                    
                    refresh_palette_color()
                    self.need_update = True

                dpg.add_slider_int(label="Palette_ID", min_value=0, max_value=len(self.palette)-1, format="%d", 
                                    default_value=self.highlight_palette_id, callback=callback_set_palette_id)
                
                def callback_change_palette(sender, app_data):
                    self.palette[self.highlight_palette_id] = torch.tensor(app_data[:3], dtype=torch.float32) # only need RGB in [0, 1]
                    refresh_palette_color()
                    self.need_update = True

                highlight_color = (self.palette[self.highlight_palette_id].detach().cpu().numpy()*255).clip(0, 255).astype(np.uint8)
                dpg.add_color_edit(tuple(highlight_color), label="Palette Color", width=200, tag="_palette_color_editor", 
                                    no_alpha=True, callback=callback_change_palette)
                
                def callback_save_palette(sender, app_data):
                    pred_basis_color = []

                    for i in range(self.opt.num_basis):
                        basis_color = self.palette[i,None,None,:].repeat(100, 100, 1)
                        basis_color = basis_color.clamp(0, 1)
                        pred_basis_color.append(basis_color.detach().cpu().numpy())

                    pred_basis_color = (np.concatenate(pred_basis_color, axis=1).clip(0,1)*255).astype(np.uint8)

                    cv2.imwrite(os.path.join("./results_gui", f'basis_color.png'), pred_basis_color[...,[2,1,0]])

                dpg.add_button(label="save_palette", tag="_button_save_palette", callback=callback_save_palette)
                dpg.bind_item_theme("_button_save_palette", theme_button)

            with dpg.collapsing_header(label="Stylization", default_open=True):                       
                def callback_select_style_image(sender, app_data):
                    filepath = app_data['selections'][list(app_data['selections'].keys())[0]]
                    image = imageio.imread(filepath)[:,:,:3]
                    print(image.shape)
                    self.style_image = cv2.resize(image, (400, 400), interpolation=cv2.INTER_LINEAR)
                    self.style_image = (self.style_image/255.0).clip(0, 1).astype(np.float32)
                    print(self.style_image.shape)
                    print(self.style_image.sum())
                    self.drawed_style_image = self.style_image
                    # dpg.set_value("_style_texture", self.style_image)
                    # self.style_W = self.style_image.shape[1] // 8
                    # self.style_H = self.style_image.shape[0] // 8
                  
                with dpg.file_dialog(directory_selector=False, show=False, width=800, height=300, default_path="/home/zhengfei/cvpr2023/experiments/2_stylization/images", callback=callback_select_style_image, tag="file_dialog_id"):
                    dpg.add_file_extension("", color=(255, 150, 150, 255))
                    dpg.add_file_extension(".*")
                    dpg.add_file_extension(".jpg", color=(255, 0, 255, 255), custom_text="[jpg]")
                    dpg.add_file_extension(".jpeg", color=(255, 0, 255, 255), custom_text="[jpeg]")
                    dpg.add_file_extension(".png", color=(255, 0, 255, 255), custom_text="[png]")

                dpg.add_button(label="select style image", tag="_button_select_style_image", callback=lambda: dpg.show_item("file_dialog_id"))
                dpg.bind_item_theme("_button_select_style_image", theme_button)
                       
                def callback_delete_correspondence(sender, app_data):
                    idx = int(sender.split("_")[-1])
                    self.style_color_list.pop(idx)
                    self.style_point_list.pop(idx)
                    update_correspondence_list()
                    
                def update_correspondence_list():
                    dpg.delete_item("_correspondence_list", children_only=True)
                    for i in range(len(self.style_color_list)):
                        dpg.add_text(f"Point: {self.style_point_list[i]}", parent="_correspondence_list")
                        dpg.add_text(f"Color: {self.style_color_list[i]}", parent="_correspondence_list")
                        dpg.add_button(label=f"Delete", parent="_correspondence_list", tag=f"_corr_delete_{i}", callback=callback_delete_correspondence)
                        dpg.bind_item_theme(f"_corr_delete_{i}", theme_button)
                        
                with dpg.group(horizontal=True):                    
                    
                    def callback_add_correspondence(sender, app_data):
                        assert(self.selected_point is not None and self.style_pixel is not None)
                        self.style_color_list.append(self.style_image[self.style_pixel[1], self.style_pixel[0]])
                        self.style_point_list.append(self.selected_point)
                        update_correspondence_list()
                            
                    def callback_save_correspondence(sender, app_data):
                        assert(self.style_point_list is not None and self.style_color_list is not None)
                        corr_dict = {
                            'points': self.style_point_list,
                            'colors': self.style_color_list
                        }
                        pickle.dump(corr_dict, open("./results_gui/style_corr.pkl", "wb"))
                        
                    def callback_load_correspondence(sender, app_data):
                        assert(self.style_point_list is not None and self.style_color_list is not None)
                        filename = app_data['selections'][list(app_data['selections'].keys())[0]]
                        corr_dict = pickle.load(open(filename, "rb"))
                        self.style_point_list = corr_dict['points']
                        self.style_color_list = corr_dict['colors']
                        update_correspondence_list()
                        
                    with dpg.file_dialog(directory_selector=False, show=False, width=800, height=300, 
                                         default_path="./results_gui", callback=callback_load_correspondence, tag="style_file_dialog"):
                        dpg.add_file_extension("", color=(255, 150, 150, 255))
                        dpg.add_file_extension(".*")
                        dpg.add_file_extension(".pkl", color=(255, 0, 255, 255), custom_text="[pkl]")

                    dpg.add_button(label="add correspondence", tag="_button_add_corr", callback=callback_add_correspondence)
                    dpg.bind_item_theme("_button_add_corr", theme_button)
                    
                    dpg.add_button(label="save", tag="_button_save_corr", callback=callback_save_correspondence)
                    dpg.bind_item_theme("_button_save_corr", theme_button)
                    
                    dpg.add_button(label="load", tag="_button_load_corr", callback=lambda: dpg.show_item("style_file_dialog"))
                    dpg.bind_item_theme("_button_load_corr", theme_button)
                    
                    
                    
                with dpg.group(horizontal=True):  
                    def callback_stylize(sender, app_data):
                            if self.stylize:
                                self.stylize = False
                                self.trainer.model.stylizer = None
                                self.need_update = True
                                dpg.configure_item("_button_stylize", label="stylize")
                            else:
                                self.stylize = True
                                self.trainer.model.stylizer = self.cached_stylizer
                                self.need_update = True
                                dpg.configure_item("_button_stylize", label="unstylize")
                                
                    def callback_optimize_stylize(sender, app_data):
                        self.need_optimize_stylize = True
                        
                    dpg.add_button(label="stylize", tag="_button_stylize", callback=callback_stylize)
                    dpg.bind_item_theme("_button_stylize", theme_button)
                    
                    dpg.add_button(label="optimize", tag="_button_optimize_stylize", callback=callback_optimize_stylize)
                    dpg.bind_item_theme("_button_optimize_stylize", theme_button)

                dpg.add_text("", tag="_img_point")
                dpg.add_text("", tag="_style_pixel")
        
            # debug info
            if self.debug:
                with dpg.collapsing_header(label="Debug"):
                    # pose
                    dpg.add_separator()
                    dpg.add_text("Camera Pose:")
                    dpg.add_text(str(self.cam.pose), tag="_log_pose")
                    
        with dpg.window(tag="_style_window", width=400, height=self.H, pos=(self.W+400, 0)):
            # add the texture
            dpg.add_image("_style_texture")

            with dpg.collapsing_header(label="Correspondence List", default_open=True, tag="_correspondence_list"):
                pass

        ### register camera handler

        def callback_camera_drag_rotate(sender, app_data):

            if not dpg.is_item_focused("_primary_window"):
                return

            dx = app_data[1]
            dy = app_data[2]

            self.cam.orbit(dx, dy)
            self.need_update = True

            if self.debug:
                dpg.set_value("_log_pose", str(self.cam.pose))


        def callback_camera_wheel_scale(sender, app_data):

            if not dpg.is_item_focused("_primary_window"):
                return

            delta = app_data

            self.cam.scale(delta)
            self.need_update = True

            if self.debug:
                dpg.set_value("_log_pose", str(self.cam.pose))


        def callback_camera_drag_pan(sender, app_data):

            if not dpg.is_item_focused("_primary_window"):
                return

            dx = app_data[1]
            dy = app_data[2]

            self.cam.pan(dx, dy)
            self.need_update = True

            if self.debug:
                dpg.set_value("_log_pose", str(self.cam.pose))

        def callback_select_point(sender, app_data):

            if not dpg.is_item_focused("_primary_window") and not dpg.is_item_focused("_style_window"):
                return
            style_img = dpg.is_item_focused("_style_window")
            x, y = dpg.get_mouse_pos()
            x = int(x)
            y = int(y)
            if not style_img:
                if x > 0 and y > 0 and x < self.W and y < self.H:
                    self.selected_pixel = [x,y]
                    self.selected_point = None
            else:
                if x > 0 and y > 0 and x < 400 and y < 400:
                    self.style_pixel = [x,y]
                    self.drawed_style_image = cv2.circle(self.style_image.copy(), self.style_pixel, 5, (255, 0, 0), -1)
                        
            self.need_update = True
                
            if self.debug:
                dpg.set_value("_log_pose", str(self.cam.pose))
                
        def callback_clear_point(sender, app_data):

            if not dpg.is_item_focused("_primary_window"):
                return
            print("Unselecting point")
            self.selected_point = None
            self.selected_pixel = None
            self.style_pixel = None
            self.need_update = True

            if self.debug:
                dpg.set_value("_log_pose", str(self.cam.pose))

        with dpg.handler_registry():
            dpg.add_mouse_click_handler(button=dpg.mvMouseButton_Right, callback=callback_select_point)
            dpg.add_mouse_double_click_handler(button=dpg.mvMouseButton_Right, callback=callback_clear_point)
            dpg.add_mouse_drag_handler(button=dpg.mvMouseButton_Left, callback=callback_camera_drag_rotate)
            dpg.add_mouse_wheel_handler(callback=callback_camera_wheel_scale)
            dpg.add_mouse_drag_handler(button=dpg.mvMouseButton_Middle, callback=callback_camera_drag_pan)


        dpg.create_viewport(title='torch-ngp', width=self.W+400+400, height=self.H, resizable=False)
        
        # TODO: seems dearpygui doesn't support resizing texture...
        # def callback_resize(sender, app_data):
        #     self.W = app_data[0]
        #     self.H = app_data[1]
        #     # how to reload texture ???

        # dpg.set_viewport_resize_callback(callback_resize)

        ### global theme
        with dpg.theme() as theme_no_padding:
            with dpg.theme_component(dpg.mvAll):
                # set all padding to 0 to avoid scroll bar
                dpg.add_theme_style(dpg.mvStyleVar_WindowPadding, 0, 0, category=dpg.mvThemeCat_Core)
                dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 0, 0, category=dpg.mvThemeCat_Core)
                dpg.add_theme_style(dpg.mvStyleVar_CellPadding, 0, 0, category=dpg.mvThemeCat_Core)
        
        dpg.bind_item_theme("_primary_window", theme_no_padding)
        dpg.bind_item_theme("_style_window", theme_no_padding)

        dpg.setup_dearpygui()

        #dpg.show_metrics()

        dpg.show_viewport()


    def render(self):
        while dpg.is_dearpygui_running():
            # update texture every frame
            if self.training:
                self.train_step()
            self.test_step()
            dpg.render_dearpygui_frame()