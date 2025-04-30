import os 
import cv2
import time
import tqdm 
import json
import trimesh 
import numpy as np
import pickle as pkl
from PIL import Image 
import skimage.io as io 
import dearpygui.dearpygui as dpg 
import torch
import torch.nn.functional as F 
import os.path as osp
import kiui
from kiui.mesh import Mesh 

import smplx 
from smplx.render import Renderer
from smplx.warp import warp_points
from smplx.lbs import batch_rodrigues  
from smplx.vis import lbs_weights_to_colors 

from src.utils.camera import Camera  
from src.utils.mesh import compute_normal 
from src.utils.helper import create_flame_model, orthogonal_projection
import src.utils.dict as dicts 


class GUI:
    def __init__(self, opt):
        self.opt = opt  # shared with the trainer's opt to support in-place modification of rendering parameters.
        self.gui = opt.gui # enable gui
        self.in_dir = opt.in_dir
        self.W = opt.W
        self.H = opt.H
        self.cam = Camera()

        self.mode = "image"   

        self.need_update = True  # update buffer_image 

        self.mouse_loc = np.array([0, 0])

        self.idx = 0 
        self.motion_mode = 'posed'  # canon, posed, dancing   
        self.auto_rotate = True # auto rotate the camera
    
        self.device = torch.device("cuda")

        # renderer
        self.renderer = Renderer(gui=False) 

        # load motion 
        self.out_h, self.out_w = self.H, self.W  
        
        self.buffer_image = np.ones((self.out_w, self.out_h, 3), dtype=np.float32) 
        # load parametric model  
        self.body_model = create_flame_model()
        self.flame_model = create_flame_model()

        # load mesh  
        self.load_data()
        
        if self.gui:
            dpg.create_context()
            self.register_dpg()
            self.test_step()
    
    def load_data(self):  
        # load flame params
        data = torch.load(osp.join(self.in_dir, 'deep3dface/flame.pth'))  
        exp = data['expression']
        betas = data['betas']
        global_orient = data['global_orient']
        jaw_pose = data['jaw_pose']
        scale = data['scale']
        transl = data['transl']
        leye_pose = data['eye_pose'] if 'eye_pose' in data else torch.zeros(1, 3).float().to(self.device)
        reye_pose = data['eye_pose'] if 'eye_pose' in data else torch.zeros(1, 3).float().to(self.device)
        
        self.expression = torch.tensor(exp).float().to(self.device)
        self.betas = torch.tensor(betas).float().to(self.device)
        self.global_orient = torch.tensor(global_orient).reshape(1, 3).float().to(self.device)
        self.jaw_pose = torch.tensor(jaw_pose).float().to(self.device)
        self.scale = torch.tensor(scale).float().to(self.device)
        self.transl = torch.tensor(transl).float().to(self.device)
        self.leye_pose = leye_pose
        self.reye_pose = reye_pose

        # backen betas and expression
        self.origin_beta = self.betas.clone()
        self.origin_expression = self.expression.clone()
        self.origin_jawpose = self.jaw_pose.clone()

        # update flame model 
        personal_attributes = torch.load(f"{self.in_dir}/newest/flame_attributes.pth") 
        personal_attributes = dicts.to_device(personal_attributes, 'cuda') 
        self.body_model.set_params(personal_attributes)
        
        self.lbs_vc = lbs_weights_to_colors(self.body_model.lbs_weights.detach().cpu()) 
        self.lbs_vc = torch.tensor(self.lbs_vc).to(self.device)  
        
        self.base_mesh = Mesh.load_obj(f"{self.in_dir}/newest/recon/recon_textured.obj")

    @property
    def mesh(self):   
        expression = self.expression 
        global_orient=self.global_orient 
        leye_pose=self.leye_pose 
        reye_pose=self.reye_pose 
        jaw_pose = self.jaw_pose 

         
        if self.motion_mode == 'natural':
            jaw_pose *= 0 
            expression *= 0  
            leye_pose *= 0 
            reye_pose *= 0
        elif self.motion_mode == 'open_mouse':
            jaw_pose = torch.tensor([[np.radians(15), 0, 0]]).float().to(self.device)

        output = self.body_model( 
            jaw_pose=jaw_pose, 
            betas=self.betas,
            expression=expression, 
            global_orient=global_orient, 
            leye_pose=leye_pose,
            reye_pose=reye_pose,
        )  
        v_posed = orthogonal_projection(output.vertices[0], self.transl, self.scale*self.cam.s) 
        vn = compute_normal(v_posed, self.base_mesh.f)  
        if self.auto_rotate:
            R = batch_rodrigues(torch.tensor([0, np.radians(self.idx), 0]).reshape(1, 3))[0].float() 
        else:
            R = batch_rodrigues(torch.tensor([np.radians(self.cam.theta), 0, 0]).reshape(1, 3))[0] @ batch_rodrigues(torch.tensor([0, np.radians(self.cam.phi), 0]).reshape(1, 3))[0] 
            
        v_posed = v_posed @ R.float().to(self.device).T   
        
        mesh = self.base_mesh.clone()
        mesh.v = v_posed
        mesh.vn = vn
        
        if self.mode == 'skinning': 
            mesh.vc = self.lbs_vc
            mesh.albedo = None  
            
        return mesh    

    @property
    def flame_mesh(self):   
        expression = self.expression 
        global_orient=self.global_orient 
        leye_pose=self.leye_pose 
        reye_pose=self.reye_pose 
        jaw_pose = self.jaw_pose  
         
        if self.motion_mode == 'natural':
            jaw_pose *= 0 
            expression *= 0  
            leye_pose *= 0 
            reye_pose *= 0
        elif self.motion_mode == 'open_mouse':
            jaw_pose = torch.tensor([[np.radians(15), 0, 0]]).float().to(self.device)

        output = self.flame_model( 
            jaw_pose=jaw_pose, 
            betas=self.betas,
            expression=expression, 
            global_orient=global_orient, 
            leye_pose=leye_pose,
            reye_pose=reye_pose,
        )  
        v_posed = orthogonal_projection(output.vertices[0], self.transl, self.scale*self.cam.s) 
        # vn = compute_normal(v_posed, self.base_mesh.f)  
        if self.auto_rotate:
            R = batch_rodrigues(torch.tensor([0, np.radians(self.idx), 0]).reshape(1, 3))[0].float() 
        else:
            R = batch_rodrigues(torch.tensor([np.radians(self.cam.theta), 0, 0]).reshape(1, 3))[0] @ batch_rodrigues(torch.tensor([0, np.radians(self.cam.phi), 0]).reshape(1, 3))[0] 
            
        v_posed = v_posed @ R.float().to(self.device).T   
        
        mesh = Mesh(v=v_posed, f=self.flame_model.faces_tensor.to(self.device).int(), vc=torch.ones_like(v_posed))
        mesh.auto_normal()
        
        if self.mode == 'skinning': 
            mesh.vc = self.lbs_vc 
            
        return mesh   

    def __del__(self):
        if self.gui:
            dpg.destroy_context()

    @torch.no_grad()
    def test_step(self):   
        # ignore if no need to update
        if not self.need_update:
            return

        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)
        starter.record()

        # render image    
        mvp = torch.eye(4).unsqueeze(0).to(self.device)  
        mvp[:, :3, :3] = batch_rodrigues(torch.tensor([np.pi, 0, 0]).reshape(1, 3))

        shading = 'lambertian' if self.mode == 'skinning' else 'albedo'
        flame_out = self.renderer(self.flame_mesh, mvp, self.H, self.W, shading_mode='lambertian')
        out = self.renderer(self.mesh, mvp, self.H, self.W, shading_mode=shading)
        buffer_image = out['image'] if self.mode == 'skinning' else out[self.mode] # [H, W, 3]  
        flame_buffer_image = flame_out['image'] if self.mode == 'skinning' else flame_out[self.mode] # [H, W, 3]
        # buffer_image = torch.cat([flame_buffer_image, buffer_image], 1)

        if not self.mode == 'alpha':
            buffer_image = buffer_image * out['alpha'] + (1 - out['alpha'])  
            flame_buffer_image = flame_buffer_image * flame_out['alpha'] + (1 - flame_out['alpha'])  
        
        if self.mode in ['depth', 'alpha']:
            buffer_image = buffer_image.repeat(1, 1, 1, 3)
            flame_buffer_image = flame_buffer_image.repeat(1, 1, 1, 3)

            if self.mode == 'depth':
                buffer_image = (buffer_image - buffer_image.min()) / (buffer_image.max() - buffer_image.min() + 1e-20)
        
        buffer_image = buffer_image.contiguous().clamp(0, 1).detach().cpu().numpy() 
        flame_buffer_image = flame_buffer_image.contiguous().clamp(0, 1).detach().cpu().numpy() 
        
        # buffer_image = np.hstack([buffer_image, flame_buffer_image])

        self.buffer_image = buffer_image 

        ender.record()
        torch.cuda.synchronize()
        t = starter.elapsed_time(ender)

        if self.gui:  
            dpg.set_value("_log_infer_time", f"{t:.4f}ms ({int(1000/t)} FPS)")
            dpg.set_value(
                "_texture", self.buffer_image
            )  # buffer must be contiguous, else seg fault!

        self.idx = (self.idx + 1) 

    def register_dpg(self):
        ### register texture

        with dpg.texture_registry(show=False):
            dpg.add_raw_texture( 
                self.out_w,  
                self.out_h,  
                self.buffer_image,
                format=dpg.mvFormat_Float_rgb,
                tag="_texture",
            )

        ### register window

        # the rendered image, as the primary window
        with dpg.window(
            tag="_primary_window", 
            width=self.out_w,  
            height=self.out_h, 
            pos=[0, 0],
            no_move=True,
            no_title_bar=True,
            no_scrollbar=True,
        ):
            # add the texture
            dpg.add_image("_texture")

        # dpg.set_primary_window("_primary_window", True)

        # control window
        with dpg.window(
            label="Control",
            tag="_control_window",
            width=600,
            height=self.out_h,
            pos=[self.out_w, 0],
            no_move=True,
            no_title_bar=True,
        ):
            # button theme
            with dpg.theme() as theme_button:
                with dpg.theme_component(dpg.mvButton):
                    dpg.add_theme_color(dpg.mvThemeCol_Button, (23, 3, 18))
                    dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (51, 3, 47))
                    dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (83, 18, 83))
                    dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 5)
                    dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 3, 3)

            # timer stuff
            with dpg.group(horizontal=True):
                dpg.add_text("Infer time: ")
                dpg.add_text("no data", tag="_log_infer_time")
  
            # rendering options
            with dpg.collapsing_header(label="Render", default_open=True):
                
                # auto rotate camera 
                with dpg.group(horizontal=True):
                    def callback_toggle_auto_rotate(sender, app_data):
                            self.auto_rotate = not self.auto_rotate
                            self.need_update = self.auto_rotate
                    dpg.add_checkbox(
                        label="auto rotate",
                        default_value=self.auto_rotate,
                        callback=callback_toggle_auto_rotate,
                    )

                # input motion file 
                def callback_select_input(sender, app_data):
                    # only one item
                    for k, v in app_data["selections"].items():
                        # dpg.set_value("_log_input", k) 
                        self.load_motions(v)
            
                    self.need_update = True

                with dpg.file_dialog(
                    directory_selector=False,
                    show=False,
                    callback=callback_select_input,
                    file_count=1,
                    tag="file_dialog_tag",
                    width=700,
                    height=400,
                ):
                    dpg.add_file_extension(".*")
                    dpg.add_file_extension("*")
                    dpg.add_file_extension("motions{.pkl,}", color=(0, 255, 0, 255))
  

                # render mode combo
                # def callback_change_subject(sender, app_data):   
                #     self.load_data(app_data)
                #     self.need_update = True
                #     self.idx = 0  

                # dpg.add_combo(
                #     (
                #         "f358d6ef5c20194d18789e09dbd2907370a14a12_high", 
                #         "4e53f42e907b164939993b1519fba53fc9143825_high", 
                #         "b88627f7e0b064346ac7b1027630802c"
                #     ),
                #     label="subject",
                #     default_value=self.opt.subject,
                #     callback=callback_change_subject,
                # )

                def callback_change_rendering_mode(sender, app_data):
                    self.mode = app_data
                    self.need_update = True
                    self.idx = 0 

                dpg.add_combo(
                    ("image", "alpha", "normal", 'skinning'),
                    label="render mode",
                    default_value=self.mode,
                    callback=callback_change_rendering_mode,
                )

                # # motion mode combo
                def callback_change_motion_mode(sender, app_data):
                    self.motion_mode = app_data
                    self.need_update = True 
                    self.betas = self.origin_beta.clone()
                    self.expression = self.origin_expression.clone()
 
                dpg.add_combo(
                    ("posed", "natural", 'open_mouse'),
                    label="motion mode",
                    default_value=self.motion_mode,
                    callback=callback_change_motion_mode,
                )
            
            # shape control, the first 10 betas 
            with dpg.collapsing_header(label='Shape Control', default_open=True):  
                def callback_shape0_control(sender, app_data):
                    self.betas[:, 0] = app_data
                    self.need_update = True
                dpg.add_slider_float(
                    label="shape_0",
                    min_value=-2,
                    max_value=2,
                    default_value=self.betas[:, 0].item(),
                    callback=callback_shape0_control,
                )

                def callback_shape1_control(sender, app_data):
                    self.betas[:, 1] = app_data
                    self.need_update = True
                dpg.add_slider_float(
                    label="shape 1",
                    min_value=-2,
                    max_value=2,
                    default_value=self.betas[:, 1].item(),
                    callback=callback_shape1_control,
                )

                def callback_shape2_control(sender, app_data):
                    self.betas[:, 2] = app_data
                    self.need_update = True
                dpg.add_slider_float(
                    label="shape 2",
                    min_value=-2,
                    max_value=2,
                    default_value=self.betas[:, 2].item(),
                    callback=callback_shape2_control,
                )

                def callback_shape3_control(sender, app_data):
                    self.betas[:, 3] = app_data
                    self.need_update = True
                dpg.add_slider_float(
                    label="shape 3",
                    min_value=-2,
                    max_value=2,
                    default_value=self.betas[:, 3].item(),
                    callback=callback_shape3_control,
                )

                def callback_shape4_control(sender, app_data):
                    self.betas[:, 4] = app_data
                    self.need_update = True
                dpg.add_slider_float(
                    label="shape 4",
                    min_value=-2,
                    max_value=2,
                    default_value=self.betas[:, 4].item(),
                    callback=callback_shape4_control,
                )

                def callback_shape5_control(sender, app_data):
                    self.betas[:, 5] = app_data
                    self.need_update = True
                dpg.add_slider_float(
                    label="shape 5",
                    min_value=-2,
                    max_value=2,
                    default_value=self.betas[:, 5].item(),
                    callback=callback_shape5_control,
                )

                def callback_shape6_control(sender, app_data):
                    self.betas[:, 6] = app_data
                    self.need_update = True
                dpg.add_slider_float(
                    label="shape 6",
                    min_value=-2,
                    max_value=2,
                    default_value=self.betas[:, 6].item(),
                    callback=callback_shape6_control,
                )

                def callback_shape7_control(sender, app_data):
                    self.betas[:, 7] = app_data
                    self.need_update = True
                dpg.add_slider_float(
                    label="shape 7",
                    min_value=-2,
                    max_value=2,
                    default_value=self.betas[:, 7].item(),
                    callback=callback_shape7_control,
                )

                def callback_shape8_control(sender, app_data):
                    self.betas[:, 8] = app_data
                    self.need_update = True
                dpg.add_slider_float(
                    label="shape 8",
                    min_value=-2,
                    max_value=2,
                    default_value=self.betas[:, 8].item(),
                    callback=callback_shape8_control,
                )

                def callback_shape9_control(sender, app_data):
                    self.betas[:, 9] = app_data
                    self.need_update = True
                dpg.add_slider_float(
                    label="shape 9",
                    min_value=-2,
                    max_value=2,
                    default_value=self.betas[:, 9].item(),
                    callback=callback_shape9_control,
                )

                # reset values 
                def reset_shape(sender, app_data):
                    self.betas = self.origin_beta.clone()
                    self.need_update = True
                    # update slider value 
                    # for i in range(10):
                    #     dpg.set_value(f"shape {i}", self.origin_beta[:, i].item()) 
                    dpg.set_value(f"shape_0", self.origin_beta[0, 0]) 
                dpg.add_button(label="reset shape", callback=reset_shape)

            # expression control, the first 10 expression
            with dpg.collapsing_header(label='Expression Control', default_open=True):  
                def callback_expression0_control(sender, app_data):
                    self.expression[:, 0] = app_data
                    self.need_update = True
                dpg.add_slider_float(
                    label="expression 0",
                    min_value=-2,
                    max_value=2,
                    default_value=self.expression[:, 0].item(),
                    callback=callback_expression0_control,
                )

                def callback_expression1_control(sender, app_data):
                    self.expression[:, 1] = app_data
                    self.need_update = True
                dpg.add_slider_float(
                    label="expression 1",
                    min_value=-2,
                    max_value=2,
                    default_value=self.expression[:, 1].item(),
                    callback=callback_expression1_control,
                )

                def callback_expression2_control(sender, app_data):
                    self.expression[:, 2] = app_data
                    self.need_update = True
                dpg.add_slider_float(
                    label="expression 2",
                    min_value=-2,
                    max_value=2,
                    default_value=self.expression[:, 2].item(),
                    callback=callback_expression2_control,
                )

                def callback_expression3_control(sender, app_data):
                    self.expression[:, 3] = app_data
                    self.need_update = True
                dpg.add_slider_float(
                    label="expression 3",
                    min_value=-2,
                    max_value=2,
                    default_value=self.expression[:, 3].item(),
                    callback=callback_expression3_control,
                )

                def callback_expression4_control(sender, app_data):
                    self.expression[:, 4] = app_data
                    self.need_update = True
                dpg.add_slider_float(
                    label="expression 4",
                    min_value=-2,
                    max_value=2,
                    default_value=self.expression[:, 4].item(),
                    callback=callback_expression4_control,
                )

                def callback_expression5_control(sender, app_data):
                    self.expression[:, 5] = app_data
                    self.need_update = True
                dpg.add_slider_float(
                    label="expression 5",
                    min_value=-2,
                    max_value=2,
                    default_value=self.expression[:, 5].item(),
                    callback=callback_expression5_control,
                )

                def callback_expression6_control(sender, app_data):
                    self.expression[:, 6] = app_data
                    self.need_update = True
                dpg.add_slider_float(
                    label="expression 6",
                    min_value=-2,
                    max_value=2,
                    default_value=self.expression[:, 6].item(),
                    callback=callback_expression6_control,
                )

                def callback_expression7_control(sender, app_data):
                    self.expression[:, 7] = app_data
                    self.need_update = True
                dpg.add_slider_float(
                    label="expression 7",
                    min_value=-2,
                    max_value=2,
                    default_value=self.expression[:, 7].item(),
                    callback=callback_expression7_control,
                )

                def callback_expression8_control(sender, app_data):
                    self.expression[:, 8] = app_data
                    self.need_update = True
                dpg.add_slider_float(
                    label="expression 8",
                    min_value=-2,
                    max_value=2,
                    default_value=self.expression[:, 8].item(),
                    callback=callback_expression8_control,
                )

                def callback_expression9_control(sender, app_data):
                    self.expression[:, 9] = app_data
                    self.need_update = True
                dpg.add_slider_float(
                    label="expression 9",
                    min_value=-2,
                    max_value=2,
                    default_value=self.expression[:, 9].item(),
                    callback=callback_expression9_control,
                )

                def reset_expression(sender, app_data):
                    self.expression = self.origin_expression.clone()
                    self.need_update = True
                    for i in range(10):
                        dpg.set_value(f"expression {i}", self.origin_expression[:, i].item()) 
                dpg.add_button(label="reset expression", callback=reset_expression)
            
            # with dpg.collapsing_header(label='Eyes Control', default_open=True):  
            #     def callback_expression0_control(sender, app_data):
            #         self.leye_pose[:, 1] = np.radians(app_data)
            #         self.need_update = True
                    
            #     dpg.add_slider_float(
            #         label="left eye pose",
            #         min_value=-30,
            #         max_value=30,
            #         default_value=self.leye_pose[:, 1].item(),
            #         callback=callback_expression0_control,
            #     )

        ### register camera handler
        def callback_camera_drag_rotate_or_draw_mask(sender, app_data):
            if not dpg.is_item_focused("_primary_window"):
                return

            dx = app_data[1]
            dy = app_data[2]
 
            self.cam.orbit(dx, dy)
            self.need_update = True 

        def callback_camera_wheel_scale(sender, app_data):
            if not dpg.is_item_focused("_primary_window"):
                return

            delta = app_data

            self.cam.scale(delta)
            self.need_update = True

        def callback_camera_drag_pan(sender, app_data):
            if not dpg.is_item_focused("_primary_window"):
                return

            dx = app_data[1]
            dy = app_data[2]

            self.cam.pan(dx, dy)
            self.need_update = True

        def callback_set_mouse_loc(sender, app_data):
            if not dpg.is_item_focused("_primary_window"):
                return

            # just the pixel coordinate in image
            self.mouse_loc = np.array(app_data)
 
        with dpg.handler_registry():
            # for camera moving
            dpg.add_mouse_drag_handler(
                button=dpg.mvMouseButton_Left,
                callback=callback_camera_drag_rotate_or_draw_mask,
            )
            dpg.add_mouse_wheel_handler(callback=callback_camera_wheel_scale)
            dpg.add_mouse_drag_handler(
                button=dpg.mvMouseButton_Middle, callback=callback_camera_drag_pan
            ) 
            dpg.add_mouse_move_handler(callback=callback_set_mouse_loc) 
 

        dpg.create_viewport(
            title="Head-Avatar-Animation",
            width=self.out_w + 600,
            height=self.out_h + (45 if os.name == "nt" else 0),
            resizable=False,
        )

        ### global theme
        with dpg.theme() as theme_no_padding:
            with dpg.theme_component(dpg.mvAll):
                # set all padding to 0 to avoid scroll bar
                dpg.add_theme_style(
                    dpg.mvStyleVar_WindowPadding, 0, 0, category=dpg.mvThemeCat_Core
                )
                dpg.add_theme_style(
                    dpg.mvStyleVar_FramePadding, 0, 0, category=dpg.mvThemeCat_Core
                )
                dpg.add_theme_style(
                    dpg.mvStyleVar_CellPadding, 0, 0, category=dpg.mvThemeCat_Core
                )

        dpg.bind_item_theme("_primary_window", theme_no_padding)

        dpg.setup_dearpygui()

        ### register a larger font
        # get it from: https://github.com/lxgw/LxgwWenKai/releases/download/v1.300/LXGWWenKai-Regular.ttf
        if os.path.exists("LXGWWenKai-Regular.ttf"):
            with dpg.font_registry():
                with dpg.font("LXGWWenKai-Regular.ttf", 18) as default_font:
                    dpg.bind_font(default_font)

        # dpg.show_metrics()

        dpg.show_viewport()

    def render(self):
        assert self.gui
        while dpg.is_dearpygui_running():
            self.test_step()
            dpg.render_dearpygui_frame()

if __name__ == "__main__":
    import argparse 
    
    parser = argparse.ArgumentParser() 
    parser.add_argument('--H', type=int, default=800, help='gui windown height') 
    parser.add_argument('--W', type=int, default=800, help="gui window width")
    parser.add_argument('-i', '--in_dir', type=str, default='output/examples/00/6-views', help="file to the smplx path")
    parser.add_argument('-g', '--gui', type=bool, default=True, help="gui mode or not")   
    opt = parser.parse_args()

    gui = GUI(opt)
    gui.render() 