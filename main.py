import os 
import cv2
import time    
import random 
from typing import Dict, List, Tuple
import numpy as np   
from PIL import Image
from tqdm import tqdm
import torch
import torch.nn as nn  
import torch.nn.functional as F 
from skimage import io
import coloredlogs, logging
from rich.progress import track
from kiui.mesh import Mesh 
from kiui.op import scale_img_hwc, scale_img_nhwc
from pytorch3d.loss import mesh_laplacian_smoothing
from pytorch3d.structures import Meshes 
from smplx.render import Renderer 
from smplx.vis import lbs_weights_to_colors 
from smplx.lbs import batch_rodrigues  

import src.utils.dict as dicts 
from src.dataset import Dataset
from src.utils.crop import uncrop   
from src.utils.mesh import split_mesh
from src.utils.vis import vis_landmarks
from src.utils.loss import lmk68_symmetry_loss
from src.geo.mesh_opt import remesh, remesh_withuv 
from src.utils.videotool import export_video, load_video 
from src.utils.mesh import simple_clean_mesh, save_obj, normalize_vertices 
from src.utils.texturing import albedo_from_images, backface_culling, albedo_from_images_possion_blending, dilate_image
from src.utils.helper import orthogonal_projection, l2_distance, create_flame_model, save_image, find_files_from_dir, ids2mask, vid2fmask
            

logger = logging.getLogger(__name__) 
coloredlogs.install(level='DEBUG', logger=logger) 


class Trainer(nn.Module):
    def __init__(self, opt):
        super(Trainer, self).__init__()  
        self.opt = opt    
        self.device = torch.device("cuda")
        self.debug = opt.debug   
        self.bg_color = torch.tensor(opt.bg_color).to(self.device) 
        style, subject = opt.image.split("/")[-2:]  
        self.subject = subject.split(".")[0]  
        self.save_dir = os.path.join(opt.save_dir, style, self.subject, f'{len(opt.views)}-views', opt.exp_name) 
        os.makedirs(f'{self.save_dir}/video', exist_ok=True)
        os.makedirs(f'{self.save_dir}/recon', exist_ok=True)
        logger.info(f"[Init]: input image: {opt.image}")
        
        if self.debug:
            os.makedirs(f'{self.save_dir}/optim_eye', exist_ok=True)

        # TODO: remove boundary/neck area 

        # dataset 
        self.dataset = Dataset(os.path.join(opt.save_dir, style, self.subject, f'{len(opt.views)}-views'), opt) 
        # renderer 
        self.renderer = Renderer()  
        
        self.flame_model = create_flame_model(
            model_path=opt.flame_model_path, create_segms=True, add_teeth=False, device=self.device
        )  

        # flame  
        self.body_model = create_flame_model(
            model_path=opt.flame_model_path, create_segms=True, add_teeth=False, device=self.device
        )  
        
        self.ds = self.body_model.shapedirs.shape[-1]   # [V, 3, 400]
        self.dp = self.body_model.posedirs.shape[0]     # [4*9, V*3]
        self.dw = self.body_model.lbs_weights.shape[-1] # [V, 5]
        self.dj = self.body_model.J_regressor.shape[0]  # [J, V]
        
        # self.faces = torch.tensor(self.body_model.faces.astype(np.int32)).to(self.device) 
        # self.flame_tri = self.faces.clone()  
        # flame uv  
        flame_template = Mesh.load_obj(opt.flame_template_path)  
        if opt.uv_interpolate:  # mapping vt to vertices 
            ft = flame_template.ft.view(-1).long()
            f = flame_template.f.view(-1).long()
            vmapping = torch.zeros(flame_template.vt.shape[0], dtype=torch.long, device=self.device)
            vmapping[ft] = f
            flame_template.v = flame_template.v[vmapping]
            flame_template.f = flame_template.ft 
            
            ids = [] 
            for i in range(5023): 
                ids.append(torch.where(vmapping == i)[0][0].item()) 
            ids = ids + list(set(range(len(vmapping))) - set(ids))
            
            flame_template.v = flame_template.v[ids]
            flame_template.vt = flame_template.vt[ids]
            
            ids = torch.tensor(ids, device=self.device)
            fids = torch.zeros_like(ids)
            fids[ids] = torch.arange(len(ids), device=ids.device) 
            flame_template.f = fids[flame_template.f].int()
            flame_template.ft = fids[flame_template.ft].int()
        
        self.vt = flame_template.vt.clone()
        self.ft = flame_template.ft.clone()
        self.faces = flame_template.f.clone()
        
        del flame_template
        
        self.face_vid = self.body_model.segment.get_vertex_ids(['face', 'neck', 'boundary'])
        self.no_face_vid = ids2mask(self.face_vid, self.body_model.v_template.shape[0], reverse=True)
        
        face_tri_mask = vid2fmask(self.face_vid, self.faces)
        self.face_tri = self.faces[face_tri_mask]
        
        if opt.learnable_eyes:
            self._init_eye_params()
        
        self.n_view = len(opt.views) 
        self._init_cameras()
        self.cnt = 0

        # building parsing labels 
        parse_labels = torch.zeros_like(self.body_model.v_template).to(self.device)
        parse_labels[self.face_vid] = torch.tensor(opt.part_color_map.face).float().to(self.device)
        parse_labels[self.eye_vid] = torch.tensor(opt.part_color_map.eye).float().to(self.device)
        parse_labels[self.neck_vid] = torch.tensor(opt.part_color_map.neck).float().to(self.device)
        parse_labels[parse_labels.sum(1) == 0] = torch.tensor(opt.part_color_map.hair).float().to(self.device)
        self.parse_labels = parse_labels.float()
        
        # update body model parameter 
        if opt.uv_interpolate:
            v_attr = torch.cat([self.pack_attributes(), self.parse_labels], 1)
            v_attr = v_attr[vmapping][ids]
            v_attr, self.parse_labels = torch.split(v_attr, [v_attr.shape[1]-3, 3], 1)
            self.body_model.set_params(self.unpack_attributes(v_attr)) 
            self.body_model.J_regressor = self.body_model.J_regressor[:, vmapping][:, ids]
            self.body_model.J_regressor[:, 5023:] *= 0  
            self.origin_template = self.body_model.v_template.clone().detach()
        
    def _init_cameras(self):
        """Initialize orthogonal cameras for rendering."""
        self.rot = batch_rodrigues(
            torch.tensor([
                [0, np.radians(angle), 0] for angle in opt.views 
            ])).float().to(self.device) 
        extrinsic = torch.eye(4)[None].expand(self.n_view, -1, -1).clone() .to(self.device)
        extrinsic[:, :3, :3] = self.rot
        intrinsic = torch.eye(4)[None].expand(self.n_view, -1, -1).clone() .to(self.device)
        R = batch_rodrigues(torch.tensor([np.pi, 0, 0]).reshape(1, 3)).reshape(3, 3).float()   
        intrinsic[:, :3, :3] = R  
        self.mvps = torch.bmm(intrinsic, extrinsic) 
        self.extrinsic = extrinsic
    
    def _init_eye_params(self):
        # eyeliad 
        self.leyelid_vid = self.body_model.segment.get_vertex_ids(['left_eyelid'])
        self.reyelid_vid = self.body_model.segment.get_vertex_ids(['right_eyelid'])
        self.eyelid_vid = np.concatenate([self.leyelid_vid, self.reyelid_vid], 0)
        
        # eyeball
        self.leyeball_vid = self.body_model.segment.get_vertex_ids(['left_eyeball'])
        self.reyeball_vid = self.body_model.segment.get_vertex_ids(['right_eyeball'])
        self.eyeball_vid = np.concatenate([self.leyeball_vid, self.reyeball_vid], 0)
        
        # eye cone
        self.leyecone_vid = self.body_model.segment.get_vertex_ids(['left_eyeball_cone'])
        self.reyecone_vid = self.body_model.segment.get_vertex_ids(['right_eyeball_cone'])
        self.eyecone_vid = np.concatenate([self.leyecone_vid, self.reyecone_vid], 0)
        
        # 
        self.leye_vid = np.concatenate([self.leyeball_vid, self.leyecone_vid], 0)
        self.reye_vid = np.concatenate([self.reyeball_vid, self.reyecone_vid], 0)
        self.eye_vid = np.concatenate([self.leye_vid, self.reye_vid], 0)
        
        self.eye_cone_fmask = vid2fmask(self.eyecone_vid, self.faces)
        self.eyeball_fmask = vid2fmask(self.eyeball_vid, self.faces)
        self.wo_eye_fmask = self.eye_cone_fmask.logical_or(self.eyeball_fmask).logical_not() 
        self.eyes_fids = self.eye_cone_fmask.logical_or(self.eyeball_fmask).nonzero().flatten()
        self.tri_wo_eyes = self.faces[self.wo_eye_fmask]
        self.tri_eyes = self.faces[self.eyeball_fmask]
        self.tri_leye = self.body_model.segment.get_triangles('left_eyeball')
        self.tri_reye = self.body_model.segment.get_triangles('right_eyeball')
        self.tri_leye = torch.tensor(self.tri_leye).int().to(self.device)
        self.tri_reye = torch.tensor(self.tri_reye).int().to(self.device)
        
        # neck 
        self.neck_vid = self.body_model.segment.get_vertex_ids(['neck', 'boundary'])
        self.neck_tri = self.body_model.segment.get_triangles(['neck', 'boundary'])
        self.neck_tri = torch.tensor(self.neck_tri).int().to(self.device)
        
        self.wo_neck_fmask = vid2fmask(self.neck_vid, self.faces).logical_not() 
        self.tri_wo_neck = self.faces[self.wo_neck_fmask]
    
    def pack_attributes(self, with_J_regressor=False):  
        attributes = torch.cat([  
            self.body_model.v_template, 
            self.body_model.shapedirs.reshape(-1, 3*self.ds),
            self.body_model.posedirs.reshape(self.dp, -1, 3).permute(1, 2, 0).reshape(-1, self.dp*3), # [V, dp*3]
            self.body_model.lbs_weights,  
        ], 1)
        if with_J_regressor:
            attributes = torch.cat(
                [
                    attributes, self.body_model.J_regressor.transpose(0, 1)
                ], 1
            )
        return attributes
    
    def unpack_attributes(self, v_attr, with_J_regressor=False):
        if not with_J_regressor: 
            v_template, shapedirs, posedirs, lbs_weights = torch.split(v_attr, [3, self.ds*3, self.dp*3, self.dw], 1)
            return { 
                'v_template': v_template.detach(),  
                'shapedirs': shapedirs.reshape(-1, 3, self.ds).detach(), 
                'posedirs': posedirs.reshape(-1, 3, self.dp).permute(2, 0, 1).reshape(self.dp, -1).detach(),
                'lbs_weights': lbs_weights.detach(),
            } 
        else:
            v_template, shapedirs, posedirs, lbs_weights, J_regressor = torch.split(v_attr, [3, self.ds*3, self.dp*3, self.dw, self.dj], 1)
            return { 
                'v_template': v_template.detach(),  
                'shapedirs': shapedirs.reshape(-1, 3, self.ds).detach(), 
                'posedirs': posedirs.reshape(-1, 3, self.dp).permute(2, 0, 1).reshape(self.dp, -1).detach(),
                'lbs_weights': lbs_weights.detach(),
                'J_regressor': J_regressor.transpose(0, 1).detach()
            } 
    
    def rotate_normal(self, normals_world, masks):
        normals_camera = normals_world * masks * 2 - 1 
        normals_camera = F.normalize(normals_camera, dim=-1)  
        normals_camera = normals_camera * masks - (1 - masks) 
        normals_camera = torch.bmm(normals_camera.reshape(len(normals_world), -1, 3), self.rot).reshape(*normals_world.shape)
        normals_camera = (normals_camera + 1) / 2 
        normals_camera = normals_camera * masks + (1 - masks) * self.bg_color
        return normals_camera
    
    def render(self, v, f, vc=None, vt=None, ft=None, H=512, W=512, mvp=None, shading_mode='albedo'):
        mesh = Mesh(v=v.float(), f=f.int(), vt=vt, ft=ft, vc=vc)
        mesh.auto_normal()  
        spp = 2 if H < 2048 else 1
        mvp = mvp if mvp is not None else self.mvps
        return self.renderer(mesh, mvp, spp=spp, bg_color=self.bg_color, h=H, w=W, shading_mode=shading_mode) 

    def inner_mouth_mask_render(self, mesh, uv_mask, **kwargs):
        # render head without inner-teeth 

        head_mesh = mesh.clone() 
        head_mesh.albedo = torch.cat([mesh.albedo, 1 - uv_mask.unsqueeze(-1)], -1)
        head_pkg = self.renderer(head_mesh, self.mvps, **kwargs, shading_mode='albedo')
        
        teeth_mesh = Mesh(
            v=mesh.v[-24847:].clone(),
            f=mesh.f[-49344:].clone() - mesh.f[-49344:].min(),
            vt=mesh.vt[-27392:].clone(),
            ft=mesh.ft[-49344:].clone() - mesh.ft[-49344:].min(),
        )
        teeth_mesh.auto_normal()
        teeth_alpha = uv_mask.clone()
        teeth_alpha[teeth_alpha.shape[0]//2:] = 1
        teeth_mesh.albedo = torch.cat([mesh.albedo, teeth_alpha.unsqueeze(-1)], -1)
        teeth_pkg = self.renderer(teeth_mesh, self.mvps, **kwargs, shading_mode='lambertian')

        teeth_alpha = (1 - head_pkg['image'][..., 3:]) * teeth_pkg['image'][..., 3:]
        head_pkg['image'] = teeth_pkg['image'][..., :3] * teeth_alpha + head_pkg['image'][..., :3] * (1 - teeth_alpha)
        return head_pkg

    def render_360view(
        self, 
        input_mesh, 
        num_views=90, loop=1, res=512, 
        shading_mode='albedo', 
        export_path=None, 
        input_pil=None, 
        skip_normal=False, 
        skip_skin=False, 
        resize=False, 
        ):
        mesh = input_mesh.clone()
        
        frames = [] 
        if input_pil is not None:
            input_img = input_pil.convert('RGB').resize((res, res))  
            frames = np.stack([np.array(input_img)[..., :3]] * num_views, 0)
            
        pkg = self.renderer.render_360views(mesh, num_views, res, loop=loop, resize=resize, shading_mode=shading_mode, size=0.95)
        
        if pkg['image'] is not None:
            rgbs = (pkg['image'].detach().cpu().numpy() * 255).astype(np.uint8) # (num_views, H, W, 3)
            frames = np.concatenate([frames, rgbs], 2) if len(frames) > 0 else rgbs
        
        if not skip_normal:
            normal = (pkg['normal'].detach().cpu().numpy() * 255).astype(np.uint8) # (num_views, H, W, 3)
            frames = np.concatenate([frames, normal], 2) if len(frames) > 0 else normal
        
        if not skip_skin:
            lbs_color = lbs_weights_to_colors(self.body_model.lbs_weights.detach().cpu())     
            mesh.vc = torch.tensor(lbs_color).to(self.device)  
            mesh.albedo = None 
            pkg = self.renderer.render_360views(mesh, num_views, res, loop=loop, resize=resize, shading_mode='lambertian', size=0.95) 
            image = (pkg['image'].detach().cpu().numpy() * 255).astype(np.uint8) # (num_views, H, W, 3)
            frames = np.concatenate([frames, image], 2) if len(frames) > 0 else image
        
        mesh.vc = self.parse_labels  
        mesh.albedo = None 
        pkg = self.renderer.render_360views(mesh, num_views, res, loop=loop, resize=resize, shading_mode='lambertian', size=0.95) 
        image = (pkg['image'].detach().cpu().numpy() * 255).astype(np.uint8) # (num_views, H, W, 3)
        frames = np.concatenate([frames, image], 2) if len(frames) > 0 else image
        
        if export_path is not None: 
            export_video(frames, export_path, fps=num_views/3/loop)
        return frames
    
    def template_deformation(
        self, 
        gt_normal, 
        gt_mask, 
        flame_dict, 
        transl, scale, 
        gt_lmk68, 
        gt_eye_mask=None,  
        render_res=512,  
        vis_interval=100, 
        learn_eyes=False, 
        neck_mask=None, 
        face_mask=None, 
        hair_mask=None, 
        parse_images:Dict=None,
        w_mask_loss=False 
    ):    
        if learn_eyes:  # prepare eyes-related 
            assert gt_eye_mask is not None, "gt_eye_mask is required when learn_eyes=True"  
            faces_all = torch.cat([self.tri_wo_eyes, self.tri_eyes], 0)
        else:
            faces_all = self.faces
        
        # 
        with torch.no_grad():
            v = self.body_model(**flame_dict).vertices[0]  
            v = normalize_vertices(v) 
            visible_mask = backface_culling(v, self.tri_wo_eyes, self.mvps[2]).logical_not() 
            vis_tri_wo_eyes = self.tri_wo_eyes[visible_mask] 
            visible_mask = backface_culling(v, self.tri_wo_neck, self.mvps[2]).logical_not() 
            # vis_tri_wo_neck = self.tri_wo_neck[visible_mask] 
        
        # base_v = self.body_model.v_template.clone().detach()
        v_template = nn.Parameter(self.body_model.v_template.clone(), requires_grad=True)
        optimizer = torch.optim.Adam([v_template], lr=0.001)   
        
        # reg_weight = 10 * (0.99 ** self.epoch)  
        reg_weight = 10 
        nml_weight = 3 * (1.05 ** self.epoch)  
        
        # neck_mask_back = torch.flip(neck_mask, [2])
        pbar = tqdm(range(self.opt.iters_per_epoch))
        for i in pbar:  
            # initial loss 
            lmk_loss = torch.tensor(0).float().to(self.device)
            semantic_loss = torch.tensor(0).float().to(self.device)
            
            # forward
            out = self.body_model(v_template=v_template, **flame_dict)  
            vertices = orthogonal_projection(out.vertices[0], transl, scale)
            
            # render full head 
            render_pkg = self.render(vertices, faces_all, H=render_res, W=render_res)
            
            if learn_eyes:   
                # no-eyes  
                alpha_wo_eyes = self.render(vertices, vis_tri_wo_eyes, H=render_res, W=render_res)['alpha']                  
                no_eye_mask_loss = ((alpha_wo_eyes[0] - gt_mask[0] * (1 - gt_eye_mask[0])) ** 2).mean()   
                semantic_loss += no_eye_mask_loss * 10     
                
                # remove eyes-related normal 
                pred_eye_mask = self.render(vertices, self.tri_eyes, H=render_res, W=render_res)['alpha']
                nml_loss = (render_pkg['normal'] - gt_normal)  
                nml_loss[0] *= (1 - gt_eye_mask[0])
                nml_loss[1] *= (1 - pred_eye_mask[1].detach())
                nml_loss[3:] *= (1 - pred_eye_mask[3:].detach()) 
                nml_loss = (nml_loss ** 2).mean() 
            else:  
                nml_loss = ((render_pkg['normal'] - gt_normal) ** 2).mean()    
            
            # landmark loss 
            if not self.opt.no_lmk_loss:   
                # landmark projection loss 
                lmks = orthogonal_projection(out.joints[0], transl, scale)
                lmks = torch.matmul(self.mvps[:, :3, :3], lmks.T).transpose(1, 2) + self.mvps[:, :3, 3].unsqueeze(1)
                lmk_prj_loss = l2_distance(lmks[0, 5:73, :2], gt_lmk68[:68])
                lmk_loss += lmk_prj_loss 
                
                # landmark symmetry loss
                out_cano = self.body_model(v_template=v_template, betas=flame_dict['betas'], 
                                           neck_pose=flame_dict['neck_pose']) 
                cano_lmks = out_cano.joints[0, 5:73, :]
                lmk_sym_loss = lmk68_symmetry_loss(cano_lmks)
                lmk_loss += lmk_sym_loss * 10  
            
            # laplacian reg 
            mesh = Meshes(verts=vertices.unsqueeze(0), faces=self.faces.unsqueeze(0))
            lap_loss = mesh_laplacian_smoothing(mesh, method="uniform")  

            bbox_loss = (vertices.abs() > 1).float().mean() * 10
            
            if self.opt.semantic_w > 0: 
                # render parsing  
                pred_parse = self.render(
                    vertices, faces_all, vc=self.parse_labels, H=render_res, W=render_res, 
                )['image']  
                semantic_loss += (((pred_parse[0] - parse_images['front'][0]) * gt_mask[0] * (1 - gt_eye_mask[0])) ** 2).mean() 
                
                if 'left_side' in parse_images and 'right_side' in parse_images:
                    semantic_loss += (
                        ((pred_parse[4] - parse_images['left_side'][0, ..., :3]) * (1 - pred_eye_mask[4].detach()) * parse_images['left_side'][0, ..., 3:]
                        ) ** 2).mean()  
                    semantic_loss += (((pred_parse[5] - parse_images['right_side'][0, ..., :3]) * (1 - pred_eye_mask[5].detach()) * parse_images['right_side'][0, ..., 3:]
                                    ) ** 2).mean()  
            
            loss = nml_loss * nml_weight + lmk_loss * self.opt.lmk_w  + lap_loss * reg_weight + semantic_loss * self.opt.semantic_w  
            loss += bbox_loss 
            # # TODO: add mask loss at the last epoch 
            # if w_mask_loss:
            #     # mask_loss = ((render_pkg['alpha'] - gt_mask) ** 2).mean() 
            #     # loss += mask_loss * 10   

            loss.backward()
            optimizer.step()
            optimizer.zero_grad() 
            
            # warm-up for hair 
            if not self.opt.no_lmk_loss and self.epoch < self.opt.hair_warmup_epoch:  
                v_template.data[self.face_vid] = self.body_model.v_template[self.face_vid]  # keep face vertices fixed
            
            if self.opt.fix_neck:
                v_template.data[self.neck_vid] = self.body_model.v_template[self.neck_vid]  
            
            pbar.set_description(
                f"loss: {loss.item():.4f} | "  
                f"nml_loss: {nml_loss.item():.4f} |" 
                f"lap_loss: {lap_loss.item():.4f} | " 
                f"semantic_loss: {semantic_loss.item():.4f} |"
                )  
            
            if vis_interval > 0 and i % vis_interval == 0 and self.debug: 
                os.makedirs(f'{self.save_dir}/optim', exist_ok=True) 
                skin_vc = lbs_weights_to_colors(self.body_model.lbs_weights.detach().cpu()) 
                skin_vc = torch.tensor(skin_vc).to(self.device)
                skin_image = self.render(vertices, faces_all, vc=skin_vc, H=render_res, W=render_res, shading_mode='lambertian')['image']

                # render canonical view 
                cano_v = orthogonal_projection(out_cano.vertices[0], transl, scale)
                cano_skin = self.render(cano_v, faces_all, vc=skin_vc, H=render_res, W=render_res, shading_mode='lambertian')['image']
                cano_lmks = orthogonal_projection(out_cano.joints[0], transl, scale)
                cano_lmks[:, 1] *= -1
                
                vis = [
                    torch.cat([vis_landmarks(gt_normal[0], gt_lmk68)] + list(gt_normal[1:].detach().cpu()), 1), 
                    torch.cat(
                        [vis_landmarks(render_pkg['normal'][0], lmks[0, :73])] + list(render_pkg['normal'][1:].detach().cpu())
                    , 1),  
                    torch.cat(list(pred_parse), 1).detach().cpu(), 
                    torch.cat(list(skin_image), 1).detach().cpu(),  
                    torch.cat([vis_landmarks(cano_skin[0], cano_lmks[:73])] + list(cano_skin[1:].detach().cpu()), 1),
                ]
                n_vis = len(vis)
                vis = (torch.cat(vis, dim=0).numpy()* 255).astype(np.uint8) 
                cv2.putText(vis, f"iter: {self.cnt}", (400, 80), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
                Image.fromarray(vis).resize((256*len(self.opt.views),256*n_vis)).save(f'{self.save_dir}/optim/{self.cnt:06d}.png')
                # exit()
            self.cnt += 1 
        
        with torch.no_grad():  
            out = self.body_model(v_template=v_template, **flame_dict)   
            vertices = orthogonal_projection(out.vertices[0], transl, scale)
        return v_template.detach(), vertices, out.joints_cano[0] 

    @torch.no_grad()
    def save_animation_video(
        self, data, base_mesh, uv_mask=None, driven='d6', res=512, flag_pasteback=True): 
        logger.info(f"Generate animation video...") 
        
        # load driven video 
        driven_frames, fps = load_video(driven, return_fps=True)
        print('LEN driven_frames:', len(driven_frames))
        
        # load motions    
        motions = np.load(driven.replace('.mp4', '.npz'))   
        # motions = {k: motions[k] for k in ['rotation', 'neck_pose', 'jaw_pose', 'eyes_pose', 'expr']}

        motions = {
            'betas': motions['shape'], 
            'expression': motions['expr'], 
            'jaw_pose': motions['jaw_pose'],
            'neck_pose': motions['neck_pose'], 
            'global_orient': motions['rotation'],
            'leye_pose': motions['eyes_pose'][:, :3],
            'reye_pose': motions['eyes_pose'][:, 3:]
        }
        motions = dicts.to_tensor(motions, self.device)  
        motions['betas'] = motions['betas'].unsqueeze(0).repeat(len(driven_frames), 1)

        # for debugging 
        flame_vertices = self.flame_model(**motions).vertices  

        assert len(driven_frames) == len(motions['global_orient']), f"len(driven_frames) != len(motions['rotation'])"
        # update pose 
        if flag_pasteback:
            motions['global_orient'][:, :3] = data['global_orient'] 
            if 'neck_pose' in data:  
                motions['neck_pose'][:, :3] = data['neck_pose']   
        elif self.opt.flag_relative_pose:
            flame_deg = torch.rad2deg(data['global_orient'])
            motion_deg = torch.rad2deg(motions['global_orient'])  
            motion_deg = (motion_deg + flame_deg).fmod(360) 
            motion_deg[:, 0] = flame_deg[0, 0] 
            motions['global_orient'] = torch.deg2rad(motion_deg) 
        
        # motions = { 
        #     'betas': data['betas'],  
        #     'expression': motions['expr'], 
        #     'jaw_pose': motions['jaw_pose'],
        #     'neck_pose': motions['neck_pose'], 
        #     'global_orient': motions['rotation'],
        #     'leye_pose': motions['eyes_pose'][:, :3],
        #     'reye_pose': motions['eyes_pose'][:, 3:]
        # } 
        motions['betas'] = data['betas']        
        # infer flame  
        vertices = self.body_model(**motions).vertices   
        vertices = orthogonal_projection(vertices, data['transl'].squeeze(), data['scale']) 
        
        # render 
        frames = []   
        mesh = base_mesh.clone()
        bg_color = torch.tensor([1.0]*3).to(self.device)
        input_img = np.array(data['input_pil'].convert('RGB').resize((res, res))) 
        rparams = dict(spp=2, h=res, w=res) 
        for i, (v, df) in enumerate(zip(vertices, driven_frames)):    
            # tz = v[:, 2].mean(0)
            # v[:, 2] -= tz
            mesh.v = v  
            if uv_mask is None:
                render_pkg = self.renderer(mesh, self.mvps, bg_color=bg_color, **rparams)
            else:
                render_pkg = self.inner_mouth_mask_render(mesh, uv_mask, **rparams)
            
            rgb = render_pkg['image'][0].detach().cpu().numpy()  
            nml = render_pkg['normal'][0].detach().cpu().numpy()
            alpha = render_pkg['alpha'][0].detach().cpu().numpy() 
            # rgb = render_pkg['image'] * render_pkg['alpha'] + 1 - render_pkg['alpha']
            # rgb = rgb[0].detach().cpu().numpy() * 255

            if flag_pasteback:  
                rgb = rgb * alpha * 255 + input_img * (1 - alpha) 
                rgb = uncrop(rgb, data['crop_info'], data['origin_pil'])
                rgb.thumbnail([1024, 1024], Image.Resampling.LANCZOS)
                # frames.append(np.hstack([input_img, np.array(rgb)])) 
                frames.append(np.array(rgb)) 
            else: 
                # df
                rgb = rgb * alpha + (1 - alpha)
                nml = nml * alpha + 1 - alpha 
                df = cv2.resize(df, (res, res)) 
                image = np.hstack([df, input_img, rgb* 255, nml * 255])
                frames.append(image.astype(np.uint8))

            # render driven flame model 
            flame_mesh = Mesh(v=flame_vertices[i], f=self.flame_model.faces_tensor.int())
            render_pkg = self.renderer(flame_mesh, self.mvps, bg_color=bg_color, **rparams)
            frames[-1] = np.hstack([frames[-1].astype(np.uint8), render_pkg['normal'][0].detach().cpu().numpy() * 255])

        export_video(frames, f"{self.save_dir}/video/animation-{driven.split('/')[-1]}", fps=fps)
        
    def refine_texture(self, mesh, full_texture_mesh=None, max_iter=1500, albedo_mask=None):
        # TODO: compare optimization vs flame-back-proj
        if mesh.albedo is None:
            mesh.albedo = torch.zeros(1024, 1024, 3).to(self.device)
        
        albedo = mesh.albedo.clamp(0, 1)   
        # inpainting 
        if full_texture_mesh is not None: 
            tmp_mesh = mesh.clone() 
            tmp_mesh.auto_size()
            full_texture_mesh.auto_size()
            inpaint_albedo = nn.Parameter(torch.zeros_like(albedo), requires_grad=True)
            optimizer = torch.optim.Adam([inpaint_albedo], lr=0.01)
            for _ in track(range(max_iter), description="Inpainting texture..."):
                elev = random.randint(-60, 60) 
                azim = random.randint(-180, 180) 
                mvp = torch.eye(4).unsqueeze(0).to(self.device)
                rot = batch_rodrigues(torch.tensor([[0, np.radians(azim), 0]])
                ) @ batch_rodrigues(torch.tensor([[np.radians(elev), 0, 0]]))  
                R = batch_rodrigues(torch.tensor([[np.pi, 0, 0]]))
                rot = rot.float() @ R.float()
                mvp[:, :3, :3] = rot.float().to(self.device) 
                target_image = self.renderer(full_texture_mesh, mvp, spp=2, bg_color=self.bg_color, h=h, w=w)['image'].detach()
                tmp_mesh.albedo = inpaint_albedo
                
                prd_img = self.renderer(tmp_mesh, mvp, spp=2, bg_color=self.bg_color, h=h, w=w)['image']
                loss = ((prd_img - target_image)).abs().mean()  
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                
            inpaint_albedo = inpaint_albedo.detach() 
            albedo_mask = albedo.sum(2) == 0 
            albedo[albedo_mask] = inpaint_albedo[albedo_mask]
            
        mesh.albedo = albedo.clamp(0, 1)
        return mesh 
    
    def geometry_optmization(self, data, size=(512, 512)):
        flame_attributes = {} 
        # rotate normal to face camera 
        gt_normal = self.rotate_normal(data['normal'], data['mask'])            
        # prepare   
        learn_eyes = self.opt.learnable_eyes and data['learnable_eyes']
        in_dict = {
            'gt_normal': scale_img_nhwc(gt_normal, size),
            'gt_mask': scale_img_nhwc(data['mask'], size),
            'gt_eye_mask': (scale_img_nhwc(data['eye_mask'], size) > 0.5).float(),
            'neck_mask': scale_img_nhwc(data['neck_mask'], size) if 'neck_mask' in data else None, 
            'parse_images': {k: scale_img_nhwc(v, size) for k,v in data['parse_images'].items()},
            'flame_dict': {
                k:v.clone() for k,v in data.items() if k in [
                    'betas', 'expression', 'global_orient', 'jaw_pose', 'leye_pose', 'reye_pose', 'neck_pose'
                    ]
            }, 
            'learn_eyes': learn_eyes, 
            'transl': data['transl'],
            'scale': data['scale'], 
            'gt_lmk68': data['lmks'],    
        }
        
        for epoch in range(self.opt.epoch):  
            self.epoch = epoch 
            
            # 1. update template 
            v_template, v, joints_cano = self.template_deformation(**in_dict, render_res=512, w_mask_loss=epoch>3)
            self.body_model.v_template = v_template
            flame_attributes['v_template'] = v_template
            
            N = len(self.body_model.v_template)
            
            # 2. remeshing and rigging 
            if not self.opt.no_remesh and epoch < self.opt.epoch - 1: #  and epoch < self.opt.epoch - 1:   
                # remeshing 
                v_attr = torch.cat([self.pack_attributes(), self.parse_labels], 1)
                if learn_eyes: 
                    non_remesh_vid = self.eye_vid
                    if self.opt.fix_neck:
                        non_remesh_vid = np.concatenate([self.eye_vid, self.neck_vid], 0) 
                    if epoch < self.opt.hair_warmup_epoch:
                        non_remesh_vid = np.concatenate([non_remesh_vid, self.face_vid], 0)
                    
                    vid_mask = ids2mask(non_remesh_vid, N, device=self.device, reverse=True)
                    if self.opt.uv_interpolate: 
                        v_attr, faces, self.vt, self.ft = remesh_withuv(
                        v_attr, self.faces, 
                        self.body_model.v_template.new_ones(N) * self.opt.min_edge_length, 
                        self.body_model.v_template.new_ones(N) * self.opt.max_edge_length,
                        flip=False, 
                        vid_mask=vid_mask, 
                        vt=self.vt, 
                        ft=self.ft
                    ) 
                    else: 
                        v_attr, faces = remesh(
                            v_attr, self.faces, 
                            self.body_model.v_template.new_ones(N) * self.opt.min_edge_length, 
                            self.body_model.v_template.new_ones(N) * self.opt.max_edge_length,
                            flip=True, 
                            vid_mask=vid_mask
                        ) 
                    v_attr, self.parse_labels = v_attr.split([v_attr.shape[1]-3, 3], 1)
                    self.faces = faces.int()  
                    # update non-eye fids 
                    fmask = ids2mask(self.eyes_fids, len(self.faces), reverse=True, device=self.device)
                    self.tri_wo_eyes = self.faces[fmask]
                
                else:
                    non_remesh_vid = self.neck_vid if self.opt.fix_neck else None 
                    if epoch < self.opt.hair_warmup_epoch:
                        non_remesh_vid = np.concatenate([non_remesh_vid, self.face_vid], 0) if non_remesh_vid is not None else self.face_vid
                    
                    vid_mask = None
                    if non_remesh_vid is not None:
                        vid_mask = ids2mask(non_remesh_vid, N, device=self.device, reverse=True)
                    
                    v_attr, faces = remesh(            
                        v_attr, self.faces, 
                        v.new_ones(N) * self.opt.min_edge_length, 
                        v.new_ones(N) * self.opt.max_edge_length,
                        flip=True, 
                        vid_mask=vid_mask
                    ) 
                    self.faces = faces.int()
                
                # update \Omega   
                flame_attributes = self.unpack_attributes(v_attr) 
                self.body_model.set_params(flame_attributes) 
                with torch.no_grad():  
                    self.body_model.update_J_regressor(joints_cano, data['betas'], data['expression'])
                    flame_attributes['J_regressor'] = self.body_model.J_regressor.detach()
                
        if self.opt.epoch > 1:
            dicts.save2pth(flame_attributes, f'{self.save_dir}/flame_attributes.pth')
        
        if opt.uv_tex == 'xatlas':
            mesh = Mesh(v=v, f=self.faces, ft=self.faces, device=self.device) 
        else:
            mesh = Mesh(v=v, f=self.faces, vt=self.vt, ft=self.ft.int(), device=self.device)
            
        if learn_eyes: 
            # self.faces = torch.cat([self.tri_wo_eyes, self.tri_eyes], 0)
            # mesh = Mesh(v=v, f=self.faces, ft=self.faces, device=self.device)
            mesh = self.eyeball_optimization(data, mesh, in_dict['flame_dict'], joints_cano)
            # joints_cano[3] = self.body_model.v_template[self.leye_vid].mean(0).detach()
            # joints_cano[4] = self.body_model.v_template[self.reye_vid].mean(0).detach() 
            # self.body_model.update_J_regressor(joints_cano, data['betas'], data['expression'])
            flame_attributes['v_template'] = self.body_model.v_template.detach()
            flame_attributes['J_regressor'] = self.body_model.J_regressor.detach()
            dicts.save2pth(flame_attributes, f'{self.save_dir}/flame_attributes.pth')

        return mesh 

    def eyeball_optimization(self, data, mesh, flame_dict, joints_cano, iters=100, size=512):
        # debuging 
        # save_obj('test1.obj', mesh.v, mesh.f)
        
        leyeball_v = self.origin_template[self.leyeball_vid].detach() 
        _, c, s = normalize_vertices(leyeball_v, max_dim=1, return_params=True)
        leye_v = self.origin_template[self.leye_vid].detach() 
        leye_v_norm = (leye_v - c) * s 
        
        reyeball_v = self.origin_template[self.reyeball_vid].detach()
        _, c, s = normalize_vertices(reyeball_v, max_dim=1, return_params=True)
        reye_v = self.origin_template[self.reye_vid].detach()
        reye_v_norm = (reye_v - c) * s
        
        # Equation (11) 
        eye_angle = np.radians(self.opt.eye_deg*0.5)   
        observe_eye_length = (data['leye_width'] + data['reye_width']) * 0.5 
        eye_radius = (observe_eye_length * 0.5) / np.sin(eye_angle) 
        eye_to_plane_dist = eye_radius * np.cos(eye_angle) 
        # scale to flame space 
        eye_radius /= data['scale'].item()
        eye_to_plane_dist /= data['scale'].item()
        
        lref_pts = self.body_model.v_template[self.leyelid_vid].mean(0).detach() 
        lref_pts[2] -= eye_to_plane_dist
        rref_pts = self.body_model.v_template[self.reyelid_vid].mean(0).detach() 
        rref_pts[2] -= eye_to_plane_dist
        
        # initial parameters
        leye_center = nn.Parameter(lref_pts[None], requires_grad=True)
        reye_center = nn.Parameter(rref_pts[None], requires_grad=True)
        eye_radius = nn.Parameter(torch.tensor(eye_radius).to(self.device), requires_grad=True) 
        optimizer = torch.optim.Adam([leye_center, reye_center, eye_radius], lr=0.0001)
        
        # set eye color as the hair color  
        vc_idff = self.parse_labels.unsqueeze(1) - torch.tensor([
            self.opt.part_color_map.face, 
            self.opt.part_color_map.hair, 
            self.opt.part_color_map.neck, 
            self.opt.part_color_map.eye, 
        ]).float().to(self.device).unsqueeze(0)
        vc_idff = vc_idff.abs().mean(2)
        labels = torch.argmin(vc_idff, 1)
        
        # set camera  
        mvp = self.renderer.get_orthogonal_cameras(n=7, yaw_range=(-30, 30), endpoint=True)
        mvp = mvp.to(self.mvps) 
        
        v_template = self.body_model.v_template.clone().detach()
        v_template[self.leye_vid] = leye_v_norm * eye_radius + leye_center
        v_template[self.reye_vid] = reye_v_norm * eye_radius + reye_center
        
        # rm eyes and render parsing images 
        with torch.no_grad():
            global_orient = flame_dict['global_orient'].clone()
            flame_dict['global_orient'] *= 0 
            v = self.body_model(v_template=v_template, **flame_dict).vertices[0] 
            vmax, vmin = v.max(0)[0], v.min(0)[0] 
            v = orthogonal_projection(v, data['transl'].squeeze(), data['scale']*0.8) 
            parse_images = self.render(v, self.tri_wo_eyes, mvp=mvp, vc=self.parse_labels, H=size, W=size)['image'].detach() 
            face_mask_wo_mask = (parse_images - torch.tensor(self.opt.part_color_map.face).to(parse_images).reshape(1, 1, 1, 3))
            face_mask_wo_mask = (face_mask_wo_mask.abs().mean(-1) < 0.1).float() 
            
            vc_lable = torch.ones_like(self.parse_labels)
            vc_lable[labels == 3] = torch.tensor([0., 0., 0.]).to(self.device)  
            vc_lable[labels == 0] = torch.tensor([0., 0., 0.]).to(self.device) 
            face_mask = self.render(v, mesh.f, mvp=mvp, vc=vc_lable, H=size, W=size)['image'].detach() 
            face_mask = (face_mask.mean(-1) < 0.5).float() 
            
            eye_mask = self.render(v, self.tri_eyes, mvp=mvp, H=size, W=size)['alpha'].detach().squeeze(-1)             
            eye_mask = ((face_mask - face_mask_wo_mask) * eye_mask) > 0.1 
            parse_images[eye_mask] = torch.tensor(self.opt.part_color_map.eye).to(parse_images)
            
            # # debug
            vis = torch.cat([
                torch.cat(list(face_mask_wo_mask), 1),
                torch.cat(list(face_mask), 1),
                torch.cat(list(eye_mask), 1),
                ], 0).detach().cpu().numpy()
            # Image.fromarray((vis * 255).astype(np.uint8)).save(f'reconstruction.png')
            # exit()
            
        vc_lable = self.parse_labels.clone() 
        # vc_lable[labels == 3] = torch.tensor(self.opt.part_color_map.hair).to(self.device)
        # vc_lable[labels == 1] = torch.tensor(self.opt.part_color_map.eye).to(self.device) 

        minimal_loss, minimal_params = None, None   
        pbar = tqdm(range(iters))
        for i in pbar:    
            v_template[self.leye_vid] = leye_v_norm * eye_radius + leye_center
            v_template[self.reye_vid] = reye_v_norm * eye_radius + reye_center
            
            with torch.no_grad(): 
                self.body_model.v_template = v_template.detach()
                joints_cano[3] = leye_center[0].detach()
                joints_cano[4] = reye_center[0].detach() 
                self.body_model.update_J_regressor(joints_cano, data['betas'], data['expression'])

            out = self.body_model(v_template=v_template, **flame_dict) 
            v = orthogonal_projection(out.vertices[0], data['transl'].squeeze(), data['scale']*0.8)

            # render 
            pkg_rnd = self.render(v, mesh.f, vc=vc_lable, mvp=mvp, H=size, W=size)      
            loss = (pkg_rnd['image'] - parse_images).abs()  
            loss = loss.sum()  
            loss.backward(retain_graph=True) 
            optimizer.step()
            optimizer.zero_grad() 
            
            if i == 0 or loss.item() < minimal_loss:
                minimal_params = (eye_radius.clone(), leye_center.clone(), reye_center.clone())
                minimal_loss = loss.item() 
                pbar.set_description(f"Minimal loss: {minimal_loss:.2f}, {i}")
            
            # vis  
            if self.debug and i % 10 == 0: 
                vis = torch.cat([   
                        torch.cat(list(parse_images), 1),   
                        torch.cat(list(pkg_rnd['image']), 1), 
                        torch.cat(list((pkg_rnd['image'] - parse_images).abs()), 1),
                        # torch.cat(list(eye_mask * (pkg_rnd['image'] - parse_images).abs()), 1), 
                    ], 0).detach().cpu().numpy() 
                cv2.putText(
                    vis, f'{loss.item():.2f}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA
                )
                vis = Image.fromarray((vis * 255).astype(np.uint8))
                vis.save(f'{self.save_dir}/optim_eye/parse_{i:04d}.png')
        
        with torch.no_grad(): 
            eye_radius, leye_center, reye_center = minimal_params
            v_template[self.leye_vid] = leye_v_norm * eye_radius + leye_center
            v_template[self.reye_vid] = reye_v_norm * eye_radius + reye_center 
            self.body_model.v_template = v_template 

            joints_cano[3] = leye_center[0].detach()
            joints_cano[4] = reye_center[0].detach() 
            self.body_model.update_J_regressor(joints_cano, data['betas'], data['expression'])

            flame_dict['global_orient'] = global_orient
            v = self.body_model(**flame_dict).vertices[0].detach()
            v = orthogonal_projection(v, data['transl'].squeeze(), data['scale'])
            mesh.v = v
            # save_obj('test1.obj', mesh.v, mesh.f) 
        return mesh 
    
    def add_teeth(self, mesh, flame_attributes): # TODO: add teeth
        teeth_mesh = Mesh.load_obj(f'./data/teeth/teeth_tri2.obj', device=self.device)
        teeth_mesh.auto_normal()
        id_mesh = self.body_model.v_template
        
        target = self.body_model().vertices[0]
        trans = (target[3797]+target[3920]-id_mesh[3797]-id_mesh[3920])/2
        scale = torch.norm(target[3797]-target[3920]) / torch.norm(id_mesh[3797]-id_mesh[3920])
        center = torch.mean(teeth_mesh.v, dim=0)
        # TODO: check bug here 
        teeth_mesh.v = (teeth_mesh.v-center)*scale+center+trans
        teeth_mesh.v[:, 2] -= 0.015 # move teeth back a little bit
        # teeth_mesh.v[:, 1] -= 0.005

        # load skinning and rigging
        with open(f'./data/teeth/teeth_mask.txt') as file:
            teeth_mask = [int(line.rstrip()) for line in file]
        flame_attributes['v_template'] = torch.cat((flame_attributes['v_template'], teeth_mesh.v), 0)
        flame_attributes['shapedirs'] = torch.cat((flame_attributes['shapedirs'], torch.zeros((teeth_mesh.v.shape[0],3,400)).to(self.device)), 0)
        flame_attributes['posedirs'] = torch.cat((flame_attributes['posedirs'], torch.zeros((36, teeth_mesh.v.shape[0]*3)).to(self.device)), 1)
        flame_attributes['J_regressor'] = torch.cat((flame_attributes['J_regressor'], torch.zeros((5, teeth_mesh.v.shape[0])).to(self.device)), 1)
        lbs_temp = []
        for i in range(teeth_mesh.v.shape[0]):
            if i in teeth_mask:
                lbs_temp.append([0,0,1,0,0])
            else:
                lbs_temp.append([0,1,0,0,0])
        flame_attributes['lbs_weights'] = torch.cat((flame_attributes['lbs_weights'], torch.tensor(lbs_temp).to(self.device)), 0)
        self.body_model.set_params(flame_attributes)

        # combine texture  
        if mesh.albedo.shape[0] != 2048:
            mesh.albedo = scale_img_hwc(mesh.albedo, (2048, 2048))
            
        tex = mesh.albedo.cpu().numpy()[..., ::-1] * 255  
        img = cv2.imread(f'./data/teeth/teeth_color_map.png') * 0.6
        tex[1024:, :1024,] = img
        img = cv2.imread(f'./data/teeth/mouth_color_map.png')
        tex[1024:, 1024:,] = img
        albedo = cv2.cvtColor(tex / 255, cv2.COLOR_BGR2RGB) 
        mesh.albedo = torch.tensor(albedo, dtype=torch.float32, device=self.device) 
        # load texture and uv  
        mesh.f = torch.cat((mesh.f, teeth_mesh.f+(mesh.v.shape[0])), 0)
        mesh.v = torch.cat((mesh.v, teeth_mesh.v), 0)
        mesh.vn = torch.cat((mesh.vn, teeth_mesh.vn), 0)
        mesh.fn = torch.cat((mesh.fn, teeth_mesh.fn), 0)
        mesh.ft = torch.cat((mesh.ft, teeth_mesh.ft+(mesh.vt.shape[0])), 0)
        mesh.vt = torch.cat((mesh.vt, teeth_mesh.vt), 0)
        # bottom left 
        mesh.vt[-23084:,] /= 2.0
        mesh.vt[-23084:,1] += 0.5
        # bottom right  
        mesh.vt[-27392:-23084,] /= 2.0
        mesh.vt[-27392:-23084,:] += 0.5  
        return mesh 
    
    def run(self):    
        # os.system(f'rm -rf {self.save_dir}/video/project.mp4')
        
        data = self.dataset.get_item()
        data = dicts.to_device(data, self.device) 
        
        if 'v_offsets' in data:
            self.body_model.v_template += data['v_offsets']  
        
        if not os.path.exists(f'{self.save_dir}/flame_attributes.pth') or not os.path.exists(f'{self.save_dir}/recon/recon_textured.obj') or \
            not os.path.exists(f'{self.save_dir}/parse_labels.pth'):
              
            mesh = self.geometry_optmization(data) 
            if self.opt.uv_tex == 'flame': 
                mesh = albedo_from_images_possion_blending(self.renderer, mesh, data['image'], dilate=True) 
            elif self.opt.uv_tex == 'xatlas':
                mesh = albedo_from_images(self.renderer, mesh, data['image']) 
            else:  
                # get the flame uv 
                mesh = albedo_from_images_possion_blending(
                    self.renderer, mesh, data['image'], self.mvps, self.extrinsic, 
                    weights=[10, 1, 1, 1, 1, 1], 
                    dilate=True
                )  

                v_attr = torch.cat([self.pack_attributes(True), self.parse_labels], 1)
                # find face area  
                label_colos = torch.tensor([
                    self.opt.part_color_map.face, 
                    self.opt.part_color_map.hair, 
                    self.opt.part_color_map.neck, 
                    self.opt.part_color_map.eye
                ]).to(self.parse_labels)
                diff = (self.parse_labels.unsqueeze(1) - label_colos.unsqueeze(0)).abs().mean(2)
                face_ids = torch.argmax(diff, dim=1)
                face_mask = (face_ids == 0) | (face_ids == 3) # face and eyes
                
                face_mesh, nface_mesh, indices = split_mesh(mesh, face_mask)  
                v_attr = torch.cat([v_attr[indices[0]], v_attr[indices[1]]])
                v_attr, self.parse_labels = v_attr.split([v_attr.shape[1]-3, 3], 1)
                flame_attributes =  self.unpack_attributes(v_attr, True)
                self.body_model.set_params(flame_attributes) 
                dicts.save2pth(flame_attributes, f'{self.save_dir}/flame_attributes.pth')

                # merge mesh 
                face_mesh.vt /= 2 
                # non-face mesh generate xatalx uv 
                nface_mesh.auto_uv(vmap=False)
                nface_mesh.vt /= 2 
                nface_mesh.vt[:, 0] += 0.5 
                v = torch.cat([face_mesh.v, nface_mesh.v], 0)
                f = torch.cat([face_mesh.f, nface_mesh.f + len(face_mesh.v)], 0)
                vt = torch.cat([face_mesh.vt, nface_mesh.vt], 0)
                ft = torch.cat([face_mesh.ft, nface_mesh.ft + len(face_mesh.v)], 0)
                    
                final_mesh = Mesh(v=v, f=f.int(), vt=vt, ft=ft.int(), device=self.device)
                final_mesh, albedo_mask = albedo_from_images_possion_blending(
                    self.renderer, final_mesh, data['image'], self.mvps, self.extrinsic, 
                    weights=[10,1,1,1,1,1], return_mask=True, albedo_res=self.opt.albedo_res) 
                final_mesh_albedo = final_mesh.albedo.clone()

                # inpainting from flame 
                mvps, extrinsic, _ = self.renderer.get_orthogonal_cameras(
                    n=8, yaw_range=(22.5, 382.5), return_all=True
                ) 
                out = self.renderer(mesh, mvps.to(v), spp=2, bg_color=self.bg_color)  
                final_mesh, inpaint_mask = albedo_from_images_possion_blending(
                    self.renderer, final_mesh, out['image'], mvps.to(v), extrinsic.to(v), 
                    weights=[1]*8, return_mask=True, albedo_res=self.opt.albedo_res
                )  
                final_mesh.albedo[albedo_mask] = final_mesh_albedo[albedo_mask]
                albedo_mask = albedo_mask | inpaint_mask

                # dilate for face area 
                h, w = final_mesh.albedo.shape[:2]
                face_albedo = final_mesh.albedo[:h//2] 
                face_albedo_mask = albedo_mask[:h//2].cpu().numpy() 
                face_albedo = dilate_image(face_albedo, face_albedo_mask, iterations=int(h*0.2)) 
                final_mesh.albedo[:h//2] = face_albedo 
                mesh = final_mesh.clone()

            # mesh.write(f'{self.save_dir}/recon/recon_textured.obj')
            # self.render_360view(mesh, export_path=f'{self.save_dir}/video/reconstruction.mp4', input_pil=data['input_pil'])  
            # torch.save(self.parse_labels, f'{self.save_dir}/parse_labels.pth') 
            # print('Save to:', f'{self.save_dir}/flame_attributes.pth')
            # exit()
        else:   
            flame_attributes = torch.load(f'{self.save_dir}/flame_attributes.pth')
            flame_attributes = dicts.to_device(flame_attributes, self.device)
            mesh = Mesh.load_obj(f'{self.save_dir}/recon/recon_textured.obj', device=self.device)
            self.body_model.set_params(flame_attributes)        
            self.parse_labels = torch.load(f'{self.save_dir}/parse_labels.pth').to(self.device)
        
        uv_mask = None 
        if self.opt.add_teeth:
            mesh = self.add_teeth(mesh, flame_attributes)  
            uv_mask = np.array(Image.open('./data/teeth/inner_mouth_mask2.png').resize((2048, 2048))) > 0 
            uv_mask = torch.tensor(uv_mask).to(self.device).float()
        
        for dr, past_back in self.opt.drivens.items():   
            self.save_animation_video(
                data, mesh, uv_mask, driven=dr, 
                res=self.opt.render_res,
                flag_pasteback=past_back)

        return 


if __name__ == "__main__": 
    import argparse
    from omegaconf import OmegaConf

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default='./configs/6views.yaml', help="path to the yaml config file")
    args, extras = parser.parse_known_args() 
    opt = OmegaConf.merge(OmegaConf.load(args.config), OmegaConf.from_cli(extras))
    
    import time
    start = time.time()
    trainer = Trainer(opt)   
    trainer.run()

    logger.info(f"Total time: {time.time() - start:.2f}s")