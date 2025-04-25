from typing import Any, Callable, Dict, List, Optional, Tuple, Union 
import os
import os.path as osp
import cv2 
import torch
import torch.nn as nn
import random 
import numpy as np
from glob import glob
import face_alignment
from PIL import Image
from tqdm import tqdm 
from rich.progress import track
from termcolor import colored 
import kiui 
from kiui.mesh import Mesh
from kiui.op import scale_img_hwc, scale_img_nhwc
from smplx.render import Renderer 
from smplx.lbs import batch_rodrigues  
from rembg import new_session, remove  

import src.utils.dict as dicts 
from src.utils.mesh import save_obj 
from src.utils.vis import vis_landmarks
from src.model.segment import Segmenter
from src.model.FaceDetector import FaceParser
from src.model.deep3dface import PoseEstimator 
from src.utils.crop import change_rgba_bg, pad_image_square, get_bbox, crop_rgba, pad_image
from src.utils.helper import pts_to_bbox, create_flame_model, orthogonal_projection, find_files_from_dir, save_image 
from src.model.unique3d_diffusion import Unique3dDiffusion


import coloredlogs, logging 
logger = logging.getLogger(__name__) 
coloredlogs.install(level='DEBUG', logger=logger) 

flame_model = create_flame_model()  

renderer = Renderer()   
providers = [
    ('CUDAExecutionProvider', {
        'device_id': 0,
        'arena_extend_strategy': 'kSameAsRequested',
        'gpu_mem_limit': 8 * 1024 * 1024 * 1024,
        'cudnn_conv_algo_search': 'HEURISTIC',
    })
]
session = new_session(providers=providers) 

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# segmentation model
rmbg = Segmenter(device)  
logger.debug(f"Loaded segmentor.")

# exit()
def emoca_flame_tracking(emoca, pil_image, landmarks, save_dir):
    '''
    Run EMOCA on a list of images
        Args:
        -----
        emoca: EMOCA
            EMOCA model
        images: list
            list of images
        save_dir: str
            path to save the results
        render_video: bool, optional
            render the result video
    '''     
    logger.info(f"EMOCA: forwad pass...") 
    vals, visdict = emoca.forward(pil_image, landmarks, save_dir=save_dir) 
    
    logger.info(f"EMOCA: rendering results...")
    overlap, emoca_rgba = render_emoca_results(vals, flame_model, renderer, pil_image)

    # TODO: use a smarter and faster way to recompute the camera
    logger.info(f"EMOCA: recomputing camera in original image size...")
    new_rgba = emoca_recompute_camera(vals, flame_model, renderer, emoca_rgba, save_dir)
    new_normal, new_mask = new_rgba[..., :3], new_rgba[..., 3] / 255 
    transparent_pil = Image.fromarray((new_mask * 255).astype(np.uint8)) 
    overlap2 = pil_image.copy()  
    overlap2.paste(Image.fromarray(new_normal), (0, 0), transparent_pil)

    vis = np.hstack([np.array(pil_image)[..., ::-1], overlap, np.array(overlap2)[..., ::-1]])
    cv2.imwrite(osp.join(save_dir, 'face_mask.png'), new_mask * 255)
    
    logger.info(f"EMOCA Done! Results saved to {save_dir}")
    return vis


def emoca_recompute_camera(vals, flame_model, renderer, normal_rgba, save_dir, iters=1000, render_res=800):
    ''''
    convert emoca camera (crop image) to align with the original image
        Args:
        -----
        vals: dict
            model outputs
        flame_model: nn.Module
            flame model
        renderer: Renderer
            renderer
        normal_rgba: np.ndarray h x w x 4, rgb values in 0-255
            normal map
        iters: int, optional
            number of iterations
        
        Returns:
        -------
        optim_normal: np.ndarray
            optimized normal map h x w x 3
    ''' 
    faces = flame_model.segment.get_triangles('face')
    faces = torch.tensor(faces.astype(np.int32)).cuda() 
    
    ori_h, ori_w = normal_rgba.shape[:2] 
    normal_rgba = pad_image_square(normal_rgba, 255) 
    normal_rgba = cv2.resize(normal_rgba, (render_res, render_res))
    normal_rgba = torch.tensor(normal_rgba).float().cuda() / 255

    transl = nn.Parameter(torch.tensor([[0, 0]]).float().cuda(), requires_grad=True)
    scale = nn.Parameter(torch.tensor([1]).float().cuda(), requires_grad=True)
    optimizer = torch.optim.Adam([transl, scale], lr=1e-2)

    mvp = torch.eye(4)[None].cuda()
    mvp[:, :3, :3] = batch_rodrigues(torch.tensor([np.pi, 0, 0]).reshape(1, 3)).reshape(3, 3) 
    
    pbar = tqdm(range(iters))
    for i in pbar: 
        optimizer.zero_grad() 
        out = flame_model(
            betas=vals['shapecode'], 
            expression=vals['expcode'], 
            global_orient=vals['posecode'][:, 0:3],  
            jaw_pose=vals['posecode'][:, 3:6]
        ) 
        vertices = orthogonal_projection(out.vertices[0], transl, scale) 
        mesh = Mesh(vertices, faces)
        mesh.auto_normal()

        render_pkg = renderer(mesh, mvp, h=render_res, w=render_res, bg_color=torch.zeros(3).cuda()) 
        nml = render_pkg['normal'][0] 
        loss = ((nml - normal_rgba[..., :3]).abs() * normal_rgba[..., 3:]).mean()
        pbar.set_description(f"loss: {loss.item():.4f}")

        if loss < 2e-3:
            break

        loss.backward()
        optimizer.step() 

    new_cam = {
        'transl': transl[0].detach().cpu().numpy(),
        'scale': scale.item()
    }
    np.save(osp.join(save_dir, "camera.npy"), new_cam) 

    # convert the normal image to the origin size 
    optim_rgba = torch.cat([nml, render_pkg['alpha'][0]], -1).detach().cpu().numpy() * 255
    pad_size = max(ori_h, ori_w)  
    optim_rgba = cv2.resize(optim_rgba.astype(np.uint8), (pad_size, pad_size))
    dx, dy = (pad_size - ori_w) // 2, (pad_size - ori_h) // 2
    optim_rgba = optim_rgba[dy:dy+ori_h, dx:dx+ori_w] 
    return optim_rgba


def landmark_detection(face_detector, image, save_path=None, vis_path=None):
    '''
    detect 68 face landmarks and save the normalized landmarks to a txt file
        Args:
        -----
        face_detectors: dict of face_alignment.FaceAlignment
            face detector
        image: np.ndarray h x w x 3 
            image
        save_path: str, endswith .txt
            path to save the landmarks
        vis_path: str, endswith .png
            path to save the visualization of landmarks
    '''
    w, h = image.shape[:2]
    
    landmarks = None
    for name in ['sfd', 'blazeface', 'dlib']:
        detector = face_detector[name]

        if detector is None:
            logger.debug(f"Loading {name} face detector...")
            detector = load_face_detector(name)
            face_detector[name] = detector

        try:
            landmarks = detector.get_landmarks(image)  
        except Exception as e:
            print(e)
            continue

        if landmarks is not None: 
            logger.debug(f"Face detected using {name} detector!")
            break

    if landmarks is None:
        logger.error(f"Fail to detected face!")   
        raise ValueError("Fail to detected face") 

    landmarks = landmarks[0] 

    if save_path is not None:
        np.savetxt(save_path, landmarks, fmt='%.4f')

    if vis_path is not None:  
        image = vis_landmarks(image / 255, landmarks, isScale=False).numpy()  
        Image.fromarray((image * 255).astype(np.uint8)).save(vis_path)

    return landmarks 


def load_face_detector(detector_name):
    if detector_name == 'sfd': 
        return face_alignment.FaceAlignment(
            face_alignment.LandmarksType.TWO_D, 
            device='cuda', 
            flip_input=False,  
            face_detector='sfd',  
        )  
    elif detector_name == 'blazeface':
        return face_alignment.FaceAlignment(
            landmarks_type=face_alignment.LandmarksType.TWO_D, 
            device='cuda',  
            face_detector='blazeface',  
        )  
    elif detector_name == 'dlib':
        return face_alignment.FaceAlignment(
            landmarks_type=face_alignment.LandmarksType.TWO_D, 
            device='cuda',  
            face_detector='dlib',  
        )  
    else:
        raise ValueError(f"Unknown face detector: {detector_name}, supported detectors: sfd, blazeface, dlib")


def process_image(image_path, debug=False, shoulder_size=0.2, pad=0.1):
    image = Image.open(image_path).convert('RGB')  
    image.thumbnail([2048, 2048], Image.Resampling.LANCZOS)  
    
    # remove bg  
    alpha = rmbg(image, output_type='pil')  
    alpha = np.array(alpha) 
    
    # crop by landmarks
    landmarks = landmark_detection(face_detectors, np.array(image)) 
    top, _, bottom, _ = pts_to_bbox(landmarks)  
    bottom = int(bottom + (bottom - top) * shoulder_size) 
    alpha[bottom:] = 0
    img = np.concatenate([np.array(image), alpha[..., None]], -1)    

    # first crop
    img, crop_bbox1 = crop_rgba(img, return_bbox=True, to_square=False)
    landmarks = landmarks - np.array([crop_bbox1[1], crop_bbox1[0]]) 
   
    parse_alpha = face_parsing(img, face_parser, landmarks)['mask']  
    y_min, x_min, y_max, x_max = get_bbox(parse_alpha, pad_size=0)
    # Image.fromarray(parse_alpha[x_min: x_max, y_min: y_max].astype(np.uint8) * 255 ).save(f'test.png')
    
    merge_alpha = np.zeros_like(parse_alpha)
    merge_alpha[x_min: x_max, y_min: y_max] = img[x_min: x_max, y_min: y_max, -1] 
    merge_alpha = (merge_alpha > 0) | parse_alpha.astype(bool)
    merge_alpha = merge_alpha.astype(np.uint8) * 255 
    img = np.concatenate([img[..., :3], merge_alpha[..., None]], -1)
    
    # second crop
    img, crop_bbox2 = crop_rgba(img, return_bbox=True, to_square=False)
    landmarks = landmarks - np.array([crop_bbox2[1], crop_bbox2[0]]) 
    crop_info = {
        'origin_size': image.size,
        'crop_bbox': [
            crop_bbox1[0] + crop_bbox2[0],
            crop_bbox1[1] + crop_bbox2[1],
            crop_bbox1[2] - crop_bbox2[2],
            crop_bbox1[3] - crop_bbox2[3],
        ] 
    } 
    
    # pad to square
    if img.shape[0] != img.shape[1]:
        img, pad_bbox = pad_image_square(img, (255, 255, 255, 0), return_bbox=True)
        landmarks = landmarks + np.array(pad_bbox[:2])
        crop_info['pad2square_bbox'] = pad_bbox 

    # pad-border  
    if pad > 0:
        pad_size = int(img.shape[0] * 0.1)
        # img = np.pad(img, ((pad_size, pad_size), (pad_size, pad_size), (0, 0)), 'constant', 
        # constant_values=((255, 255), (255, 255), (255, 255), (0, 0)))
        img = pad_image(img, pad_size, (255, 255, 255, 0))
        landmarks += pad_size  
        crop_info['pad_bbox'] = [pad_size] * 4  
    
    crop_info['size'] = img.shape[:2] 
    face_parsing(img, face_parser, landmarks, save_dir=save_dir)
    
    return img, landmarks, crop_info


def process_image2(image_path, shoulder_size=0.2, pad=0.1):
    image = Image.open(image_path)   
    image.thumbnail([2048, 2048], Image.Resampling.LANCZOS)  
     
   
    # crop by landmarks
    landmarks = landmark_detection(face_detectors, np.array(image)[..., :3])  
    img = np.array(image) 
    
    
    crop_info = {
        'origin_size': image.size,
        'crop_bbox': [
            0,
            0,
            img.shape[1],
            img.shape[0],
        ] 
    }  
    crop_info['size'] = img.shape[:2] 
    face_parsing(img, face_parser, landmarks, save_dir=save_dir)
    
    return img, landmarks, crop_info


def face_parsing(image, face_parser, lmk68, remove_bg=False, pad=0, save_dir=None):
    if remove_bg: 
        rgb, a = image[:, :, :3], image[:, :, 3:] / 255 
        rgb = rgb * a 
        rgb = Image.fromarray(rgb.astype(np.uint8))
    else:
        rgb = Image.fromarray(image[..., :3].astype(np.uint8))
    output = face_parser.run(rgb)   
    # output['parse_image'].save(f'{debug_parse_dir}/{image_name}.png')
    
    eye_mask = output['eye_mask']
    left_eyes = lmk68[36:42, 0].mean(0) 
    right_eyes = lmk68[42:48, 0].mean(0) 
    center = int((left_eyes + right_eyes) * 0.5) 
    leye_mask = eye_mask.copy()
    leye_mask[:, center:] = 0 
    reye_mask = eye_mask.copy()
    reye_mask[:, :center] = 0

    if save_dir is not None: 
        save_image(leye_mask, f'{save_dir}/process/leye_mask.png')
        save_image(reye_mask, f'{save_dir}/process/reye_mask.png')
        save_image(output['parse_image'], f'{save_dir}/process/parse.png')
        save_image(output['face_mask'], f'{save_dir}/process/face_mask.png')
        save_image(output['neck_mask'], f'{save_dir}/process/neck_mask.png')
        save_image(output['hair_mask'], f'{save_dir}/process/hair_mask.png')

    new_rgba = np.concatenate([image[:, :, :3], output['mask'][:, :, None]*255], -1)
    # Image.fromarray(new_rgba.astype(np.uint8)).save(f'test.png')
    
    return output

def bfm2flame(data, iters=1500, debug=False):   
    # gt data 
    global_orient = data['global_orient'].reshape(1, 3).detach()
    bfm_mask = data['bfm_mask']
    normal, mask  = torch.split(data['mvnml'], [3, 1], dim=-1)
    lmk68 = data['lmk68'] 
    
    # parameters 
    neck_pose = nn.Parameter(torch.zeros(1, 3).to(device), requires_grad=True)
    jaw_pose = nn.Parameter(torch.zeros(1, 3).to(device), requires_grad=True) 
    betas = nn.Parameter(torch.zeros(1, 300).to(device), requires_grad=True)
    expression = nn.Parameter(torch.zeros(1, 100).to(device), requires_grad=True)
    txy = nn.Parameter(torch.zeros(1, 2).to(device), requires_grad=True)
    tz = nn.Parameter(torch.zeros(1, 1).to(device), requires_grad=True)
    scale = nn.Parameter(torch.ones(1).to(device) * 5, requires_grad=True)
    
    optimizer = torch.optim.Adam([txy, scale], lr=1e-2)
    
    # camera TODO: change  
    if len(data['mvnml']) == 4:
        mvp = renderer.get_orthogonal_cameras(n=4, yaw_range=(0, -360)).to(device)
    elif len(data['mvnml']) == 6:
        mvp = renderer.get_orthogonal_cameras(n=8, yaw_range=(0, -360))[[0, 2, 4, 6, 1, 7]].to(device) 
    else:
        raise ValueError(f"Unknown camera: {len(data['mvnml'])}")
    
    # flame triangles  
    faces = flame_model.segment.get_triangles(['face', 'left_eyeball', 'right_eyeball'])
    face_tri = torch.tensor(faces.astype(np.int32)).to(device)
    
    for i in track(range(iters), description='bfm2flame optimizing...'): 
        if i == 200:
            txy = txy.detach()
            scale = scale.detach()
            optimizer = torch.optim.Adam([neck_pose, betas, tz], lr=1e-2)
        
        if i == 700:
            neck_pose=neck_pose.detach() 
            # betas = betas.detach()
            optimizer = torch.optim.Adam([expression], lr=1e-3)
        
        # if i == iters - 200:
        #     print('starting optimize tz') 
        #     neck_pose = neck_pose.detach()
        #     expression = expression.detach() 
        #     # betas = betas.detach()
        #     optimizer = torch.optim.Adam([tz, betas], lr=1e-2) 
        
        optimizer.zero_grad()
        out = flame_model(
            betas=betas, 
            expression=expression, 
            global_orient=global_orient,  
            jaw_pose=jaw_pose,  
        )
        transl = torch.cat([txy, tz], dim=1)
        vertices = orthogonal_projection(out.vertices[0], transl, scale)
        joints = orthogonal_projection(out.joints[0, 5:73], transl, scale)
        joints = joints[:, :2]
        joints[:, 1] *= -1
        
        mesh = Mesh(v=vertices, f=face_tri) 
        pkg_face = renderer(mesh, mvp) 
        
        # render for faces
        mesh = Mesh(v=vertices, f=flame_model.faces_tensor.int()) 
        pkg_head = renderer(mesh, mvp) 
        
        # * render_pkg['alpha'] + (0.5 - render_pkg['alpha'])
        # pred_normal = render_pkg['normal']  
        lmk_loss = ((joints - lmk68) ** 2).mean()  
        if i < 200:
            mask_loss = ((pkg_face['alpha'][0] - bfm_mask).abs()).mean() 
            loss = mask_loss + lmk_loss * 10 
        else:    
            if len(normal) == 4: 
                face_nml_loss = (((pkg_face['normal'] - normal) * pkg_face['alpha'])[[0, 1, 3]] ** 2).mean()   
                head_nml_loss = (((pkg_head['normal'] - normal) * pkg_head['alpha'])[[0, 1, 3]] ** 2).mean()   
            else: 
                face_nml_loss = (((pkg_face['normal'] - normal) * pkg_face['alpha'])[[0, 1, 3, 4, 5]] ** 2).mean()   
                head_nml_loss = (((pkg_head['normal'] - normal) * mask)[[0, 1, 3, 4, 5]] ** 2).mean()   
                
            # msk_loss = ((pkg_head['alpha'] - mask)[1:] ** 2).mean()
            reg_loss = (torch.sum(betas ** 2) + torch.sum(expression ** 2)) 
            loss = lmk_loss * 10 + reg_loss * 5e-4 + (face_nml_loss + head_nml_loss) * 5 * (1 + i / iters)   

        loss.backward()
        optimizer.step()

        if debug and (i == iters - 1 or i % 100 == 0): 
            with torch.no_grad(): 
                mesh = Mesh(v=vertices.detach(), f=flame_model.faces_tensor.int())
                mesh.auto_normal()  
                out = renderer(mesh, mvp) 

                vis = [
                    torch.cat(
                        [vis_landmarks(normal[0], lmk68, isScale=True)] + list(normal[1:].detach().cpu()),
                    1), 
                    torch.cat(
                        [vis_landmarks(pkg_head['normal'][0], joints, isScale=True),]+list(pkg_head['normal'][1:].detach().cpu()), 
                    1),
                ]
                vis.append(
                    (vis[0] + vis[1]) * 0.5 
                )  
                nml = (torch.cat(vis, 0) * 255).numpy().astype(np.uint8)
                os.makedirs(f'{debug_flame_dir}/{image_name}', exist_ok=True)
                Image.fromarray(nml).resize((256*len(mvp), 256*len(vis))).save(f'{debug_flame_dir}/{image_name}/iter-{i+1:04d}.png')   
                print(f"saved to {debug_flame_dir}/{image_name}-iter-{i+1:04d}.png")

    return {
        'global_orient': global_orient,
        'jaw_pose': jaw_pose,
        'neck_pose': neck_pose, 
        'betas': betas,
        'expression': expression,
        'transl': transl,
        'scale': scale, 
    }


def flame_estimation(image: Image, lmk68:np.ndarray, method='emoca', mv_normal=None): 
    if method == 'emoca':
        return emoca_flame_tracking(flame_estimator, image.convert('RGB'), lmk68, emoca_dir) 
    elif method == 'deep3dface':
        bfm_out = flame_estimator.run(image.convert('RGB'), lmk68) 
        
        bfm_mask = scale_img_hwc(bfm_out['mask'], (512, 512))  
        bfm_normal = scale_img_hwc(bfm_out['normal'], (512, 512)) 
        
        mv_normal = scale_img_nhwc(mv_normal, (512, 512)) 
        bfm_out.update({
            'bfm_mask': bfm_mask,
            'bfm_normal': bfm_normal, 
            'mvnml': mv_normal.to(device), 
            'lmk68': torch.tensor(normalized_lmk68).float().to(device) * 2 - 1,  
        })  
        flame_params = bfm2flame(bfm_out, debug=opt.debug) 
        dicts.save2pth(flame_params, osp.join(save_dir, 'deep3dface', 'flame.pth'))
    else:
        raise ValueError(f"Unknown flame estimation method: {method}")
 

def lazy_remove(image: torch.Tensor):
    '''
    image: [batch, h, w, 3]
    
    Returns:
    --------
    alpha: [batch, h, w]
    '''
    base = image[:, :1, :1, :] # [batch, 1, 1, 3]
    diffs = (image - base.expand_as(image)).abs().sum(-1)
    alpha = diffs > (20 / 255)
    return alpha.unsqueeze(-1).float()

def rotate_normal(normal: torch.Tensor, view_angles: List[int]):  
    assert len(view_angles) == len(normal), f"view angles {len(view_angles)} != normal {len(normal)}"
    rot = batch_rodrigues(
            torch.tensor([
                [0, np.radians(angle), 0] for angle in view_angles
            ])
        ).float()
    if normal.shape[-1] == 3:
        alpha = lazy_remove(normal)
    else:
        normal, alpha = normal[..., :3], normal[..., 3:]
    normal = (normal * 2 - 1) * alpha  
    normal = normal * alpha
    
    a = np.concatenate(list(normal.numpy()), axis=1)  
    normal = torch.bmm(normal.reshape(len(view_angles), -1, 3), rot).reshape(*normal.shape)
    normal = (normal + 1) / 2
    normal = torch.cat([normal, alpha], -1)
    return normal 

# CUDA_VISIBLE_DEVICES=3 python process.py 
if __name__ == "__main__":
    import sys
    import argparse
    from omegaconf import OmegaConf 

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default='./configs/6views.yaml', help="path to the yaml config file")
    args, extras = parser.parse_known_args()
    opt = OmegaConf.merge(OmegaConf.load(args.config), OmegaConf.from_cli(extras))
    
    if opt.debug:
        os.makedirs('debug', exist_ok=True)
        logger.setLevel(logging.DEBUG)

    device = torch.device(f'cuda:{opt.gpu_id}')
    torch.cuda.set_device(device)
    # ----------------- 
    # Loading Models 
    # ----------------- 
    logger.debug(f"Loading models...")

    # face parser 
    face_parser = FaceParser(device=device, color_map=opt.part_color_map)
    logger.debug(f"Loaded face parser.")

    # face detectors 
    face_detectors = {
        'sfd': load_face_detector('sfd'), 
        'blazeface': None,
        'dlib': None 
    }  
    logger.debug(f"Loaded face detector.")

    # load mv-diffusion model
    from src.model.pipe import MVDiffusionImagePipeline
    logger.info(f"Loading multiview diffusion...")    
    images_pipe = MVDiffusionImagePipeline.from_pretrained(
        opt.image_diffusion.model,  
        trust_remote_code=True,  
        class_labels=opt.image_diffusion.class_labels,
    ).to(device) 
    normal_pipe = MVDiffusionImagePipeline.from_pretrained(
        opt.normal_diffusion.model,  
        trust_remote_code=True,  
        class_labels=opt.normal_diffusion.class_labels,
    ).to(device)
    
    # FLAM Estimation Model 
    if opt.flame_estimation_method == 'deep3dface':
        logger.info(f"Loading deep3dface...")   
        flame_estimator = PoseEstimator(
            device=device,
            checkpoints_dir=opt.deep3dface_ckptdir, 
            bfm_folder=opt.bfm_folder
        )
    elif opt.flame_estimation_method == 'emoca':
        from src.model.emoca import EMOCA, render_emoca_results
        logger.info(f"Loading EMOCA from {opt.emoca_ckptdir}...")    
        flame_estimator = EMOCA(
            device=device, 
            model_path=opt.emoca_ckptdir, 
            model_name=opt.emoca_model_name, 
            mode=opt.emoca_mode,  
        )
        logger.debug(f"Finihsed loading models!") 
    elif opt.flame_estimation_method == 'vhap':
        raise NotImplementedError("VHAP is not supported yet!") 
    else:
        raise ValueError(
            f"Unknown flame estimation method: {opt.flame_estimation_method}, supported methods: deep3dface, emoca")
    
    # find images TODO: add style 
    if os.path.isdir(opt.image):
        style = opt.image.split('/')[-1]
        image_files = find_files_from_dir(opt.image, exts=('.png', '.jpg', '.jpeg', '.webp'))
    elif opt.image.endswith('.txt'):
        style = opt.image.split('/')[-1].split('.')[0]
        image_files = np.loadtxt(opt.image, dtype=str) 
        image_files = [f"./data/examples/{style}/{f}" for f in image_files] 
    elif opt.image.endswith('.png') or opt.image.endswith('.jpg') or opt.image.endswith('.jpeg') or opt.image.endswith('.webp'):
        style = opt.image.split('/')[-2] 
        image_files = [opt.image]

    opt.save_dir = f"{opt.save_dir}/{style}"

    # prepare dir 
    if opt.debug:
        opt.debug_dir = os.path.join(opt.save_dir, '../debug')
        debug_lmk_dir = osp.join(opt.debug_dir, 'landmarks')  
        debug_eyeseg_dir = osp.join(opt.debug_dir, 'eyeseg') 
        debug_mvimg_dir = osp.join(opt.debug_dir, 'multi-view') 
        debug_mvnormal_dir = osp.join(opt.debug_dir, 'normal') 
        debug_flame_dir = osp.join(opt.debug_dir, opt.flame_estimation_method)
        debug_parse_dir = osp.join(opt.debug_dir, 'parse')

        os.makedirs(debug_lmk_dir, exist_ok=True)
        os.makedirs(debug_mvimg_dir, exist_ok=True) 
        os.makedirs(debug_flame_dir, exist_ok=True) 
        os.makedirs(debug_eyeseg_dir, exist_ok=True)
        os.makedirs(debug_mvnormal_dir, exist_ok=True)
        os.makedirs(debug_parse_dir, exist_ok=True)

    # process  
    for image_file in image_files: 
        logger.debug('Processing image: %s' % image_file)

        image_name = osp.basename(image_file).split("/")[-1].split(".")[0]
        save_dir = osp.join(opt.save_dir, image_name, f'{len(opt.views)}-views')  
        os.makedirs(osp.join(save_dir, opt.flame_estimation_method), exist_ok=True) 
        os.makedirs(osp.join(save_dir, 'process'), exist_ok=True)
        os.makedirs(f"{save_dir}/images", exist_ok=True) 
        os.makedirs(f"{save_dir}/normals", exist_ok=True)
        os.makedirs(f"{save_dir}/parse", exist_ok=True)
        print('saving to ', save_dir)
        
        # --------------------- 
        # Image Processing 
        # ---------------------  
        if not os.path.exists(osp.join(save_dir, 'process', 'lmk68.txt')) or not os.path.exists(osp.join(save_dir, 'process', 'input.png')):
            os.system(f"cp {image_file} {save_dir}/process/origin.png")
            print('processing image...')
            try: 
                rgba, lmk68, params = process_image(
                    image_file, 
                    shoulder_size=opt.shoulder_size,  
                    pad=0.1
                )    
            except Exception as e:
                logger.error(f"Error in processing image: {image_file}, {e}")
                continue
            # save input 
            Image.fromarray(rgba.astype(np.uint8)).save(f'{save_dir}/process/input.png')
            np.save(osp.join(save_dir, 'process', 'crop_info.npy'), params)

            # save landmarks 
            normalized_lmk68 = lmk68 / np.array([rgba.shape[1], rgba.shape[0]])
            np.savetxt(f'{save_dir}/process/lmk68.txt', normalized_lmk68, fmt='%.4f')
            
            # if opt.debug: 
                # visualize landmarks
            vis = rgba[..., :3] / 255  
            seg = vis * rgba[..., 3:] / 255 + 1 - rgba[..., 3:] / 255
            vis = vis_landmarks(vis, normalized_lmk68 * 2 - 1)   
            vis = np.concatenate([vis, rgba[..., 3:]/255], -1)
            save_image(vis, f"{save_dir}/process/vis-lmk68.png")
            # if opt.debug: 
            #     vis = np.concatenate([vis, seg], 1) 
            #     save_image(vis, f"{debug_lmk_dir}/{image_name}.png")  
        else: 
            rgba = np.array(Image.open(osp.join(save_dir, 'process', 'input.png'))) 
            normalized_lmk68 = np.loadtxt(osp.join(save_dir, 'process', 'lmk68.txt'))
            lmk68 = normalized_lmk68 * np.array([rgba.shape[1], rgba.shape[0]]) 
        
        # continue  
        pil_img = change_rgba_bg(Image.fromarray(rgba.astype(np.uint8)), "WHITE")
        
        # -----------------------------------
        # MultiView RGB & Normal Prediction  
        # -----------------------------------  
        if len(os.listdir(osp.join(save_dir, f'images'))) < len(opt.views):            
            mv_image = images_pipe(
                [pil_img] * len(opt.views), **opt.image_diffusion.forward_args, 
                generator = torch.Generator(device='cuda').manual_seed(opt.seed)
            ).images  
            
            mv_image = [
                Image.fromarray(kiui.sr.sr(np.array(img), 4, device)) for img in mv_image
            ] 
            
            mv_normal = normal_pipe(
                mv_image, **opt.normal_diffusion.forward_args, 
                generator = torch.Generator(device='cuda').manual_seed(opt.seed)
            ).images   
            mv_normal = [
                Image.fromarray(kiui.sr.sr(np.array(img), 4, device)) for img in mv_normal
            ]
            
            for i, (img, nml) in enumerate(zip(mv_image, mv_normal)):
                
                img.save(f"{save_dir}/images/{i}.png")
                nml.save(f"{save_dir}/normals/{i}.png")
                
                # face-parser 
                if i in [4, 5]: 
                    output = face_parser.run(img)
                    if opt.debug:   
                        output['parse_image'].save(f'{debug_parse_dir}/{image_name}.png')
                    save_image(output['parse_image'], f'{save_dir}/parse/vis_{i}.png')
                    save_image(output['face_mask'], f'{save_dir}/parse/face_mask_{i}.png')
                    save_image(output['neck_mask'], f'{save_dir}/parse/neck_mask_{i}.png')
                    save_image(output['hair_mask'], f'{save_dir}/parse/hair_mask_{i}.png') 
                    save_image(output['eye_mask'], f'{save_dir}/parse/eye_mask_{i}.png')
            
            if opt.debug:  
                mvimg = np.hstack([np.array(img.resize((256, 256))) for img in mv_image])
                mvnml = np.hstack([np.array(img.resize((256, 256))) for img in mv_normal])
                mvimg = np.vstack([mvimg, mvnml])[..., :3] 
                Image.fromarray(mvimg).save(f"{debug_mvimg_dir}/{image_name}.png") 
                # Image.fromarray(mvimg).save(f"out.png") 

            # rotate normal 
            mv_normal = rotate_normal(
                torch.stack([torch.tensor(np.array(img)) for img in mv_normal]).float() / 255, 
                opt.views 
                )  
                
            # --------------------------------------- 
            # Optimize Flame using Multi-View Images 
            # ---------------------------------------  
            flame_estimation(pil_img, lmk68, opt.flame_estimation_method, mv_normal)