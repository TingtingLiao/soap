import os 
import cv2
import smplx
import torch  
import trimesh 
import numpy as np 
from PIL import Image
from termcolor import colored 

import coloredlogs, logging 
logger = logging.getLogger(__name__) 
coloredlogs.install(level='DEBUG', logger=logger) 


def orthogonal_projection(vertices, transl, scale):
    """
    Orthogonal projection of vertices
        Args:
        -----
            vertices: torch.Tensor [N, 3]
            transl: torch.Tensor [1, 2] or [1, 3]
            scale: float number
            
        Returns:
        -------
            v_proj: [N, 3]
    """  
    # if transl.shape[-1] == 2:
    #     transl = torch.cat([transl, torch.zeros(1, 1).to(transl)], dim=1)
    v_proj = (vertices + transl) * scale  
    return v_proj


def perspective(vertices, mvp):
    """
    perspective projection of vertices
        Args:
        -----
            vertices: torch.Tensor [N, 3]
            mvp: torch.Tensor [B, 4, 4]
            
        Returns:
        -------
            v_proj: [B, N, 3]
    """
    
    N = vertices.shape[0]
    ones = torch.ones((N, 1), dtype=vertices.dtype, device=vertices.device)
    vertices_homogeneous = torch.cat([vertices, ones], dim=-1)  # [N, 4]
    
    # Perform the perspective projection for each batch
    v_proj_homogeneous = torch.matmul(mvp, vertices_homogeneous.T)  # [B, 4, N]
    
    # Convert from homogeneous to 3D coordinates by dividing by w (perspective divide)
    v_proj_homogeneous = v_proj_homogeneous.permute(0, 2, 1)  # [B, N, 4]
    v_proj = v_proj_homogeneous[..., :3] / v_proj_homogeneous[..., 3:4]  # [B, N, 3]

    return v_proj
    
    

def l2_distance(verts1, verts2):
    """
    Args:
    -----
        verts1: [N, 3]
        verts2: [N, 3]
    Returns:
    -------
        distance: float
    """
    return torch.sqrt(((verts1 - verts2) ** 2).sum(1)).mean()


def pts_to_bbox(points):
    """ Convert points to bounding box 
        Args:
        -----
        points: np.ndarray or torch.Tensor [N, 3]

        Returns:
        -------
        bbox: [min_x, min_y, max_x, max_y]
    """   
    if isinstance(points, list):
        points = np.array(points)
    if isinstance(points, torch.Tensor):
        points = points.cpu().numpy() 

    assert len(points.shape) == 2, logger.error(f"points should be [N, 3], input shape:{points.shape}")
    min_y, min_x = points.min(0)[:2]
    max_y, max_x = points.max(0)[:2]

    return [min_x, min_y, max_x, max_y]


def create_flame_model(model_path="./data/flame/flame2020.pkl", upsample=False, create_segms=True, add_teeth=False, device='cuda'): 
    model = smplx.create(
        model_path=model_path, 
        model_type='flame',   
        batch_size=1,
        upsample=upsample, 
        create_segms=create_segms, 
        add_teeth=add_teeth
    ).to(device)
    return model 


def find_files_from_dir(dir_path, exts=[], depth=-1):
    """ Find files endswith exts under a directory 
        Args:
        -----
            dir_path: str, path to the directory
            exts: list, list of image extensions
        Returns:
        --------
            files: list, list of image files
    """ 
    files = [] 
    for root, _, fnames in os.walk(dir_path):
        for fname in fnames:
            if len(exts) == 0:
                files.append(os.path.join(root, fname))
            elif fname.lower().endswith(tuple(exts)):
                files.append(os.path.join(root, fname))
    return files


def parse_lmk5_from_lmk68(lmk68, use_lip=True):
    """ Parse [left_eye, right_eye, nose, mouth_left, mouth_right] from 68 landmarks
        Args:
        -----
            lmk68: np.ndarray [68, 2]
            use_lip: bool, use lip or not
        Returns:
        --------
            lmk5: np.ndarray [5, 2]
    """
    left_eye = lmk68[36:42].mean(axis=0)
    right_eye = lmk68[42:48].mean(axis=0)
    nose = lmk68[30]
    mouth_left = lmk68[48]
    mouth_right = lmk68[54]
    
    lmk5 = np.array([left_eye, right_eye, nose, mouth_left, mouth_right])
    return lmk5


def save_image(image, save_path, scale=1.0, size=None): 
    if isinstance(image, Image.Image):
        image.save(save_path)
        return 
    
    assert len(image.shape) in [2, 3], logger.error(f"image should be [H, W] or [C, H, W], input shape:{image.shape}")
    
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()

    if isinstance(image, np.ndarray): 
        if len(image.shape) == 3:  
            if image.shape[0] in [1, 3, 4]:
                image = image.transpose(1, 2, 0)
                
            if image.shape[2] == 1:
                image = np.squeeze(image, axis=2)

        if image.max() <= 1:
            image = (image * 255) 
        
        image = Image.fromarray(image.astype(np.uint8)) 
    
    if size is not None:
        image = image.resize(size)
        
    elif scale != 1.0:
        w, h = image.size
        w, h = int(w * scale), int(h * scale)
        image = image.resize((w, h))
        
    image.save(save_path) 

def ids2mask(vids, N, reverse=False, device='cpu'):
    mask = torch.zeros(N).to(device)
    mask[vids] = 1 
    
    if reverse:
        mask = 1 - mask
        
    return mask.bool()

def mask2ids(mask):
    return torch.nonzero(mask).flatten()

def vmask2fmask(vmask, faces):
    """ Convert vertex mask to face mask
        Args:
        -----
            vmask: torch.Tensor [N]
            faces: torch.Tensor [M, 3]
        Returns:
        --------
            fmask: torch.Tensor [M]
    """ 
    return vmask[faces].any(1)

def vid2fmask(vid, faces):
    """ Convert vertex id to face id
        Args:
        -----
            vid: int, vertex id
            faces: torch.Tensor [M, 3]
        Returns:
        --------
            fmask: torch.Tensor [M]
    """ 
    vmask = ids2mask([vid], faces.max() + 1, device=faces.device)
    fmask = vmask2fmask(vmask, faces)
    return fmask
    