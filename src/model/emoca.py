import os 
import torch 
import numpy as np 
from torch import nn
from pathlib import Path
from kiui.mesh import Mesh
from smplx.lbs import batch_rodrigues
from smplx.segment import FlameSeg
from PIL import Image
from skimage.transform import rescale, estimate_transform, warp
from gdl_apps.EMOCA.utils.load import load_model
from gdl_apps.EMOCA.utils.io import save_obj, save_images, save_codes
from gdl.datasets.ImageDatasetHelpers import bbox2point
import coloredlogs, logging

from lib.utils.helper import pts_to_bbox 

logger = logging.getLogger(__name__) 
coloredlogs.install(level='DEBUG', logger=logger)

def point2bbox(center, size):
    size2 = size / 2 
    src_pts = np.array([
        [center[0] - size2, center[1] - size2], 
        [center[0] - size2, center[1] + size2],
        [center[0] + size2, center[1] - size2]
    ])
    return src_pts

def point2transform(center, size, target_size_height, target_size_width):
    target_size_width = target_size_width or target_size_height
    src_pts = point2bbox(center, size)
    dst_pts = np.array([[0, 0], [0, target_size_width - 1], [target_size_height - 1, 0]])
    tform = estimate_transform('similarity', src_pts, dst_pts)
    return tform

def bbpoint_warp(image, center, size, target_size_height, target_size_width=None, output_shape=None, inv=True, landmarks=None, 
        order=3 # order of interpolation, bicubic by default
        ):
    target_size_width = target_size_width or target_size_height
    tform = point2transform(center, size, target_size_height, target_size_width)
    tf = tform.inverse if inv else tform
    output_shape = output_shape or (target_size_height, target_size_width)
    dst_image = warp(image, tf, output_shape=output_shape, order=order)
    if landmarks is None:
        return dst_image
    # points need the matrix
    if isinstance(landmarks, np.ndarray):
        assert isinstance(landmarks, np.ndarray) 
        tf_lmk = tform if inv else tform.inverse
        dst_landmarks = tf_lmk(landmarks[:, :2])
    elif isinstance(landmarks, list): 
        tf_lmk = tform if inv else tform.inverse
        dst_landmarks = [] 
        for i in range(len(landmarks)):
            dst_landmarks += [tf_lmk(landmarks[i][:, :2])]
    elif isinstance(landmarks, dict): 
        tf_lmk = tform if inv else tform.inverse
        dst_landmarks = {}
        for key, value in landmarks.items():
            dst_landmarks[key] = tf_lmk(landmarks[key][:, :2])
    else: 
        raise ValueError("landmarks must be np.ndarray, list or dict")
    return dst_image, dst_landmarks


def render_emoca_results(vals, flame_model, renderer, pil_image):
    '''
    render the predicted flame from EMOCA onto the input image 
        Args:
        -----
        vals: dict
            model outputs
        flame_model: FLAME 
            flame parametric model
        renderer: Renderer
            renderer object
        origin_image: np.array h x w x 3
            input image
        
        Returns:
        --------
        np.array: rendered image h x w x 3
    '''
    camera = vals['cam']
    size = vals['size'] 
    center = vals['center'] 

    out = flame_model(
        betas=vals['shapecode'], 
        expression=vals['expcode'], 
        global_orient=vals['posecode'][:, 0:3],  
        jaw_pose=vals['posecode'][:, 3:6]
    )  
    vertices = out.vertices[0] 
    vertices[:, :2] += camera[:, 1:]
    vertices *= camera[0, 0] 
    # zmax, zmin = vertices[:, 2].max(), vertices[:, 2].min() 
    # vertices[:, 2] = (vertices[:, 2] - zmin) / (zmax - zmin) * 2 - 1 
    faces = flame_model.segment.get_triangles('face')
    faces = torch.tensor(faces.astype(np.int32)).cuda() 
    mesh = Mesh(vertices, faces)
    mesh.auto_normal()
 
    mvp = torch.eye(4)[None].cuda()
    flipY = batch_rodrigues(torch.tensor([np.pi, 0, 0]).reshape(1, 3)).reshape(3, 3) 
    mvp[:, :3, :3] = flipY
    
    render_pkg = renderer(mesh, mvp, h=224, w=224, bg_color=torch.zeros(3).cuda()) 
    nml = render_pkg['normal'][0].detach().cpu().numpy() 
    mask = render_pkg['alpha'][0].detach().cpu().squeeze().numpy() 
  
    h, w = pil_image.size
    warped_img = bbpoint_warp(nml, center, size, 224, output_shape=(w, h), inv=False)
    warped_mask = bbpoint_warp(mask, center, size, 224, output_shape=(w, h), inv=False)
    
    vis_pil = Image.fromarray((warped_img * 255).astype(np.uint8))
    mask_pil = Image.fromarray((warped_mask * 255).astype(np.uint8))
    mask_pil_transparent = Image.fromarray((warped_mask * 180).astype(np.uint8)) 
    outpil = pil_image.copy()
    outpil.paste(vis_pil, (0, 0), mask_pil)

    merge_pil = pil_image.copy()
    merge_pil.paste(vis_pil, (0, 0), mask_pil_transparent)

    normal_rgba = np.concatenate([warped_img, warped_mask[..., None]], axis=-1) * 255 
    return np.array(merge_pil)[..., ::-1], normal_rgba.astype(np.uint8)
      


class EMOCA(nn.Module):
    def __init__(
        self, 
        device, model_path, 
        model_name='EMOCA_v2_lr_mse_20', 
        mode='detail', 
        scale=1.25,
        res_input=224, 
        detector_type='mediapipe'
    ): 
        super(EMOCA, self).__init__()
        self.res_input = res_input  
        self.scale = scale
        self.device = device
        self.model, _ = load_model(model_path, model_name, mode)
        self.model.to(device)
        self.model.eval() 
        self.detector_type = detector_type
         
    def warp_image(self, pil_image, landmarks=None):
        """
        Warps an input image based on detected face bounding box.

        Args:
            pil_image (PIL.Image): Input image.
            landmarks (np.array): Detected landmarks. Nx2 array.

        Returns:
            np.array: Warped image.
            dict: Warp information including size, center, and crop flag.
        """
        np_image = np.asarray(pil_image)
        h, w = np_image.shape[:2]
        
        if landmarks is not None: 
            # left, top, right, bottom = pts_to_bbox(landmarks)
            top, left, bottom, right = pts_to_bbox(landmarks)
            crop = True 
        else: 
            left, top, right, bottom = 0, 0, w-1, h-1 
            crop = False 
 
        old_size, center = bbox2point(left, right, top, bottom, type=self.detector_type)
        size = int(old_size * self.scale)

        image = np_image / 255.0 
        dst_image = bbpoint_warp(image, center, size, self.res_input)
 
        warp_dict = {'size': size, 'center': center, 'crop': crop}
        return dst_image, warp_dict

    def forward(self, pil_image, landmarks=None, save_dir=None): 
        '''
        forward pass of the model
            Args:
            -----
            pil_image: PIL.Image
                input image
            save_dir: str, optional 
                path to save the results
            
            Returns:
            --------
            dict: model outputs
            dict: visualization dictionary
        '''
        dst_image, warp_dict = self.warp_image(pil_image, landmarks) 
        data = {
            'image': torch.tensor(dst_image.transpose(2, 0, 1)).float().to(self.device).unsqueeze(0)
        }

        # encode 
        vals = self.model.encode(data, training=False)

        # decode 
        with torch.no_grad():
            values = self.model.decode(vals, training=False)  
            uv_detail_normals = values['uv_detail_normals'] if 'uv_detail_normals' in values else None
            visdict, grid_image = self.model._visualization_checkpoint(
                values['verts'], values['trans_verts'], values['ops'], uv_detail_normals, values, 0, "", "", save=False
            )
        
        # save results
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            save_obj(self.model, os.path.join(save_dir, "mesh_coarse.obj"), vals, 0)
            save_images(save_dir, '', visdict, with_detection=True, i=0)
            save_codes(Path(save_dir), '', vals, i=0) 
            np.save(os.path.join(save_dir, 'warp.npy'), warp_dict)
        
        vals.update(warp_dict)
        return vals, visdict