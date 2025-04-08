import os
from PIL import Image
import numpy as np
import torch 
from scipy.io import loadmat, savemat

from deep3dface.options.test_options import TestOptions
from deep3dface.data import create_dataset
from deep3dface.models import create_model
from deep3dface.util.visualizer import MyVisualizer
from deep3dface.util.preprocess import align_img 
from deep3dface.util.load_mats import load_lm3d 
from deep3dface.data.flist_dataset import default_flist_reader


class PoseEstimator:
    def __init__(self, device, checkpoints_dir, bfm_folder):
        # exit()
        opt = TestOptions() 
        opt.bfm_folder = bfm_folder
        opt.checkpoints_dir = checkpoints_dir
        
        self.device = device 
        self.model = create_model(opt)
        self.model.setup(opt)
        self.model.device = device
        self.model.parallelize()
        self.model.eval()

        self.lm3d_std = load_lm3d(opt.bfm_folder)
        

    def prepare_data(self, im, lm):
        W, H = im.size
        lm = lm.reshape([-1, 2])
        lm[:, -1] = H - 1 - lm[:, -1]
        crop_params, im, lm, _ = align_img(im, lm, self.lm3d_std)
        # print(lm.shape)
        # exit()

        im = torch.tensor(np.array(im)/255., dtype=torch.float32).permute(2, 0, 1).to(self.device).unsqueeze(0)
        lm = torch.tensor(lm).to(self.device).unsqueeze(0)

        return im, lm, crop_params
 
    def parse_camera(self):
        # Extrinsic
        trans = pred_coeffs_dict['trans'][0]
        trans[2] += -10
        c = -np.dot(R, trans)
        c *= 0.27  # normalize camera radius
        c[1] += 0.006  # additional offset used in submission
        c[2] += 0.161  # additional offset used in submission
        radius = np.linalg.norm(c)
        c = c / radius * 2.7

        Rot = np.eye(3)
        Rot[0, 0] = 1
        Rot[1, 1] = -1
        Rot[2, 2] = -1
        R = np.dot(R, Rot)

        pose = np.eye(4)
        pose[0, 3] = c[0]
        pose[1, 3] = c[1]
        pose[2, 3] = c[2]
        pose[:3, :3] = R

        # Intrinsics
        focal = 2985.29 / CENTER_CROP_SIZE
        cx, cy = 0.5, 0.5
        K = np.eye(3)
        K[0][0] = focal
        K[1][1] = focal
        K[0][2] = cx
        K[1][2] = cy 
        return K, pose
    
    
    def unwarp(self, crop_params, image, lmk=None): 
        pil_img = image[0].permute(1, 2, 0).cpu().numpy()
        pil_img = Image.fromarray((pil_img * 255).astype(np.uint8))
        new_img = Image.new('RGB', crop_params['size'], (0, 0, 0))
        new_img.paste(pil_img, (crop_params['bbox'][0], crop_params['bbox'][1]))       
        new_img = new_img.resize(crop_params['origin_size'])
        new_img = np.array(new_img) / 255.
        
        return new_img
    
    @torch.no_grad()
    def run(self, image, landmarks):
        print(image.size)
        img_proc, lm_proc, crop_params = self.prepare_data(image, landmarks)
        data = {
            'imgs': img_proc,
            'lms': lm_proc,
        }
        

        self.model.set_input(data)
        self.model.test()
        coef_dict = self.model.pred_coeffs_dict
        visuals = self.model.get_current_visuals()['output_vis']   
        # self.model.save_mesh(os.path.join('vis-debug', 'ting.obj')) # save reconstruction meshes
        # self.model.save_coeff(os.path.join('vis-debug', 'ting.mat')) # save predicted coefficients

        normal = self.unwarp(crop_params, self.model.pred_face)
        mask = normal.sum(-1) > 0
        normal = torch.tensor(normal).float().to(self.device) 
        mask = torch.tensor(mask).float().to(self.device).unsqueeze(-1)
        
        return {
            'global_orient': coef_dict['angle'][0], 
            'vis': visuals[0].permute(1, 2, 0), 
            'mask': mask,   
            'normal': normal, 
        }
