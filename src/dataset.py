import os 
import cv2
import torch
import numpy as np
import os.path as osp 
from PIL import Image, ImageFilter
import torchvision.transforms as transforms 
from src.utils.crop import pad_image_square 
from smplx.lbs import batch_rodrigues   
import coloredlogs, logging 
logger = logging.getLogger(__name__) 
coloredlogs.install(level='DEBUG', logger=logger) 

def lazy_remove(image: torch.Tensor):
    '''
    image: [batch, h, w, 3]
    
    Returns:
    --------
    alpha: [batch, h, w]
    '''
    base = image[:, :1, :1, :] # [batch, 1, 1, 3]
    diffs = (image - base.expand_as(image)).abs().sum(-1)
    alpha = diffs > 20 / 255
    return alpha.unsqueeze(-1).float()
    
class Dataset(torch.utils.data.Dataset):
    def __init__(self, root, opt):
        self.root = root
        self.subject_name = osp.basename(opt.image).split(".")[0]
        self.transform = transforms.Compose([
            transforms.Resize(opt.img_res),
            transforms.ToTensor()
        ])   
        self.data_type = opt.flame_estimation_method
        self.near = opt.near
        self.far = opt.far
        self.opt = opt 
        
        self.rot = batch_rodrigues(
            torch.tensor([
                [0, np.radians(angle), 0] for angle in [0, 270, 180, 90]
            ])
        ).float()

    def load_landmarks(self, landmark_file):
        landmarks = torch.tensor(np.loadtxt(landmark_file)).float()
        return landmarks * 2 - 1 

    def get_eye_points(self):
        eye_mask = cv2.imread(osp.join(self.root, 'eye_mask.png'), cv2.IMREAD_GRAYSCALE)
        contours, _ = cv2.findContours(eye_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) 
        left_pts = np.array([point[0] for point in contours[0]])
        right_pts = np.array([point[0] for point in contours[1]])

        w, h = eye_mask.shape
        left_pts = left_pts / [w, h] * 2 - 1
        right_pts = right_pts / [w, h] * 2 - 1

        if left_pts.mean(0)[0] > right_pts.mean(0)[0]:
            left_pts, right_pts = right_pts, left_pts
    
        return {
            'left_eye': torch.tensor(left_pts).float(),
            'right_eye': torch.tensor(right_pts).float()
        }

    def rotate_normal(self, normal, alpha):  
        normal = (normal * 2 - 1) * alpha   
        
        a = np.concatenate(list(normal.numpy()), axis=1)  
        normal = torch.bmm(normal.reshape(4, -1, 3), self.rot).reshape(*normal.shape)
        normal = (normal + 1) / 2
        normal = normal * alpha + 1 - alpha # set bg as white  
        return normal 
    
    def load_images(self, data_dir):  
        # load origin image
        origin_pil = Image.open(self.opt.image).convert('RGB')
        origin_pil.thumbnail((2048, 2048), Image.Resampling.LANCZOS)
        # load input 
        input_image = Image.open(osp.join(data_dir, 'process', 'input.png')) 
        
        # load crop-param 
        if os.path.exists(osp.join(data_dir, 'process', 'crop_info.npy')):
            crop_param = np.load(osp.join(data_dir, 'process', 'crop_info.npy'), allow_pickle=True).item()
        else:
            crop_param = None
        
        mv_images, mv_normals = [], []  
        for i in range(len(self.opt.views)):  
            images = Image.open(osp.join(data_dir, 'images', f'{i}.png')).convert('RGB')  
            normal = Image.open(osp.join(data_dir, f'normals', f'{i}.png')).convert('RGB')  
            mv_normals.append(self.transform(normal)) 
            mv_images.append(self.transform(images))
            
        mv_normals = torch.stack(mv_normals).permute(0, 2, 3, 1) 
        mv_images = torch.stack(mv_images).permute(0, 2, 3, 1)   
        mv_masks = lazy_remove(mv_normals) 

        # replace the front mask to the input mask 
        # front_mask = Image.open(osp.join(data_dir, 'process', 'input.png')).split()[-1]
        # front_mask = front_mask.resize(mv_masks.shape[1:3]) 
        # mv_masks[0] = torch.tensor(np.array(front_mask) > 0).float().unsqueeze(-1)   
        
        mv_images = torch.cat([mv_images, mv_masks], -1)
        # mv_normals = mv_normals * mv_masks + 1 - mv_masks
        # Image.fromarray((torch.cat(list(mv_masks), 1).squeeze().numpy() * 255).astype(np.uint8)).save('test.png')
        
        # vis = torch.cat(list(mv_masks), 1).squeeze(-1).numpy()         
        return_dict = { 
            'input_pil': input_image,
            'origin_pil': origin_pil, 
            'image': mv_images,
            'normal': mv_normals,
            'mask': mv_masks, 
            'crop_info': crop_param
        }
        
        if osp.exists(osp.join(data_dir, 'face_mask.png')):
            face_mask = Image.open(osp.join(self.root, 'face_mask.png')).convert('L')
            face_mask = pad_image_square(face_mask, 0)  
            face_mask = self.transform(face_mask)
            face_mask = face_mask.permute(1, 2, 0).unsqueeze(0) 
            return_dict['face_mask'] = face_mask
        
        def load_eye_mask(fp): 
            eye_mask = Image.open(fp).convert('L') 
            bbox = eye_mask.getbbox() 
            return eye_mask, bbox
        
        # parse_image = torch.ones_like(mv_images[0, :, :, :3]) * torch.tensor(self.opt.bg_color).reshape(1, 1, 3)
        parse_image = torch.zeros_like(mv_images[0, :, :, :3])  
        parse_image_back = torch.zeros_like(mv_images[0, :, :, :3])
        for part, pc in self.opt.part_color_map.items(): 
            if osp.exists(osp.join(self.root, 'process', f'{part}_mask.png')):
                part_mask = Image.open(osp.join(self.root, 'process', f'{part}_mask.png')).convert('L')
                part_mask = self.transform(part_mask).permute(1, 2, 0) 
                return_dict[f'{part}_mask'] = part_mask.unsqueeze(0)  
                parse_image[part_mask.squeeze().bool()] = torch.tensor(pc).float().reshape(1, 1, 3)
                
                if part == 'face':
                    bbox = Image.fromarray((part_mask.squeeze(-1).numpy() * 255).astype(np.uint8)).getbbox()
                    dh = int(bbox[3] - (bbox[3] - bbox[1]) / 4.)  
                    bach_hair_mask = part_mask.clone()
                    bach_hair_mask[dh:, :] = 0
                    parse_image_back += bach_hair_mask * torch.tensor(self.opt.part_color_map.hair).reshape(1, 1, 3)
                    parse_image_back += (part_mask - bach_hair_mask) * torch.tensor(self.opt.part_color_map.neck).reshape(1, 1, 3)
                    # parse_image_back[bach_hair_mask.squeeze().bool()] = torch.FloatTensor(self.opt.part_color_map.hair).reshape(1, 1, 3)
                    # parse_image_back[(part_mask - bach_hair_mask).squeeze().bool()] = torch.FloatTensor(self.opt.part_color_map.neck).reshape(1, 1, 3)
                else:
                    parse_image_back += part_mask * torch.tensor(pc).reshape(1, 1, 3)
        
        bg_color = torch.tensor(self.opt.bg_color).reshape(1, 1, 3)
        parse_image = parse_image * mv_masks[0] + (1 - mv_masks[0]) * bg_color
        
        parse_image_back = parse_image_back * mv_masks[0] + (1 - mv_masks[0]) * bg_color
        parse_image_back = torch.flip(parse_image_back, [1])
        
        parse_image_dict = {
            'front': parse_image.unsqueeze(0),
            'back': parse_image_back.unsqueeze(0)
        }
        
        if len(mv_images) == 6:
            parse_image_left_side = Image.open(osp.join(data_dir, 'parse', 'vis_4.png')).convert('RGB')
            parse_image_left_side = self.transform(parse_image_left_side).permute(1, 2, 0) 
            # mask = (parse_image_left_side.sum(-1) > 0).float().unsqueeze(-1)
            # parse_image_left_side = parse_image_left_side * mask + (1 - mask) * bg_color
            mask = np.array(Image.open(osp.join(data_dir, 'parse', 'eye_mask_4.png'))) > 0 
            mask |= np.array(Image.open(osp.join(data_dir, 'parse', 'hair_mask_4.png'))) > 0
            mask |= np.array(Image.open(osp.join(data_dir, 'parse', 'face_mask_4.png'))) > 0
            mask = torch.tensor(mask).float().unsqueeze(-1)
            
            parse_image_left_side = parse_image_left_side * mask + (1 - mask) * bg_color
            
            parse_image_left_side = torch.cat([parse_image_left_side, mask], -1)
            parse_image_dict[f'left_side'] = parse_image_left_side.unsqueeze(0)  
            
            parse_image_right_side = Image.open(osp.join(data_dir, 'parse', 'vis_5.png')).convert('RGB')
            parse_image_right_side = self.transform(parse_image_right_side).permute(1, 2, 0) 
            # mask = (parse_image_right_side.sum(-1) > 0).float().unsqueeze(-1)
            # parse_image_right_side = parse_image_right_side * mask + (1 - mask) * bg_color 
            mask = np.array(Image.open(osp.join(data_dir, 'parse', 'eye_mask_5.png')))  > 0
            mask |= np.array(Image.open(osp.join(data_dir, 'parse', 'hair_mask_5.png'))) > 0
            mask |= np.array(Image.open(osp.join(data_dir, 'parse', 'face_mask_5.png'))) > 0
            mask = torch.tensor(mask).float().unsqueeze(-1)
            parse_image_right_side = parse_image_right_side * mask + (1 - mask) * bg_color 
            parse_image_right_side = torch.cat([parse_image_right_side, mask], -1)
            parse_image_dict[f'right_side'] = parse_image_right_side.unsqueeze(0)  
        
        return_dict['parse_images'] = parse_image_dict
        
        if osp.exists(osp.join(self.root, 'process', 'leye_mask.png')):
            # try:
            leye_mask, leye_bbox = load_eye_mask(osp.join(self.root, 'process', 'leye_mask.png'))
            reye_mask, reye_bbox = load_eye_mask(osp.join(self.root, 'process', 'reye_mask.png'))
            eye_mask = Image.fromarray(np.array(leye_mask) + np.array(reye_mask))
            eye_mask = self.transform(eye_mask)   # [1, h, w]
            leye_width = (leye_bbox[2] - leye_bbox[0]) / (leye_mask.size[0] * 0.5)
            reye_width = (reye_bbox[2] - reye_bbox[0]) / (reye_mask.size[0] * 0.5)
            
            parse_image[eye_mask[0].bool()] = torch.tensor(self.opt.part_color_map.eye).to(parse_image)
            parse_image_dict['front'] = parse_image.unsqueeze(0)
            # Image.fromarray((parse_image.numpy() * 255).astype(np.uint8)).save('test.png')
            
            return_dict.update({
                'eye_mask': eye_mask.permute(1, 2, 0).unsqueeze(0), 
                'leye_width': leye_width,  
                'reye_width': reye_width, 
                'learnable_eyes': True, 
                'parse_images': parse_image_dict 
            })
            # except Exception as e:
            #     print(e)
            #     return_dict['learnable_eyes'] = False 
        
        return return_dict
    
    def load_flame(self, data_dir):    
        if self.data_type == 'emoca':
            if osp.exists(osp.join(data_dir, 'flame_params.pth')):
            # if False:
                data = torch.load(osp.join(data_dir, 'flame_params.pth'))
                data['reoptim'] = False
                return data 
            else:
                exp = np.load(osp.join(data_dir, 'exp.npy'))
                pose = np.load(osp.join(data_dir, 'pose.npy'))  
                betas = np.load(osp.join(data_dir, 'shape.npy'))
                
                try:
                    camera = np.load(osp.join(data_dir, 'camera.npy'), allow_pickle=True).item()
                    transl = torch.tensor(camera['transl']).reshape(1, -1).float() 
                    scale = camera['scale']
                except:
                    camera = np.load(osp.join(data_dir, 'camera.npy'), allow_pickle=True)
                    transl = torch.tensor(camera[:2]).reshape(1, -1).float()
                    scale = camera[2]
                
                betas = torch.tensor(betas).reshape(1, -1).float()
                expression = torch.tensor(exp).reshape(1, -1).float()
                global_orient = torch.tensor(pose[:3]).reshape(1, -1).float() 
                jaw_pose = torch.tensor(pose[3:6]).reshape(1, -1).float()  
                
                return {
                    'betas': betas,
                    'expression': expression,
                    'global_orient': global_orient,
                    'jaw_pose': jaw_pose,
                    'transl': transl,
                    'scale': scale, 
                    'reoptim': True 
                }
        elif self.data_type == 'deep3dface':
            data = torch.load(osp.join(data_dir, 'deep3dface', 'flame.pth'))  
            data['reoptim'] = False 
            return data 
        else: 
            fp = osp.join(data_dir, 'vhap', 'flame.npz') 
            data = np.load(fp)
            
            w2c = torch.eye(4) 
            w2c[2, 3] = -1
            
            focal_length = data['focal_length'][0]
            proj = torch.tensor([
                [2 * focal_length, 0, 0, 0],
                [0, -2 * focal_length, 0, 0],
                [0, 0, -(self.far+self.near)/(self.far-self.near), -2*self.far*self.near/(self.far-self.near)],
                [0, 0, -1, 0]
            ], dtype=torch.float32) 
            
            data = {
                'betas': torch.tensor(data['shape']).reshape(1, -1).float(), 
                'expression': torch.tensor(data['expr']).float(),
                'global_orient': torch.tensor(data['rotation']).float(),
                'neck_pose': torch.tensor(data['neck_pose']).float(),
                'jaw_pose': torch.tensor(data['jaw_pose']).float(),
                'leye_pose': torch.tensor(data['eyes_pose'][:, 3:]).float(),
                'reye_pose': torch.tensor(data['eyes_pose'][:, :3]).float(),
                'v_offsets': torch.tensor(data['static_offset'][0, :5023]).float(),
                'transl': torch.tensor(data['translation']).float(),
                'proj': proj,
                'w2c': w2c,
                'reoptim': True
            }
            return data  
   
    def get_item(self):   
        data = {
            'subject': self.subject_name
        }    
        data['lmks'] = self.load_landmarks(osp.join(self.root, 'process', 'lmk68.txt'))
        data.update(self.load_images(self.root)) 
        data.update(self.load_flame(self.root))   
        return data

    def __getitem__(self, index):
        return self.get_item(index)

    def __len__(self):
        return 1



if __name__ == '__main__': 
    import argparse
    from omegaconf import OmegaConf

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default='./configs/default.yaml', help="path to the yaml config file")
    args, extras = parser.parse_known_args() 
    opt = OmegaConf.merge(OmegaConf.load(args.config), OmegaConf.from_cli(extras))

    data_dir = 'results/f358d6ef5c20194d18789e09dbd2907370a14a12_high'
    # dataset = Dataset(data_dir) 
    dataset = VideoDataset(data_dir, opt)
    print('LEN: ', len(dataset))
    for i in range(len(dataset)):
        data = dataset.get_item(i)
        for k, v in data.items():
            if isinstance(v, torch.Tensor):
                logger.info(f'{k}: {v.shape}')

        exit()
