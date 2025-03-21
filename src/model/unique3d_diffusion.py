import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import kiui 
from kiui.op import scale_img_nhwc
from smplx.lbs import batch_rodrigues  
from diffusers import EulerAncestralDiscreteScheduler 
from .unique3d_pipeline import Unique3dDiffusionPipeline



class Unique3dDiffusion(nn.Module):
    def __init__(self, opt, device, dtype=torch.float16):
        super().__init__() 
        self.opt = opt
        self.device = device
        self.dtype = dtype
        self.generator = torch.Generator(device).manual_seed(opt.seed)  
        self.mvimg_pipe = Unique3dDiffusionPipeline.from_pretrained(  
            "Luffuly/unique3d-mvimage-diffuser", 
            torch_dtype=dtype, 
            trust_remote_code=True,  
            class_labels=torch.tensor(range(4)),
        ).to(device)  
        self.mvimg_pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(
            self.mvimg_pipe.scheduler.config
        ) 
        
        self.normal_pipe = Unique3dDiffusionPipeline.from_pretrained( 
            "Luffuly/unique3d-normal-diffuser", 
            torch_dtype=dtype, 
            trust_remote_code=True,  
        ).to(device)  
        self.normal_pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(
            self.normal_pipe.scheduler.config
        )

    
    def forward_image(self, image, output_type: str='pil', upscale: int=1):
        out = self.mvimg_pipe(
                image, 
                generator=self.generator, 
                output_type=output_type, 
                **self.opt.image_diffusion.forward_args 
            ).images  
        
        if upscale > 1: 
            pass 
        
        return out 
    
    def forward_normal(self, image, output_type='pil'):
        return self.normal_pipe(
                image, 
                generator=self.generator, 
                output_type=output_type, 
                **self.opt.normal_diffusion.forward_args,  
            ).images 
        
    def lazy_remove(self, image: torch.Tensor):
        '''
        image: [batch, h, w, 3]
        
        Returns:
        --------
        alpha: [batch, h, w, 1]
        '''
        base = image[:, :1, :1, :] # [batch, 1, 1, 3]
        diffs = (image - base.expand_as(image)).abs().sum(-1)
        alpha = diffs > 80 / 255
        return alpha.unsqueeze(-1)
    
    @property 
    def rot(self):
        return batch_rodrigues(
            torch.tensor([
                [0, np.radians(angle), 0] for angle in [0, 270, 180, 90]
            ])
        ) 
    
    def forward(self, image: Image, rot: bool=True, sr=False, resolution=2048, condition_images=None, strength=0.2):
        # inference
        if condition_images is None:
            mv_image = self.forward_image(image, output_type='pil')   
        else:
            mv_image = self.mvimg_pipe.refine(
                condition_images, image.convert('RGB'), strength=strength, output_type="pil", generator=self.generator, **self.opt.image_diffusion.forward_args 
            ).images  # [4, 3, 256, 256] 
        
        if sr:  # super resolution  
            scale = resolution // 256
            front_image = image.convert('RGB')
            if front_image.size[0] <= 512:
                s = resolution // front_image.size[0] 
                front_image = Image.fromarray(kiui.sr.sr(np.array(front_image), scale, self.device)) 
            mv_image = [front_image] + [
                Image.fromarray(kiui.sr.sr(np.array(img), scale, self.device)) for img in mv_image[1:]
            ]
        
        # inference normal 
        mv_normal = self.forward_normal(mv_image, output_type='torch')
        
        if sr:
            scale = resolution // 512  
            mv_normal = torch.cat([
                kiui.sr.sr(img[None], scale, self.device) for img in mv_normal.permute(0, 3, 1, 2) 
            ]).to(mv_normal).permute(0, 2, 3, 1)

        # remove bg 
        mv_alpha = self.lazy_remove(mv_normal).to(mv_normal) 
        
        if rot:  
            # rotate normal to world space
            mv_normal = (mv_normal * 2 - 1) * mv_alpha  
            mv_normal = torch.bmm(mv_normal.reshape(4, -1, 3), self.rot.to(mv_normal)).reshape(*mv_normal.shape)
            mv_normal = (mv_normal + 1) / 2
        
        # add alpha channel
        mv_normal = torch.cat([mv_normal, mv_alpha], -1) * 255 
        
        if not sr: 
            mv_alpha = scale_img_nhwc(mv_alpha, (256, 256)) 
        
        mv_alpha = mv_alpha.cpu().numpy()*255
                
        mv_image = [
            np.concatenate([np.array(mv_image[0]), np.array(image.resize(mv_image[0].size))[..., 3:4]], -1)
            ] + [
            np.concatenate([np.array(img), alpha], -1) for img, alpha in zip(mv_image[1:], mv_alpha[1:])
            ] 
        
        # to pil 
        mv_image = [
            Image.fromarray(img.astype(np.uint8)) for img in mv_image 
        ]  
        mv_normal = [
            Image.fromarray(img) for img in (mv_normal.cpu().numpy()).astype(np.uint8)
        ]
        
        return mv_image, mv_normal