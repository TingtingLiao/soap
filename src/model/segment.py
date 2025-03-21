from typing import Any, Callable, Dict, List, Optional, Tuple, Union 
import torch 
import numpy as np 
from PIL import Image  
import torch.nn as nn 
import torch.nn.functional as F
from .briarmbg import BriaRMBG 


class Segmenter(nn.Module): 
    def __init__(self, device, dtype=torch.float32):
        super().__init__()
        self.model = BriaRMBG.from_pretrained(
            "briaai/RMBG-1.4" 
        ).to(device, dtype=dtype) 
        self.device = device 
        self.dtype = dtype

    def input_processor(self, image: Union[Image.Image, torch.FloatTensor, np.ndarray]):
        if isinstance(image, Image.Image):
            image = torch.tensor(np.array(image)) 
            
        elif isinstance(image, np.ndarray):
            image = torch.tensor(image)
            
        if isinstance(image, torch.Tensor):
            image = image.to(self.device, dtype=self.dtype)
            
            # add batch dimension
            if image.dim() == 3:
                image = image.unsqueeze(0)
            
            # swap channels
            if image.shape[-1] == 3:
                image = image.permute(0, 3, 1, 2) 
            
            # normalize
            if image.mean() > 1:
                image = image / 127.0 - 1.0 
            
            # resize 
            _, C, H, W = image.shape 
            k = (256.0 / float(H * W)) ** 0.5  
            in_size = (int(64 * round(H * k)), int(64 * round(W * k)))
            image = F.interpolate(image, in_size, mode="bilinear")
            
            return image, (H, W)

        raise ValueError(f"Unknown input type: {type(image)}")
    
    def output_processor(self, alpha: torch.FloatTensor, origin_size: Tuple[int, int], output_type: str):
        # un-resize 
        alpha = F.interpolate(alpha, size=origin_size, mode="bilinear")
        # swap channels
        alpha = alpha.movedim(1, -1).clip(0, 1) > 0.05  # [B, H, W, 1]
        
        if output_type == "torch":
            return alpha
        elif output_type == "np":
            return alpha.detach().cpu().numpy()
        elif output_type == "pil":
            return Image.fromarray((alpha.squeeze().detach().cpu().numpy() * 255).astype(np.uint8))
        
        raise ValueError(f"Unknown output type: {output_type}")
    
    @torch.no_grad()
    def forward(self, image: Union[Image.Image, torch.FloatTensor, np.ndarray], output_type: Optional[str] = "torch"):  
        image, origin_size  = self.input_processor(image)   
        alpha = self.model(image)[0][0]   
        out = self.output_processor(alpha, origin_size, output_type)
        return out