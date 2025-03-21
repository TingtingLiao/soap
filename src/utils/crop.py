import numpy as np
from PIL import Image

import coloredlogs, logging 
logger = logging.getLogger(__name__) 
coloredlogs.install(level='DEBUG', logger=logger) 


def get_bbox(mask, to_square=False, pad_size=0.):
    ''' Get bounding box from mask
    Args:
        mask: np.ndarray [H, W]
    Returns:
        bbox: list [y_min, x_min, y_max, x_max]
    '''

    # Find the coordinates of the non-zero elements
    y_indices, x_indices = np.nonzero(mask)
    coords = np.nonzero(mask)
    x_min, x_max = coords[0].min(), coords[0].max()
    y_min, y_max = coords[1].min(), coords[1].max()
    
    if to_square:
        max_size = max(x_max - x_min, y_max - y_min)
        if pad_size > 0:
            max_size = int(max_size * (1 + pad_size))
        x_min = max(0, x_min - (max_size - (x_max - x_min)) // 2)
        x_max = min(mask.shape[0], max_size + x_min)
        y_min = max(0, y_min - (max_size - (y_max - y_min)) // 2)
        y_max = min(mask.shape[1], max_size + y_min)
    
    # PIL bbox  
    bbox = [y_min, x_min, y_max, x_max]
    
    return bbox


def pad_image_square(image, bg_color, return_bbox=False):
    '''
    pad image to square
        Args:
            pil_img: PIL.Image or np.ndarray
            bg_color: int 0-255 
            return_bbox: bool, return bbox or not [left, top, right, bottom]
        Returns:
            PIL.Image or np.ndarray
    ''' 
    if isinstance(image, np.ndarray):
        pil_img = Image.fromarray(image.astype(np.uint8))
    else:
        pil_img = image 

    w, h = pil_img.size

    if w == h:
        result = pil_img  
        bbox = [0, 0, 0, 0]
    elif w > h:
        d = (w - h) // 2
        result = Image.new(pil_img.mode, (w, w), bg_color)
        result.paste(pil_img, (0, d))  
        bbox = [0, d, 0, w - h - d]
    else:
        d = (h - w) // 2
        result = Image.new(pil_img.mode, (h, h), bg_color)
        result.paste(pil_img, (d, 0)) 
        bbox = [d, 0, h - w - d, 0]
    if isinstance(image, np.ndarray):
        result = np.array(result)

    if return_bbox:
        return result, bbox
    
    return result


def pad_image(image, size, bg_color):
    if isinstance(image, np.ndarray):
        pil_img = Image.fromarray(image.astype(np.uint8))
    else:
        pil_img = image 
    w, h = pil_img.size
    if size < 1: 
        dw, dh = int(size * w), int(size * h) 
    else:
        dw, dh = size, size 
    result = Image.new(pil_img.mode, (w + dw * 2, h + dh * 2), bg_color)
    result.paste(pil_img, (dw, dh))
    if isinstance(image, np.ndarray):
        result = np.array(result)
    return result


def change_rgba_bg(rgba: Image.Image, bkgd="WHITE"):
    rgb_white = rgba_to_rgb(rgba, bkgd)
    new_rgba = Image.fromarray(np.concatenate([np.array(rgb_white), np.array(rgba)[:, :, 3:4]], axis=-1))
    return new_rgba

def rgba_to_rgb(rgba: Image.Image, bkgd="WHITE"):
    new_image = Image.new("RGBA", rgba.size, bkgd)
    new_image.paste(rgba, (0, 0), rgba)
    new_image = new_image.convert('RGB')
    return new_image


def crop_rgba(image, to_square=False, pad_size=0, return_bbox=False):
    if isinstance(image, Image.Image):
        arr = np.asarray(image.astype(np.uint8))
    else:
        arr = image

    alpha = image[:, :, -1]  
    y_min, x_min, y_max, x_max = get_bbox(alpha, to_square=to_square, pad_size=pad_size)
    
    arr = arr[x_min: x_max, y_min: y_max]
    bbox = [x_min, y_min, x_max, y_max]
    
    if isinstance(image, Image.Image):
        arr = Image.fromarray(arr) 

    if return_bbox:
        return arr, bbox
        
    return arr

def uncrop(image: Image.Image, params, origin_image: Image.Image=None, border:int=10):
    
    if isinstance(image, np.ndarray):
        img = Image.fromarray(image.astype(np.uint8)) 
    else:
        img = image
    # un-resize
    img = img.resize(params['size'])

    # un-pad border 
    if 'pad_bbox' in params:
        l, t, r, b = params['pad_bbox']  
        img = img.crop((l, t, img.size[0]-r, img.size[1]-b)) 

    # un-pad-square
    if 'pad2square_bbox' in params:
        l, t, r, b = params['pad2square_bbox'] 
        img = img.crop((l, t, img.size[0]-r, img.size[1]-b))

    if origin_image is None:  
        bgc = (255, 255, 255) if c == 3 else (255, 255, 255, 0)
        new_img = Image.new('RGBA', params['origin_size'], bgc) 
    else:
        new_img = origin_image.copy()

    # past back to the original image
    l, t, r, b = params['crop_bbox']  
    if border > 0:
        img = img.crop((border, border, img.size[0]-border, img.size[1]-border))
        new_img.paste(img, (t+border, l+border)) 
    else:
        new_img.paste(img, (t, l)) 
    return new_img