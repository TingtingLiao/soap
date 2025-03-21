from abc import abstractmethod, ABC
import numpy as np
import pickle as pkl
from PIL import Image
import torch
import torch.nn as nn 
import mediapipe as mp  
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from face_alignment.utils import flip, get_preds_fromhm
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
from src.utils.vis import parsing_from_label_image
from src.utils.helper import pts_to_bbox


class FaceParser(ABC):
    def __init__(self, device, color_map=None):
        if color_map is None:
            self.color_map = dict(
                face=[1, 0, 0.33],
                hair=[0, 0, 1],
                neck=[0, 1, 0.33] ,
                lips=[1, 0, 1],
            )
        else:
            self.color_map = color_map
        self.device = device
        self.seg_processor = SegformerImageProcessor.from_pretrained("jonathandinu/face-parsing")
        self.seg_model = SegformerForSemanticSegmentation.from_pretrained("jonathandinu/face-parsing")
        self.seg_model.to(device)
    
    def run(self, image: Image.Image):         
        seg_inputs = self.seg_processor(images=image, return_tensors="pt").to(self.device)
        outputs = self.seg_model(**seg_inputs) 
        upsampled_logits = nn.functional.interpolate(outputs.logits, size=image.size[::-1], mode='bilinear', align_corners=False)
        labels = upsampled_logits.argmax(dim=1)[0].cpu().numpy()    
        
        w, h = labels.shape 

        # mask for face, hair, neck 
        face_mask = np.zeros((w, h), dtype=np.uint8) 
        for idx in [1, 2, 3, 6, 7, 10, 11, 12]: # colors[1] 
            face_mask[labels == idx] = 1
        
        eye_mask = np.zeros((w, h), dtype=np.uint8) 
        # do not use left and right eye seperatly, its not accurate
        # split them manually 
        for idx in [4, 5]:  
            eye_mask[labels == idx] = 1 

        hair_mask = np.zeros((w, h), dtype=np.uint8)
        for idx in [8, 9, 13, 14, 15]:
            hair_mask[labels == idx] = 1
        
        neck_mask = np.zeros((w, h), dtype=np.uint8)
        for idx in [16, 17, 18]:
            neck_mask[labels == idx] = 1
        
        mask = face_mask | hair_mask | neck_mask | eye_mask 
        parse_image = (
            np.stack([face_mask]*3, -1) * self.color_map.face + 
            np.stack([eye_mask]*3, -1) * self.color_map.eye +  
            np.stack([hair_mask]*3, -1) * self.color_map.hair + 
            np.stack([neck_mask]*3, -1) * self.color_map.neck
        )
        parse_image = parse_image * 255  
        vis = parse_image * 0.5 + np.array(image) * 0.5
        vis = Image.fromarray(vis.astype(np.uint8))
        
        # mask for lips, eyes, nose  
        nose_mask = np.zeros((w, h), dtype=np.uint8)
        for idx in [9]:
            nose_mask[labels == idx] = 1
        
        lips_mask = np.zeros((w, h), dtype=np.uint8)
        for idx in [11, 12]:
            lips_mask[labels == idx] = 1
        
        # blend the masks 
        
        return {
            'mask': mask,
            'face_mask': face_mask,
            'eye_mask': eye_mask, 
            'hair_mask': hair_mask, 
            'neck_mask': neck_mask, 
            'parse_image': Image.fromarray(parse_image.astype(np.uint8)), 
            'vis': vis 
        }  

    def get_face_mask(self, image: Image.Image):
        seg_inputs = self.seg_processor(images=image, return_tensors="pt").to(self.device)
        outputs = self.seg_model(**seg_inputs) 
        logits = nn.functional.interpolate(outputs.logits, size=image.size[::-1], mode='bilinear', align_corners=False)
        labels = logits.argmax(dim=1)[0].cpu().numpy()    
        h, w = labels.shape

        mask = np.zeros((w, h), dtype=np.uint8) 
        for idx in [1, 2, 3, 6, 7, 11, 12]: # colors[1] 
            mask[labels == idx] = 1

        return mask


class MediapipeLmks(ABC):
    silhouette = [
        10,  338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
        397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
        172, 58,  132, 93,  234, 127, 162, 21,  54,  103, 67,  109
    ] 
    lipsUpperOuter = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291]
    lipsLowerOuter = [146, 91, 181, 84, 17, 314, 405, 321, 375, 291],
    lipsUpperInner = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308]
    lipsLowerInner = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308]

    rightEyeUpper0 = [246, 161, 160, 159, 158, 157, 173]
    rightEyeLower0 = [33, 7, 163, 144, 145, 153, 154, 155, 133]
    rightEyeUpper1 = [247, 30, 29, 27, 28, 56, 190],
    rightEyeLower1 = [130, 25, 110, 24, 23, 22, 26, 112, 243]
    rightEyeUpper2 = [113, 225, 224, 223, 222, 221, 189]
    rightEyeLower2 = [226, 31, 228, 229, 230, 231, 232, 233, 244]
    rightEyeLower3 = [143, 111, 117, 118, 119, 120, 121, 128, 245]

    rightEyebrowUpper = [156, 70, 63, 105, 66, 107, 55, 193]
    rightEyebrowLower = [35, 124, 46, 53, 52, 65]
    rightEyeIris = [473, 474, 475, 476, 477]

    leftEyeUpper0 = [466, 388, 387, 386, 385, 384, 398]
    leftEyeLower0 = [263, 249, 390, 373, 374, 380, 381, 382, 362]
    leftEyeUpper1 = [467, 260, 259, 257, 258, 286, 414]
    leftEyeLower1 = [359, 255, 339, 254, 253, 252, 256, 341, 463]
    leftEyeUpper2 = [342, 445, 444, 443, 442, 441, 413]
    leftEyeLower2 = [446, 261, 448, 449, 450, 451, 452, 453, 464]
    leftEyeLower3 = [372, 340, 346, 347, 348, 349, 350, 357, 465] 

    leftEyebrowUpper = [383, 300, 293, 334, 296, 336, 285, 417] 
    leftEyebrowLower = [265, 353, 276, 283, 282, 295] 
    leftEyeIris =[468, 469, 470, 471, 472] 

    midwayBetweenEyes = [168] 

    noseTip = [1] 
    noseBottom = [2] 
    noseRightCorner = [98] 
    noseLeftCorner = [327] 

    rightCheek = [205] 
    leftCheek = [425]

    def __init__(self, mp_landmarks, h, w):
        self.h = h 
        self.w = w
        self._mp_lmks = mp_landmarks
        self._lmks = torch.tensor([[l.x, l.y, l.z]  for l in mp_landmarks[0]], dtype=torch.float32)  
        self._bbox = pts_to_bbox(self._lmks.cpu().numpy())  * np.array([w, h, w, h])
    
    @property
    def lmks(self):
        return self._lmks

    @property
    def mp_lmks(self):
        return self._mp_lmks

    @property
    def bbox(self):
        return self._bbox
    
    def pad_bbox(self, size):
        x1, y1, x2, y2 = self._bbox
        dw = (x2 - x1) * size
        dh = (y2 - y1) * size
        x1 = max(0, x1 - dw)
        y1 = max(0, y1 - dh)
        x2 = min(self.w, x2 + dw)
        y2 = min(self.h, y2 + dh)
        return [x1, y1, x2, y2]

    @property
    def left_iris(self):
        return self._lmks[self.leftEyeIris] 
    
    @property
    def right_iris(self):
        return self._lmks[self.rightEyeIris]

    @property
    def kpt68(self):
        return self._lmks[self.silhouette]

class Mediapipe(ABC): 
    def __init__(self): 
        base_options = python.BaseOptions(model_asset_path='data/mediapipe/face_landmarker_v2_with_blendshapes.task')
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=mp.tasks.vision.FaceDetectorOptions.running_mode.IMAGE,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.detector = vision.FaceLandmarker.create_from_options(options)

    def run(self, image, return_bbox=False):
        """
        Input:
            image: np.array, [h, w, 3], uint8, 0-255

        return: 
            mp_lmks: mediapipe landmarks
            lmks: torch.tensor, [68, 3]
            bbox: [left, top, right, bottom]
        """
        image_mp = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
        mp_lmks = self.detector.detect(image_mp).face_landmarks     
        if len(mp_lmks) == 0:
            return None 
        return MediapipeLmks(mp_lmks, h=image.shape[0], w=image.shape[1])
    

    def __call__(self, *args, **kwargs):
        return self.run(*args, **kwargs)
