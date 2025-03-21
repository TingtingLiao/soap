import cv2 
import numpy as np
import mediapipe as mp
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
from matplotlib import cm as mpl_cm, colors as mpl_colors
import torch 

 
def draw_mediapipe_landmarks(rgb_image: object, face_landmarks_list: object, save_path=None) -> object:
    annotated_image = np.copy(rgb_image)

    # Loop through the detected faces to visualize.
    for idx in range(len(face_landmarks_list)):
        face_landmarks = face_landmarks_list[idx]

        # Draw the face landmarks.
        face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        face_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in face_landmarks
        ])

        solutions.drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
            # connections=FACEMESH_NOSE,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_tesselation_style()
        )
        solutions.drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_contours_style())
        solutions.drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=mp.solutions.face_mesh.FACEMESH_IRISES,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_iris_connections_style())

    if save_path is not None:
        cv2.imwrite(save_path, annotated_image[..., ::-1])

    return annotated_image


def draw_landmarks(canvas, landmarks, eps=1e-4, fill=(0, 0, 0), thickness=-1):
    h, w, c = canvas.shape
    for lmk in landmarks:
        x, y = lmk
        x = int(x * w)
        y = int(y * h)
        if eps < x <= w and eps < y <= h:
            cv2.circle(canvas, (x, y), 3, fill, thickness=thickness)
    return canvas

def draw_smplx_landmarks(image, landmarks):
    rgb_np = draw_landmarks(image * 255, smplx_lmk_eye[0], fill=(255, 255, 255))
    rgb_np = draw_landmarks(rgb_np, smplx_lmk_nose[0], fill=(255, 255, 255))
    rgb_np = draw_landmarks(rgb_np, torch.cat([smplx_lmk_mouth, smplx_lmk_lips], 1)[0], fill=(255, 255, 255))
    rgb_np = draw_landmarks(rgb_np, smplx_lmk_eye_brow[0], fill=(255, 255, 255))
    rgb_np = draw_landmarks(rgb_np * 255, out['smplx_landmarks'][0], fill=(0, 0, 255)) 


def parsing_from_label_image(labels, bg_color=[0, 0, 0], save_path=None):
    """
    label: hxw numpy image, each pixel is an integer 0-19 
    """
    labels_viz = labels.cpu().numpy()   
    h, w = labels_viz.shape[:2]

    part_colors = [
        [255, 255, 255], 
        [255, 85, 0], 
        [255, 170, 0],
        [255, 0, 85], [255, 0, 170],
        [0, 255, 0], [85, 255, 0], [170, 255, 0],
        [0, 255, 85], [0, 255, 170],
        [0, 0, 255], [85, 0, 255], [170, 0, 255],
        [0, 85, 255], [0, 170, 255],
        [255, 255, 0], [255, 255, 85], [255, 255, 170],
        [255, 0, 255], [255, 85, 255], [255, 170, 255],
        [0, 255, 255], [85, 255, 255], [170, 255, 255]
        ]
    
    # remove eye, lips, mouth, ears 
    # full face 
    face_mask = np.zeros((h, w, 3), dtype=np.uint8) 
    for idx in [1, 2, 3, 6, 7, 8, 9, 11, 12]: # colors[1] 
        face_mask[labels_viz == idx] = 1
    
    mouse_mask = np.zeros((h, w, 3), dtype=np.uint8) 
    for idx in [10, 11, 12]:
        mouse_mask[labels_viz == idx] = 1
    
    eyes_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for idx in [4, 5]:
        eyes_mask[labels_viz == idx] = 1

    hair_mask = np.zeros((h, w, 3), dtype=np.uint8)
    hair_mask[labels_viz == 13] = 1 # hair 
    hair_mask[labels_viz == 14] = 1 # hat 

    body_mask = np.zeros((h, w, 3), dtype=np.uint8)  # neck part_colors[17]
    for idx in [15, 16, 17]:
        body_mask[labels_viz == idx] = 1

    cloth_mask = np.zeros((h, w, 3), dtype=np.uint8)  # part_colors[18] 
    cloth_mask[labels_viz == 18] = 1

    out_rgb = face_mask * part_colors[1] + hair_mask * part_colors[13] + body_mask * part_colors[17] + mouse_mask * part_colors[10] + eyes_mask * part_colors[4]
    out_rgb += cloth_mask * part_colors[18]

    if save_path is not None: 
        cv2.imwrite(save_path, out_rgb)
    
    head_mask = face_mask + hair_mask + mouse_mask + eyes_mask + body_mask
    return head_mask, out_rgb


def plot_kpts(image, kpts, color = 'r'):
    ''' Draw 68 key points
    Args:
        image: the input image
        kpt: (68, 3).
    '''
    if color == 'r':
        c = (255, 0, 0)
    elif color == 'g':
        c = (0, 255, 0)
    elif color == 'b':
        c = (255, 0, 0)
    image = image.copy()
    kpts = kpts.copy()
    
    end_list = np.array([17, 22, 27, 31, 36, 42, 48, 60, 68], dtype = np.int32) - 1
    end_list = end_list.tolist()

    colors = [
        (0, 0, 0), 
        (255, 0, 0),   #red  left-eyebrow 17-22
        (0, 255, 0),   # green right-eyebrow 22-27
        (0, 0, 255),   # blue nose 27-31
        (255, 255, 0),  #yellow nose 31-36
        (0, 255, 255),  #cyan left-eye 36-42
        (255, 0, 255),  #magenta right-eye 42-48
        (255, 165, 0),  #orange outer-mouth 48-60
        (128, 0, 128),  #purple inner-mouth 60-68
        (128, 128, 0)   #purple inner-mouth 60-68
        ]   
    
    cir_strength = np.ceil(image.shape[0] / 64).astype(np.int32)
    lin_strength = np.ceil(image.shape[0] / 256).astype(np.int32)
    c = colors[0]
    for i in range(kpts.shape[0]):
        st = kpts[i, :2].astype(np.int32)
        if kpts.shape[1]==4:
            if kpts[i, 3] > 0.5:
                c = (0, 255, 0)
            else:
                c = (0, 0, 255)
        image = cv2.circle(image,(st[0], st[1]), 1, c, cir_strength)
        if i in end_list:
            c = colors[end_list.index(i) + 1]
            continue
        ed = kpts[i + 1, :2].astype(np.int32)
        image = cv2.line(image, (st[0], st[1]), (ed[0], ed[1]), (255, 255, 255), lin_strength)

    return image

def plot_verts(image, verts, color = 'r'):
    ''' Draw 68 key points
    Args:
        image: the input image
        kpt: (68, 3).
    '''
    if color == 'r':
        c = (255, 0, 0)
    elif color == 'g':
        c = (0, 255, 0)
    elif color == 'b':
        c = (255, 0, 0)
    image = image.copy()
    verts = verts.copy() 
    for i in range(verts.shape[0]):
        st = verts[i, :2].astype(np.int32)
        if verts.shape[1]==4:
            if verts[i, 3] > 0.5:
                c = (0, 255, 0)
            else:
                c = (0, 0, 255)
    
        image = cv2.circle(image,(st[0], st[1]), 3, c, 2)
    
    return image

def vis_landmarks(image, landmarks, color='g', isScale=True):
    """
    visualize landmarks on image
        Args:
        -----
        image: torch.tensor, [B, H, W, C] or [H, W, C]
        landmarks: torch.tensor, [B, 68, 3] or [68, 3]
        color: str, color of landmarks
        isScale: bool, whether scale landmarks to image size
        
        Returns:
        --------
        out_imgs: list of torch.tensor, [H, W, C] 
    """
    assert len(image.shape) == 4 or len(image.shape) == 3, "image should be [B, H, W, C] or [H, W, C]"

    # visualize landmarks
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()
    if isinstance(landmarks, torch.Tensor):
        landmarks = landmarks.detach().cpu().numpy()

    if isScale:
        landmarks = (landmarks * 0.5 + 0.5) * image.shape[1]  
    
    def drw_single_image(image, landmarks):
        if landmarks.shape[0] == 68:
            img_lmks = plot_kpts(image*255, landmarks, color) 
        elif landmarks.shape[0] == 70:  # lmk68 with eyes 
            img_lmks = plot_kpts(image*255, landmarks[:68], color)
            img_lmks = plot_verts(img_lmks, landmarks[68:], color)  
        elif landmarks.shape[0] == 73:  # lmk68 with eyes  
            img_lmks = plot_verts(image*255, landmarks[:5], color)
            img_lmks = plot_kpts(img_lmks, landmarks[5:73], color)  
        else:
            img_lmks = plot_verts(image*255, landmarks, color) 
        return torch.as_tensor(img_lmks, dtype=torch.float32) / 255
    
    if len(image.shape) == 3: 
        return drw_single_image(image, landmarks)
    
    if len(image.shape) == 4:
        out_imgs = [drw_single_image(img, lmks) for img, lmks in zip(image, landmarks)]   
        return out_imgs
