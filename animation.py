import os 
import torch 
from kiui.mesh import Mesh 
import smplx 
from smplx.render import Renderer 
import numpy as np
import imageio 
from rich.progress import track


device = 'cuda'
model = smplx.create(
        model_path='./data/flame/flame2020.pkl', 
        model_type='flame',   
        batch_size=1, 
        create_segms=True 
    ).to(device)
renderer = Renderer() 



def export_video(images, wfp, **kwargs): 
    fps = kwargs.get('fps', 30)
    video_format = kwargs.get('format', 'mp4')  # default is mp4 format
    codec = kwargs.get('codec', 'libx264')  # default is libx264 encoding
    quality = kwargs.get('quality')  # video quality
    pixelformat = kwargs.get('pixelformat', 'yuv420p')  # video pixel format
    image_mode = kwargs.get('image_mode', 'rgb')
    macro_block_size = kwargs.get('macro_block_size', 2)
    ffmpeg_params = ['-crf', str(kwargs.get('crf', 18))]

    writer = imageio.get_writer(
        wfp, fps=fps, format=video_format,
        codec=codec, quality=quality, ffmpeg_params=ffmpeg_params, pixelformat=pixelformat, macro_block_size=macro_block_size
    )

    for i in track(range(len(images)), description='Writing', transient=True):
        if image_mode.lower() == 'bgr':
            writer.append_data(images[i][..., ::-1])
        else:
            writer.append_data(images[i])

    writer.close()


@torch.no_grad()
def run_animation(flame_params, base_mesh, motions, res=512):   
    # update shape  
    motions = { 
        'betas': flame_params['betas'],  
        'expression': motions['expr'], 
        'jaw_pose': motions['jaw_pose'],
        'neck_pose': motions['neck_pose'], 
        'global_orient': motions['rotation'],
        'leye_pose': motions['eyes_pose'][:, :3],
        'reye_pose': motions['eyes_pose'][:, 3:]
    } 
    
    # infer  
    vertices = model(**motions).vertices  
    # project
    vertices = (vertices + flame_params['transl']) * flame_params['scale'] 
    # render 
    frames = []   
    mesh = base_mesh.clone()
    bg_color = torch.tensor([1.0]*3).to(device) 
    mvps = renderer.get_orthogonal_cameras(1).to(device)
    for i, v in enumerate(vertices):    
        tz = v[:, 2].mean(0)
        v[:, 2] -= tz
        mesh.v = v 
        render_pkg = renderer(mesh, mvps, spp=2, bg_color=bg_color, h=res, w=res) 
        rgb = render_pkg['image'][0].detach().cpu().numpy()  
        nml = render_pkg['normal'][0].detach().cpu().numpy() 
        image = np.hstack([rgb, nml]) * 255
        frames.append(image.astype(np.uint8))
    
    return frames


subject = '1728079749210' 
flame_path = f'data/{subject}/flame.pth'
rigging_path = f'data/{subject}/w-eye/flame_attributes.pth'
mesh_path = f'data/{subject}/w-eye/recon_textured.obj'
driven_path = f'data/motions/d11.npz'

# load flame params
flame_params = torch.load(flame_path)  
flame_params = {k: v.to(device) for k, v in flame_params.items()}


# load skinning and rigging 
flame_attributes = torch.load(rigging_path)
flame_attributes = {k: v.to(device) for k, v in flame_attributes.items()}
model.set_params(flame_attributes)

# load texture and uv 
mesh = Mesh.load_obj(mesh_path, device=device)

# load driven motion sequence
motions = np.load(driven_path)  
motions = {k: motions[k] for k in ['rotation', 'neck_pose', 'jaw_pose', 'eyes_pose', 'expr']} 
motions = {k: torch.tensor(v).to(device) for k, v in motions.items()}


frames = run_animation(flame_params, mesh, motions, res=512) 
export_video(frames, "out.mp4", fps=30)


# save_base_mesh 
# v = model().vertices[0]
# mesh.v = v
# mesh.write('base_mesh.obj')





