import torch 
import numpy as np
import torch.nn.functional as F
from sklearn.neighbors import NearestNeighbors
from scipy.ndimage import binary_dilation, binary_erosion
import random
import kiui 
from kiui.mesh import Mesh   
from kiui.grid_put import mipmap_linear_grid_put_2d 
from smplx.lbs import batch_rodrigues
from PIL import Image
from pytorch3d.renderer.mesh import rasterize_meshes
from pytorch3d.renderer import TexturesVertex
from pytorch3d.structures import Meshes 
from pytorch3d.renderer import RasterizationSettings, MeshRasterizer
from .mesh import normalize_vertices 


def dilate_image(image, mask, iterations):
    # image: [H, W, C], current image
    # mask: [H, W], region with content (~mask is the region to inpaint)
    # iterations: int

    if torch.is_tensor(mask):
        mask = mask.detach().cpu().numpy()
    
    if mask.dtype != bool:
        mask = mask > 0.5

    inpaint_region = binary_dilation(mask, iterations=iterations)
    inpaint_region[mask] = 0

    search_region = mask.copy()
    not_search_region = binary_erosion(search_region, iterations=3)
    search_region[not_search_region] = 0

    search_coords = np.stack(np.nonzero(search_region), axis=-1)
    inpaint_coords = np.stack(np.nonzero(inpaint_region), axis=-1)

    knn = NearestNeighbors(n_neighbors=1, algorithm="kd_tree").fit(search_coords)
    _, indices = knn.kneighbors(inpaint_coords)

    image[tuple(inpaint_coords.T)] = image[tuple(search_coords[indices[:, 0]].T)]
    return image


def get_visibility_pytorch3d(vertices, faces, img_res=256, blur_radius=0.0, faces_per_pixel=1):
    """get the visibility of vertices

    Args:
        vertices (torch.tensor): [B, N, 3] in [-1, 1]
        faces (torch.tensor): [B, F, 3]
        size (int): resolution of rendered image
    """
    
    xyz = vertices.clone()
    
    if vertices.dim() == 2 and faces.dim() == 2:
        xyz = xyz.unsqueeze(0)
        faces = faces.unsqueeze(0)
    elif vertices.dim() == 3 and faces.dim() == 2:
        faces = faces.unsqueeze(0).expand(xyz.shape[0], -1, -1)
    else:
        raise ValueError(f"vertices shape:{vertices.shape}, faces shape:{faces.shape}")
        
    xyz[:, :, 2] *= -1  
    batch, N, _ = xyz.shape  
    vis_mask = torch.zeros(size=(batch, N)).to(xyz.device)
    
    meshes_screen = Meshes(verts=xyz, faces=faces)
    # TODO: use nvdiffrast 
    pix_to_face, zbuf, bary_coords, dists = rasterize_meshes(
        meshes_screen,
        image_size=img_res,
        blur_radius=blur_radius,
        faces_per_pixel=faces_per_pixel,
        bin_size=-1,
        max_faces_per_bin=None,
        perspective_correct=False,
        cull_backfaces=True,
    )

    pix_to_face = pix_to_face.detach().cpu().view(batch, -1)
    faces = faces.detach().cpu()

    for idx in range(batch):
        F = len(faces[idx])
        vis_vertices_id = torch.unique(
            faces[idx][torch.unique(pix_to_face[idx][pix_to_face[idx] != -1]) - F * idx, :]
        )
        vis_mask[idx, vis_vertices_id] = 1.0 

    if vertices.dim() == 2:
        vis_mask = vis_mask[0]
        
    return vis_mask


def backface_culling(vertices, faces, mvp):
    '''
    backface culling  
    Args:
        vertices: torch.Tensor [N, 3]
        faces: torch.Tensor [F, 3]
        mvp: torch.Tensor [V, 4, 4] or [B, V, 4, 4]
    
    Returns:
        tri_mask: torch.Tensor [B, F] or [F]
    '''
    assert len(vertices.shape) == 2 and len(faces.shape) == 2
    assert len(mvp.shape) in [2, 3], f"mvp shape:{mvp.shape} should be [V, 4, 4] or [B, V, 4, 4]"
    
    if len(mvp.shape) == 3:  
        num_view = mvp.shape[0]
        v_homo = F.pad(vertices, pad=(0, 1), mode='constant', value=1.0).unsqueeze(0).expand(num_view, -1, -1) 
        screen_v = torch.bmm(v_homo, torch.transpose(mvp, 1, 2))[:, :, :3]  # [V, N, 4]
    else:
        screen_v = torch.matmul(vertices, mvp[:3, :3].transpose(0, 1)) + mvp[:3, 3] 
    
    visibility = get_visibility_pytorch3d(screen_v, faces)  # [B, F] 
    tri_mask = visibility[..., faces].any(axis=-1) # [B, F] 
    return tri_mask


def query_color(verts, faces, image, device):
    """query colors from points and image

    Args:
        verts ([N, 3]): [query verts]
        faces ([M, 3]): [query faces]
        image ([B, D, H, W]): [full image]

    Returns:
        colors ([N, D]): [colors of query verts]
        visibility ([N]): [visibility of query verts]
    """

    verts = verts.float().to(device)
    faces = faces.long().to(device)

    (xy, z) = verts.split([2, 1], dim=1)
    visibility = get_visibility_pytorch3d(verts, faces[:, [0, 2, 1]]).flatten()
    uv = xy.unsqueeze(0).unsqueeze(2)    # [B, N, 2]
    uv = uv * torch.tensor([1.0, -1.0]).type_as(uv)
    colors = torch.nn.functional.grid_sample(image, uv, align_corners=True)[0, :, :, 0].permute(1, 0)   
    colors[visibility == 0.0] = (torch.tensor([0.]*image.shape[1])).to(device)
    # visibility  = (visibility > 0).float()
    # print(verts.shape, colors.shape, visibility.shape)
    # exit()
    
    # v_valid = verts[visibility > 0]
    
    return colors.detach().cpu(), visibility.cpu()


def complete_unseen_vertex_color(meshes, valid_index: torch.Tensor) -> dict:
    """
    meshes: the mesh with vertex color to be completed.
    valid_index: the index of the valid vertices, where valid means colors are fixed. [V, 1]
    """
    valid_index = valid_index.to(meshes.device)
    colors = meshes.textures.verts_features_packed()    # [V, 3]
    V = colors.shape[0]
    
    invalid_index = torch.ones_like(colors[:, 0]).bool()    # [V]
    invalid_index[valid_index] = False
    invalid_index = torch.arange(V).to(meshes.device)[invalid_index]
    
    L = meshes.laplacian_packed()
    E = torch.sparse_coo_tensor(torch.tensor([list(range(V))] * 2), torch.ones((V,)), size=(V, V)).to(meshes.device)
    L = L + E 
    colored_count = torch.ones_like(colors[:, 0])   # [V]
    colored_count[invalid_index] = 0
    L_invalid = torch.index_select(L, 0, invalid_index)    # sparse [IV, V]
    
    total_colored = colored_count.sum()
    coloring_round = 0
    stage = "uncolored"
    from tqdm import tqdm
    pbar = tqdm(miniters=100)
    while stage == "uncolored" or coloring_round > 0:
        new_color = torch.matmul(L_invalid, colors * colored_count[:, None])    # [IV, 3]
        new_count = torch.matmul(L_invalid, colored_count)[:, None]             # [IV, 1]
        colors[invalid_index] = torch.where(new_count > 0, new_color / new_count, colors[invalid_index])
        colored_count[invalid_index] = (new_count[:, 0] > 0).float()
        
        new_total_colored = colored_count.sum()
        if new_total_colored > total_colored:
            total_colored = new_total_colored
            coloring_round += 1
        else:
            stage = "colored"
            coloring_round -= 1
        pbar.update(1)
        if coloring_round > 10000:
            print("coloring_round > 10000, break")
            break
    assert not torch.isnan(colors).any()
    meshes.textures = TexturesVertex(verts_features=[colors])
    return meshes


def vertex_color_from_images(vertices, faces, images, conf_thresh=0.2, weights=[1, 0, 1, 0.]):
    '''
    get vertex color from 4 orthogal images 

    Args:
    -----
        vertices: torch.Tensor [N, 3]
        faces: torch.Tensor [M, 3]
        images: torch.Tensor [B, H, W, 4] with alpha channel 
        conf_thresh: float 
            The confidence threshold for visibility (default: 0.2).
        weights: list of float
            Weights for each view (default: [2, 0.2, 1, 0.2]).
    Returns:
    --------
        colors: torch.Tensor [N, 3]
    '''
    device = vertices.device
    
    # rotate vertices to 4 orthographic views
    R = batch_rodrigues(torch.tensor([[0, np.pi*0.5 * i, 0] for i in range(4)]).reshape(-1, 3)) 
    verts_4view = torch.matmul(vertices.unsqueeze(0).expand(4, -1, -1), R.to(device)) 
    
    images = images.permute(0, 3, 1, 2).float()  
    colors = torch.zeros_like(vertices).cpu() 
    vis_count = torch.zeros(vertices.shape[0])  
    
    # get vertex color from images for visible vertices
    for v, img, w in zip(verts_4view, images, weights):  
        color, visibility = query_color(v, faces, img.unsqueeze(0), device)   
        colors += color * w
        vis_count += visibility.float() * w
    
    valid_ids = vis_count >= conf_thresh
    invalid_ids = torch.logical_not(valid_ids)
    # Normalize visible colors
    colors[valid_ids] /= vis_count[valid_ids].unsqueeze(-1) 
    # colors[invalid_ids] = (0.1 * (conf_thresh - vis_count[invalid_ids].unsqueeze(-1)) + colors[invalid_ids]) / conf_thresh

    # complete unseen vertex colors
    mesh = Meshes(verts=[vertices.to(device)], faces=[faces.to(device)], textures=TexturesVertex(verts_features=[colors.to(device)]))
    valid_ids = torch.arange(len(vis_count)).to(device)[vis_count >= conf_thresh]
    mesh = complete_unseen_vertex_color(mesh, valid_ids)
    verts = mesh.verts_packed()
    colors = mesh.textures.verts_features_packed()

    return colors 

def nvdiff_backface_culling():
    # get the visibility of vertices in each view 
    pix_to_face = render_pkg['pix_to_face'].view(batch, -1) # [B, H, w]  
    vis_albedo_mask = torch.zeros(size=(1, albedo_res, albedo_res, 1)).to(device)
    for idx in range(batch):  
        fids = torch.unique(pix_to_face[idx][pix_to_face[idx] != -1]) - 1 
        vis_uv_ids = torch.unique(mesh.ft[fids, :])
        # vis_uv_ids = mesh.ft[fids, :].reshape(-1)
        
        # uv_vis_mask[idx, vis_uv_ids] = 1.0 
        vis_uvs = mesh.vt.detach()[vis_uv_ids]  * 2 - 1 
        print(vis_uvs.shape)
        
        # # print(mesh.vt[vis_uv_ids].shape)
        # vis_uvs[:, 0] = vis_uvs[:, 0] * (albedo_res - 1)  # Convert u to x-coordinate
        # vis_uvs[:, 1] = vis_uvs[:, 1] * (albedo_res - 1)  # Convert v to y-coordinate
        # vis_uvs = vis_uvs.round().long()  
        
        from kiui.grid_put import scatter_add_nd, linear_grid_put_2d, mipmap_linear_grid_put_2d
        
        # scatter_add_nd(vis_albedo_mask[idx], vis_uvs, torch.ones_like(vis_uvs[:, :1]).float())
        # vis_albedo_mask[idx].index_put_((vis_uvs[:, 0], vis_uvs[:, 1]), torch.ones_like(vis_uvs[:, 0]).float(), accumulate=True)
        vis_albedo_mask = mipmap_linear_grid_put_2d(
            albedo_res, albedo_res, 
            vis_uvs[..., [1, 0]], 
            torch.ones_like(vis_uvs[:, :1]).float(), 
            min_resolution=256
            )
        
        # linear_grid_put_2d(cur_H, cur_W, coords, values, return_count=True)
        
        # mesh.vc = vertex_vis_mask[1].unsqueeze(-1).repeat(1, 3).float() * 0.5 
        # mesh.albedo = vis_albedo_mask.repeat(1, 1, 3).to(device)   
        # mesh.albedo[:, :, 1:] *= 0 
        # return mesh  


def albedo_from_images(
    renderer, mesh, images, weights=[4, 1, 1, 1], 
    albedo_res=1024, dilate=False, 
    deblur=False, return_mask=False):
    '''
    un-projecting images back to the mesh albedo map 

    Args:
    -----
        renderer: Renderer
        mesh: Mesh
        mvp: torch.Tensor [B, 4, 4]
        images: torch.Tensor [B, H, W, 4] with alpha channel 
        weights: list of float
            Weights for each view (default: [2, 0.2, 1, 0.2]).
        albedo_res: int, albedo resolution (default: 2048).
        dilate: bool, whether to dilate the mask and albedo (default: False).
        deblur: bool, whether to deblur the albedo (default: False).
    Returns:
    --------
        mesh with albedo map
    ''' 
    if mesh.vn is None:
        mesh.auto_normal()
    if mesh.vt is None or mesh.ft is None:
        mesh.auto_uv(vmap=False)

    device = mesh.v.device 
    batch, h, w, _ = images.shape
    if batch == 4:
        mvp, extrinsic, intrinsic = renderer.get_orthogonal_cameras(n=batch, yaw_range=(0, -360), return_all=True)    
        weights = [4, 1, 1, 1]
    else: # batch=6 
        mvp, extrinsic, intrinsic = renderer.get_orthogonal_cameras(n=8, yaw_range=(0, -360), return_all=True)
        ids = [0, 2, 4, 6, 1, 7]
        mvp, extrinsic, intrinsic = mvp[ids], extrinsic[ids], intrinsic[ids]
        weights = [4, 1, 1, 1, 1, 1]
    mvp = mvp.to(device)
    extrinsic = extrinsic.to(device)

    render_pkg = renderer(mesh, mvp.to(device), spp=1, h=h, w=w)
    uvs = render_pkg['uvs'] 
    alphas = render_pkg['alpha'][..., -1] > 0 
    normal = render_pkg['normal'] * 2 - 1 # [b, h, w, 3]
    view_cos = torch.bmm(normal.reshape(batch, -1, 3), extrinsic[:, :3, :3].to(device))
    view_cos = view_cos[..., 2].abs().reshape(*normal.shape[:3])  # [b, h, w]
    
    if images.shape[-1] == 4:
        images, image_alphas = images[..., :3], images[..., 3] > 0
        # alphas = torch.logical_and(alphas, image_alphas)

    # erod mask 
    alphas = alphas.unsqueeze(1).float()
    for _ in range(1):
        alphas[1:] = 1 - F.max_pool2d(1 - alphas[1:], kernel_size=5, stride=1, padding=2) 
    alphas = alphas.squeeze(1) > 0
    
    alphas = alphas.view(batch, -1)
    images = images.view(batch, -1, 3) 
    uvs = uvs.view(batch, -1, 2) * 2 - 1  
    view_cos = view_cos.view(batch, -1, 1)
    images = torch.cat([images, view_cos], dim=-1)

    cnt = torch.zeros(albedo_res, albedo_res, 1).to(device)
    albedo = torch.zeros(albedo_res, albedo_res, 3).to(device)
    view_cos = torch.zeros(albedo_res, albedo_res, 1).to(device)
    ids =[0, 2, 4, 5, 3, 1] 
    # ids =[0, 1, 2, 3]  
    for idx in ids: 
        uv, image, alpha, w = uvs[idx], images[idx], alphas[idx], weights[idx]
        img, uv = image[alpha], uv[alpha] 
        cur_albedo, cur_cnt = mipmap_linear_grid_put_2d(
            albedo_res, albedo_res, 
            uv[..., [1, 0]],  img, 
            min_resolution=256, 
            return_count=True
            )
        cur_albedo, cur_view_cos = torch.split(cur_albedo, [3, 1], dim=-1)
        vis_mask = (cur_cnt > 0) & (cur_view_cos > view_cos)   
        vis_mask = vis_mask.squeeze(-1) 

        albedo[vis_mask] += cur_albedo[vis_mask] * (cur_view_cos[vis_mask] ) * w  * cur_cnt[vis_mask] 
        cnt[vis_mask] += (cur_cnt[vis_mask]**2) * (cur_view_cos[vis_mask] ) * w
        view_cos[vis_mask] = cur_view_cos[vis_mask]
    
    mask = cnt.squeeze(-1) > 0 
    albedo[mask] /= cnt[mask].repeat(1, 3)

    if dilate:  # dilate only works for continuous uv 
        print('dilating ')
        mask = mask.detach().cpu().numpy() 
        albedo = dilate_image(albedo, mask, iterations=int(albedo_res*0.2))
        # cnt = dilate_image(cnt, mask, iterations=int(albedo_res*0.2))
    
    if deblur:  
        h = w = albedo_res
        ratio = 4 
        cur_albedo = F.interpolate(albedo.permute(2, 0, 1).unsqueeze(0), (h//ratio, w//ratio), mode='bilinear', align_corners=False) 
        cur_albedo = kiui.sr.sr(cur_albedo, scale=ratio).squeeze(0).permute(1, 2, 0)
    
    mesh.albedo = albedo.clamp(0, 1)
    if return_mask:
        return mesh, mask
    
    return mesh

def albedo_from_images_possion_blending(
    renderer, mesh, images, 
    mvp, extrinsic, 
    weights=None, 
    albedo_res=1024, dilate=False, 
    deblur=False, return_mask=False):
    '''
    un-projecting images back to the mesh albedo map 

    Args:
    -----
        renderer: Renderer
        mesh: Mesh
        mvp: torch.Tensor [B, 4, 4]
        images: torch.Tensor [B, H, W, 4] with alpha channel 
        weights: list of float
            Weights for each view (default: [2, 0.2, 1, 0.2]).
        albedo_res: int, albedo resolution (default: 2048).
        dilate: bool, whether to dilate the mask and albedo (default: False).
        deblur: bool, whether to deblur the albedo (default: False).
    Returns:
    --------
        mesh with albedo map
    ''' 
    if mesh.vn is None:
        mesh.auto_normal()
    if mesh.vt is None or mesh.ft is None:
        mesh.auto_uv(vmap=True)
    
    origin_images = images.clone()
    device = mesh.v.device 
    batch, h, w, _ = images.shape
    # if batch == 4:
    #     mvp, extrinsic, _ = renderer.get_orthogonal_cameras(n=batch, yaw_range=(0, -360), return_all=True)    
    #     weights = [4, 1, 1, 1]
    # else: # batch=6 
    #     mvp, extrinsic, _ = renderer.get_orthogonal_cameras(n=8, yaw_range=(0, -360), return_all=True)
    #     ids = [0, 2, 4, 6, 1, 7]
    #     mvp, extrinsic = mvp[ids], extrinsic[ids] 
    #     weights = [10, 1, 1, 1, 1, 1]
    mvp = mvp.to(device)
    extrinsic = extrinsic.to(device)
    
    render_pkg = renderer(mesh, mvp, spp=1, h=h, w=w)
    uvs = render_pkg['uvs'] 
    alphas = render_pkg['alpha'][..., -1] > 0 
    normal = render_pkg['normal'] * 2 - 1 # [b, h, w, 3]
    view_cos = torch.bmm(normal.reshape(batch, -1, 3), extrinsic[:, :3, :3])
    view_cos = view_cos[..., 2].abs().reshape(*normal.shape[:3])  # [b, h, w]
    
    if images.shape[-1] == 4: # TODO: test this 
        images, image_alphas = images[..., :3], images[..., 3] > 0
        # alphas = torch.logical_and(alphas, image_alphas)

    alphas = alphas.unsqueeze(1).float()
    for _ in range(1):
        alphas = 1 - F.max_pool2d(1 - alphas, kernel_size=5, stride=1, padding=2) 
    alphas = alphas.squeeze(1) > 0
    # Image.fromarray((alphas[0].detach().cpu().numpy() * 255).astype(np.uint8)).save('test.png')
    # exit() 
    
    alphas = alphas.view(batch, -1)
    images = images.view(batch, -1, 3)
    uvs = uvs.view(batch, -1, 2) * 2 - 1  
    view_cos = view_cos.view(batch, -1, 1)
    images = torch.cat([images, view_cos], dim=-1)

    cnt = torch.zeros(albedo_res, albedo_res, 1).to(device)
    albedo = torch.zeros(albedo_res, albedo_res, 3).to(device)
    view_cos = torch.zeros(albedo_res, albedo_res, 1).to(device)
    ids = [0, 2, 4, 5, 3, 1] 
    for idx in ids: 
        # if idx > 0: 
        #     mask = cnt.squeeze(-1) > 0 
        #     albedo[mask] /= cnt[mask].repeat(1, 3)
        #     mesh.albedo = albedo  
        #     out = renderer(mesh, mvp, spp=1, h=h, w=w)
        #     img = out['image'].detach().cpu().numpy() * 255  
        #     # img = np.concatenate(list(img), 1) * 255
        #     mask = (img[idx].sum(-1) > 0) * 255  
        #     Image.fromarray(mask.astype(np.uint8)).save('mask.png')
        #     Image.fromarray(img[idx].astype(np.uint8)).save('source.png')
        #     tgt = origin_images[idx, ..., :3].detach().cpu().numpy() * 255
        #     Image.fromarray(tgt.astype(np.uint8)).save('target.png')
        
        uv, image, alpha, wt = uvs[idx], images[idx], alphas[idx], weights[idx]
        
        img, uv = image[alpha], uv[alpha] 
        cur_albedo, cur_cnt = mipmap_linear_grid_put_2d(
            albedo_res, albedo_res, 
            uv[..., [1, 0]],  img, 
            min_resolution=256, 
            return_count=True
            )
        cur_albedo, cur_view_cos = torch.split(cur_albedo, [3, 1], dim=-1)
        
        vis_mask = (cur_cnt > 0) & (cur_view_cos > view_cos) 
        vis_mask = vis_mask.squeeze(-1) 
        albedo[vis_mask] += cur_albedo[vis_mask] * (cur_view_cos[vis_mask] ) * wt  * cur_cnt[vis_mask] 
        cnt[vis_mask] += (cur_cnt[vis_mask]**2) * (cur_view_cos[vis_mask] ) * wt
        view_cos[vis_mask] = cur_view_cos[vis_mask]
    
    mask = cnt.squeeze(-1) > 0 
    albedo[mask] /= cnt[mask].repeat(1, 3)
    
    if dilate:  # dilate only works for continuous uv 
        print('dilating ')
        mask = mask.detach().cpu().numpy() 
        albedo = dilate_image(albedo, mask, iterations=int(albedo_res*0.2))
        cnt = dilate_image(cnt, mask, iterations=int(albedo_res*0.2))
    
    if deblur:  
        h = w = albedo_res
        ratio = 4 
        cur_albedo = F.interpolate(albedo.permute(2, 0, 1).unsqueeze(0), (h//ratio, w//ratio), mode='bilinear', align_corners=False) 
        cur_albedo = kiui.sr.sr(cur_albedo, scale=ratio).squeeze(0).permute(1, 2, 0)
    
    mesh.albedo = albedo.clamp(0, 1)
    if return_mask:
        return mesh, mask
    
    return mesh 

def albedo_from_images2(renderer, mesh, images, weights=[1, 0., 1, 0.], albedo_res=1024, dilate=False, deblur=False):
    
    '''
    un-projecting images back to the mesh albedo map 

    Args:
    -----
        renderer: Renderer
        mesh: Mesh
        mvp: torch.Tensor [B, 4, 4]
        images: torch.Tensor [B, H, W, 4] with alpha channel 
        weights: list of float
            Weights for each view (default: [2, 0.2, 1, 0.2]).
        albedo_res: int, albedo resolution (default: 2048).
        dilate: bool, whether to dilate the mask and albedo (default: False).
        deblur: bool, whether to deblur the albedo (default: False).
    Returns:
    --------
        mesh with albedo map
    '''
    # TODO: add backface culling 
    if mesh.vn is None:
        mesh.auto_normal()
    
    if mesh.vt is None or mesh.ft is None:
        mesh.auto_uv() 
    device = mesh.v.device 
    batch = images.shape[0]
    
    # render views 
    mvp = renderer.get_orthogonal_cameras(n=batch, yaw_range=(0, -360))    
    render_pkg = renderer(mesh, mvp.to(device), spp=1, h=albedo_res, w=albedo_res)
    uvs = render_pkg['uvs'] 
    alphas = render_pkg['alpha'][..., -1] > 0.5 
    normal = render_pkg['normal'] * 2 - 1 
    
    if images.shape[-1] == 4:
        images, image_alphas = images[..., :3], images[..., 3] > 0
        alphas = torch.logical_and(alphas, image_alphas)
    
    alphas = alphas.view(batch, -1)
    images = images.view(batch, -1, 3) 
    uvs = uvs.view(batch, -1, 2) * 2 - 1   
    
    cnt = torch.zeros(albedo_res, albedo_res, 1).to(device)
    albedo = torch.zeros(albedo_res, albedo_res, 3).to(device)
    # for idx in [2, 1, 3, 0]:
    for idx in [2, 1, 3, 0]:
        uv, image, alpha, weight = uvs[idx], images[idx], alphas[idx], weights[idx]  
        img, uv = image[alpha], uv[alpha]  
        
        cur_albedo, cur_cnt = mipmap_linear_grid_put_2d(albedo_res, albedo_res, uv[..., [1, 0]], img, min_resolution=128, return_count=True)

        # thresh = 1 if idx in [1, 3] else 0  
        if idx == 0:
            vis_mask = cur_cnt.squeeze(-1) > 0
        else:
            vis_mask = torch.logical_and(cur_cnt > cnt, cur_cnt > 1).squeeze(-1)
        # vis_mask = cur_cnt.squeeze(-1) > 0
        albedo[vis_mask] = cur_albedo[vis_mask]   
        cnt[vis_mask] = cur_cnt[vis_mask]             

        # break
        
    mask = cnt.squeeze(-1) > 0
    albedo[mask] /= cnt[mask].repeat(1, 3)

    # TODO: inpaint the missing regions
    
    if dilate: 
        print('dilating ')
        mask = mask.detach().cpu().numpy() 
        albedo = dilate_image(albedo, mask, iterations=int(albedo_res*0.2))
        cnt = dilate_image(cnt, mask, iterations=int(albedo_res*0.2))
    
    if deblur:  
        h = w = albedo_res
        ratio = 4 
        cur_albedo = F.interpolate(albedo.permute(2, 0, 1).unsqueeze(0), (h//ratio, w//ratio), mode='bilinear', align_corners=False) 
        cur_albedo = kiui.sr.sr(cur_albedo, scale=ratio).squeeze(0).permute(1, 2, 0)

    mesh.albedo = albedo.clamp(0, 1) 

    # mesh.write_obj('test.obj')
    # exit()
    return mesh

def inpaint_mesh_albedo(mesh, mask, iterations=5):
    """
    Inpaints the albedo map directly on the 3D mesh by propagating colors.

    Args:
    -----
        mesh: Mesh
            The 3D mesh object containing vertices, faces, and albedo.
        mask: torch.Tensor [num_vertices]
            Binary mask indicating valid regions (1 for valid, 0 for missing).
        iterations: int
            Number of propagation iterations for inpainting.

    Returns:
    --------
        inpainted_vertex_colors: torch.Tensor [num_vertices, 3]
            The inpainted vertex colors on the 3D mesh.
    """
    device = mesh.v.device

    # Map albedo texture to vertex colors
    vertex_colors = map_texture_to_vertices(mesh, mesh.albedo)
    print(vertex_colors.shape, mesh.v.shape)
    exit()
    # Initialize mask for valid vertex colors
    valid_mask = mask.clone().to(device).float()

    # Compute adjacency matrix
    adjacency_matrix = compute_mesh_adjacency(mesh.v, mesh.f).to(device)

    for _ in range(iterations):
        # Propagate colors to neighbors
        neighbor_colors = adjacency_matrix @ vertex_colors
        neighbor_mask = adjacency_matrix @ valid_mask.unsqueeze(-1)

        # Normalize by the number of valid neighbors
        valid_neighbors = neighbor_mask > 0
        neighbor_colors[valid_neighbors] /= neighbor_mask[valid_neighbors]

        # Fill missing values with propagated neighbor colors
        to_fill = (valid_mask == 0).bool()
        vertex_colors[to_fill] = neighbor_colors[to_fill]

        # Update the mask
        valid_mask = torch.max(valid_mask, (neighbor_mask > 0).squeeze(-1).float())

    return vertex_colors


def compute_mesh_adjacency(vertices, faces):
    """
    Computes the adjacency matrix for a given mesh.
    """
    import scipy.sparse

    num_vertices = vertices.shape[0]
    edges = torch.cat([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], dim=0).cpu().numpy()
    adj_matrix = scipy.sparse.coo_matrix(
        (np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
        shape=(num_vertices, num_vertices),
    )
    adj_matrix = adj_matrix + adj_matrix.T  # Ensure symmetry
    return torch.tensor(adj_matrix.toarray(), dtype=torch.float32)


def map_texture_to_vertices(mesh, texture):
    """
    Maps a 2D texture to unique vertex colors based on UV mapping.

    Args:
    -----
        mesh: Mesh
            The mesh containing UV coordinates (`mesh.vt`) and faces (`mesh.ft`).
        texture: torch.Tensor [H, W, 3]
            The texture image mapped to the mesh.

    Returns:
    --------
        vertex_colors: torch.Tensor [num_vertices, 3]
            Colors for each vertex based on the texture.
    """
    device = texture.device

    # UV coordinates for all vertices in face order
    uv = mesh.vt[mesh.ft.flatten()].to(device)  # [num_faces * 3, 2]

    # Scale UVs to texture resolution
    h, w = texture.shape[:2]
    uv = uv * torch.tensor([w - 1, h - 1], device=device)  # Scale to pixel indices

    # Separate into x and y coordinates
    u, v = uv[:, 0].long(), uv[:, 1].long()

    # Clamp indices to ensure they are valid
    u = u.clamp(0, w - 1)
    v = v.clamp(0, h - 1)

    # Sample texture using the UV indices
    sampled_colors = texture[v, u]  # [num_faces * 3, 3]

    # Initialize vertex colors and a count buffer
    vertex_colors = torch.zeros((mesh.v.shape[0], 3), device=device)
    vertex_count = torch.zeros((mesh.v.shape[0], 1), device=device)

    # Map sampled UV colors back to vertex colors
    for uv_index, vertex_index in enumerate(mesh.ft.flatten()):
        vertex_colors[vertex_index] += sampled_colors[uv_index]
        vertex_count[vertex_index] += 1

    # Avoid division by zero
    valid_vertices = vertex_count.squeeze() > 0
    vertex_colors[valid_vertices] /= vertex_count[valid_vertices]

    return vertex_colors
