import trimesh
import xatlas 
import torch 
import numpy as np
import pymeshlab as ml 
from kiui.op import safe_normalize, dot  
from kiui.mesh import Mesh 


@torch.no_grad()
def save_obj(path, vertices, faces=None, vertex_colors=None):
    if isinstance(vertices, torch.Tensor):
        vertices = vertices.cpu().numpy().astype(np.float32)
    if isinstance(faces, torch.Tensor):
        faces = faces.cpu().numpy().astype(np.int32)
    if vertex_colors is not None:
        if isinstance(vertex_colors, torch.Tensor):
            vertex_colors = vertex_colors.cpu().numpy()  
    
    trimesh.Trimesh(vertices=vertices, faces=faces, vertex_colors=vertex_colors, process=False).export(path)

def split_mesh(mesh: Mesh,  v_mask: torch.Tensor):
    """
    Splits a mesh into two separate meshes based on given vertex indices.

    Parameters:
    - mesh (Mesh):  
    - v_mask (set or list): Indices of vertices belonging to the first split mesh.

    Returns:
    - (vertices1, faces1), (vertices2, faces2): Tuple of two split meshes.
    """ 
    # Create masks for vertex ownership
    is_in_mesh1 = v_mask.bool()  
    is_in_mesh2 = ~is_in_mesh1  # The rest belong to the second mesh

    # Create face masks
    face_mask1 = torch.any(is_in_mesh1[mesh.f], dim=1)
    face_mask2 = torch.any(is_in_mesh2[mesh.f], dim=1)
    faces1 = mesh.f[face_mask1]
    faces2 = mesh.f[face_mask2]

    # Ensure all vertices are included (even if not part of a face)
    vert_ids1 = torch.unique(torch.cat([faces1.flatten(), torch.nonzero(is_in_mesh1, as_tuple=True)[0]]))
    vert_ids2 = torch.unique(torch.cat([faces2.flatten(), torch.nonzero(is_in_mesh2, as_tuple=True)[0]]))
    
    # Create mapping from old indices to new ones
    id_map1 = torch.full((mesh.v.shape[0],), -1, dtype=torch.long, device=mesh.v.device)
    id_map1[vert_ids1] = torch.arange(len(vert_ids1), device=mesh.v.device)

    id_map2 = torch.full((mesh.v.shape[0],), -1, dtype=torch.long, device=mesh.v.device)
    id_map2[vert_ids2] = torch.arange(len(vert_ids2), device=mesh.v.device)

    # Re-index faces
    faces1 = id_map1[faces1]
    faces2 = id_map2[faces2]

    # Extract new vertices ensuring all original ones are included
    vertices1 = mesh.v[vert_ids1]
    vertices2 = mesh.v[vert_ids2]
    
    mesh1 = Mesh(v=vertices1, f=faces1.int(), device=mesh.v.device)
    mesh2 = Mesh(v=vertices2, f=faces2.int(), device=mesh.v.device)
    
    if mesh.vt is not None: 
        mesh1.vt = mesh.vt[vert_ids1]
        mesh2.vt = mesh.vt[vert_ids2]
    
    if mesh.ft is not None:
        mesh1.ft = id_map1[mesh.ft[face_mask1].int()]
        mesh2.ft = id_map2[mesh.ft[face_mask2].int()]

    return mesh1, mesh2, [vert_ids1, vert_ids2]


def generate_uv(vertices, faces, mapping=False): 
    if isinstance(vertices, torch.Tensor):
        vertices = vertices.detach().cpu().numpy()
    if isinstance(faces, torch.Tensor):
        faces = faces.detach().cpu().numpy()

    atlas = xatlas.Atlas()
    atlas.add_mesh(vertices, faces)
    chart_options = xatlas.ChartOptions()
    chart_options.max_iterations = 4
    atlas.generate(chart_options=chart_options)
    _, ft, vt = atlas[0]  # [N], [M, 3], [N, 2]  
    return vt, ft



def simple_clean_mesh(vertices, faces, apply_smooth=True, stepsmoothnum=1, apply_sub_divide=False, sub_divide_threshold=0.25):
    if isinstance(vertices, torch.Tensor):
        vertices = vertices.detach().cpu().numpy()
    if isinstance(faces, torch.Tensor):
        faces = faces.detach().cpu().numpy()
    mesh = ml.Mesh(
        vertex_matrix=vertices.astype(np.float64),
        face_matrix=faces.astype(np.int32),
    )
    ms = ml.MeshSet()
    ms.add_mesh(mesh, "cube_mesh")
    
    if apply_smooth:
        ms.apply_filter("apply_coord_laplacian_smoothing", stepsmoothnum=stepsmoothnum, cotangentweight=False)
    if apply_sub_divide:    # 5s, slow
        ms.apply_filter("meshing_repair_non_manifold_vertices")
        ms.apply_filter("meshing_repair_non_manifold_edges", method='Remove Faces')
        ms.apply_filter("meshing_surface_subdivision_loop", iterations=2, threshold=ml.PercentageValue(sub_divide_threshold))
    
    mesh = ms.current_mesh() 

    return mesh.vertex_matrix(), mesh.face_matrix()


def compute_normal(vertices, faces): 
    """ calculate the vertex normals """
    i0, i1, i2 = faces[:, 0].long(), faces[:, 1].long(), faces[:, 2].long()
    v0, v1, v2 = vertices[i0, :], vertices[i1, :], vertices[i2, :]

    face_normals = torch.cross(v1 - v0, v2 - v0, dim=-1)

    # Splat face normals to vertices
    vn = torch.zeros_like(vertices)
    vn.scatter_add_(0, i0[:, None].repeat(1, 3), face_normals)
    vn.scatter_add_(0, i1[:, None].repeat(1, 3), face_normals)
    vn.scatter_add_(0, i2[:, None].repeat(1, 3), face_normals)

    # Normalize, replace zero (degenerated) normals with some default value
    vn = torch.where(
        dot(vn, vn) > 1e-20,
        vn,
        torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32, device=vn.device),
    )
    vn = safe_normalize(vn)
    return vn


def merge_meshes(v1, f1, v2, f2):
    """
    merge two meshes 
    """
    v = torch.cat([v1, v2], dim=0) 
    f = torch.cat([f1, f2+v1.shape[0]], dim=0)
    return v, f


def construct_new_mesh(vertices, faces, v_mask):  
    v = vertices[v_mask]  
    vids = torch.argwhere(v_mask).flatten()
    f = torch.searchsorted(vids, faces) 
    return v, f 


def normalize_vertices(vertices, bound=1.0, max_dim=None, return_params=False):
    """
    normalize vertices to [-1, 1]
    
    Args:
        vertices (np.ndarray or torch.Tensor): mesh vertices, float [N, 3]
        bound (float, optional): the bounding box size. Defaults to 1.0.
        max_dim (float, optional): the maximum dimension of the bounding box. Defaults to None.
    Returns:
        np.ndarray or torch.Tensor: normalized vertices.
    """
    if isinstance(vertices, torch.Tensor): 
        vmax = vertices.max(0)[0]
        vmin = vertices.min(0)[0] 
    elif isinstance(vertices, np.ndarray): 
        vmax = vertices.max(0)
        vmin = vertices.min(0)
    else:
        raise ValueError("vertices must be either torch.Tensor or np.ndarray")
    center = (vmax + vmin) / 2
    
    if max_dim is not None:
        scale = 2 * bound / (vmax - vmin)[max_dim]
    else:
        scale = 2 * bound / (vmax - vmin).max()
    v = (vertices - center) * scale
    
    if return_params:
        return v, center, scale
    
    return v 
    
    