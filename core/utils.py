import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import roma
from kiui.op import safe_normalize
import cv2
from basicsr.utils import img2tensor

def get_rays(pose, h, w, fovy, opengl=True):

    x, y = torch.meshgrid(
        torch.arange(w, device=pose.device),
        torch.arange(h, device=pose.device),
        indexing="xy",
    )
    x = x.flatten()
    y = y.flatten()

    cx = w * 0.5
    cy = h * 0.5

    focal = h * 0.5 / np.tan(0.5 * np.deg2rad(fovy))

    camera_dirs = F.pad(
        torch.stack(
            [
                (x - cx + 0.5) / focal,
                (y - cy + 0.5) / focal * (-1.0 if opengl else 1.0),
            ],
            dim=-1,
        ),
        (0, 1),
        value=(-1.0 if opengl else 1.0),
    )  # [hw, 3]

    rays_d = camera_dirs @ pose[:3, :3].transpose(0, 1)  # [hw, 3]
    rays_o = pose[:3, 3].unsqueeze(0).expand_as(rays_d) # [hw, 3]

    rays_o = rays_o.view(h, w, 3)
    rays_d = safe_normalize(rays_d).view(h, w, 3)

    return rays_o, rays_d

def orbit_camera_jitter(poses, strength=0.1):
    # poses: [B, 4, 4], assume orbit camera in opengl format
    # random orbital rotate

    B = poses.shape[0]
    rotvec_x = poses[:, :3, 1] * strength * np.pi * (torch.rand(B, 1, device=poses.device) * 2 - 1)
    rotvec_y = poses[:, :3, 0] * strength * np.pi / 2 * (torch.rand(B, 1, device=poses.device) * 2 - 1)

    rot = roma.rotvec_to_rotmat(rotvec_x) @ roma.rotvec_to_rotmat(rotvec_y)
    R = rot @ poses[:, :3, :3]
    T = rot @ poses[:, :3, 3:]

    new_poses = poses.clone()
    new_poses[:, :3, :3] = R
    new_poses[:, :3, 3:] = T
    
    return new_poses

def grid_distortion(images, strength=0.5):
    # images: [B, C, H, W]
    # num_steps: int, grid resolution for distortion
    # strength: float in [0, 1], strength of distortion

    B, C, H, W = images.shape

    num_steps = np.random.randint(8, 17)
    grid_steps = torch.linspace(-1, 1, num_steps)

    # have to loop batch...
    grids = []
    for b in range(B):
        # construct displacement
        x_steps = torch.linspace(0, 1, num_steps) # [num_steps], inclusive
        x_steps = (x_steps + strength * (torch.rand_like(x_steps) - 0.5) / (num_steps - 1)).clamp(0, 1) # perturb
        x_steps = (x_steps * W).long() # [num_steps]
        x_steps[0] = 0
        x_steps[-1] = W
        xs = []
        for i in range(num_steps - 1):
            xs.append(torch.linspace(grid_steps[i], grid_steps[i + 1], x_steps[i + 1] - x_steps[i]))
        xs = torch.cat(xs, dim=0) # [W]

        y_steps = torch.linspace(0, 1, num_steps) # [num_steps], inclusive
        y_steps = (y_steps + strength * (torch.rand_like(y_steps) - 0.5) / (num_steps - 1)).clamp(0, 1) # perturb
        y_steps = (y_steps * H).long() # [num_steps]
        y_steps[0] = 0
        y_steps[-1] = H
        ys = []
        for i in range(num_steps - 1):
            ys.append(torch.linspace(grid_steps[i], grid_steps[i + 1], y_steps[i + 1] - y_steps[i]))
        ys = torch.cat(ys, dim=0) # [H]

        # construct grid
        grid_x, grid_y = torch.meshgrid(xs, ys, indexing='xy') # [H, W]
        grid = torch.stack([grid_x, grid_y], dim=-1) # [H, W, 2]

        grids.append(grid)
    
    grids = torch.stack(grids, dim=0).to(images.device) # [B, H, W, 2]

    # grid sample
    images = F.grid_sample(images, grids, align_corners=False)

    return images

def resize_numpy_image(image, max_resolution=768 * 768, resize_short_edge=None):
    h, w = image.shape[:2]
    w_org = image.shape[1]
    if resize_short_edge is not None:
        k = resize_short_edge / min(h, w)
    else:
        k = max_resolution / (h * w)
        k = k**0.5
    h = int(np.round(h * k / 64)) * 64
    w = int(np.round(w * k / 64)) * 64
    image = cv2.resize(image, (w, h), interpolation=cv2.INTER_LANCZOS4)
    scale = w/w_org
    return image, scale


def process_drag(path_mask, h, w, x, y, x_cur, y_cur, scale, input_scale, up_scale, up_ft_index, w_edit, w_inpaint, w_content, precision, latent_in):
    if isinstance(path_mask, str):
        mask_x0 = cv2.imread(path_mask)
    else:
        mask_x0 = path_mask
    mask_x0 = cv2.resize(mask_x0, (h, w))
    mask_x0 = img2tensor(mask_x0)[0]
    dict_mask = {}
    dict_mask['base'] = mask_x0
    mask_x0 = (mask_x0>0.5).float().to('cuda', dtype=precision)

    mask_other = F.interpolate(mask_x0[None,None], (int(mask_x0.shape[-2]//scale), int(mask_x0.shape[-1]//scale)))<0.5
    mask_tar = []
    mask_cur = []
    for p_idx in range(len(x)):
        mask_tar_i = torch.zeros(int(mask_x0.shape[-2]//scale), int(mask_x0.shape[-1]//scale)).to('cuda', dtype=precision)
        mask_cur_i = torch.zeros(int(mask_x0.shape[-2]//scale), int(mask_x0.shape[-1]//scale)).to('cuda', dtype=precision)
        y_tar_clip = int(np.clip(y[p_idx]//scale, 1, mask_tar_i.shape[0]-2))
        x_tar_clip = int(np.clip(x[p_idx]//scale, 1, mask_tar_i.shape[0]-2))
        y_cur_clip = int(np.clip(y_cur[p_idx]//scale, 1, mask_cur_i.shape[0]-2))
        x_cur_clip = int(np.clip(x_cur[p_idx]//scale, 1, mask_cur_i.shape[0]-2))
        mask_tar_i[y_tar_clip-1:y_tar_clip+2,x_tar_clip-1:x_tar_clip+2]=1
        mask_cur_i[y_cur_clip-1:y_cur_clip+2,x_cur_clip-1:x_cur_clip+2]=1
        mask_tar_i = mask_tar_i>0.5
        mask_cur_i=mask_cur_i>0.5
        mask_tar.append(mask_tar_i)
        mask_cur.append(mask_cur_i)
        latent_in[:,:,y_cur_clip//up_scale-1:y_cur_clip//up_scale+2, x_cur_clip//up_scale-1:x_cur_clip//up_scale+2] = latent_in[:,:, y_tar_clip//up_scale-1:y_tar_clip//up_scale+2, x_tar_clip//up_scale-1:x_tar_clip//up_scale+2] 
        

    return {
        "dict_mask":dict_mask,
        "mask_x0":mask_x0,
        "mask_tar":mask_tar,
        "mask_cur":mask_cur,
        "mask_other":mask_other,
        "up_scale":up_scale,
        "up_ft_index":up_ft_index,
        "w_edit": w_edit,
        "w_inpaint": w_inpaint,
        "w_content": w_content,
        "latent_in":latent_in,
    }

