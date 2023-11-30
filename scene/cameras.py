#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
from torch import nn
import numpy as np
from utils.graphics_utils import getWorld2View2, getProjectionMatrix, fov2focal, get_rays

class Camera(nn.Module):
    def __init__(self, colmap_id, R, T, FoVx, FoVy, image, gt_alpha_mask,
                 image_name, uid,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda", 
                 normal_image=None,
                 ):
        super(Camera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")

        self.original_image = image.clamp(0.0, 1.0).to(self.data_device)
        if normal_image is not None:
            self.original_normal_image = normal_image.clamp(0.0, 1.0).to(self.data_device)
            original_normal_image = self.original_normal_image * 2 - 1. # (0, 1) -> (-1, 1)
            original_normal_image = torch.nn.functional.normalize(original_normal_image, p=2, dim=0)
            fg_mask = torch.where((self.original_normal_image==0.).sum(0)<3, True, False)[None,...].repeat(3, 1, 1)
            original_normal_image = torch.where(fg_mask, original_normal_image, torch.ones_like(original_normal_image))            
            self.original_normal_image = original_normal_image
        else:
            self.original_normal_image = None
        self.image_width = self.original_image.shape[2]
        self.image_height = self.original_image.shape[1]
        self.gt_alpha_mask = gt_alpha_mask

        # if gt_alpha_mask is not None:
        if False:
            self.original_image *= gt_alpha_mask.to(self.data_device)
            if self.original_normal_image is not None:
                self.original_normal_image *= gt_alpha_mask.to(self.data_device)
        else:
            self.original_image *= torch.ones((1, self.image_height, self.image_width), device=self.data_device)
            if self.original_normal_image is not None:
                    self.original_normal_image *= torch.ones((1, self.image_height, self.image_width), device=self.data_device)

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]
    
    def get_calib_matrix_nerf(self):
        focal = fov2focal(self.FoVx, self.image_width)  # original focal length
        intrinsic_matrix = torch.tensor([[focal, 0, self.image_width / 2], [0, focal, self.image_height / 2], [0, 0, 1]]).float()
        extrinsic_matrix = self.world_view_transform.transpose(0,1).contiguous() # cam2world
        return intrinsic_matrix, extrinsic_matrix

    def get_rays(self):
        intrinsic_matrix, extrinsic_matrix = self.get_calib_matrix_nerf()

        viewdirs = get_rays(self.image_width, self.image_height, intrinsic_matrix, extrinsic_matrix[:3,:3])
        return viewdirs

class MiniCam:
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform):
        self.image_width = width
        self.image_height = height    
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]

