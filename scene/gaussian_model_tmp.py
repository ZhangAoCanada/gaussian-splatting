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
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH, SH2RGB
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation

from scene.tools import *
import cv2

class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize


    def __init__(self, sh_degree : int):
        self.z_near = 0.1
        self.z_far = 100.0
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree  
        self._xyz = torch.empty(0)
        self._xyz_tensor = torch.empty(0)
        self._rays_o = torch.empty(0)
        self._rays_d = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._features_dc_tensor = torch.empty(0)
        self._features_rest_tensor = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._scaling_tensor = torch.empty(0)
        self._rotation_tensor = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.setup_functions()

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            # self.rays_o, 
            # self.rays_d,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )
    
    def restore(self, model_args, training_args):
        (self.active_sh_degree, 
        self._xyz, 
        # self.rays_o, 
        # self.rays_d,
        self._features_dc, 
        self._features_rest,
        self._scaling, 
        self._rotation, 
        self._opacity,
        self.max_radii2D, 
        xyz_gradient_accum, 
        denom,
        opt_dict, 
        self.spatial_lr_scale) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    @property
    def get_scaling(self):
        # return self.scaling_activation(self._scaling)
        return self.scaling_activation(self._scaling_tensor)
    
    @property
    def get_rotation(self):
        # return self.rotation_activation(self._rotation)
        return self.rotation_activation(self._rotation_tensor)
    
    @property
    def get_xyz(self):
        # return self._xyz
        return self._xyz_tensor
        # z = torch.sigmoid(self._xyz) * (self.z_far - self.z_near) + self.z_near
        # return self._rays_o + self._rays_d * z
    
    # @property
    # def get_xyz(self):
    #     return self._rays_o + self._rays_d * torch.exp(self._xyz)
    
    @property
    def get_features(self):
        # features_dc = self._features_dc
        # features_rest = self._features_rest
        features_dc = self._features_dc_tensor
        features_rest = self._features_rest_tensor
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_opacity(self):
        # return self.opacity_activation(self._opacity)
        W = 979
        H = 546
        downsample_ratio = 8
        W, H = W // downsample_ratio, H // downsample_ratio
        N_sample = 100
        if self._opacity.shape[0] == W * H * N_sample:
            return self._opacity.view(H, W, N_sample).softmax(dim=-1).view(-1, 1)
        return self.opacity_activation(self._opacity)

    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    #########################################################################
    #########################################################################
    #########################################################################
    def create_with_all_viewingrays(self, cams, spatial_lr_scale):
        self.spatial_lr_scale = spatial_lr_scale

        downsample_ratio = 8
        xyz = []
        rays_origin = []
        rays_direction = []
        features_dc = []
        features_rest = []
        scaling = []
        rotation = []
        opacity = []

        for cam in cams:
            W, H = cam.image_width, cam.image_height
            W, H = W // downsample_ratio, H // downsample_ratio

            fov_x, fov_y = cam.FoVx, cam.FoVy
            focal_x = fov2focal(fov_x, W)
            focal_y = fov2focal(fov_y, H)
            w2c = cam.world_view_transform.transpose(0, 1)

            rays_o, rays_d = get_rays_wh(H, W, focal_y, focal_x, w2c.inverse())
            rays_o, rays_d = rays_o.float(), rays_d.float()
            rays_o = rays_o + rays_d * downsample_ratio ### NOTE: origin on image plane
            rays_o = rays_o.reshape(-1, 3)
            rays_d = rays_d.reshape(-1, 3)

            gt_image = cam.original_image.cuda()
            gt_color = torch.nn.functional.interpolate(gt_image.unsqueeze(0), (H, W), mode='bilinear', align_corners=False).squeeze(0)
            gt_color = gt_color.permute(1, 2, 0).reshape(-1, 3)
            
            if rays_o.shape[0] == 0:
                return

            rays_distance = torch.randn((rays_o.shape[0], 1), device="cuda") * 0.1
            new_xyz_points = rays_o + rays_d * torch.exp(rays_distance)

            fused_color = np.random.normal(0, 1.0, size=[rays_distance.shape[0], 3])
            fused_color = RGB2SH(gt_color)
            features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
            features[:, :3, 0 ] = fused_color
            features[:, 3:, 1:] = 0.0

            dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(new_xyz_points.cpu())).float().cuda()), 0.0000001)
            scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
            rots = torch.zeros((rays_distance.shape[0], 4), device="cuda")
            rots[:, 0] = 1

            opacities = inverse_sigmoid(0.1 * torch.ones((rays_distance.shape[0], 1), dtype=torch.float, device="cuda"))

            xyz.append(rays_distance)
            rays_origin.append(rays_o)
            rays_direction.append(rays_d)
            features_dc.append(features[:,:,0:1].transpose(1, 2).contiguous())
            features_rest.append(features[:,:,1:].transpose(1, 2).contiguous())
            scaling.append(scales)
            rotation.append(rots)
            opacity.append(opacities)

        xyz = torch.cat(xyz, dim=0)
        rays_origin = torch.cat(rays_origin, dim=0)
        rays_direction = torch.cat(rays_direction, dim=0)
        features_dc = torch.cat(features_dc, dim=0)
        features_rest = torch.cat(features_rest, dim=0)
        scaling = torch.cat(scaling, dim=0)
        rotation = torch.cat(rotation, dim=0)
        opacity = torch.cat(opacity, dim=0)

        self._xyz = nn.Parameter(xyz.requires_grad_(True))
        self._rays_o = rays_origin
        self._rays_d = rays_direction
        self._features_dc = nn.Parameter(features_dc.requires_grad_(True))
        self._features_rest = nn.Parameter(features_rest.requires_grad_(True))
        self._scaling = nn.Parameter(scaling.requires_grad_(True))
        self._rotation = nn.Parameter(rotation.requires_grad_(True))
        self._opacity = nn.Parameter(opacity.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

        print("Number of points at initialized : ", self._xyz.shape[0])
        

    def create_simple(self, pcd : BasicPointCloud, spatial_lr_scale : float):
        self.spatial_lr_scale = spatial_lr_scale
        points_scale = np.random.normal(0, 1.0, size=[1, 1])
        fused_color = np.random.normal(0, 1.0, size=[1, 3])
        fused_point_cloud = torch.tensor(points_scale).float().cuda()
        fused_color = RGB2SH(torch.tensor(fused_color).float().cuda())
        rays_o = torch.randn((1, 3), device="cuda").float()
        rays_d = torch.randn((1, 3), device="cuda").float()
        points = rays_o + rays_d * torch.exp(fused_point_cloud)
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0
        print("Number of points at initialisation : ", fused_point_cloud.shape[0])
        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(points.cpu())).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1
        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))
        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._rays_o = rays_o
        self._rays_d = rays_d
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda") 


    def create_from_ply(self, ply_path : str, spatial_lr_scale : float):
        self.spatial_lr_scale = spatial_lr_scale
        ### NOTE: load from ply ###
        plydata = PlyData.read(ply_path)
        points = np.stack((np.asarray(plydata.elements[0]["x"]), np.asarray(plydata.elements[0]["y"]), np.asarray(plydata.elements[0]["z"])), axis=1).astype(np.float32)
        fused_color = np.stack((np.asarray(plydata.elements[0]["red"]), np.asarray(plydata.elements[0]["green"]), np.asarray(plydata.elements[0]["blue"])), axis=1).astype(np.float32)
        fused_color = fused_color / 255.0
        ### NOTE: normal distribution ###
        # points = np.random.normal(0, 100.0, size=[5000000, 3])
        # fused_color = np.random.normal(0, 1.0, size=[5000000, 3])
        # points = np.random.uniform(-100.0, 100.0, size=[5000000, 3])
        # fused_color = np.random.uniform(0, 1.0, size=[5000000, 3])
        #################################
        fused_point_cloud = torch.tensor(points).float().cuda()
        fused_color = RGB2SH(torch.tensor(fused_color).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
    #########################################################################
    #########################################################################
    #########################################################################


    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            # {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            # {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            # {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            # {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            # {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        # for param_group in self.optimizer.param_groups:
        #     if param_group["name"] == "xyz":
        #         lr = self.xyz_scheduler_args(iteration)
        #         param_group['lr'] = lr
        #         return lr
        lr = self.xyz_scheduler_args(iteration)
        return lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def reset_opacity(self):
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

        self.active_sh_degree = self.max_sh_degree

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            # if group["name"] in ["xyz", "f_dc", "f_rest"]:
            if group["name"] in ["f_dc", "f_rest", "scaling", "rotation"]:
                continue
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        # self._xyz = optimizable_tensors["xyz"]
        # self._features_dc = optimizable_tensors["f_dc"]
        # self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        # self._scaling = optimizable_tensors["scaling"]
        # self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation):
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)

        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation)

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()

    ########################################################################
    ########################################################################
    ########################################################################
    # def ray_points_densify(self, image, gt_image, cam, source_path, visibility_filter, max_grad, min_opacity, extent, size_threshold):
    #     downsample_ratio = 8
    #     ssim_threshold = 0.3

    #     ssim_mask = ssim_nograd(image.detach(), gt_image.detach())
    #     W, H = cam.image_width, cam.image_height
    #     W, H = W // downsample_ratio, H // downsample_ratio

    #     fov_x, fov_y = cam.FoVx, cam.FoVy
    #     focal_x = fov2focal(fov_x, W)
    #     focal_y = fov2focal(fov_y, H)
    #     w2c = cam.world_view_transform.transpose(0, 1)

    #     # depth_map = cv2.imread(os.path.join(source_path, "depth", f"{cam.image_name}.png"), cv2.IMREAD_UNCHANGED)
    #     # depth_map = cv2.resize(depth_map, (W, H), interpolation=cv2.INTER_NEAREST)
    #     # depth_map = depth_map / 255. * 20.
    #     # depth_tensor = torch.from_numpy(depth_map).float().cuda().unsqueeze(-1)

    #     ssim_mask = torch.nn.functional.interpolate(ssim_mask.unsqueeze(0).unsqueeze(0), (H, W), mode='bilinear', align_corners=False).squeeze(0).squeeze(0)
    #     gt_color = torch.nn.functional.interpolate(gt_image.clone().unsqueeze(0), (H, W), mode='bilinear', align_corners=False).squeeze(0)

    #     rays_o, rays_d = get_rays_wh(H, W, focal_y, focal_x, w2c.inverse())
    #     rays_o, rays_d = rays_o.float(), rays_d.float()
    #     # rays_o = rays_o + rays_d * downsample_ratio ### NOTE: origin on image plane
    #     rays_o = rays_o.reshape(-1, 3)
    #     rays_d = rays_d.reshape(-1, 3)
    #     # depth_tensor = depth_tensor.reshape(-1, 1)
    #     ssim_mask = ssim_mask.reshape(-1)
    #     gt_color = gt_color.permute(1, 2, 0).reshape(-1, 3)

    #     mask = ssim_mask < ssim_threshold
    #     rays_o = rays_o[mask]
    #     rays_d = rays_d[mask]
    #     gt_color = gt_color[mask]
    #     # depth_tensor = depth_tensor[mask]

    #     # mask = torch.ones(rays_o.shape[0], dtype=bool, device="cuda")
    #     # for ray_o in self._rays_o:
    #     #     mask = torch.logical_and(mask, torch.norm(rays_o - ray_o, dim=-1) > 0.0001)

    #     # if len(self._rays_o[visibility_filter]) != 0:
    #     #     mask = self._rays_o[visibility_filter].unsqueeze(1) - rays_o.unsqueeze(0)
    #     #     mask = torch.norm(mask, dim=-1).min(dim=0).values > 0.0001
    #     #     rays_o = rays_o[mask]
    #     #     rays_d = rays_d[mask]
    #     #     gt_color = gt_color[mask]
    #     #     depth_tensor = depth_tensor[mask]

    #     if rays_o.shape[0] == 0:
    #         return

    #     rays_distance = torch.randn((rays_o.shape[0], 1), device="cuda") * 0.1
    #     # rays_distance = torch.tensor(torch.log(depth_tensor + 1e-6), device="cuda")
    #     new_xyz_points = rays_o + rays_d * torch.exp(rays_distance)

    #     fused_color = np.random.normal(0, 1.0, size=[1, 3])
    #     fused_color = RGB2SH(gt_color)
    #     features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
    #     features[:, :3, 0 ] = fused_color
    #     features[:, 3:, 1:] = 0.0

    #     print("Number of points at densified : ", rays_distance.shape[0])

    #     dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(new_xyz_points.cpu())).float().cuda()), 0.0000001)
    #     scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
    #     rots = torch.zeros((rays_distance.shape[0], 4), device="cuda")
    #     rots[:, 0] = 1

    #     opacities = inverse_sigmoid(0.1 * torch.ones((rays_distance.shape[0], 1), dtype=torch.float, device="cuda"))

    #     new_xyz = nn.Parameter(rays_distance.requires_grad_(True))
    #     self._rays_o = torch.cat((self._rays_o, rays_o), dim=0)
    #     self._rays_d = torch.cat((self._rays_d, rays_d), dim=0)
    #     new_features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
    #     new_features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
    #     new_scaling = nn.Parameter(scales.requires_grad_(True))
    #     new_rotation = nn.Parameter(rots.requires_grad_(True))
    #     new_opacities = nn.Parameter(opacities.requires_grad_(True))
    #     self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation)


    def create_with_firstview(self, cam_1st, spatial_lr_scale):
        self.spatial_lr_scale = spatial_lr_scale
        downsample_ratio = 4
        near = 1
        far = 100.0
        N_sample = 100

        W, H = cam_1st.image_width, cam_1st.image_height
        W, H = W // downsample_ratio, H // downsample_ratio

        fov_x, fov_y = cam_1st.FoVx, cam_1st.FoVy
        focal_x = fov2focal(fov_x, W)
        focal_y = fov2focal(fov_y, H)
        w2c = cam_1st.world_view_transform.transpose(0, 1)
        # w2c = torch.tensor(getWorld2View(cam_1st.R, cam_1st.T), device="cuda")
        # c2w = torch.inverse(w2c)

        image = cam_1st.original_image.cuda()
        color = torch.nn.functional.interpolate(image.unsqueeze(0), (H, W), mode='bilinear', align_corners=False).squeeze(0).permute(1, 2, 0)

        rays_o, rays_d = get_rays_wh(H, W, focal_y, focal_x, w2c.inverse())
        rays_o, rays_d = rays_o.float(), rays_d.float()

        # z_val = torch.randn((H, W, 1), device="cuda") + 1.
        # z_pred = torch.sigmoid(z_val) * (self.z_far - self.z_near) + self.z_near
        # pts = rays_o + rays_d * z_pred

        z_val = torch.linspace(near, far, N_sample, device="cuda")
        pts = rays_o[..., None, :] + rays_d[..., None, :] * z_val[..., :, None]
        color = color[:, :, None, :].repeat(1, 1, N_sample, 1)

        pts = pts.view(-1, 3)
        color = color.view(-1, 3)
        # plot_points(pts, color, "rays_check")
        # self._rays_o = rays_o.view(-1, 3).float().contiguous().cuda()
        # self._rays_d = rays_d.view(-1, 3).float().contiguous().cuda()

        fused_color = RGB2SH(color)
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pts.cpu())).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        rots = torch.zeros((pts.shape[0], 4), device="cuda")
        rots[:, 0] = 1
        opacities = inverse_sigmoid(0.1 * torch.ones((pts.shape[0], 1), dtype=torch.float, device="cuda"))

        # self._rays_o = rays_o
        # self._rays_d = rays_d
        # self._xyz = nn.Parameter(pts.requires_grad_(True))
        # self._xyz = nn.Parameter(torch.randn((pts.shape[0], 1), device="cuda").requires_grad_(True))
        self._xyz_tensor = pts
        # self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        # self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_dc_tensor = features[:,:,0:1].transpose(1, 2).contiguous()
        self._features_rest_tensor = features[:,:,1:].transpose(1, 2).contiguous()
        # self._scaling = nn.Parameter(scales.requires_grad_(True))
        # self._rotation = nn.Parameter(rots.requires_grad_(True))
        # self._scaling_tensor = torch.ones((pts.shape[0], 3), device="cuda") * 0.01
        self._scaling_tensor = torch.log(torch.ones((pts.shape[0], 3), device="cuda") * 0.001)
        # self._scaling_tensor = scales
        self._rotation_tensor = rots
        # self._opacity = nn.Parameter(torch.randn((pts.shape[0], 1), device="cuda").requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        print("Number of points at initialized : ", self._xyz.shape[0])


    def prune_ray(self, cam, visibility_mask, max_grad, min_opacity, extent, max_screen_size):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        downsample_ratio = 4
        N_sample = 100
        W, H = cam.image_width, cam.image_height
        W, H = W // downsample_ratio, H // downsample_ratio

        opacity = self.get_opacity
        opacity = opacity.view(H, W, N_sample)

        max_opacity = torch.max(opacity, dim=-1).values
        mask = opacity == max_opacity.unsqueeze(-1)
        mask = mask.view(-1)
        mask = torch.logical_and(mask, visibility_mask)

        self._xyz_tensor = self._xyz_tensor[mask]
        self._features_dc_tensor = self._features_dc_tensor[mask]
        self._features_rest_tensor = self._features_rest_tensor[mask]
        self._scaling_tensor = self._scaling_tensor[mask]
        self._rotation_tensor = self._rotation_tensor[mask]
        self.prune_points(~mask)

        ################################################################
        points = self.get_xyz
        colors = SH2RGB(self.get_features.transpose(1,2)[:, :3, 0])
        plot_points(points, colors)

        world_view_transform = cam.world_view_transform.cuda().t()
        R = world_view_transform[:3, :3]
        T = world_view_transform[:3, 3]
        points_view = R.matmul(points.unsqueeze(-1)).squeeze(-1) + T.unsqueeze(0)

        full_proj_transform = cam.full_proj_transform.cuda().t()
        points = torch.cat([points, torch.ones((points.shape[0], 1), device="cuda")], dim=-1)
        points = full_proj_transform.matmul(points.t()).t()
        point_star = points[:, :3].clone()
        points[:, :3] = points[:, :3] / points[:, 3:4]
        points = points[:, :3]

        W, H = cam.image_width, cam.image_height
        S = torch.tensor([W, H], device="cuda", dtype=torch.float)
        points_2d = ndc2pix(points[:, :2], S)
        points_2d = points_2d.round().long()
        mask = (points_2d[:, 0] >= 0) & (points_2d[:, 0] < W) & (points_2d[:, 1] >= 0) & (points_2d[:, 1] < H)
        points_2d = points_2d[mask]
        points = points[mask]
        points_view = points_view[mask]
        point_star = point_star[mask]
        depth_map = torch.zeros((H, W), device="cuda")
        depth_map[points_2d[:, 1], points_2d[:, 0]] = point_star[:, 2]
        # depth_map[points_2d[:, 1], points_2d[:, 0]] = points_view[:, 2]
        depth_map = depth_map.detach()
        depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())
        depth_map = depth_map[::downsample_ratio, ::downsample_ratio]
        depth_map = depth_map.unsqueeze(-1)
        depth_map_np = depth_map.cpu().numpy()
        depth_map_np = (depth_map_np * 255).astype(np.uint8)
        cv2.imwrite("output/depth_map.png", depth_map_np)
        #################################################################
        
        torch.cuda.empty_cache()


    def ray_points_densify(self, image, gt_image, cam, viewspace_point_tensor, visibility_filter, source_path):
        downsample_ratio = 8
        near = 0.1  
        far = 100.0
        N_sample = 1000
        ssim_threshold = 0.5

        ssim_mask = ssim_nograd(image.detach(), gt_image.detach())
        W, H = cam.image_width, cam.image_height
        W, H = W // downsample_ratio, H // downsample_ratio
        ssim_mask = torch.nn.functional.interpolate(ssim_mask.unsqueeze(0).unsqueeze(0), (H, W), mode='bilinear', align_corners=False).squeeze(0).squeeze(0)

        viewspace_point_tensor = viewspace_point_tensor.view(H, W, N_sample, 3)
        visibility_filter = visibility_filter.view(H, W, N_sample)

        fov_x, fov_y = cam.FoVx, cam.FoVy
        focal_x = fov2focal(fov_x, W)
        focal_y = fov2focal(fov_y, H)
        w2c = cam.world_view_transform.transpose(0, 1)

        gt_color = torch.nn.functional.interpolate(gt_image.detach().clone().unsqueeze(0), (H, W), mode='bilinear', align_corners=False).squeeze(0)

        rays_o, rays_d = get_rays_wh(H, W, focal_y, focal_x, w2c.inverse())
        rays_o, rays_d = rays_o.float(), rays_d.float()
        # rays_o = rays_o + rays_d * downsample_ratio ### NOTE: origin on image plane
        rays_o = rays_o.reshape(-1, 3)
        rays_d = rays_d.reshape(-1, 3)
        # depth_tensor = depth_tensor.reshape(-1, 1)
        ssim_mask = ssim_mask.reshape(-1)
        gt_color = gt_color.permute(1, 2, 0).reshape(-1, 3)

        mask = ssim_mask < ssim_threshold
        rays_o = rays_o[mask]
        rays_d = rays_d[mask]
        gt_color = gt_color[mask]
        # depth_tensor = depth_tensor[mask]

        if rays_o.shape[0] == 0:
            return

        rays_distance = torch.randn((rays_o.shape[0], 1), device="cuda") * 0.1
        # rays_distance = torch.tensor(torch.log(depth_tensor + 1e-6), device="cuda")
        new_xyz_points = rays_o + rays_d * torch.exp(rays_distance)

        fused_color = np.random.normal(0, 1.0, size=[1, 3])
        fused_color = RGB2SH(gt_color)
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at densified : ", rays_distance.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(new_xyz_points.cpu())).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        rots = torch.zeros((rays_distance.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((rays_distance.shape[0], 1), dtype=torch.float, device="cuda"))

        new_xyz = nn.Parameter(rays_distance.requires_grad_(True))
        self._rays_o = torch.cat((self._rays_o, rays_o), dim=0)
        self._rays_d = torch.cat((self._rays_d, rays_d), dim=0)
        new_features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        new_features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        new_scaling = nn.Parameter(scales.requires_grad_(True))
        new_rotation = nn.Parameter(rots.requires_grad_(True))
        new_opacities = nn.Parameter(opacities.requires_grad_(True))
        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation)



    def prune_only(self, max_grad, min_opacity, extent, max_screen_size):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()
    ########################################################################
    ########################################################################
    ########################################################################


    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1