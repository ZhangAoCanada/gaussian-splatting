import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from scene.tools import *

from scene.median_pool import MedianPool2d


H = 546
W = 979
ds = 4


points = np.load("output/points.npy")
colors = np.load("output/colors.npy")
opacities = np.load("output/opacities.npy")
full_proj_transform = np.load("output/full_proj_transform.npy")
world_view_transform = np.load("output/world_view_transform.npy")

points = torch.from_numpy(points).float().cuda()
colors = torch.from_numpy(colors).float().cuda()
opacities = torch.from_numpy(opacities).float().cuda()
full_proj_transform = torch.from_numpy(full_proj_transform).float().cuda()
world_view_transform = torch.from_numpy(world_view_transform).float().cuda()


sorts = torch.argsort(opacities.view(-1), descending=True)
points = points[sorts]
colors = colors[sorts]
opacities = opacities[sorts]

world_view_transform = world_view_transform.cuda().t()
R = world_view_transform[:3, :3]
T = world_view_transform[:3, 3]
points_view = R.matmul(points.detach().unsqueeze(-1)).squeeze(-1) + T.unsqueeze(0)

full_proj_transform = full_proj_transform.cuda().t()
points = torch.cat([points, torch.ones((points.shape[0], 1), device="cuda")], dim=-1)
points = full_proj_transform.matmul(points.t()).t()
points[:, :3] = points[:, :3] / points[:, 3:4]
points = points[:, :3]

S = torch.tensor([W, H], device="cuda", dtype=torch.float)
points_2d = ndc2pix(points[:, :2], S)
points_2d = points_2d.floor().long()
mask = (points_2d[:, 0] >= 0) & (points_2d[:, 0] < W) & (points_2d[:, 1] >= 0) & (points_2d[:, 1] < H)
points_2d = points_2d[mask]
points = points[mask]
points_view = points_view[mask]
colors = colors[mask]
opacities = opacities[mask]


# W, H = W // ds, H // ds
# points_2d = points_2d // ds - 1


ranks = points_2d[:, 0] * H + points_2d[:, 1]
sorts = torch.argsort(ranks, stable=True) 
points_2d, points_view, colors, ranks = points_2d[sorts], points_view[sorts], colors[sorts], ranks[sorts]
points_2d, points_view, colors = quickfind(points_2d, points_view, colors, ranks)

show_depth_map(points_2d, points_view, H, W, "depth_map_noisy")


######## NOTE: filter out noisy points ########
mask3 = torch.logical_and(points_view[:, 2] < 100, points_view[:, 2] > 0.1)
mask4 = torch.logical_and(points_2d[:, 0] % 8 == 0, points_2d[:, 1] % 8 == 0)
mask4 = torch.logical_or(mask4, torch.logical_and(points_2d[:, 0] % 8 == 1, points_2d[:, 1] % 8 == 0))
mask4 = torch.logical_or(mask4, torch.logical_and(points_2d[:, 0] % 8 == 0, points_2d[:, 1] % 8 == 1))
mask4 = torch.logical_or(mask4, torch.logical_and(points_2d[:, 0] % 8 == 1, points_2d[:, 1] % 8 == 1))
mask4 = torch.logical_or(mask4, torch.logical_and(points_2d[:, 0] % 8 == 2, points_2d[:, 1] % 8 == 0))
mask4 = torch.logical_or(mask4, torch.logical_and(points_2d[:, 0] % 8 == 2, points_2d[:, 1] % 8 == 1))
mask3 = torch.logical_and(mask3, ~mask4)
points_view, points_2d, colors = points_view[mask3], points_2d[mask3], colors[mask3]
##############################################
# win = 7
# assert win % 2 == 1
# depth = F.pad(points_view[:, 2], ((win-1) // 2, (win-1) // 2), mode="constant", value=0)
# depth = depth.unfold(0, win, 1).median(-1).values
# points_view[:, 2] = depth
##############################################

show_depth_map(points_2d, points_view, H, W, "depth_map_filter")
mask2 = points_view[:, 2] < 20
plot_points(points_view[mask2], colors[mask2])

