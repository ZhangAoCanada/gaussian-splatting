import os
import torch
import torch.nn.functional as F
import numpy as np
import math
import matplotlib.pyplot as plt
import cv2

def get_rays(H, W, focal, c2w):
    """
    H: int, can be as H / downscale
    W: int, can be as W / downscale
    focal: float, can be as focal / downscale
    """
    i, j = torch.meshgrid(torch.linspace(0, W - 1, W), torch.linspace(0, H - 1, H))
    i = i.t()
    j = j.t()
    dirs = torch.stack([(i - W * .5) / focal, -(j - H * .5) / focal, -torch.ones_like(i)], -1)
    rays_d = torch.sum(dirs[..., None, :] * c2w[:3, :3], -1)
    rays_o = c2w[:3, -1].expand(rays_d.shape)
    return rays_o, rays_d


def get_rays_wh(H, W, focal_y, focal_x, c2w):
    """
    H: int, can be as H / downscale
    W: int, can be as W / downscale
    focal_y: float, can be as focal_y / downscale
    focal_x: float, can be as focal_x / downscale
    """
    i, j = torch.meshgrid(torch.linspace(0, W - 1, W), torch.linspace(0, H - 1, H))
    i = i.t()
    j = j.t()
    # dirs = torch.stack([(i - W * .5) / focal_x, -(j - H * .5) / focal_y, -torch.ones_like(i)], -1).cuda()
    dirs = torch.stack([(i - W * .5) / focal_x, (j - H * .5) / focal_y, torch.ones_like(i)], -1).cuda()
    rays_d = torch.sum(dirs[..., None, :] * c2w[:3, :3], -1)
    rays_o = c2w[:3, -1].expand(rays_d.shape)
    # R = c2w[:3, :3]
    # rays_d = R[None, None, :, :].expand(H, W, 3, 3) @ dirs[..., None]
    # rays_d = rays_d.squeeze(-1)
    # rays_o = c2w[:3, -1].expand(rays_d.shape)
    return rays_o, rays_d


def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))


def focal2fov(focal, pixels):
    return 2*math.atan(pixels/(2*focal))


def ndc2pix(v, S):
    return ((v + 1) * S - 1) / 2


def rotation_align_vectors(v1, v2):
    v1 = v1 / torch.norm(v1, dim=-1, keepdim=True)
    v2 = v2 / torch.norm(v2, dim=-1, keepdim=True)
    v = torch.cross(v1, v2)
    s = torch.norm(v, dim=-1)
    c = torch.dot(v1, v2)
    # c = torch.sum(v1 * v2, dim=-1)
    v_skew = torch.tensor([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]], device=v.device)
    R = torch.eye(3, device=v.device) + v_skew + v_skew @ v_skew * (1 - c) / (s ** 2)
    return R

def rotation_align_vectors_batch(v1, v2):
    v1 = v1 / torch.norm(v1, dim=-1, keepdim=True)
    v2 = v2 / torch.norm(v2, dim=-1, keepdim=True)
    v = torch.cross(v1, v2)
    s = torch.norm(v, dim=-1)
    c = torch.sum(v1 * v2, dim=-1)
    v_skew = torch.zeros((v1.shape[0], 3, 3), device=v1.device)
    v_skew[:, 0, 1] = -v[:, 2]
    v_skew[:, 0, 2] = v[:, 1]
    v_skew[:, 1, 0] = v[:, 2]
    v_skew[:, 1, 2] = -v[:, 0]
    v_skew[:, 2, 0] = -v[:, 1]
    v_skew[:, 2, 1] = v[:, 0]
    R = torch.eye(3, device=v.device).unsqueeze(0).expand(v1.shape[0], 3, 3) + v_skew + v_skew @ v_skew * ((1 - c) / (s ** 2))[..., None, None]
    return R

def quaternion2rotation(r):
    norm = torch.sqrt(r[:,0]*r[:,0] + r[:,1]*r[:,1] + r[:,2]*r[:,2] + r[:,3]*r[:,3])

    q = r / norm[:, None]

    R = torch.zeros((q.size(0), 3, 3), device='cuda')

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y*y + z*z)
    R[:, 0, 1] = 2 * (x*y - r*z)
    R[:, 0, 2] = 2 * (x*z + r*y)
    R[:, 1, 0] = 2 * (x*y + r*z)
    R[:, 1, 1] = 1 - 2 * (x*x + z*z)
    R[:, 1, 2] = 2 * (y*z - r*x)
    R[:, 2, 0] = 2 * (x*z - r*y)
    R[:, 2, 1] = 2 * (y*z + r*x)
    R[:, 2, 2] = 1 - 2 * (x*x + y*y)
    return R

def rotation2quaternion(R):
    q = torch.zeros((R.size(0), 4), device='cuda')
    q[:, 0] = 0.5 * torch.sqrt(1 + R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2])
    q[:, 1] = (R[:, 2, 1] - R[:, 1, 2]) / (4 * q[:, 0])
    q[:, 2] = (R[:, 0, 2] - R[:, 2, 0]) / (4 * q[:, 0])
    q[:, 3] = (R[:, 1, 0] - R[:, 0, 1]) / (4 * q[:, 0])
    return q


def gaussian(window_size, sigma):
    gauss = torch.Tensor([math.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = torch.tensor(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def ssim_nograd(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim_nograd(img1, img2, window, window_size, channel, size_average)


def _ssim_nograd(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    return ssim_map.mean(0)


def getWorld2View(R, t):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0
    return np.float32(Rt)


def plot_points(points, colors, name="points", image_name=None):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    points_np = points.detach().cpu().numpy()
    colors_np = colors.detach().clamp(0., 1.).cpu().numpy()
    ax.scatter(points_np[:, 0], points_np[:, 2], -points_np[:, 1], s=0.2, c=colors_np)
    if image_name == None:
        plt.savefig(f"output/{name}.png")
    else:
        save_dir = f"output/{name}"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        plt.savefig(f"{save_dir}/{image_name}.png")


def quickfind(points, points_view, colors, ranks):
    kept = torch.ones(points.shape[0], device=points.device, dtype=torch.bool)
    kept[:-1] = (ranks[1:] != ranks[:-1])

    points, points_view, colors = points[kept], points_view[kept], colors[kept]

    return points, points_view, colors


def show_depth_map(points_2d, points_view, H, W, name="depth_map"):
    depth_map = torch.zeros((H, W), device="cuda")
    depth_map[points_2d[:, 1], points_2d[:, 0]] = points_view[:, 2]
    depth_map1 = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())
    depth_map1 = (depth_map1.unsqueeze(-1).cpu().numpy() * 255.).astype(np.uint8)
    cv2.imwrite(f"output/{name}.png", depth_map1)
    depth_map_log = torch.log2(depth_map.detach() + 1.)
    depth_map_log = (depth_map_log - depth_map_log.min()) / (depth_map_log.max() - depth_map_log.min())
    depth_map_log = (depth_map_log.unsqueeze(-1).cpu().numpy() * 255.).astype(np.uint8)
    cv2.imwrite(f"output/{name}_log.png", depth_map_log)
    depth_map2 = cv2.applyColorMap(depth_map1, cv2.COLORMAP_JET)
    cv2.imwrite(f"output/{name}_color.png", depth_map2)

