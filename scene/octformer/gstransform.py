import ocnn
import torch
import random
import scipy.interpolate
import scipy.ndimage
import numpy as np

from ocnn.octree import Points, Octree
from ocnn.dataset import CollateBatch
from thsolver import Dataset
from typing import List

from .utils import ReadFile, Transform

from utils.sh_utils import RGB2SH, SH2RGB, eval_sh


class gsTransform():

  def __init__(self, sh_degree):
    # super().__init__(flags)
    # The `self.scale_factor` is used to normalize the input point cloud to the
    # range of [-1, 1]. If this parameter is modified, the `self.elastic_params`
    # and the `jittor` in the data augmentation should be scaled accordingly.
    # self.scale_factor = 5.12    # depth 9: voxel size 2cm
    self.scale_factor = 10.24     # depth 10: voxel size 2cm; depth 11: voxel size 1cm
    self.sh_degree = sh_degree
    self.center = None

  def update_scale_factor(self, scale_factor):
    self.scale_factor = scale_factor

  def __call__(self, feat, normalized=False):
    """
    FEATURE2CHANNEL = {
            'means': 3,
            'features_dc': 3,
            'features_rest': 3,
            'opacities': 1,
            'scales': 3,
            'quats': 4, }
    """

    # normalize points
    xyz = feat[:, :3]

    if not normalized:
        self.center = (xyz.min(dim=0).values + xyz.max(dim=0).values) / 2.0
        xyz = (xyz - self.center) / self.scale_factor  # xyz in [-1, 1]

    # get features
    rgb = feat[:, 3:]

    # construct points [points, normals, features, labels]
    points = Points(points=xyz, features=rgb)

    return {'points': points}

  def inverse_transform(self, xyz, normalized=False):
    if not normalized:
        return xyz * self.scale_factor + self.center
    return xyz
