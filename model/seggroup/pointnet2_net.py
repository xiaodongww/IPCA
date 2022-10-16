import torch
import torch.nn as nn
import os
import sys

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ROOT_DIR)  # model
sys.path.append(os.path.join(ROOT_DIR, '../../'))

from lib.pointnet2.pointnet2_modules import PointnetSAModule, PointnetFPModule


class segmentPointNet2(nn.Module):
    def __init__(self, fea_dim=6, out_fea_dim=64, use_bn=False):
        super(segmentPointNet2, self).__init__()
        self.out_fea_dim = out_fea_dim
        self.sa1 = PointnetSAModule(mlp=[fea_dim, self.out_fea_dim], npoint=128, radius=0.4, nsample=32, bn=use_bn)
        self.sa2 = PointnetSAModule(mlp=[self.out_fea_dim, self.out_fea_dim], npoint=None, radius=None, nsample=None, bn=use_bn)

    def forward(self, xyz, points):
        """

        :param xyz:  shape (bs, N, 3)
        :param points:  shape (bs, fea_dim, N)
        :return:
        """

        l1_xyz, l1_points = self.sa1(xyz.contiguous(), points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        global_features = l2_points.view(-1, self.out_fea_dim)

        return global_features



if __name__ == '__main__':
    """
        :param xyz: (B, N, 3) tensor of the xyz coordinates of the features
        :param features: (B, N, C) tensor of the descriptors of the the features
        :param new_xyz:
    """
    xyz = torch.rand(1, 16, 3).cuda()  # B, N, C
    points = torch.rand(1, 6, 16).cuda()  # B, C, N
    model = segmentPointNet2().cuda()
    point_features, global_features = model(xyz, points)
