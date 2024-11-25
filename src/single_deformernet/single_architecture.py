import torch.nn as nn
import torch
import torch.nn.functional as F

import os
import sys

src_path = os.path.join(os.path.dirname(__file__), '..')
print(f"Adding {src_path} to sys.path")
sys.path.append(src_path)

import utils.tools as tools
from utils.pointconv_util_groupnorm import PointConvDensitySetAbstraction
from utils.explain import visualize_pointclouds_simple_from_tensor

class DeformerNetSingle(nn.Module):
    def __init__(self, normal_channel=False, use_mp_input=True):
        super(DeformerNetSingle, self).__init__()

        if normal_channel:
            additional_channel = 3
        else:
            additional_channel = 0
        self.normal_channel = normal_channel
        self.use_mp_input = use_mp_input

        # Set Abstraction layers
        if self.use_mp_input:
            self.sa1 = PointConvDensitySetAbstraction(
                npoint=512,
                nsample=32,
                in_channel=4 + 3 + additional_channel, # 4 = 3 (xyz) + 1 (manipulation point). There are two point clouds but only one with a manipulation point, so 4 + 3
                mlp=[64],
                bandwidth=0.1,
                group_all=False,
            )
        else:
            self.sa1 = PointConvDensitySetAbstraction(
                npoint=512,
                nsample=32,
                in_channel=3 + 3 + additional_channel, # 3 (xyz current) + 3 (xyz goal)
                mlp=[64],
                bandwidth=0.1,
                group_all=False,
            )

        self.sa2 = PointConvDensitySetAbstraction(
            npoint=128,
            nsample=64,
            in_channel=64 + 3,
            mlp=[128],
            bandwidth=0.2,
            group_all=False,
        )
        self.sa3 = PointConvDensitySetAbstraction(
            npoint=1,
            nsample=None,
            in_channel=128 + 3,
            mlp=[256],
            bandwidth=0.4,
            group_all=True,
        )

        self.sa1_g = PointConvDensitySetAbstraction(
            npoint=512,
            nsample=32,
            in_channel=3 + 3 + additional_channel,
            mlp=[64],
            bandwidth=0.1,
            group_all=False,
        )
        self.sa2_g = PointConvDensitySetAbstraction(
            npoint=128,
            nsample=64,
            in_channel=64 + 3,
            mlp=[128],
            bandwidth=0.2,
            group_all=False,
        )
        self.sa3_g = PointConvDensitySetAbstraction(
            npoint=1,
            nsample=None,
            in_channel=128 + 3,
            mlp=[256],
            bandwidth=0.4,
            group_all=True,
        )

        self.fc1 = nn.Linear(512, 256)
        self.bn1 = nn.GroupNorm(1, 256)

        self.fc3 = nn.Linear(256, 128)
        self.bn3 = nn.GroupNorm(1, 128)

        self.fc4 = nn.Linear(128, 64)
        self.bn4 = nn.GroupNorm(1, 64)

        self.fc5 = nn.Linear(64, 9)

    def forward(self, xyz, xyz_goal):
        # Set Abstraction layers
        B, C, N = xyz.shape

        if self.normal_channel:
            l0_points = xyz
            l0_xyz = xyz[:, :3, :] # remove manipulation point channel
        else:
            l0_points = xyz
            l0_xyz = xyz[:, :3, :] # remove manipulation point channel

        # Encode current point cloud
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)

        # visualize_pointclouds_simple_from_tensor(l1_xyz, l1_points, plot_origin=True)

        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        x = l3_points.view(B, 256) # reshape to (B, 256)

        if self.normal_channel:
            l0_points = xyz_goal
            l0_xyz = xyz_goal[:, :3, :] # remove manipulation point channel
        else:
            l0_points = xyz_goal
            l0_xyz = xyz_goal

        # Encode goal point cloud
        l1_xyz, l1_points = self.sa1_g(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2_g(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3_g(l2_xyz, l2_points)
        g = l3_points.view(B, 256) # reshape to (B, 256)

        x = torch.cat((x, g), dim=-1) # x.shape = (B, 512)

        # Fully connected layers for reasoning over features
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn3(self.fc3(x)))
        x = F.relu(self.bn4(self.fc4(x)))

        x = self.fc5(x)

        position = x[:, :3]
        out_rotation_matrix = tools.compute_rotation_matrix_from_ortho6d(x[:, 3:])

        return position, out_rotation_matrix

    def compute_geodesic_loss(self, gt_r_matrix, out_r_matrix):
        theta = tools.compute_geodesic_distance_from_two_matrices(
            gt_r_matrix, out_r_matrix
        )
        error = theta.mean()
        return error


if __name__ == "__main__":

    device = torch.device("cuda")
    input = torch.randn((8, 4, 2048)).float().to(device)
    goal = torch.randn((8, 3, 2048)).float().to(device)
    model = DeformerNetSingle().to(device)
    out = model(input, goal)
    print(out.shape)
