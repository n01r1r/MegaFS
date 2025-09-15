"""
Feature Pyramid Network (FPN) implementation for MegaFS
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FPN(nn.Module):
    def __init__(self, backbone, depth=50, omega_only=False):
        super(FPN, self).__init__()
        self.omega_only = omega_only
        self.backbone = backbone
        
        if not omega_only:
            self.output4 = nn.Conv2d(2048, 512, 1)
        self.output3 = nn.Conv2d(1024, 512, 1)
        self.output2 = nn.Conv2d(512, 512, 1)
        self.output1 = nn.Conv2d(256, 512, 1)

    def forward(self, x):
        if not self.omega_only:
            c1, c2, c3, c4 = self.backbone.forward_fpn(x)
            p4 = self.output4(c4)                                               # N, 512, 4, 4
        else:
            c1, c2, c3 = self.backbone.forward_fpn(x)[:3]  # 처음 3개만 사용
            p4 = None
        p3 = self.output3(c3)                                                   # N, 512, 8, 8
        p2 = self.output2(c2) + F.upsample(p3, scale_factor=2, mode='bilinear', align_corners=True) # N, 512, 16, 16
        p1 = self.output1(c1) + F.upsample(p2, scale_factor=2, mode='bilinear', align_corners=True) # N, 512, 32, 32

        return p1, p2, p3, p4
