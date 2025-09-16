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
        scale = 4 if depth == 50 else 1
        # Reference FPN uses conv_bn; here we approximate with 1x1 for channel alignment
        if not omega_only:
            self.output4 = nn.Sequential(
                nn.Conv2d(512*scale, 512, 3, 1, padding=1, bias=True),
                nn.BatchNorm2d(512)
            )
        self.output3 = nn.Conv2d(512*scale, 512, 1)
        self.output2 = nn.Conv2d(256*scale, 512, 1)
        self.output1 = nn.Conv2d(128*scale, 512, 1)

    def forward(self, x):
        if not self.omega_only:
            c1, c2, c3, c4 = self.backbone.forward_fpn(x)
            p4 = self.output4(c4)                                               # N, 512, 4, 4
        else:
            c1, c2, c3 = self.backbone.forward_fpn(x)
            p4 = None
        p3 = self.output3(c3)                                                   # N, 512, 8, 8
        p2 = self.output2(c2) + F.interpolate(p3, scale_factor=2, mode='bilinear', align_corners=True) # N, 512, 16, 16
        p1 = self.output1(c1) + F.interpolate(p2, scale_factor=2, mode='bilinear', align_corners=True) # N, 512, 32, 32

        return p4, p3, p2, p1
