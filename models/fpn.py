"""
Feature Pyramid Network (FPN) implementation for MegaFS
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_bn(inp, oup, stride=1):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=True),
        nn.BatchNorm2d(oup),
        nn.ReLU()
    )


class FPN(nn.Module):
    def __init__(self, backbone, depth=50, omega_only=False):
        super(FPN, self).__init__()
        scale = 4 if depth == 50 else 1
        self.omega_only = omega_only
        self.backbone = backbone
        self.output1 = conv_bn(128*scale, 512, stride=1)
        self.output2 = conv_bn(256*scale, 512, stride=1)
        self.output3 = conv_bn(512*scale, 512, stride=1)
        if not omega_only:
            self.output4 = nn.Sequential(
                nn.Conv2d(512*scale, 512, 3, 1, padding=1, bias=True),
                nn.BatchNorm2d(512)
            )

    def forward(self, x):
        if not self.omega_only:
            c1, c2, c3, c4 = self.backbone(x)
            p4 = self.output4(c4)                                               # N, 512, 4, 4
        else:
            c1, c2, c3 = self.backbone(x)
            p4 = None
        p3 = self.output3(c3)                                                   # N, 512, 8, 8
        p2 = self.output2(c2) + F.upsample(p3, scale_factor=2, mode='bilinear', align_corners=True) # N, 512, 16, 16
        p1 = self.output1(c1) + F.upsample(p2, scale_factor=2, mode='bilinear', align_corners=True) # N, 512, 32, 32

        return p4, p3, p2, p1
