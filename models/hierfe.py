"""
HieRFE (Hierarchical Region Feature Encoder) implementation for MegaFS
Based on One-Shot-Face-Swapping-on-Megapixels repository
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .fpn import FPN


class StyleMapping(nn.Module):
    def __init__(self, size):
        super(StyleMapping, self).__init__()
        num_layers = int(math.log(size, 2))
        convs = []
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            convs.append(nn.Conv2d(512, 512, 3, 2, padding=1, bias=True))
            convs.append(nn.BatchNorm2d(512))
            convs.append(nn.LeakyReLU(inplace=True))
        self.convs = nn.Sequential(*convs)
    
    def forward(self, x):
        x = self.convs(x).squeeze(2).squeeze(2).unsqueeze(1)
        return x


class HieRFE(nn.Module):
    def __init__(self, backbone, num_latents=[4,6,8], depth=50, omega_only=False):
        super(HieRFE, self).__init__()
        self.fpn = FPN(backbone, depth, omega_only)
        self.act = nn.Tanh()
        
        # num_latents의 합이 18이 되도록 동적으로 계산
        if len(num_latents) != 3:
            raise ValueError("num_latents must be a list of 3 integers.")
        
        # 각 레벨별 StyleMapping 모듈을 생성 (reference: sizes 8,16,32)
        self.mapping1 = nn.ModuleList([StyleMapping(8) for _ in range(num_latents[0])])
        self.mapping2 = nn.ModuleList([StyleMapping(16) for _ in range(num_latents[1])])
        self.mapping3 = nn.ModuleList([StyleMapping(32) for _ in range(num_latents[2])])

        
    def forward(self, x):
        latents = []
        f4, f8, f16, f32 = self.fpn(x)
        # Build per-scale latents like reference
        for maps in self.mapping1:
            latents.append(maps(f8))
        for maps in self.mapping2:
            latents.append(maps(f16))
        for maps in self.mapping3:
            latents.append(maps(f32))
        latents = torch.cat(latents, 1)
        return self.act(latents), f4
