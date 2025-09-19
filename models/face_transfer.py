"""
Face Transfer Module implementations for MegaFS
Based on One-Shot-Face-Swapping-on-Megapixels repository
"""

import torch
import torch.nn as nn
from typing import Tuple


class TransferCell(nn.Module):
    def __init__(self, num_blocks):
        super(TransferCell, self).__init__()
        self.num_blocks = num_blocks
        self.idd_selectors = nn.ModuleList()
        self.idd_shifters = nn.ModuleList()
        self.att_selectors = nn.ModuleList()
        self.att_shifters = nn.ModuleList()

        self.act = nn.LeakyReLU(True)

        for i in range(self.num_blocks):
            self.idd_selectors.append(nn.Sequential(nn.Linear(1024, 256), nn.ReLU(True), nn.Linear(256, 512), nn.Sigmoid()))
            # Note: Original paper implementation used nn.Tanh(). This was changed to LeakyReLU.
            self.idd_shifters.append(nn.Sequential(nn.Linear(1024, 256), nn.ReLU(True), nn.Linear(256, 512), nn.Tanh()))

            self.att_selectors.append(nn.Sequential(nn.Linear(1024, 256), nn.ReLU(True), nn.Linear(256, 512), nn.Sigmoid()))
            # Note: Original paper implementation used nn.Tanh(). This was changed to LeakyReLU.
            self.att_shifters.append(nn.Sequential(nn.Linear(1024, 256), nn.ReLU(True), nn.Linear(256, 512), nn.Tanh()))

    def forward(self, idd, att):
        for i in range(self.num_blocks):
            fuse = torch.cat([idd, att], dim=1)
            idd = self.act(idd * self.idd_selectors[i](fuse) + self.idd_shifters[i](fuse))
            att = self.act(att * self.att_selectors[i](fuse) + self.att_shifters[i](fuse))
        return idd.unsqueeze(1), att.unsqueeze(1)

class InjectionBlock(nn.Module):
    def __init__(self):
        super(InjectionBlock, self).__init__()
        self.idd_linears = nn.Sequential(nn.Linear(512, 512), nn.ReLU(True))
        self.idd_selectors = nn.Linear(512, 512)
        self.idd_shifters = nn.Linear(512, 512)
        self.att_bns = nn.BatchNorm1d(512, affine=False)

    def forward(self, x: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        idd, att = x[0], x[1]
        normalized = self.att_bns(att)
        actv = self.idd_linears(idd)
        gamma = self.idd_selectors(actv)
        beta = self.idd_shifters(actv)
        out = normalized * (1 + gamma) + beta
        return out


class InjectionResBlock(nn.Module):
    def __init__(self, num_blocks):
        super(InjectionResBlock, self).__init__()
        self.num_blocks = num_blocks

        self.att_path1 = nn.ModuleList()
        self.att_path2 = nn.ModuleList()

        self.act = nn.LeakyReLU()  # Changed from Tanh to LeakyReLU as per user request

        for i in range(self.num_blocks):
            self.att_path1.append(InjectionBlock())
            self.att_path2.append(InjectionBlock())

    def forward(self, idd: torch.Tensor, att: torch.Tensor) -> torch.Tensor:
        for i in range(self.num_blocks):
            att_bias = att * 1
            att = self.att_path1[i]((idd, att))
            att = self.att_path2[i]((idd, att))
            att = att + att_bias
        return self.act(att.unsqueeze(1))


def LCR(idd: torch.Tensor, att: torch.Tensor, swap_indice: int = 4) -> torch.Tensor:
    """Latent Code Replacement function"""
    swapped = torch.cat([att[:, :swap_indice], idd[:, swap_indice:]], 1)
    return swapped


class FaceTransferModule(nn.Module):
    def __init__(self, num_blocks=1, swap_indice=4, num_latents=14, typ="ftm"):
        super(FaceTransferModule, self).__init__()
        self.type = typ
        if self.type == "ftm":
            self.swap_indice = swap_indice
            self.num_latents = num_latents - swap_indice
            self.blocks = nn.ModuleList()
            for i in range(self.num_latents):
                self.blocks.append(TransferCell(num_blocks))

            self.weight = nn.Parameter(torch.randn(1, self.num_latents, 512))
        
        elif self.type == "injection":
            self.swap_indice = swap_indice
            self.num_latents = num_latents - swap_indice
            self.blocks = nn.ModuleList()
            for i in range(self.num_latents):
                self.blocks.append(InjectionResBlock(num_blocks))
        
        elif self.type == "lcr":
            self.swap_indice = swap_indice
        
        else:
            raise NotImplementedError()
        
    def forward(self, idd, att):
        if self.type == "ftm":
            att_low = att[:, :self.swap_indice]
            idd_high = idd[:, self.swap_indice:]
            att_high = att[:, self.swap_indice:]

            N = idd.size(0)
            idds = []
            atts = []
            for i in range(self.num_latents):
                new_idd, new_att = self.blocks[i](idd_high[:, i], att_high[:, i])
                idds.append(new_idd)
                atts.append(new_att)
            idds = torch.cat(idds, 1)  # [N, num_latents, 512]
            atts = torch.cat(atts, 1)  # [N, num_latents, 512]
            scale = torch.sigmoid(self.weight).expand(N, -1, -1)
            latents = scale * idds + (1-scale) * atts

            return torch.cat([att_low, latents], 1)
            
        elif self.type == "injection":
            att_low = att[:, :self.swap_indice]
            idd_high = idd[:, self.swap_indice:]
            att_high = att[:, self.swap_indice:]

            N = idd.size(0)
            latents = []
            for i in range(self.num_latents):
                # InjectionBlock expects a tuple-like (idd, att)
                new_latent = self.blocks[i]((idd_high[:, i], att_high[:, i]))
                latents.append(new_latent.unsqueeze(1))
            latents = torch.cat(latents, 1)
            return torch.cat([att_low, latents], 1)
        
        elif self.type == "lcr":
            return LCR(idd, att, swap_indice=self.swap_indice)
        
        else:
            raise NotImplementedError()


class LCRBlock(nn.Module):
    def __init__(self, num_blocks):
        super(LCRBlock, self).__init__()
        self.num_blocks = num_blocks
        self.idd_linears = nn.ModuleList()
        self.att_linears = nn.ModuleList()
        for i in range(self.num_blocks):
            self.idd_linears.append(nn.Sequential(nn.Linear(512, 512), nn.ReLU(True), nn.Linear(512, 512)))
            self.att_linears.append(nn.Sequential(nn.Linear(512, 512), nn.ReLU(True), nn.Linear(512, 512)))

    def forward(self, idd, att):
        for i in range(self.num_blocks):
            idd = idd + self.idd_linears[i](idd)
            att = att + self.att_linears[i](att)
        return idd, att
