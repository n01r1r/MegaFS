# MegaFS Models Package
from .resnet import ResNet, BasicBlock, Bottleneck, resnet50
from .fpn import FPN
from .hierfe import HieRFE, StyleMapping
from .face_transfer import FaceTransferModule, InjectionBlock, InjectionResBlock, LCRBlock, TransferCell, LCR
from .stylegan2 import Generator, Discriminator, PixelNorm, EqualLinear, EqualConv2d, StyledConv, ToRGB, ConstantInput, Blur, Upsample, Downsample, ModulatedConv2d, NoiseInjection, ScaledLeakyReLU, ConvLayer, ResBlock
from .soft_erosion import SoftErosion
from .megafs import MegaFS, encode_segmentation_rgb

__all__ = [
    'ResNet', 'BasicBlock', 'Bottleneck', 'resnet50',
    'FPN', 'HieRFE', 'StyleMapping', 'FaceTransferModule', 'InjectionBlock', 'InjectionResBlock', 'LCRBlock', 'TransferCell', 'LCR',
    'Generator', 'Discriminator', 'PixelNorm', 'EqualLinear', 'EqualConv2d', 'StyledConv', 'ToRGB', 'ConstantInput', 'Blur', 'Upsample', 'Downsample', 'ModulatedConv2d', 'NoiseInjection', 'ScaledLeakyReLU', 'ConvLayer', 'ResBlock',
    'SoftErosion', 'MegaFS', 'encode_segmentation_rgb'
]
