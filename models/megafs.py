"""
MegaFS main class implementation
Based on One-Shot-Face-Swapping-on-Megapixels repository
"""

import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.functional as tF
from .resnet import resnet50
from .hierfe import HieRFE
from .face_transfer import FaceTransferModule
from .stylegan2 import Generator
from .soft_erosion import SoftErosion


def encode_segmentation_rgb(segmentation, no_neck=True):
    """Encode segmentation mask to RGB format"""
    parse = segmentation[:,:,0]

    face_part_ids = [1, 2, 3, 4, 5, 6, 10, 12, 13] if no_neck else [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 13, 14]
    mouth_id = 11
    hair_id = 17
    face_map = np.zeros([parse.shape[0], parse.shape[1]])
    mouth_map = np.zeros([parse.shape[0], parse.shape[1]])
    hair_map = np.zeros([parse.shape[0], parse.shape[1]])

    for valid_id in face_part_ids:
        valid_index = np.where(parse==valid_id)
        face_map[valid_index] = 255
    valid_index = np.where(parse==mouth_id)
    mouth_map[valid_index] = 255
    valid_index = np.where(parse==hair_id)
    hair_map[valid_index] = 255

    return np.stack([face_map, mouth_map, hair_map], axis=2)


class MegaFS(object):
    """MegaFS class for face swapping"""
    
    def __init__(self, swap_type, img_root, mask_root, checkpoint_dir="weights", data_map=None):
        # Inference Parameters
        self.size = 1024
        self.swap_type = swap_type
        self.img_root = img_root
        self.mask_root = mask_root
        self.checkpoint_dir = checkpoint_dir
        self.data_map = data_map

        # Model - MegaFS 방식
        num_blocks = 3 if self.swap_type == "ftm" else 1
        latent_split = [4, 6, 8]
        num_latents = 18
        swap_indice = 4
        self.encoder = HieRFE(resnet50(False), num_latents=latent_split, depth=50).cuda()
        self.swapper = FaceTransferModule(num_blocks=num_blocks, swap_indice=swap_indice, num_latents=num_latents, typ=self.swap_type).cuda()

        # 체크포인트 로드
        ckpt_e = os.path.join(self.checkpoint_dir, "{}_final.pth".format(self.swap_type))
        if ckpt_e is not None and os.path.exists(ckpt_e):
            print("load encoder & swapper:", ckpt_e)
            ckpts = torch.load(ckpt_e, map_location=torch.device("cpu"))

            # strict=False로 로드하여 누락된 키 무시
            self.encoder.load_state_dict(ckpts["e"], strict=False)
            self.swapper.load_state_dict(ckpts["s"], strict=False)
            del ckpts

        self.generator = Generator(self.size, 512, 8, channel_multiplier=2).cuda()
        ckpt_f = os.path.join(self.checkpoint_dir, "stylegan2-ffhq-config-f.pth")
        if ckpt_f is not None and os.path.exists(ckpt_f):
            print("load generator:", ckpt_f)
            ckpts = torch.load(ckpt_f, map_location=torch.device("cpu"))
            self.generator.load_state_dict(ckpts["g_ema"], strict=False)
            del ckpts

        self.smooth_mask = SoftErosion(kernel_size=17, threshold=0.9, iterations=7).cuda()

        # 모든 모델을 eval 모드로 설정
        self.encoder.eval()
        self.swapper.eval()
        self.generator.eval()
        self.smooth_mask.eval()

    def read_pair(self, src_idx, tgt_idx):
        """Read source and target image pair"""
        # If a data_map was provided, prefer it to resolve exact file paths
        if self.data_map is not None and isinstance(self.data_map, dict):
            if src_idx in self.data_map and tgt_idx in self.data_map:
                src_path = self.data_map[src_idx]["image_path"]
                tgt_path = self.data_map[tgt_idx]["image_path"]
                mask_path = self.data_map[tgt_idx]["mask_path"]
            else:
                # Fallback to legacy convention when ids are not present
                src_path = os.path.join(self.img_root, "{}.jpg".format(src_idx))
                tgt_path = os.path.join(self.img_root, "{}.jpg".format(tgt_idx))
                mask_path = os.path.join(self.mask_root, "{}.png".format(tgt_idx))
        else:
            src_path = os.path.join(self.img_root, "{}.jpg".format(src_idx))
            tgt_path = os.path.join(self.img_root, "{}.jpg".format(tgt_idx))
            mask_path = os.path.join(self.mask_root, "{}.png".format(tgt_idx))
        
        # 파일 존재 확인
        if not os.path.exists(src_path):
            raise FileNotFoundError(f"Source image not found: {src_path}")
        if not os.path.exists(tgt_path):
            raise FileNotFoundError(f"Target image not found: {tgt_path}")
        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"Target mask not found: {mask_path}")
        
        src_face = cv2.imread(src_path)
        tgt_face = cv2.imread(tgt_path)
        tgt_mask = cv2.imread(mask_path)
        
        # 이미지 읽기 확인
        if src_face is None:
            raise ValueError(f"Cannot read source image: {src_path}")
        if tgt_face is None:
            raise ValueError(f"Cannot read target image: {tgt_path}")
        if tgt_mask is None:
            raise ValueError(f"Cannot read target mask: {mask_path}")

        src_face_rgb = src_face[:, :, ::-1]
        tgt_face_rgb = tgt_face[:, :, ::-1]
        tgt_mask = encode_segmentation_rgb(tgt_mask)
        return src_face_rgb, tgt_face_rgb, tgt_mask

    def preprocess(self, src, tgt):
        """Preprocess images for model input"""
        src = cv2.resize(src.copy(), (256, 256))
        tgt = cv2.resize(tgt.copy(), (256, 256))
        src = torch.from_numpy(src.transpose((2, 0, 1))).float().mul_(1/255.0)
        tgt = torch.from_numpy(tgt.transpose((2, 0, 1))).float().mul_(1/255.0)

        src = tF.normalize(src, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), False)
        tgt = tF.normalize(tgt, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), False)

        return src.unsqueeze_(0), tgt.unsqueeze_(0)

    def run(self, src_idx, tgt_idx, refine=True, save_path=None):
        """Run face swapping"""
        src_face_rgb, tgt_face_rgb, tgt_mask = self.read_pair(src_idx, tgt_idx)
        source, target = self.preprocess(src_face_rgb, tgt_face_rgb)
        swapped_face = self.swap(source, target)
        swapped_face = self.postprocess(swapped_face, tgt_face_rgb, tgt_mask)

        result = np.hstack((src_face_rgb[:,:,::-1], tgt_face_rgb[:,:,::-1], swapped_face))

        if refine:
            swapped_tensor, _ = self.preprocess(swapped_face[:,:,::-1], swapped_face)
            refined_face = self.refine(swapped_tensor)
            refined_face = self.postprocess(refined_face, tgt_face_rgb, tgt_mask)
            result = np.hstack((result, swapped_face))

        if save_path:
            cv2.imwrite(save_path, result)
            return save_path, result
        else:
            return None, result

    def swap(self, source, target):
        """Perform face swapping"""
        with torch.no_grad():
            ts = torch.cat([target, source], dim=0).cuda()
            lats, struct = self.encoder(ts)

            # lats는 [2, num_latents, 512] 형태의 텐서
            # struct는 [2, C, H, W] 형태의 텐서
            idd_lats = lats[1:]  # 소스 이미지의 latent [1, num_latents, 512]
            att_lats = lats[0].unsqueeze_(0)  # 타겟 이미지의 latent [1, num_latents, 512]
            att_struct = struct[0].unsqueeze_(0)  # 타겟 이미지의 구조 [1, C, H, W]

            swapped_lats = self.swapper(idd_lats, att_lats)

            # StyleGAN2 Generator는 styles 리스트만 받습니다.
            # 단일 스타일만 사용할 경우 한 개의 텐서만 담아 전달합니다.
            fake_swap, _ = self.generator([swapped_lats], randomize_noise=True)

            fake_swap_max = torch.max(fake_swap)
            fake_swap_min = torch.min(fake_swap)
            denormed_fake_swap = (fake_swap[0] - fake_swap_min) / (fake_swap_max - fake_swap_min) * 255.0
            fake_swap_numpy = denormed_fake_swap.permute((1, 2, 0)).cpu().numpy()
        return fake_swap_numpy

    def refine(self, swapped_tensor):
        """Refine swapped face by re-encoding and generating."""
        with torch.no_grad():
            # 스왑 결과를 재인코딩하여 latent만 사용해 재생성합니다.
            lats, struct = self.encoder(swapped_tensor.cuda())

            # Generator는 styles만 입력으로 받습니다. 결정적 출력을 위해 randomize_noise=False.
            fake_refine, _ = self.generator([lats], randomize_noise=False)

            # Denormalization process remains the same.
            fake_refine_max = torch.max(fake_refine)
            fake_refine_min = torch.min(fake_refine)
            denormed_fake_refine = (fake_refine[0] - fake_refine_min) / (fake_refine_max - fake_refine_min) * 255.0
            fake_refine_numpy = denormed_fake_refine.permute((1, 2, 0)).cpu().numpy()
        return fake_refine_numpy

    def postprocess(self, swapped_face, target, target_mask):
        """Postprocess swapped face with mask blending"""
        target_mask = cv2.resize(target_mask, (self.size,  self.size))

        mask_tensor = torch.from_numpy(target_mask.copy().transpose((2, 0, 1))).float().mul_(1/255.0).cuda()
        face_mask_tensor = mask_tensor[0] + mask_tensor[1]

        soft_face_mask_tensor, _ = self.smooth_mask(face_mask_tensor.unsqueeze_(0).unsqueeze_(0))
        soft_face_mask_tensor.squeeze_()

        soft_face_mask = soft_face_mask_tensor.cpu().numpy()
        soft_face_mask = soft_face_mask[:, :, np.newaxis]
        result =  swapped_face * soft_face_mask + target * (1 - soft_face_mask)
        result = result[:,:,::-1].astype(np.uint8)
        return result
