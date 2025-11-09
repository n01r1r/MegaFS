"""
BlazeFace face detection model for MegaFS
Official implementation from hollance/BlazeFace-PyTorch
Adapted for integration with attack pipeline
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List
import os
import cv2


class BlazeBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super(BlazeBlock, self).__init__()

        self.stride = stride
        self.channel_pad = out_channels - in_channels

        # TFLite uses slightly different padding than PyTorch 
        # on the depthwise conv layer when the stride is 2.
        if stride == 2:
            self.max_pool = nn.MaxPool2d(kernel_size=stride, stride=stride)
            padding = 0
        else:
            padding = (kernel_size - 1) // 2

        self.convs = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, 
                      kernel_size=kernel_size, stride=stride, padding=padding, 
                      groups=in_channels, bias=True),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, 
                      kernel_size=1, stride=1, padding=0, bias=True),
        )

        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        if self.stride == 2:
            h = F.pad(x, (0, 2, 0, 2), "constant", 0)
            x = self.max_pool(x)
        else:
            h = x

        if self.channel_pad > 0:
            x = F.pad(x, (0, 0, 0, 0, 0, self.channel_pad), "constant", 0)

        return self.act(self.convs(h) + x)


class FinalBlazeBlock(nn.Module):
    def __init__(self, channels, kernel_size=3):
        super(FinalBlazeBlock, self).__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=channels,
                      kernel_size=kernel_size, stride=2, padding=0,
                      groups=channels, bias=True),
            nn.Conv2d(in_channels=channels, out_channels=channels,
                      kernel_size=1, stride=1, padding=0, bias=True),
        )

        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        h = F.pad(x, (0, 2, 0, 2), "constant", 0)
        return self.act(self.convs(h))


class BlazeFace(nn.Module):
    """The BlazeFace face detection model from MediaPipe.
    
    Official implementation from hollance/BlazeFace-PyTorch
    Adapted for MegaFS attack pipeline integration.
    """
    def __init__(self, back_model=False):
        super(BlazeFace, self).__init__()

        # These are the settings from the MediaPipe example graphs
        self.num_classes = 1
        self.num_anchors = 896
        self.num_coords = 16
        self.score_clipping_thresh = 100.0
        self.back_model = back_model
        if back_model:
            self.x_scale = 256.0
            self.y_scale = 256.0
            self.h_scale = 256.0
            self.w_scale = 256.0
            self.min_score_thresh = 0.65
        else:
            self.x_scale = 128.0
            self.y_scale = 128.0
            self.h_scale = 128.0
            self.w_scale = 128.0
            self.min_score_thresh = 0.75
        self.min_suppression_threshold = 0.3

        self._define_layers()
        self.anchors = None

    def _define_layers(self):
        if self.back_model:
            self.backbone = nn.Sequential(
                nn.Conv2d(in_channels=3, out_channels=24, kernel_size=5, stride=2, padding=0, bias=True),
                nn.ReLU(inplace=True),
                BlazeBlock(24, 24),
                BlazeBlock(24, 24),
                BlazeBlock(24, 24),
                BlazeBlock(24, 24),
                BlazeBlock(24, 24),
                BlazeBlock(24, 24),
                BlazeBlock(24, 24),
                BlazeBlock(24, 24),
                BlazeBlock(24, 24, stride=2),
                BlazeBlock(24, 24),
                BlazeBlock(24, 24),
                BlazeBlock(24, 24),
                BlazeBlock(24, 24),
                BlazeBlock(24, 24),
                BlazeBlock(24, 24),
                BlazeBlock(24, 24),
                BlazeBlock(24, 48, stride=2),
                BlazeBlock(48, 48),
                BlazeBlock(48, 48),
                BlazeBlock(48, 48),
                BlazeBlock(48, 48),
                BlazeBlock(48, 48),
                BlazeBlock(48, 48),
                BlazeBlock(48, 48),
                BlazeBlock(48, 96, stride=2),
                BlazeBlock(96, 96),
                BlazeBlock(96, 96),
                BlazeBlock(96, 96),
                BlazeBlock(96, 96),
                BlazeBlock(96, 96),
                BlazeBlock(96, 96),
                BlazeBlock(96, 96),
            )
            self.final = FinalBlazeBlock(96)
            self.classifier_8 = nn.Conv2d(96, 2, 1, bias=True)
            self.classifier_16 = nn.Conv2d(96, 6, 1, bias=True)
            self.regressor_8 = nn.Conv2d(96, 32, 1, bias=True)
            self.regressor_16 = nn.Conv2d(96, 96, 1, bias=True)
        else:
            self.backbone1 = nn.Sequential(
                nn.Conv2d(in_channels=3, out_channels=24, kernel_size=5, stride=2, padding=0, bias=True),
                nn.ReLU(inplace=True),
                BlazeBlock(24, 24),
                BlazeBlock(24, 28),
                BlazeBlock(28, 32, stride=2),
                BlazeBlock(32, 36),
                BlazeBlock(36, 42),
                BlazeBlock(42, 48, stride=2),
                BlazeBlock(48, 56),
                BlazeBlock(56, 64),
                BlazeBlock(64, 72),
                BlazeBlock(72, 80),
                BlazeBlock(80, 88),
            )
            self.backbone2 = nn.Sequential(
                BlazeBlock(88, 96, stride=2),
                BlazeBlock(96, 96),
                BlazeBlock(96, 96),
                BlazeBlock(96, 96),
                BlazeBlock(96, 96),
            )
            self.classifier_8 = nn.Conv2d(88, 2, 1, bias=True)
            self.classifier_16 = nn.Conv2d(96, 6, 1, bias=True)
            self.regressor_8 = nn.Conv2d(88, 32, 1, bias=True)
            self.regressor_16 = nn.Conv2d(96, 96, 1, bias=True)

    def forward(self, x):
        # TFLite uses slightly different padding on the first conv layer
        # than PyTorch, so do it manually.
        x = F.pad(x, (1, 2, 1, 2), "constant", 0)
        
        b = x.shape[0]      # batch size, needed for reshaping later

        if self.back_model:
            x = self.backbone(x)           # (b, 16, 16, 96)
            h = self.final(x)              # (b, 8, 8, 96)
        else:
            x = self.backbone1(x)           # (b, 88, 16, 16)
            h = self.backbone2(x)           # (b, 96, 8, 8)
        
        # Note: Because PyTorch is NCHW but TFLite is NHWC, we need to
        # permute the output from the conv layers before reshaping it.
        
        c1 = self.classifier_8(x)       # (b, 2, 16, 16)
        c1 = c1.permute(0, 2, 3, 1)     # (b, 16, 16, 2)
        c1 = c1.reshape(b, -1, 1)       # (b, 512, 1)

        c2 = self.classifier_16(h)      # (b, 6, 8, 8)
        c2 = c2.permute(0, 2, 3, 1)     # (b, 8, 8, 6)
        c2 = c2.reshape(b, -1, 1)       # (b, 384, 1)

        c = torch.cat((c1, c2), dim=1)  # (b, 896, 1)

        r1 = self.regressor_8(x)        # (b, 32, 16, 16)
        r1 = r1.permute(0, 2, 3, 1)     # (b, 16, 16, 32)
        r1 = r1.reshape(b, -1, 16)      # (b, 512, 16)

        r2 = self.regressor_16(h)       # (b, 96, 8, 8)
        r2 = r2.permute(0, 2, 3, 1)     # (b, 8, 8, 96)
        r2 = r2.reshape(b, -1, 16)      # (b, 384, 16)

        r = torch.cat((r1, r2), dim=1)  # (b, 896, 16)
        return [r, c]

    def _device(self):
        """Which device (CPU or GPU) is being used by this model?"""
        return self.classifier_8.weight.device
    
    def load_weights(self, path):
        """Load model weights from file."""
        self.load_state_dict(torch.load(path, map_location=self._device()))
        self.eval()
    
    def load_anchors(self, path):
        """Load anchor boxes from file."""
        self.anchors = torch.tensor(np.load(path), dtype=torch.float32, device=self._device())
        assert(self.anchors.ndimension() == 2)
        assert(self.anchors.shape[0] == self.num_anchors)
        assert(self.anchors.shape[1] == 4)

    def _preprocess(self, x):
        """Converts the image pixels to the range [-1, 1]."""
        return x.float() / 127.5 - 1.0

    def _tensors_to_detections(self, raw_box_tensor, raw_score_tensor, anchors):
        """Convert raw network outputs to detections."""
        assert raw_box_tensor.ndimension() == 3
        assert raw_box_tensor.shape[1] == self.num_anchors
        assert raw_box_tensor.shape[2] == self.num_coords

        assert raw_score_tensor.ndimension() == 3
        assert raw_score_tensor.shape[1] == self.num_anchors
        assert raw_score_tensor.shape[2] == self.num_classes

        assert raw_box_tensor.shape[0] == raw_score_tensor.shape[0]
        
        detection_boxes = self._decode_boxes(raw_box_tensor, anchors)
        
        thresh = self.score_clipping_thresh
        raw_score_tensor = raw_score_tensor.clamp(-thresh, thresh)
        detection_scores = raw_score_tensor.sigmoid().squeeze(dim=-1)
        
        mask = detection_scores >= self.min_score_thresh

        output_detections = []
        for i in range(raw_box_tensor.shape[0]):
            boxes = detection_boxes[i, mask[i]]
            scores = detection_scores[i, mask[i]].unsqueeze(dim=-1)
            output_detections.append(torch.cat((boxes, scores), dim=-1))

        return output_detections

    def _decode_boxes(self, raw_boxes, anchors):
        """Converts the predictions into actual coordinates using the anchor boxes."""
        boxes = torch.zeros_like(raw_boxes)

        x_center = raw_boxes[..., 0] / self.x_scale * anchors[:, 2] + anchors[:, 0]
        y_center = raw_boxes[..., 1] / self.y_scale * anchors[:, 3] + anchors[:, 1]

        w = raw_boxes[..., 2] / self.w_scale * anchors[:, 2]
        h = raw_boxes[..., 3] / self.h_scale * anchors[:, 3]

        boxes[..., 0] = y_center - h / 2.  # ymin
        boxes[..., 1] = x_center - w / 2.  # xmin
        boxes[..., 2] = y_center + h / 2.  # ymax
        boxes[..., 3] = x_center + w / 2.  # xmax

        for k in range(6):
            offset = 4 + k*2
            keypoint_x = raw_boxes[..., offset    ] / self.x_scale * anchors[:, 2] + anchors[:, 0]
            keypoint_y = raw_boxes[..., offset + 1] / self.y_scale * anchors[:, 3] + anchors[:, 1]
            boxes[..., offset    ] = keypoint_x
            boxes[..., offset + 1] = keypoint_y

        return boxes

    def _weighted_non_max_suppression(self, detections):
        """Weighted NMS as mentioned in the BlazeFace paper."""
        if len(detections) == 0:
            return []

        output_detections = []
        remaining = torch.argsort(detections[:, 16], descending=True)

        while len(remaining) > 0:
            detection = detections[remaining[0]]

            first_box = detection[:4]
            other_boxes = detections[remaining, :4]
            ious = overlap_similarity(first_box, other_boxes)

            mask = ious > self.min_suppression_threshold
            overlapping = remaining[mask]
            remaining = remaining[~mask]

            weighted_detection = detection.clone()
            if len(overlapping) > 1:
                coordinates = detections[overlapping, :16]
                scores = detections[overlapping, 16:17]
                total_score = scores.sum()
                weighted = (coordinates * scores).sum(dim=0) / total_score
                weighted_detection[:16] = weighted
                weighted_detection[16] = total_score / len(overlapping)

            output_detections.append(weighted_detection)

        return output_detections


# IOU code from https://github.com/amdegroot/ssd.pytorch/blob/master/layers/box_utils.py

def intersect(box_a, box_b):
    """Compute intersection area between two sets of boxes."""
    A = box_a.size(0)
    B = box_b.size(0)
    max_xy = torch.min(box_a[:, 2:].unsqueeze(1).expand(A, B, 2),
                       box_b[:, 2:].unsqueeze(0).expand(A, B, 2))
    min_xy = torch.max(box_a[:, :2].unsqueeze(1).expand(A, B, 2),
                       box_b[:, :2].unsqueeze(0).expand(A, B, 2))
    inter = torch.clamp((max_xy - min_xy), min=0)
    return inter[:, :, 0] * inter[:, :, 1]


def jaccard(box_a, box_b):
    """Compute the jaccard overlap of two sets of boxes."""
    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, 2]-box_a[:, 0]) *
              (box_a[:, 3]-box_a[:, 1])).unsqueeze(1).expand_as(inter)
    area_b = ((box_b[:, 2]-box_b[:, 0]) *
              (box_b[:, 3]-box_b[:, 1])).unsqueeze(0).expand_as(inter)
    union = area_a + area_b - inter
    return inter / union


def overlap_similarity(box, other_boxes):
    """Computes the IOU between a bounding box and set of other boxes."""
    return jaccard(box.unsqueeze(0), other_boxes).squeeze(0)


# API compatibility functions for MegaFS integration

def load_blazeface_weights(model: BlazeFace, checkpoint_dir: str = "weights") -> bool:
    """
    Load BlazeFace weights from local file.
    
    Args:
        model: BlazeFace model instance
        checkpoint_dir: Directory containing weights
        
    Returns:
        True if weights loaded successfully, False otherwise
    """
    weight_path = os.path.join(checkpoint_dir, "blazeface.pth")
    
    if not os.path.exists(weight_path):
        print(f"ERROR: BlazeFace weights not found: {weight_path}")
        print(f"INFO: Please download blazeface.pth from:")
        print(f"      https://github.com/hollance/BlazeFace-PyTorch")
        print(f"      and place it in {checkpoint_dir}/")
        return False
    
    try:
        model.load_weights(weight_path)
        print(f"SUCCESS: Loaded BlazeFace weights from {weight_path}")
        return True
    except Exception as e:
        print(f"ERROR: Failed to load BlazeFace weights: {e}")
        return False


def detect_faces(
    model: BlazeFace,
    image: np.ndarray,
    anchors: Optional[np.ndarray] = None,
    threshold: float = 0.5,
    nms_threshold: float = 0.3
) -> List[Tuple[int, int, int, int]]:
    """
    Detect faces in an image using BlazeFace with proper anchor-based decoding.
    
    Args:
        model: BlazeFace model instance (must have anchors loaded)
        image: Input image as numpy array [H, W, 3] in RGB format, range [0, 255]
        anchors: Anchor boxes (optional, uses model.anchors if not provided)
        threshold: Detection confidence threshold (overrides model.min_score_thresh)
        nms_threshold: Non-maximum suppression threshold
        
    Returns:
        List of bounding boxes as (x, y, w, h) tuples in image coordinates
    """
    if model.anchors is None:
        raise RuntimeError("Anchors not loaded. Call model.load_anchors() first.")
    
    model.eval()
    device = model._device()
    h, w = image.shape[:2]
    
    # Resize image to 128x128 (BlazeFace input size)
    image_resized = cv2.resize(image, (128, 128))
    
    # Convert to tensor [1, 3, 128, 128]
    image_tensor = torch.from_numpy(image_resized.transpose(2, 0, 1)).float().unsqueeze(0).to(device)
    
    # Preprocess: normalize to [-1, 1]
    image_tensor = model._preprocess(image_tensor)
    
    # Run inference
    with torch.no_grad():
        out = model(image_tensor)
    
    # Postprocess: convert to detections
    detections = model._tensors_to_detections(out[0], out[1], model.anchors)
    
    # Apply NMS
    filtered_detections = []
    for i in range(len(detections)):
        faces = model._weighted_non_max_suppression(detections[i])
        filtered_detections.append(faces)
    
    # Convert to (x, y, w, h) format and scale to original image size
    bboxes = []
    if len(filtered_detections) > 0 and len(filtered_detections[0]) > 0:
        for face in filtered_detections[0]:
            # Face detection format: [ymin, xmin, ymax, xmax, ...keypoints..., confidence]
            ymin, xmin, ymax, xmax = face[0].item(), face[1].item(), face[2].item(), face[3].item()
            
            # Scale from 128x128 to original image size
            xmin = int(xmin * w / 128.0)
            ymin = int(ymin * h / 128.0)
            xmax = int(xmax * w / 128.0)
            ymax = int(ymax * h / 128.0)
            
            # Convert to (x, y, w, h) format
            bbox_x = max(0, xmin)
            bbox_y = max(0, ymin)
            bbox_w = min(w, xmax) - bbox_x
            bbox_h = min(h, ymax) - bbox_y
            
            if bbox_w > 0 and bbox_h > 0:
                bboxes.append((bbox_x, bbox_y, bbox_w, bbox_h))
    
    return bboxes


def get_blazeface_model(device: str = 'cuda', checkpoint_dir: str = "weights") -> Optional[BlazeFace]:
    """
    Get BlazeFace model instance with loaded weights and anchors.
    
    Args:
        device: Device to load model on ('cuda' or 'cpu')
        checkpoint_dir: Directory containing weights and anchors
        
    Returns:
        BlazeFace model instance or None if loading failed
    """
    model = BlazeFace(back_model=False)  # Use front-facing camera model
    model = model.to(device)
    
    # Load weights
    if not load_blazeface_weights(model, checkpoint_dir):
        return None
    
    # Load anchors
    anchors_path = os.path.join(checkpoint_dir, "anchors.npy")
    if not os.path.exists(anchors_path):
        print(f"ERROR: Anchors file not found: {anchors_path}")
        return None
    
    try:
        model.load_anchors(anchors_path)
        print(f"SUCCESS: Loaded anchors from {anchors_path}")
    except Exception as e:
        print(f"ERROR: Failed to load anchors: {e}")
        return None
    
    return model
