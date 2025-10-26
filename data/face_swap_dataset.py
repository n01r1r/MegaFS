"""
PyTorch Dataset for face swapping experiments.
Provides train/val/test splits and batching capabilities.
"""

import torch
import torch.utils.data
import numpy as np
from typing import Dict, Optional, Tuple
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from utils.data_utils import DataMapManager
from utils.image_utils import ImageProcessor


class FaceSwapDataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset for face swap pairs.
    Compatible with DataLoader for batching/shuffling.
    """
    
    def __init__(self, 
                 data_map_path: str,
                 dataset_root: str,
                 mode: str = 'train',  # 'train', 'val', 'test'
                 split_ratio: tuple = (0.7, 0.15, 0.15),
                 transform=None,
                 return_tensors: bool = True,
                 random_pairs: bool = False,
                 seed: int = 42):
        """
        Args:
            data_map_path: Path to data_map.json
            dataset_root: Root directory for dataset
            mode: 'train', 'val', or 'test'
            split_ratio: (train, val, test) split proportions
            transform: Optional torchvision transforms
            return_tensors: If True, return torch.Tensor. If False, return numpy
            random_pairs: If True, randomly pair images. If False, use fixed pairs
            seed: Random seed for reproducibility
        """
        self.data_manager = DataMapManager(data_map_path)
        self.dataset_root = dataset_root
        self.mode = mode
        self.transform = transform
        self.return_tensors = return_tensors
        self.random_pairs = random_pairs
        self.seed = seed
        
        # Get valid IDs and split
        valid_ids = self.data_manager.get_valid_ids(dataset_root)
        self.train_ids, self.val_ids, self.test_ids = self._split_ids(
            valid_ids, split_ratio, seed
        )
        
        # Set current split
        self.ids = self._get_split_ids()
        
        # Generate pairs
        self.pairs = self._generate_pairs(random_pairs, seed)
    
    def _split_ids(self, ids, split_ratio, seed):
        """Split IDs into train/val/test"""
        np.random.seed(seed)
        shuffled = np.random.permutation(ids)
        
        n_train = int(len(shuffled) * split_ratio[0])
        n_val = int(len(shuffled) * split_ratio[1])
        
        train = shuffled[:n_train]
        val = shuffled[n_train:n_train+n_val]
        test = shuffled[n_train+n_val:]
        
        return train.tolist(), val.tolist(), test.tolist()
    
    def _get_split_ids(self):
        """Get IDs for current mode"""
        if self.mode == 'train':
            return self.train_ids
        elif self.mode == 'val':
            return self.val_ids
        elif self.mode == 'test':
            return self.test_ids
        else:
            raise ValueError(f"Invalid mode: {self.mode}")
    
    def _generate_pairs(self, random_pairs, seed):
        """Generate source-target pairs"""
        if random_pairs:
            # Random pairing for each epoch
            return None  # Generate on-the-fly in __getitem__
        else:
            # Fixed pairs
            np.random.seed(seed)
            sources = np.random.choice(self.ids, len(self.ids), replace=True)
            targets = np.random.choice(self.ids, len(self.ids), replace=True)
            return list(zip(sources, targets))
    
    def __len__(self):
        return len(self.ids)
    
    def __getitem__(self, idx):
        """
        Returns:
            dict with keys:
                'source': Source image (tensor or numpy)
                'target': Target image (tensor or numpy)
                'source_id': Source image ID
                'target_id': Target image ID
                'source_mask': Source mask (if available)
                'target_mask': Target mask (if available)
        """
        # Get pair
        if self.pairs is not None:
            src_id, tgt_id = self.pairs[idx]
        else:
            src_id = np.random.choice(self.ids)
            tgt_id = np.random.choice(self.ids)
        
        # Load images
        src_path, src_mask_path = self.data_manager.resolve_paths_for_id(
            src_id, self.dataset_root
        )
        tgt_path, tgt_mask_path = self.data_manager.resolve_paths_for_id(
            tgt_id, self.dataset_root
        )
        
        src_img = ImageProcessor.load_image(src_path, target_size=None)
        tgt_img = ImageProcessor.load_image(tgt_path, target_size=None)
        
        # Load masks if available
        src_mask = ImageProcessor.load_image(src_mask_path) if src_mask_path else None
        tgt_mask = ImageProcessor.load_image(tgt_mask_path) if tgt_mask_path else None
        
        # Convert to tensors if needed
        if self.return_tensors:
            src_img = torch.from_numpy(src_img.transpose(2, 0, 1)).float()
            tgt_img = torch.from_numpy(tgt_img.transpose(2, 0, 1)).float()
            
            if src_mask is not None:
                src_mask = torch.from_numpy(src_mask.transpose(2, 0, 1)).float()
            if tgt_mask is not None:
                tgt_mask = torch.from_numpy(tgt_mask.transpose(2, 0, 1)).float()
        
        # Apply transforms
        if self.transform:
            src_img = self.transform(src_img)
            tgt_img = self.transform(tgt_img)
        
        return {
            'source': src_img,
            'target': tgt_img,
            'source_id': src_id,
            'target_id': tgt_id,
            'source_mask': src_mask,
            'target_mask': tgt_mask
        }


def create_dataloaders(data_map_path: str, dataset_root: str, batch_size: int = 8, 
                       num_workers: int = 4, split_ratio: tuple = (0.7, 0.15, 0.15),
                       seed: int = 42):
    """
    Factory function to create train/val/test dataloaders.
    
    Returns:
        dict with keys 'train', 'val', 'test' containing DataLoaders
    """
    from torch.utils.data import DataLoader
    
    loaders = {}
    
    for mode in ['train', 'val', 'test']:
        dataset = FaceSwapDataset(
            data_map_path=data_map_path,
            dataset_root=dataset_root,
            mode=mode,
            return_tensors=True,
            split_ratio=split_ratio,
            seed=seed
        )
        
        loaders[mode] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(mode == 'train'),
            num_workers=num_workers,
            pin_memory=True
        )
    
    return loaders


