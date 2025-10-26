"""
PyTorch Dataset for face swapping experiments.
Provides train/val/test splits and batching capabilities.
"""

import torch
import torch.utils.data
import numpy as np
from typing import Dict, Optional, Tuple, List
import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from utils.data_utils import DataMapManager
from utils.image_utils import ImageProcessor


class FaceSwapDataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset for face swap pairs.
    Compatible with DataLoader for batching/shuffling.
    
    Now supports both data_map.json-based loading and simple folder-based loading
    for greater flexibility.
    """
    
    def __init__(self, 
                 dataset_root: str,
                 mode: str = 'train',  # 'train', 'val', 'test'
                 data_map_path: Optional[str] = None,
                 split_ratio: tuple = (0.7, 0.15, 0.15),
                 transform=None,
                 return_tensors: bool = True,
                 random_pairs: bool = False,
                 seed: int = 42,
                 use_data_map: bool = True):
        """
        Args:
            dataset_root: Root directory for dataset
            mode: 'train', 'val', or 'test'
            data_map_path: Path to data_map.json (optional if use_data_map=True)
            split_ratio: (train, val, test) split proportions
            transform: Optional torchvision transforms
            return_tensors: If True, return torch.Tensor. If False, return numpy
            random_pairs: If True, randomly pair images. If False, use fixed pairs
            seed: Random seed for reproducibility
            use_data_map: If True, use data_map.json. If False, auto-discover images from folder
        """
        self.dataset_root = dataset_root
        self.mode = mode
        self.transform = transform
        self.return_tensors = return_tensors
        self.random_pairs = random_pairs
        self.seed = seed
        self.use_data_map = use_data_map
        
        # Initialize data manager if using data_map.json
        if use_data_map and data_map_path:
            self.data_manager = DataMapManager(data_map_path)
        elif use_data_map:
            # Try default location
            default_map = os.path.join(dataset_root, "data_map.json")
            if os.path.exists(default_map):
                self.data_manager = DataMapManager(default_map)
            else:
                print(f"WARNING: data_map.json not found at {default_map}")
                print("Falling back to folder-based discovery")
                self.use_data_map = False
                self.data_manager = None
        else:
            self.data_manager = None
        
        # Get valid IDs and split
        if self.use_data_map and self.data_manager:
            valid_ids = self.data_manager.get_valid_ids(dataset_root)
        else:
            valid_ids = self._discover_ids_from_folder()
        
        self.train_ids, self.val_ids, self.test_ids = self._split_ids(
            valid_ids, split_ratio, seed
        )
        
        # Set current split
        self.ids = self._get_split_ids()
        
        # Generate pairs
        self.pairs = self._generate_pairs(random_pairs, seed)
    
    def _discover_ids_from_folder(self) -> List[int]:
        """Discover image IDs from folder structure without data_map.json"""
        img_dir = os.path.join(self.dataset_root, "CelebA-HQ-img")
        if not os.path.exists(img_dir):
            # Try alternative locations
            img_dir = self.dataset_root
        
        # Find all image files
        img_extensions = ['.jpg', '.jpeg', '.png']
        image_files = []
        
        for ext in img_extensions:
            for img_file in Path(img_dir).glob(f"*{ext}"):
                image_files.append(img_file.stem)
        
        # Extract numeric IDs
        ids = []
        for filename in image_files:
            try:
                # Assume filename is numeric ID
                img_id = int(filename)
                ids.append(img_id)
            except ValueError:
                # Skip non-numeric filenames
                continue
        
        return sorted(ids) if ids else [i for i in range(len(image_files))]
    
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
    
    def _resolve_paths_for_id(self, img_id: int) -> Tuple[Optional[str], Optional[str]]:
        """Resolve image and mask paths for a given ID"""
        if self.use_data_map and self.data_manager:
            # Use DataMapManager if available
            return self.data_manager.resolve_paths_for_id(img_id, self.dataset_root)
        else:
            # Fallback to folder-based discovery
            img_extensions = ['.jpg', '.jpeg', '.png']
            img_path = None
            
            # Try CelebA-HQ-img folder
            for ext in img_extensions:
                potential_path = os.path.join(self.dataset_root, "CelebA-HQ-img", f"{img_id}{ext}")
                if os.path.exists(potential_path):
                    img_path = potential_path
                    break
            
            # Try root directory if not found
            if not img_path:
                for ext in img_extensions:
                    potential_path = os.path.join(self.dataset_root, f"{img_id}{ext}")
                    if os.path.exists(potential_path):
                        img_path = potential_path
                        break
            
            return img_path, None  # Masks not supported in folder mode
    
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
        
        # Load images using flexible path resolution
        src_path, src_mask_path = self._resolve_paths_for_id(src_id)
        tgt_path, tgt_mask_path = self._resolve_paths_for_id(tgt_id)
        
        # Load images
        if src_path and os.path.exists(src_path):
            src_img = ImageProcessor.load_image(src_path, target_size=None)
        else:
            raise FileNotFoundError(f"Source image not found for ID {src_id}: {src_path}")
        
        if tgt_path and os.path.exists(tgt_path):
            tgt_img = ImageProcessor.load_image(tgt_path, target_size=None)
        else:
            raise FileNotFoundError(f"Target image not found for ID {tgt_id}: {tgt_path}")
        
        # Load masks if available
        src_mask = ImageProcessor.load_image(src_mask_path) if src_mask_path and os.path.exists(src_mask_path) else None
        tgt_mask = ImageProcessor.load_image(tgt_mask_path) if tgt_mask_path and os.path.exists(tgt_mask_path) else None
        
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


def create_dataloaders(dataset_root: str, 
                       data_map_path: Optional[str] = None,
                       batch_size: int = 8, 
                       num_workers: int = 4, 
                       split_ratio: tuple = (0.7, 0.15, 0.15),
                       seed: int = 42,
                       use_data_map: bool = True):
    """
    Factory function to create train/val/test dataloaders.
    
    Args:
        dataset_root: Root directory for dataset
        data_map_path: Path to data_map.json (optional)
        batch_size: Batch size for data loading
        num_workers: Number of worker processes for data loading
        split_ratio: (train, val, test) split proportions
        seed: Random seed for reproducibility
        use_data_map: Whether to use data_map.json (auto-fallback if not found)
    
    Returns:
        dict with keys 'train', 'val', 'test' containing DataLoaders
    """
    from torch.utils.data import DataLoader
    
    loaders = {}
    
    for mode in ['train', 'val', 'test']:
        dataset = FaceSwapDataset(
            dataset_root=dataset_root,
            mode=mode,
            data_map_path=data_map_path,
            return_tensors=True,
            split_ratio=split_ratio,
            seed=seed,
            use_data_map=use_data_map
        )
        
        loaders[mode] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(mode == 'train'),
            num_workers=num_workers,
            pin_memory=True
        )
    
    return loaders


