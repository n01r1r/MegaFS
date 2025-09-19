"""
Data management utilities for MegaFS
"""

import os
import json
from typing import Dict, Any, Optional, Tuple
import random


class DataMapManager:
    """Manages data mapping for image and mask paths"""
    
    def __init__(self, data_map_path: str = "data_map.json"):
        self.data_map_path = data_map_path
        self.data_map = {}
        self.load_data_map()
    
    def load_data_map(self) -> bool:
        """Load data map from JSON file"""
        if not os.path.exists(self.data_map_path):
            print(f"ERROR: Data map file not found: {self.data_map_path}")
            return False
        
        try:
            with open(self.data_map_path, 'r', encoding='utf-8') as f:
                loaded_map = json.load(f)
            
            # Convert string keys to integers
            self.data_map = {int(k): v for k, v in loaded_map.items()}
            print(f"SUCCESS: Data map loaded: {len(self.data_map)} entries")
            return True
            
        except Exception as e:
            print(f"ERROR: Error loading data map: {e}")
            return False
    
    def resolve_paths_for_id(self, img_id: int, dataset_root: str = "") -> Tuple[Optional[str], Optional[str]]:
        """Resolve image and mask paths for a given ID"""
        if img_id not in self.data_map:
            return None, None
        
        record = self.data_map[img_id]
        
        def _normalize_path(rel_path: Optional[str], is_image: bool) -> Optional[str]:
            if not rel_path or not isinstance(rel_path, str):
                return None
            
            # Normalize path separators
            rel_path = rel_path.replace("\\", "/").lstrip("/\\")
            
            # If path starts with dataset folder, use dataset root
            if rel_path.startswith("CelebA-HQ-img/") or rel_path.startswith("CelebAMask-HQ-mask-anno/"):
                if dataset_root:
                    abs_path = os.path.normpath(os.path.join(dataset_root, rel_path))
                    if os.path.exists(abs_path):
                        return abs_path
            
            return None
        
        # Get image path
        image_path = _normalize_path(record.get("image_path"), True)
        
        # Get mask path (try both possible keys)
        mask_path = None
        if "mask_paths" in record and isinstance(record["mask_paths"], list) and record["mask_paths"]:
            mask_path = _normalize_path(record["mask_paths"][0], False)
        elif "mask_path" in record:
            mask_path = _normalize_path(record["mask_path"], False)
        
        return image_path, mask_path
    
    def get_valid_ids(self, dataset_root: str = "", sample_size: int = None) -> list:
        """Get list of valid IDs with existing files"""
        valid_ids = []
        
        for img_id in self.data_map.keys():
            image_path, mask_path = self.resolve_paths_for_id(img_id, dataset_root)
            
            # Check if image exists (mask is optional)
            if image_path and os.path.exists(image_path):
                valid_ids.append(img_id)
        
        if sample_size and len(valid_ids) > sample_size:
            valid_ids = random.sample(valid_ids, sample_size)
        
        print(f"SUCCESS: Found {len(valid_ids)} valid IDs")
        return valid_ids
    
    def verify_sample(self, sample_size: int = 20, dataset_root: str = "") -> Dict[str, int]:
        """Verify a sample of IDs and return statistics"""
        if not self.data_map:
            return {"total": 0, "valid": 0, "missing": 0}
        
        sample_ids = random.sample(list(self.data_map.keys()), 
                                 min(sample_size, len(self.data_map)))
        
        valid_count = 0
        missing_count = 0
        
        for img_id in sample_ids:
            image_path, mask_path = self.resolve_paths_for_id(img_id, dataset_root)
            
            if image_path and os.path.exists(image_path):
                valid_count += 1
            else:
                missing_count += 1
        
        stats = {
            "total": len(sample_ids),
            "valid": valid_count,
            "missing": missing_count
        }
        
        print(f"INFO: Sample verification: {stats}")
        return stats
