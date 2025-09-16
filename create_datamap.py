import os
import sys
import json
from glob import glob
from time import time


def try_import_tqdm():
    try:
        from tqdm import tqdm  # type: ignore
        return tqdm
    except Exception:
        def _noop_iter(it, **kwargs):
            return it
        return _noop_iter


def build_data_map(base_dir: str, img_dir: str, mask_dir: str) -> dict:
    """Scan image and mask directories and build a mapping of id -> relative paths.

    - image ids are inferred from image filenames like '<id>.jpg'
    - masks are searched under subfolders using pattern '<id>_*.png'
    - paths in the mapping are stored relative to base_dir for portability
    - includes ALL images and ALL masks for each image
    """
    base_dir = os.path.abspath(base_dir)
    img_dir_abs = os.path.abspath(img_dir)
    mask_dir_abs = os.path.abspath(mask_dir)

    if not os.path.isdir(img_dir_abs):
        raise FileNotFoundError(f"Image directory not found: {img_dir_abs}")
    if not os.path.isdir(mask_dir_abs):
        raise FileNotFoundError(f"Mask directory not found: {mask_dir_abs}")

    image_paths = glob(os.path.join(img_dir_abs, '*.jpg'))
    image_map = {os.path.splitext(os.path.basename(p))[0]: p for p in image_paths}

    tqdm = try_import_tqdm()
    data_map: dict[int, dict[str, str | list[str]]] = {}

    print(f"Found {len(image_map)} images. Searching corresponding masks...")
    for img_id_str, img_path in tqdm(image_map.items(), desc="MAPPING"):
        mask_search_pattern = os.path.join(mask_dir_abs, '*', f"{img_id_str}_*.png")
        mask_paths = glob(mask_search_pattern)
        
        relative_img_path = os.path.relpath(img_path, base_dir)
        
        try:
            img_id = int(img_id_str)
        except ValueError:
            # skip non-numeric ids to keep consistency with numeric indexing
            continue

        if mask_paths:
            # Include all mask paths for this image
            relative_mask_paths = [os.path.relpath(mask_path, base_dir) for mask_path in mask_paths]
            data_map[img_id] = {
                "image_path": relative_img_path,
                "mask_paths": relative_mask_paths,
                "mask_count": len(relative_mask_paths)
            }
        else:
            # Include image even if no masks found
            data_map[img_id] = {
                "image_path": relative_img_path,
                "mask_paths": [],
                "mask_count": 0
            }

    return data_map


def main():
    # Assume running inside the dataset root directory
    # base_dir and output are './'
    base_dir = os.path.abspath(".")
    img_dir = os.path.join(base_dir, "CelebA-HQ-img")
    mask_dir = os.path.join(base_dir, "CelebAMask-HQ-mask-anno")
    output_path = os.path.abspath(os.path.join(base_dir, "data_map.json"))

    print("Creating data map...")
    start = time()
    try:
        data_map = build_data_map(base_dir, img_dir, mask_dir)
    except FileNotFoundError as e:
        print(str(e))
        sys.exit(1)

    if not data_map:
        print("No valid image-mask pairs found. Exiting.")
        sys.exit(1)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data_map, f, indent=4)

    elapsed = time() - start
    print(f"Saved {len(data_map)} items to '{output_path}' in {elapsed:.2f}s")


if __name__ == "__main__":
    main()


