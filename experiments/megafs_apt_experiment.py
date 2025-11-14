import argparse
import os
from typing import Optional

import numpy as np
from PIL import Image
import torch

from config import DEFAULT_CONFIGS
from models.megafs import MegaFS
from experiments.stylegan2_apt_pti import (
    build_generator,
    load_image,
    inversion_stage,
    classifier_setup,
    pti_adversarial_stage,
    save_image,
)


def load_numpy_rgb(path: str) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    return np.array(img)


def run_megafs_swap(config_name: str, src_idx: int, tgt_idx: int, save_path: str) -> str:
    config = DEFAULT_CONFIGS.get(config_name, DEFAULT_CONFIGS["local"])  # fallback
    megafs = MegaFS(config=config, debug=False)
    result_path, _ = megafs.run(
        src_idx=src_idx,
        tgt_idx=tgt_idx,
        refine=True,
        save_path=save_path,
    )
    return result_path


def main():
    parser = argparse.ArgumentParser(description="Run MegaFS then APT-style refinement to induce classifier failure")
    parser.add_argument("--config", type=str, default="local")
    parser.add_argument("--src-idx", type=int, required=True)
    parser.add_argument("--tgt-idx", type=int, required=True)
    parser.add_argument("--swap-out", type=str, default="outputs/megafs_swap.jpg")
    parser.add_argument("--apt-outdir", type=str, default="outputs/megafs_apt")
    parser.add_argument("--weights-dir", type=str, default="weights")
    parser.add_argument("--image-size", type=int, default=1024)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--inv-steps", type=int, default=400)
    parser.add_argument("--inv-lr", type=float, default=0.05)
    parser.add_argument("--pti-steps", type=int, default=300)
    parser.add_argument("--pti-lr", type=float, default=1e-4)
    parser.add_argument("--mode", type=str, choices=["untargeted", "targeted"], default="untargeted")
    parser.add_argument("--true-class", type=int, default=None)
    parser.add_argument("--target-class", type=int, default=None)
    parser.add_argument("--classifier", type=str, default="resnet50")

    args = parser.parse_args()
    device = torch.device(args.device)

    # 1) Run MegaFS to get a swapped image
    swap_path = run_megafs_swap(args.config, args.src_idx, args.tgt_idx, args.swap_out)
    os.makedirs(args.apt_outdir, exist_ok=True)

    # 2) Build generator and classifier for refinement
    g = build_generator(device, args.weights_dir, args.image_size)
    clf = classifier_setup(device, args.classifier)

    # 3) Use the swapped image as the PTI target
    target = load_image(swap_path, args.image_size, device)

    # 4) Inversion to get pivot
    w = inversion_stage(g, target, steps=args.inv_steps, lr=args.inv_lr, device=device)
    pivot_path = os.path.join(args.apt_outdir, "pivot_w.pt")
    torch.save({"w": w.detach().cpu()}, pivot_path)

    # 5) PTI-style adversarial refinement
    final_img, g_tuned = pti_adversarial_stage(
        g=g,
        w=w,
        target_img=target,
        clf=clf,
        steps=args.pti_steps,
        lr=args.pti_lr,
        device=device,
        mode=args.mode,
        true_class=args.true_class,
        target_class=args.target_class,
    )

    # 6) Save results
    save_image(final_img, os.path.join(args.apt_outdir, "apt_on_megafs.png"))
    save_image(target, os.path.join(args.apt_outdir, "megafs_input_target.png"))
    torch.save(g_tuned.state_dict(), os.path.join(args.apt_outdir, "stylegan2_tuned.pt"))

    print("Done. MegaFS swap created and APT-style refinement applied.")


if __name__ == "__main__":
    main()


