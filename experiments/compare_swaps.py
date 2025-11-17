import argparse
import json
import os
import sys
from typing import Dict, List

import cv2

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from utils.attack_utils import compute_metrics  # noqa: E402


def load_image(path: str):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def compare_swaps(exp_dir: str) -> Dict[str, Dict[str, float]]:
    pairs = [
        ("swap_CC.jpg", "swap_AC.jpg", "CC_vs_AC"),
        ("swap_CA.jpg", "swap_AA.jpg", "CA_vs_AA"),
    ]
    results = {}
    for clean_name, adv_name, label in pairs:
        clean_path = os.path.join(exp_dir, clean_name)
        adv_path = os.path.join(exp_dir, adv_name)
        if not (os.path.exists(clean_path) and os.path.exists(adv_path)):
            continue
        clean_img = load_image(clean_path)
        adv_img = load_image(adv_path)
        metrics = compute_metrics(clean_img, adv_img)
        results[label] = metrics
    if not results:
        raise RuntimeError(f"No swap images found in {exp_dir}")
    out_path = os.path.join(exp_dir, "swap_comparison_metrics.json")

    def _to_serializable(obj):
        if isinstance(obj, (float, int, str)):
            return obj
        try:
            return float(obj)
        except Exception:
            return obj

    serializable_results = {
        label: {k: _to_serializable(v) for k, v in metrics.items()}
        for label, metrics in results.items()
    }

    with open(out_path, "w") as f:
        json.dump(serializable_results, f, indent=2)
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Compare swap outputs (CC vs AC, CA vs AA) and compute metrics."
    )
    parser.add_argument(
        "exp_dirs",
        nargs="+",
        help="One or more experiment directories (each containing swap_*.jpg files)",
    )
    args = parser.parse_args()

    for exp_dir in args.exp_dirs:
        print(f"\n=== {exp_dir} ===")
        try:
            metrics = compare_swaps(exp_dir)
        except Exception as exc:
            print(f"[ERROR] {exc}")
            continue
        for label, vals in metrics.items():
            l2 = vals.get("L2_norm", 0.0)
            linf = vals.get("Linf_norm", 0.0)
            ssim = vals.get("SSIM", 0.0)
            lpips = vals.get("LPIPS", "n/a")
            if isinstance(lpips, (int, float)):
                lpips = f"{lpips:.6f}"
            print(
                f"{label}: L2={l2:.2f} Linf={linf:.2f} SSIM={ssim:.6f} LPIPS={lpips}"
            )
        print(f"Metrics saved to {os.path.join(exp_dir, 'swap_comparison_metrics.json')}")


if __name__ == "__main__":
    main()

