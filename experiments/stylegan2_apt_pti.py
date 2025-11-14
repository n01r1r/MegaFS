import argparse
import os
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
from PIL import Image

from models.stylegan2 import Generator
from models.weight_loaders import StyleGAN2WeightLoader


def load_image(image_path: str, image_size: int, device: torch.device) -> torch.Tensor:
    transform = T.Compose([
        T.Resize((image_size, image_size), interpolation=T.InterpolationMode.BILINEAR),
        T.ToTensor(),
        # StyleGAN2 expects inputs in [-1, 1]
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    img = Image.open(image_path).convert("RGB")
    return transform(img).unsqueeze(0).to(device)


def save_image(tensor: torch.Tensor, out_path: str) -> None:
    inv = T.Compose([
        T.Normalize(mean=[0.0, 0.0, 0.0], std=[2.0, 2.0, 2.0]),
        T.Normalize(mean=[-0.5, -0.5, -0.5], std=[1.0, 1.0, 1.0]),
        T.ToPILImage(),
    ])
    img = tensor.detach().clamp_(-1, 1).cpu().squeeze(0)
    img = inv(img)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    img.save(out_path)


def build_generator(device: torch.device, weights_dir: str, image_size: int) -> Generator:
    # Typical StyleGAN2 FFHQ config
    g = Generator(size=image_size, style_dim=512, n_mlp=8, channel_multiplier=2).to(device)
    loader = StyleGAN2WeightLoader(weights_dir)
    ckpt = loader.load_stylegan2_weights()
    if ckpt and "g_ema" in ckpt:
        missing, unexpected = g.load_state_dict(ckpt["g_ema"], strict=False)
        if missing:
            print("WARNING: Missing G keys:", missing)
        if unexpected:
            print("WARNING: Unexpected G keys:", unexpected)
    else:
        print("WARNING: StyleGAN2 weights not found; using randomly initialized generator.")
    g.eval()
    return g


def synthesize(g: Generator, w: torch.Tensor) -> torch.Tensor:
    # w: [1, 512]
    styles: List[torch.Tensor] = [w]
    image, _ = g(
        strucs=None,
        styles=styles,
        return_latents=False,
        input_is_latent=True,
        randomize_noise=False,
    )
    return image


def get_lpips(device: torch.device):
    try:
        import lpips  # type: ignore
        return lpips.LPIPS(net='vgg').to(device)
    except Exception:
        print("WARNING: lpips not available; falling back to L2 only.")
        return None


def inversion_stage(
    g: Generator,
    target_img: torch.Tensor,
    steps: int,
    lr: float,
    device: torch.device,
) -> torch.Tensor:
    g.eval()
    w = torch.randn(1, 512, device=device, requires_grad=True)
    optimizer = optim.Adam([w], lr=lr)
    perceptual = get_lpips(device)
    l2 = nn.MSELoss()

    for step in range(steps):
        optimizer.zero_grad()
        synth = synthesize(g, w)
        loss = l2(synth, target_img)
        if perceptual is not None:
            loss = loss + 0.1 * perceptual((synth + 1) / 2, (target_img + 1) / 2)
        loss.backward()
        optimizer.step()
        if (step + 1) % max(1, steps // 10) == 0:
            print(f"[Inversion] step {step+1}/{steps} loss={float(loss):.4f}")
    return w.detach()


def select_tunable_params(g: Generator, tune_last_n: int = 2) -> List[nn.Parameter]:
    # Tune the last N ToRGB + conv blocks (higher resolutions) for stability
    params: List[nn.Parameter] = []
    # Collect last N to_rgb modules
    to_rgbs = list(g.to_rgbs)[-tune_last_n:]
    for m in to_rgbs:
        params += list(m.parameters())
    # Collect last N conv pairs
    convs = list(g.convs)[-2 * tune_last_n:]
    for m in convs:
        params += list(m.parameters())
    return params


def classifier_setup(device: torch.device, classifier_name: Optional[str]) -> nn.Module:
    if classifier_name in (None, "resnet50"):
        import torchvision.models as models
        clf = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        clf.eval().to(device)
        return clf
    else:
        # Load torchscript or state dict path
        if os.path.isfile(classifier_name):
            try:
                clf = torch.jit.load(classifier_name, map_location=device)
                clf.eval()
                return clf
            except Exception:
                pass
            sd = torch.load(classifier_name, map_location=device)
            import torchvision.models as models
            clf = models.resnet50()
            clf.load_state_dict(sd)
            clf.eval().to(device)
            return clf
        raise ValueError(f"Unsupported classifier spec: {classifier_name}")


def logits_from_classifier(clf: nn.Module, img: torch.Tensor, size: int) -> torch.Tensor:
    # Convert StyleGAN2 range [-1,1] to classifier expected [0,1] and resize to 224
    x = (img + 1) / 2
    x = torch.clamp(x, 0.0, 1.0)
    x = nn.functional.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
    # Normalization for ImageNet models
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    x = normalize(x.squeeze(0)).unsqueeze(0)
    with torch.no_grad():
        out = clf(x)
    return out


def pti_adversarial_stage(
    g: Generator,
    w: torch.Tensor,
    target_img: torch.Tensor,
    clf: nn.Module,
    steps: int,
    lr: float,
    device: torch.device,
    mode: str,
    true_class: Optional[int],
    target_class: Optional[int],
) -> Tuple[torch.Tensor, Generator]:
    g.train()
    params = select_tunable_params(g, tune_last_n=2)
    optimizer = optim.Adam(params, lr=lr)
    perceptual = get_lpips(device)
    l2 = nn.MSELoss()

    for step in range(steps):
        optimizer.zero_grad()
        synth = synthesize(g, w)
        # Perceptual preservation loss
        loss_recon = l2(synth, target_img)
        if perceptual is not None:
            loss_recon = loss_recon + 0.1 * perceptual((synth + 1) / 2, (target_img + 1) / 2)

        # Classifier adversarial objective
        logits = logits_from_classifier(clf, synth, size=target_img.shape[-1])
        if mode == "untargeted" and true_class is not None:
            # minimize confidence of true class
            adv_loss = -nn.functional.log_softmax(logits, dim=1)[0, true_class]
        elif mode == "targeted" and target_class is not None:
            # maximize confidence of target class
            adv_loss = -nn.functional.log_softmax(logits, dim=1)[0, target_class]
        else:
            raise ValueError("Invalid adversarial config: set mode and class indices correctly")

        loss = loss_recon + 0.5 * adv_loss
        loss.backward()
        optimizer.step()
        if (step + 1) % max(1, steps // 10) == 0:
            print(f"[PTI] step {step+1}/{steps} recon={float(loss_recon):.4f} adv={float(adv_loss):.4f}")

    g.eval()
    final_img = synthesize(g, w)
    return final_img.detach(), g


def main():
    parser = argparse.ArgumentParser(description="PTI-style adversarial refinement on StyleGAN2 (MegaFS)")
    parser.add_argument("--image", type=str, required=True, help="Path to target image")
    parser.add_argument("--weights-dir", type=str, default="weights", help="Directory containing StyleGAN2 weights")
    parser.add_argument("--image-size", type=int, default=1024, help="Generator image size")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--inv-steps", type=int, default=400)
    parser.add_argument("--inv-lr", type=float, default=0.05)
    parser.add_argument("--pti-steps", type=int, default=300)
    parser.add_argument("--pti-lr", type=float, default=1e-4)
    parser.add_argument("--mode", type=str, choices=["untargeted", "targeted"], default="untargeted")
    parser.add_argument("--true-class", type=int, default=None)
    parser.add_argument("--target-class", type=int, default=None)
    parser.add_argument("--classifier", type=str, default="resnet50")
    parser.add_argument("--outdir", type=str, default="outputs/apt_sg2")

    args = parser.parse_args()
    device = torch.device(args.device)

    # Build generator and classifier
    g = build_generator(device, args.weights_dir, args.image_size)
    clf = classifier_setup(device, args.classifier)

    # Load target image
    target = load_image(args.image, args.image_size, device)

    # Inversion to get pivot w
    w = inversion_stage(g, target, steps=args.inv_steps, lr=args.inv_lr, device=device)
    pivot_path = os.path.join(args.outdir, "pivot_w.pt")
    os.makedirs(args.outdir, exist_ok=True)
    torch.save({"w": w.detach().cpu()}, pivot_path)

    recon_img = synthesize(g, w)
    save_image(recon_img, os.path.join(args.outdir, "reconstruction.png"))

    # PTI-style adversarial refinement
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

    save_image(final_img, os.path.join(args.outdir, "apt_pti_result.png"))
    torch.save(g_tuned.state_dict(), os.path.join(args.outdir, "stylegan2_tuned.pt"))
    print("Done. Saved pivot, reconstruction, adversarial result, and tuned generator state.")


if __name__ == "__main__":
    main()


