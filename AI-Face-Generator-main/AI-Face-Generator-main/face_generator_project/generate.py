"""Generate synthetic face images using the included trained generator weights."""
import argparse
from pathlib import Path
import torch
from torchvision.utils import save_image
from model import Generator

def main():
    parser = argparse.ArgumentParser(description="Generate 64x64 images with the supplied DCGAN generator.")
    parser.add_argument("--count", type=int, default=1, help="Number of images to generate")
    parser.add_argument("--output", default="outputs", help="Output directory")
    parser.add_argument("--seed", type=int, default=None, help="Optional random seed for repeatable output")
    args = parser.parse_args()

    if args.count < 1:
        parser.error("--count must be at least 1")
    if args.seed is not None:
        torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = Path(__file__).with_name("generator.pth")
    model = Generator().to(device)
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        noise = torch.randn(args.count, 100, 1, 1, device=device)
        images = model(noise)
        for i, image in enumerate(images, start=1):
            path = output_dir / f"generated_face_{i:03d}.png"
            save_image(image, path, normalize=True)
            print(f"Saved: {path}")
    print(f"Device: {device}")

if __name__ == "__main__":
    main()
