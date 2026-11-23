from __future__ import annotations
import argparse, json
from .inference import predict_image


def main():
    p = argparse.ArgumentParser(description="Predict visible wheel attributes from a photograph")
    p.add_argument("--image", required=True)
    p.add_argument("--checkpoint", default="./runs/best.pt")
    p.add_argument("--reference-diameter-mm", type=float, default=None,
                   help="Known/measured outer diameter. Without it only ratios are returned, not invented millimetres.")
    p.add_argument("--export-dir", default=None)
    a = p.parse_args()
    result = predict_image(a.image, a.checkpoint, a.reference_diameter_mm, a.export_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
