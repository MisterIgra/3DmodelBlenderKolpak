"""Preprocessing shared exactly by training, validation and inference."""
from __future__ import annotations

from PIL import Image, ImageOps
from torchvision import transforms
from .constants import IMAGENET_MEAN, IMAGENET_STD, IMAGE_SIZE


class PadToSquare:
    """Resize without distorting the wheel and pad to a square canvas."""
    def __init__(self, size: int, fill=(32, 32, 32)):
        self.size = int(size)
        self.fill = fill

    def __call__(self, image: Image.Image) -> Image.Image:
        image = image.convert("RGB")
        # ImageOps.pad preserves aspect ratio and centres the full object.
        return ImageOps.pad(image, (self.size, self.size), method=Image.Resampling.LANCZOS, color=self.fill, centering=(0.5, 0.5))


def build_transforms(image_size: int = IMAGE_SIZE, train: bool = False):
    ops = [PadToSquare(image_size)]
    if train:
        # A wheel's family and visible-tip count do not change under rotation.
        ops.extend([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=180, interpolation=transforms.InterpolationMode.BILINEAR, fill=(32, 32, 32)),
            transforms.RandomAffine(degrees=0, translate=(0.04, 0.04), scale=(0.75, 1.05), shear=3,
                                    interpolation=transforms.InterpolationMode.BILINEAR, fill=(32, 32, 32)),
            transforms.ColorJitter(brightness=0.18, contrast=0.18, saturation=0.12, hue=0.015),
        ])
    ops.extend([
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return transforms.Compose(ops)
