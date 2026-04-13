"""
DeeperForensics-style augmentations for deepfake detection training.
Adapted from gasbench's processing/transforms.py.

Augmentation levels:
    0 — none (clean)
    1 — basic (light JPEG, mild blur)
    2 — medium (moderate distortions)
    3 — hard (heavy compression, noise, blur)
"""

import random
import numpy as np
from PIL import Image
import cv2


def jpeg_compress(img: np.ndarray, quality: int) -> np.ndarray:
    """Apply JPEG compression artifact."""
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, buf = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR), encode_param)
    decoded = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    return cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)


def gaussian_blur(img: np.ndarray, sigma: float) -> np.ndarray:
    """Apply Gaussian blur."""
    ksize = int(6 * sigma + 1) | 1  # ensure odd
    return cv2.GaussianBlur(img, (ksize, ksize), sigma)


def gaussian_noise(img: np.ndarray, sigma: float) -> np.ndarray:
    """Add Gaussian noise."""
    noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
    noisy = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return noisy


def color_shift(img: np.ndarray, magnitude: float) -> np.ndarray:
    """Random per-channel color shift."""
    shifts = np.random.uniform(-magnitude, magnitude, 3).astype(np.float32)
    shifted = np.clip(img.astype(np.float32) + shifts[None, None, :], 0, 255)
    return shifted.astype(np.uint8)


def color_contrast(img: np.ndarray, factor: float) -> np.ndarray:
    """Adjust contrast."""
    mean = img.mean(axis=(0, 1), keepdims=True)
    adjusted = np.clip(mean + (img.astype(np.float32) - mean) * factor, 0, 255)
    return adjusted.astype(np.uint8)


AUGMENTATION_LEVELS = {
    0: [],
    1: [
        lambda img: jpeg_compress(img, random.randint(70, 95)),
        lambda img: gaussian_blur(img, random.uniform(0.3, 0.8)),
    ],
    2: [
        lambda img: jpeg_compress(img, random.randint(40, 75)),
        lambda img: gaussian_blur(img, random.uniform(0.5, 1.5)),
        lambda img: gaussian_noise(img, random.uniform(3, 10)),
        lambda img: color_shift(img, random.uniform(5, 15)),
    ],
    3: [
        lambda img: jpeg_compress(img, random.randint(15, 45)),
        lambda img: gaussian_blur(img, random.uniform(1.0, 3.0)),
        lambda img: gaussian_noise(img, random.uniform(8, 25)),
        lambda img: color_shift(img, random.uniform(10, 30)),
        lambda img: color_contrast(img, random.uniform(0.5, 1.5)),
    ],
}


def apply_random_augmentations(
    img: np.ndarray,
    level: int = 1,
    max_augs: int = 2,
) -> np.ndarray:
    """
    Apply random augmentations from the specified level.

    Args:
        img: HWC uint8 RGB image.
        level: 0 (none) through 3 (hard).
        max_augs: Maximum augmentations to apply per image.

    Returns:
        Augmented HWC uint8 RGB image.
    """
    if level == 0:
        return img

    candidates = AUGMENTATION_LEVELS.get(level, AUGMENTATION_LEVELS[1])
    n = random.randint(1, min(max_augs, len(candidates)))
    chosen = random.sample(candidates, n)

    for aug_fn in chosen:
        img = aug_fn(img)

    return img


def resize_image(img: np.ndarray, target_size: tuple[int, int] = (224, 224)) -> np.ndarray:
    """Resize image to target HxW, preserving uint8 dtype."""
    pil_img = Image.fromarray(img)
    pil_img = pil_img.resize((target_size[1], target_size[0]), Image.BILINEAR)
    return np.array(pil_img)


def random_horizontal_flip(img: np.ndarray, p: float = 0.5) -> np.ndarray:
    """Random horizontal flip."""
    if random.random() < p:
        return np.ascontiguousarray(img[:, ::-1])
    return img
