"""Image loading and preprocessing utilities."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from ocr.core.logger import get_logger

logger = get_logger(__name__)

# Supported image extensions
SUPPORTED_EXTENSIONS: frozenset[str] = frozenset(
    {".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp", ".webp"}
)


def load_image_rgb(image_path: Path) -> np.ndarray:
    """Load an image as an RGB NumPy array using OpenCV.

    Args:
        image_path: Path to the image file.

    Returns:
        RGB image array of shape (H, W, 3), dtype uint8.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file cannot be decoded as an image.
    """
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"OpenCV could not decode image: {image_path}")

    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def load_image_pil(image_path: Path) -> Image.Image:
    """Load an image as a PIL Image.

    Args:
        image_path: Path to the image file.

    Returns:
        PIL Image in RGB mode.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    img = Image.open(image_path).convert("RGB")
    return img


def get_image_dimensions(image_path: Path) -> tuple[int, int]:
    """Return (width, height) for the given image without loading pixel data.

    Uses PIL's lazy loading — does not decode the full image.

    Args:
        image_path: Path to the image file.

    Returns:
        (width, height) in pixels.
    """
    with Image.open(image_path) as img:
        return img.size  # (width, height)


def preprocess_for_tesseract(
    image: Image.Image,
    grayscale: bool = True,
    max_long_edge: int | None = 4096,
) -> Image.Image:
    """Apply preprocessing steps recommended for Tesseract accuracy.

    Steps (in order):
      1. Resize if the long edge exceeds *max_long_edge*.
      2. Convert to grayscale if *grayscale* is True.

    Args:
        image: Input PIL Image (RGB).
        grayscale: Convert to grayscale before passing to Tesseract.
        max_long_edge: Maximum length of the long edge in pixels.
                       None = no resize.

    Returns:
        Preprocessed PIL Image.
    """
    # Resize if needed
    if max_long_edge is not None:
        w, h = image.size
        long_edge = max(w, h)
        if long_edge > max_long_edge:
            scale = max_long_edge / long_edge
            new_w = int(w * scale)
            new_h = int(h * scale)
            image = image.resize((new_w, new_h), Image.LANCZOS)
            logger.debug(
                "Resized image from %dx%d to %dx%d for Tesseract.", w, h, new_w, new_h
            )

    if grayscale:
        image = image.convert("L")

    return image


def draw_bboxes_on_image(
    image: np.ndarray,
    bboxes: list[tuple[float, float, float, float]],
    colors: list[tuple[int, int, int]],
    thickness: int = 2,
) -> np.ndarray:
    """Draw bounding boxes on an RGB image (modifies a copy).

    Args:
        image: RGB image as numpy array.
        bboxes: List of (x_min, y_min, x_max, y_max) tuples.
        colors: List of (R, G, B) tuples, one per bbox.
        thickness: Line thickness in pixels.

    Returns:
        New RGB image with bounding boxes drawn.
    """
    output = image.copy()
    for (x_min, y_min, x_max, y_max), color in zip(bboxes, colors):
        bgr_color = (color[2], color[1], color[0])  # RGB → BGR for OpenCV
        cv2.rectangle(
            output,
            (int(x_min), int(y_min)),
            (int(x_max), int(y_max)),
            bgr_color,
            thickness,
        )
    return output


def save_image(image: np.ndarray, output_path: Path) -> None:
    """Save an RGB numpy array image to *output_path*.

    Args:
        image: RGB image array.
        output_path: Destination path (parent directory must exist).
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(output_path), bgr)
