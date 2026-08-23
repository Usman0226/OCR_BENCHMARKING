#!/usr/bin/env python3
import json
import sys
from pathlib import Path

from PIL import Image
import pytesseract
from pytesseract import Output
from paddleocr import PaddleOCR
from tqdm import tqdm


def run_tesseract(path: Path) -> list[dict]:
    """Run Tesseract and return boxes in a uniform format."""
    d = pytesseract.image_to_data(
        Image.open(path), lang="fra", output_type=Output.DICT
    )
    boxes = []
    for i in range(len(d["text"])):
        if d["text"][i].strip():
            # Tesseract returns left, top, width, height
            x, y, w, h = (
                d["left"][i],
                d["top"][i],
                d["width"][i],
                d["height"][i],
            )
            boxes.append(
                {
                    "bbox": {
                        "x_min": x,
                        "y_min": y,
                        "x_max": x + w,
                        "y_max": y + h,
                    }
                }
            )
    return boxes


def run_paddle(path: Path, ocr: PaddleOCR) -> list[dict]:
    """Run PaddleOCR and return boxes in a uniform format."""
    result = ocr.ocr(str(path))
    boxes = []
    if not result or result[0] is None:
        return boxes

    for line in result[0]:
        pts = line[0]
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]

        x_min, y_min = min(xs), min(ys)
        x_max, y_max = max(xs), max(ys)
        boxes.append(
            {
                "bbox": {
                    "x_min": x_min,
                    "y_min": y_min,
                    "x_max": x_max,
                    "y_max": y_max,
                }
            }
        )
    return boxes


def main():
    images_dir = Path("images")
    output_dir = Path("ocr_output")

    if not images_dir.exists():
        print(f"Error: {images_dir} does not exist.")
        sys.exit(1)

    images = sorted(list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png")))

    if not images:
        print(f"No images found in {images_dir}")
        sys.exit(1)

    tess_dir = output_dir / "tesseract"
    tess_dir.mkdir(parents=True, exist_ok=True)

    paddle_dir = output_dir / "paddle"
    paddle_dir.mkdir(parents=True, exist_ok=True)

    print(f"Found {len(images)} images.")

    print("Initializing PaddleOCR...")
    paddle_model = PaddleOCR(lang="fr", use_angle_cls=True, show_log=False)

    for img_path in tqdm(images, desc="Processing images"):
        # Run Tesseract
        tess_boxes = run_tesseract(img_path)
        with open(tess_dir / f"{img_path.stem}.json", "w", encoding="utf-8") as f:
            json.dump({"words": tess_boxes}, f, indent=2)

        # Run Paddle
        paddle_boxes = run_paddle(img_path, paddle_model)
        with open(paddle_dir / f"{img_path.stem}.json", "w", encoding="utf-8") as f:
            json.dump({"words": paddle_boxes}, f, indent=2)

    print(f"Done! Results saved in {output_dir}/")


if __name__ == "__main__":
    main()
