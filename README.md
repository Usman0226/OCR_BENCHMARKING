# OCR Position Benchmark

Compare **PaddleOCR** and **Tesseract OCR** against manually-annotated ground truth.

## Project Structure

```
ocr-position-benchmark/
├── annotations/               # answer key: one Label Studio JSON per image
├── ocr_output/
│   ├── paddle/                # PaddleOCR boxes: IMG_051.json ...
│   └── tesseract/             # Tesseract boxes: IMG_051.json ...
├── run_ocr.py                 # runs both tools, saves their boxes
├── score.py                   # compares boxes to annotations, writes results
├── results.csv                # per-image scores (generated)
├── results_summary.md         # final table + observations
├── notes.txt                  # documentation of excluded/odd images
├── README.md                  # this file
└── requirements.txt           # dependencies
```

## Installation

```bash
# 1. Install Tesseract 
sudo apt-get update
sudo apt-get install tesseract-ocr tesseract-ocr-eng tesseract-ocr-fra

# 2. Install Python dependencies
pip install paddlepaddle==2.6.2 -f https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html
pip install -r requirements.txt
```

*(Note: PaddleOCR model download occurs automatically on the first run)*

## Usage

1. **Run OCR on all images in `images/`:**
   ```bash
   python run_ocr.py
   ```
   *This processes all `.jpg` and `.png` files in the `images/` directory and outputs the results as JSON into `ocr_output/paddle/` and `ocr_output/tesseract/`.*

2. **Score the results against annotations:**
   ```bash
   python score.py --images images/ --ann annotations/ --out results.csv
   ```
   *This compares the generated OCR boxes with the Label Studio annotations and writes the per-image metrics to `results.csv`.*

3. **Check the final summary:**
   Review `results_summary.md` for aggregate statistics, the list of worst-performing images, and detailed observations.
