# OCR Benchmark Framework

Compare **PaddleOCR** and **Tesseract OCR** against manually-annotated ground truth.

```
ocr-position-benchmark/
├── annotations/               # your answer key: one Label Studio JSON per image
├── ocr_output/
│   ├── paddle/                # PaddleOCR boxes:   IMG_051.json  IMG_052.json ...
│   └── tesseract/             # Tesseract boxes:   IMG_051.json  IMG_052.json ...
├── run_ocr.py                 # runs both tools, saves their boxes
├── score.py                   # compares boxes to annotations, writes results
├── results.csv                # per-image scores  (generated)
├── results_summary.md         # final table + observations  (generated)
├── notes.txt                  # anything odd you found in the images
├── README.md
└── requirements.txt
```

---

## Quick Start (Docker)

### Prerequisites

- **Docker Desktop 4.x** with WSL2 backend (Windows 11)
- **Git**

### 1. Clone and configure

```powershell
git clone <repo-url> ocr-position-benchmark
cd ocr-position-benchmark
Copy-Item .env.example .env
notepad .env          # set LABEL_STUDIO_PASSWORD at minimum
```

### 2. Build and start

```powershell
$env:DOCKER_BUILDKIT="1"
docker compose build   # ~15 min first time
docker compose up -d
docker compose ps      # both services should show "healthy"
```

### 3. Open Label Studio → annotate → export

1. Go to [http://localhost:8080](http://localhost:8080)
2. Create a project, upload images, draw bounding boxes + type the text
3. **Export → JSON** → save to `annotations/export.json`

### 4. Place images

```
images/
├── IMG_051.jpg
├── IMG_052.png
└── ...
```

### 5. Run OCR (both engines)

```powershell
docker compose exec ocr python run_ocr.py --engine all --images /workspace/images
```

Outputs written to:
```
ocr_output/paddle/IMG_051.json   ocr_output/paddle/IMG_052.json   ...
ocr_output/tesseract/IMG_051.json  ocr_output/tesseract/IMG_052.json  ...
```

### 6. Score

```powershell
docker compose exec ocr python score.py
```

Produces at project root:
- `results.csv` — Precision / Recall / F1 / Coverage per image per engine
- `results_summary.md` — aggregate comparison table

---

## CLI Reference

```bash
# Run a specific engine
python run_ocr.py --engine paddle    --images images/
python run_ocr.py --engine tesseract --images images/
python run_ocr.py --engine all       --images images/

# Override Tesseract language (default: fra+eng)
python run_ocr.py --engine tesseract --lang fra --images images/

# Score all engines
python score.py

# Score one engine
python score.py --engine paddle

# Point to a specific annotation file
python score.py --annotations annotations/my_export.json

# Optional: visualize bounding boxes
python scripts/visualize.py --engine paddle
```

---

## Make Targets

```bash
make setup          # first-time: copy .env + build + start
make run-all        # OCR both engines
make run-paddle     # OCR PaddleOCR only
make run-tesseract  # OCR Tesseract only
make score          # score + generate reports
make pipeline       # run-all + score + visualize in one step
make test           # run unit tests
make lint           # run ruff linter
make shell          # bash inside the ocr container
make clean-outputs  # delete generated outputs (keeps source)
make help           # list all targets
```

---

## Installation (without Docker)

If you prefer to run locally:

```bash
# Tesseract (Ubuntu/Debian)
sudo apt-get install tesseract-ocr tesseract-ocr-eng tesseract-ocr-fra

# Python deps
pip install paddlepaddle==2.6.2 -f https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html
pip install -r requirements.txt

# Run
python run_ocr.py --engine all --images images/
python score.py
```

---

## Adding a New OCR Engine

1. Create `ocr/engines/my_engine.py` implementing the `OCREngine` ABC
2. Add `MY_ENGINE = "my_engine"` to `EngineType` in `ocr/core/models.py`
3. Register in `ocr/engines/base.py → create_engine()`
4. Run: `python run_ocr.py --engine my_engine`

Output goes to `ocr_output/my_engine/` automatically.

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| Docker not running | Start Docker Desktop from Start menu |
| `healthy` never shows | `docker compose logs ocr` to inspect |
| Label Studio at 8080 unreachable | `docker compose ps` — check state |
| PaddleOCR re-downloads models | `paddle_models` volume was deleted — normal on first run |
| Tesseract missing language | Inside container: `tesseract --list-langs` |
| Permission errors on outputs | All output dirs are owned by bind-mount user |

---

## Architecture

See [docs/architecture.md](docs/architecture.md) for the full design.  
See [docs/setup.md](docs/setup.md) for the step-by-step Windows setup guide.

---

## License

MIT — see [LICENSE](LICENSE).
