# Architecture

## Overview

The OCR Benchmark Framework is organized into five layers with strict dependency flow:

```
┌─────────────────────────────────────────────────────────────┐
│  CLI Scripts (scripts/)                                      │
│  run_ocr.py │ score.py │ visualize.py │ export_results.py   │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────┐
│  OCR Engines (ocr/engines/)                                  │
│  OCREngine ABC → PaddleEngine │ TesseractEngine             │
└──────────────────────┬──────────────────────────────────────┘
                       │ NormalizedWord[]
┌──────────────────────▼──────────────────────────────────────┐
│  Core Pipeline (ocr/core/)                                   │
│  models │ geometry │ matching │ scoring │ io │ config │ log  │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────┐
│  Utilities (ocr/utils/)                                      │
│  image.py │ files.py                                         │
└─────────────────────────────────────────────────────────────┘
```

Dependencies only flow **downward**. The CLI layer knows about engines. Engines know about core models. Core knows about utils. Utils have no internal dependencies.

---

## Design Decisions

### Strategy Pattern for OCR Engines

The `OCREngine` abstract base class defines a **fixed interface** that every engine must implement:

```python
class OCREngine(ABC):
    def initialize(self) -> None: ...     # Load models
    def run(self, path: Path) -> OCRResult: ...  # Public entry point
    def _run_raw(self, path: Path) -> list[NormalizedWord]: ...  # Engine-specific
    def shutdown(self) -> None: ...       # Release resources
```

`run()` is **final** (not overrideable): it handles timing, error capture, and logging. Only `_run_raw()` is implemented per-engine. This ensures:
- All engines log consistently
- Error handling is uniform
- Timing is always measured

### Normalized Output Format

Every engine outputs `NormalizedWord`:

```python
@dataclass
class NormalizedWord:
    text: str
    bbox: BoundingBox      # axis-aligned, pixel coordinates
    confidence: float      # 0.0 – 1.0
    page: int              # 1-indexed
    image_name: str
    engine: EngineType
    polygon: Polygon | None  # original quadrilateral if available
```

All downstream code (matcher, scorer, reporter, visualizer) only knows about `NormalizedWord`. Adding a new engine never requires changing the downstream layers.

### Greedy Bipartite Matching

Word matching is a bipartite assignment problem:
- Left nodes: OCR words
- Right nodes: annotation words
- Edge weight: composite score = (IoU + text_similarity) / 2

The greedy algorithm:
1. Enumerate all candidate pairs (OCR × annotation) where the matching criterion is met
2. Sort by composite score descending
3. Greedily assign: once a node is claimed, skip it

**Why greedy instead of Hungarian algorithm?**
For typical OCR document pages (100–500 words), greedy is O(n²) and fast enough. The optimal Hungarian solution is O(n³). Greedy produces near-optimal results when the IoU threshold filters most non-matching pairs, leaving a sparse bipartite graph. If needed, the matcher can be swapped for `scipy.optimize.linear_sum_assignment` without changing the interface.

### Three Matching Strategies

| Strategy | Matches on | Use when |
|----------|-----------|----------|
| `iou_only` | Spatial overlap ≥ threshold | Text extraction is unreliable |
| `iou_and_text` (default) | Spatial + text similarity | Normal benchmarking |
| `text_only` | Text similarity only | Layout-free matching |

Controlled via `configs/config.yaml → matching.strategy`.

### Configuration Hierarchy

```
Environment variables  (highest precedence)
    ↓
configs/config.yaml
    ↓
Python dataclass defaults  (lowest precedence)
```

All paths accept environment variable overrides, enabling Docker volumes to override YAML paths without modifying the config file.

---

## Data Flow

### 1. OCR Pipeline

```
Image file
    │
    ▼
OCREngine.run(image_path)
    │  (timing, error handling)
    ▼
OCREngine._run_raw(image_path)
    │  (engine-specific: PaddleOCR, Tesseract, ...)
    ▼
list[NormalizedWord]
    │
    ▼
OCRResult saved as JSON
(ocr_output/{image_stem}_{engine}.json)
```

### 2. Scoring Pipeline

```
OCR JSON files           Label Studio export.json
       │                          │
       ▼                          ▼
 load_all_ocr_results()   load_label_studio_annotations()
       │                          │
       └──────────┬───────────────┘
                  ▼
           match_words()
           (greedy bipartite)
                  │
                  ▼
           score_image()
           (P/R/F1/Coverage)
                  │
                  ▼
        AggregateScoringResult
                  │
         ┌────────┼────────┐
         ▼        ▼        ▼
        CSV      JSON   Markdown
```

---

## Label Studio Integration

Label Studio uses **percentage-based coordinates** (0–100) for bounding boxes. The `geometry.denormalize_bbox()` function converts them to pixel coordinates before matching.

Supported annotation types:
- `rectanglelabels` — standard bounding box
- `polygonlabels` — arbitrary polygon (converted to AABB for matching)

Text transcription is read from the `text` field within each result value. In Label Studio, this requires a `<TextArea perRegion="true">` component in the labeling template.

---

## Adding a New OCR Engine

See the [README](../README.md#adding-a-new-ocr-engine) for the step-by-step guide.

The key principle: **one new file, zero changes to existing files** (except `EngineType` enum + factory registration).

### Example: Adding EasyOCR

```python
# ocr/engines/easyocr_engine.py
from ocr.engines.base import OCREngine
from ocr.core.models import EngineType, NormalizedWord, BoundingBox
from pathlib import Path

class EasyOCREngine(OCREngine):
    engine_type = EngineType.EASYOCR

    def initialize(self) -> None:
        import easyocr
        cfg = self._config.engines.easyocr  # add config section
        self._reader = easyocr.Reader(
            cfg.languages, gpu=False
        )
        self._initialized = True

    def shutdown(self) -> None:
        self._reader = None
        self._initialized = False

    def _run_raw(self, image_path: Path) -> list[NormalizedWord]:
        results = self._reader.readtext(str(image_path))
        words = []
        for bbox_pts, text, conf in results:
            from ocr.core.models import Polygon
            polygon = Polygon.from_list(bbox_pts)
            words.append(NormalizedWord(
                text=text,
                bbox=polygon.to_bounding_box(),
                confidence=float(conf),
                page=1,
                image_name=image_path.name,
                engine=self.engine_type,
                polygon=polygon,
            ))
        return words
```

---

## Logging Architecture

Three rotating log files, all structured (timestamp | level | logger | message):

| File | Content | Level |
|------|---------|-------|
| `application.log` | All events | ≥ configured level |
| `ocr.log` | Engine events only | ≥ configured level |
| `errors.log` | Warnings and errors | WARNING+ |

All loggers are under the `ocr_benchmark` hierarchy, ensuring `propagate=True` delivers events to the root handler (console + application.log) automatically.

---

## Testing Strategy

| Layer | Test file | Approach |
|-------|-----------|---------|
| Geometry | `test_geometry.py` | Pure function tests, parametrized edge cases |
| Matching | `test_matching.py` | Known-output scenarios for all three strategies |
| Scoring | `test_scoring.py` | Perfect/partial/empty cases, aggregate metrics |
| Models | `test_models.py` | Serialization roundtrips, validation errors |

**No mocking of the OCR engines** in unit tests — engine tests are integration-level and run inside the container against real images. Unit tests are fully deterministic with no I/O.
