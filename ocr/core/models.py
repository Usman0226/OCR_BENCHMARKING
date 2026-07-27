"""Core data models for the OCR Benchmark Framework.

All OCR engines normalize their output to the types defined here.
Downstream pipeline (matching, scoring, reporting) only ever sees these types.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


# =============================================================================
# Enums
# =============================================================================


class EngineType(str, Enum):
    """Supported OCR engine identifiers."""

    PADDLE = "paddle"
    TESSERACT = "tesseract"
    # Extend here when adding new engines — no other files need changing.

    @classmethod
    def from_str(cls, value: str) -> "EngineType":
        """Parse engine name case-insensitively."""
        try:
            return cls(value.lower())
        except ValueError:
            valid = [e.value for e in cls]
            raise ValueError(f"Unknown engine '{value}'. Valid options: {valid}") from None


class MatchStatus(str, Enum):
    """Result of matching an OCR word against an annotation word."""

    TRUE_POSITIVE = "true_positive"
    FALSE_POSITIVE = "false_positive"
    FALSE_NEGATIVE = "false_negative"


# =============================================================================
# Bounding Box
# =============================================================================


@dataclass(frozen=True)
class BoundingBox:
    """Axis-aligned bounding box in pixel coordinates.

    Origin is top-left corner (standard image coordinate system).
    """

    x_min: float
    y_min: float
    x_max: float
    y_max: float

    def __post_init__(self) -> None:
        if self.x_min > self.x_max:
            raise ValueError(f"x_min ({self.x_min}) must be <= x_max ({self.x_max})")
        if self.y_min > self.y_max:
            raise ValueError(f"y_min ({self.y_min}) must be <= y_max ({self.y_max})")

    @property
    def width(self) -> float:
        return self.x_max - self.x_min

    @property
    def height(self) -> float:
        return self.y_max - self.y_min

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center(self) -> tuple[float, float]:
        return (self.x_min + self.x_max) / 2, (self.y_min + self.y_max) / 2

    @classmethod
    def from_points(cls, points: list[list[float]]) -> "BoundingBox":
        """Build an AABB from an arbitrary list of [x, y] points."""
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        return cls(x_min=min(xs), y_min=min(ys), x_max=max(xs), y_max=max(ys))

    def to_dict(self) -> dict[str, float]:
        return asdict(self)

    def to_xywh(self) -> tuple[float, float, float, float]:
        """Return (x, y, width, height) format."""
        return self.x_min, self.y_min, self.width, self.height


# =============================================================================
# Polygon
# =============================================================================


@dataclass(frozen=True)
class Polygon:
    """Arbitrary polygon defined by an ordered list of (x, y) vertices.

    Used to store the raw quadrilateral output from OCR engines.
    """

    points: tuple[tuple[float, float], ...]

    @classmethod
    def from_list(cls, points: list[list[float]]) -> "Polygon":
        """Construct from a list of [x, y] pairs."""
        return cls(points=tuple((float(p[0]), float(p[1])) for p in points))

    def to_bounding_box(self) -> BoundingBox:
        """Convert polygon to its enclosing AABB."""
        xs = [p[0] for p in self.points]
        ys = [p[1] for p in self.points]
        return BoundingBox(
            x_min=min(xs), y_min=min(ys), x_max=max(xs), y_max=max(ys)
        )

    def to_list(self) -> list[list[float]]:
        return [[p[0], p[1]] for p in self.points]


# =============================================================================
# Normalized OCR Word (unified output format for all engines)
# =============================================================================


@dataclass
class NormalizedWord:
    """Single OCR-detected word in normalized format.

    Every OCR engine must produce a list of these objects.
    Downstream matching and scoring only operates on this type.
    """

    text: str
    bbox: BoundingBox
    confidence: float                   # 0.0 – 1.0
    page: int                           # 1-indexed page number
    image_name: str                     # Filename of source image (no path)
    engine: EngineType
    polygon: Polygon | None = None      # Original quadrilateral if available

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(
                f"Confidence must be in [0, 1], got {self.confidence}"
            )
        if self.page < 1:
            raise ValueError(f"Page must be >= 1, got {self.page}")

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dictionary."""
        return {
            "text": self.text,
            "bbox": self.bbox.to_dict(),
            "confidence": self.confidence,
            "page": self.page,
            "image_name": self.image_name,
            "engine": self.engine.value,
            "polygon": self.polygon.to_list() if self.polygon else None,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "NormalizedWord":
        """Deserialize from a dictionary (e.g. loaded JSON)."""
        bbox_data = data["bbox"]
        bbox = BoundingBox(
            x_min=bbox_data["x_min"],
            y_min=bbox_data["y_min"],
            x_max=bbox_data["x_max"],
            y_max=bbox_data["y_max"],
        )
        polygon: Polygon | None = None
        if data.get("polygon"):
            polygon = Polygon.from_list(data["polygon"])
        return cls(
            text=data["text"],
            bbox=bbox,
            confidence=float(data["confidence"]),
            page=int(data["page"]),
            image_name=data["image_name"],
            engine=EngineType(data["engine"]),
            polygon=polygon,
        )


# =============================================================================
# OCR Result (per image)
# =============================================================================


@dataclass
class OCRResult:
    """All normalized words extracted from a single image by one engine."""

    image_name: str
    image_path: Path
    engine: EngineType
    words: list[NormalizedWord] = field(default_factory=list)
    processing_time_s: float = 0.0
    error: str | None = None

    @property
    def word_count(self) -> int:
        return len(self.words)

    @property
    def succeeded(self) -> bool:
        return self.error is None

    def to_dict(self) -> dict[str, Any]:
        return {
            "image_name": self.image_name,
            "image_path": str(self.image_path),
            "engine": self.engine.value,
            "words": [w.to_dict() for w in self.words],
            "processing_time_s": self.processing_time_s,
            "error": self.error,
        }


# =============================================================================
# Annotation (ground truth from Label Studio)
# =============================================================================


@dataclass
class AnnotationWord:
    """A single annotated word from Label Studio."""

    text: str
    bbox: BoundingBox
    label: str                    # Label Studio label class
    annotation_id: str
    image_name: str
    page: int = 1
    polygon: Polygon | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "bbox": self.bbox.to_dict(),
            "label": self.label,
            "annotation_id": self.annotation_id,
            "image_name": self.image_name,
            "page": self.page,
            "polygon": self.polygon.to_list() if self.polygon else None,
        }


@dataclass
class ImageAnnotation:
    """All ground-truth words for a single image."""

    image_name: str
    words: list[AnnotationWord] = field(default_factory=list)
    task_id: int | None = None

    @property
    def word_count(self) -> int:
        return len(self.words)


# =============================================================================
# Matched Word (output of the matching step)
# =============================================================================


@dataclass
class MatchedWord:
    """Result of matching one OCR word to one annotation word."""

    status: MatchStatus
    ocr_word: NormalizedWord | None        # None for false negatives
    annotation_word: AnnotationWord | None  # None for false positives
    iou: float = 0.0
    text_similarity: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "ocr_word": self.ocr_word.to_dict() if self.ocr_word else None,
            "annotation_word": (
                self.annotation_word.to_dict() if self.annotation_word else None
            ),
            "iou": self.iou,
            "text_similarity": self.text_similarity,
        }


# =============================================================================
# Scoring Result (per image, per engine)
# =============================================================================


@dataclass
class ImageScoringResult:
    """Scoring metrics for a single image and engine pair."""

    image_name: str
    engine: EngineType

    # Counts
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0

    # Metrics (0–1)
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    coverage: float = 0.0          # Fraction of annotation words matched

    # Word-level lists
    missed_words: list[str] = field(default_factory=list)    # In annotation, not in OCR
    invented_words: list[str] = field(default_factory=list)  # In OCR, not in annotation
    matched_words: list[MatchedWord] = field(default_factory=list)

    # Timing
    processing_time_s: float = 0.0
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "image_name": self.image_name,
            "engine": self.engine.value,
            "true_positives": self.true_positives,
            "false_positives": self.false_positives,
            "false_negatives": self.false_negatives,
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "coverage": round(self.coverage, 4),
            "missed_words": self.missed_words,
            "invented_words": self.invented_words,
            "processing_time_s": self.processing_time_s,
            "error": self.error,
        }

    def to_flat_dict(self) -> dict[str, Any]:
        """Flat dict suitable for CSV rows (excludes word lists)."""
        return {
            "image_name": self.image_name,
            "engine": self.engine.value,
            "true_positives": self.true_positives,
            "false_positives": self.false_positives,
            "false_negatives": self.false_negatives,
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "coverage": round(self.coverage, 4),
            "missed_word_count": len(self.missed_words),
            "invented_word_count": len(self.invented_words),
            "processing_time_s": self.processing_time_s,
            "error": self.error,
        }


@dataclass
class AggregateScoringResult:
    """Macro-averaged metrics across all images for one engine."""

    engine: EngineType
    image_count: int
    total_true_positives: int = 0
    total_false_positives: int = 0
    total_false_negatives: int = 0
    mean_precision: float = 0.0
    mean_recall: float = 0.0
    mean_f1: float = 0.0
    mean_coverage: float = 0.0
    total_missed_words: int = 0
    total_invented_words: int = 0
    per_image: list[ImageScoringResult] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "engine": self.engine.value,
            "image_count": self.image_count,
            "total_true_positives": self.total_true_positives,
            "total_false_positives": self.total_false_positives,
            "total_false_negatives": self.total_false_negatives,
            "mean_precision": round(self.mean_precision, 4),
            "mean_recall": round(self.mean_recall, 4),
            "mean_f1": round(self.mean_f1, 4),
            "mean_coverage": round(self.mean_coverage, 4),
            "total_missed_words": self.total_missed_words,
            "total_invented_words": self.total_invented_words,
            "per_image": [r.to_dict() for r in self.per_image],
        }

    def save_json(self, path: Path) -> None:
        """Serialize to a JSON file."""
        path.write_text(
            json.dumps(self.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
