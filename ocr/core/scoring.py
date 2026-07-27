"""Scoring engine — computes Precision, Recall, F1, Coverage and word lists.

Takes the output of the matching layer and produces ImageScoringResult and
AggregateScoringResult objects ready for report generation.
"""

from __future__ import annotations

import statistics
from typing import TYPE_CHECKING

from ocr.core.logger import get_logger
from ocr.core.models import (
    AggregateScoringResult,
    EngineType,
    ImageAnnotation,
    ImageScoringResult,
    MatchStatus,
    MatchedWord,
    NormalizedWord,
    OCRResult,
)
from ocr.core.matching import match_words

if TYPE_CHECKING:
    from ocr.core.config import MatchingConfig

logger = get_logger(__name__)


# =============================================================================
# Per-image scoring
# =============================================================================


def score_image(
    ocr_result: OCRResult,
    annotation: ImageAnnotation,
    matching_cfg: "MatchingConfig",
) -> ImageScoringResult:
    """Score a single image's OCR output against its annotation.

    Args:
        ocr_result: Normalized OCR output for one image.
        annotation: Ground-truth annotation for the same image.
        matching_cfg: Matching thresholds and strategy.

    Returns:
        ImageScoringResult with all metrics populated.
    """
    if not ocr_result.succeeded:
        logger.warning(
            "OCR failed for '%s' (%s): %s — returning zero-score result.",
            ocr_result.image_name,
            ocr_result.engine.value,
            ocr_result.error,
        )
        return ImageScoringResult(
            image_name=ocr_result.image_name,
            engine=ocr_result.engine,
            error=ocr_result.error,
            processing_time_s=ocr_result.processing_time_s,
        )

    matched_words = match_words(
        ocr_words=ocr_result.words,
        annotation_words=annotation.words,
        cfg=matching_cfg,
    )

    tp = sum(1 for m in matched_words if m.status == MatchStatus.TRUE_POSITIVE)
    fp = sum(1 for m in matched_words if m.status == MatchStatus.FALSE_POSITIVE)
    fn = sum(1 for m in matched_words if m.status == MatchStatus.FALSE_NEGATIVE)

    precision = _safe_divide(tp, tp + fp)
    recall = _safe_divide(tp, tp + fn)
    f1 = _safe_divide(2 * precision * recall, precision + recall)
    coverage = _safe_divide(tp, len(annotation.words))

    missed_words = [
        m.annotation_word.text
        for m in matched_words
        if m.status == MatchStatus.FALSE_NEGATIVE and m.annotation_word
    ]
    invented_words = [
        m.ocr_word.text
        for m in matched_words
        if m.status == MatchStatus.FALSE_POSITIVE and m.ocr_word
    ]

    result = ImageScoringResult(
        image_name=ocr_result.image_name,
        engine=ocr_result.engine,
        true_positives=tp,
        false_positives=fp,
        false_negatives=fn,
        precision=precision,
        recall=recall,
        f1=f1,
        coverage=coverage,
        missed_words=missed_words,
        invented_words=invented_words,
        matched_words=matched_words,
        processing_time_s=ocr_result.processing_time_s,
    )

    logger.info(
        "Scored '%s' [%s] — P=%.3f R=%.3f F1=%.3f TP=%d FP=%d FN=%d",
        ocr_result.image_name,
        ocr_result.engine.value,
        precision,
        recall,
        f1,
        tp,
        fp,
        fn,
    )
    return result


# =============================================================================
# Aggregate scoring
# =============================================================================


def score_all(
    ocr_results: list[OCRResult],
    annotations: dict[str, ImageAnnotation],
    matching_cfg: "MatchingConfig",
    engine: EngineType,
) -> AggregateScoringResult:
    """Score all OCR results for one engine and aggregate metrics.

    Images present in OCR results but lacking annotations are skipped
    with a warning.

    Args:
        ocr_results: All OCR results for *engine*.
        annotations: Annotation map (image_name → ImageAnnotation).
        matching_cfg: Matching configuration.
        engine: The engine being scored.

    Returns:
        AggregateScoringResult with macro-averaged metrics.
    """
    per_image: list[ImageScoringResult] = []

    for result in ocr_results:
        annotation = annotations.get(result.image_name)
        if annotation is None:
            logger.warning(
                "No annotation found for '%s' — skipping.", result.image_name
            )
            continue
        per_image.append(score_image(result, annotation, matching_cfg))

    if not per_image:
        logger.warning("No images were scored for engine '%s'.", engine.value)
        return AggregateScoringResult(engine=engine, image_count=0)

    total_tp = sum(r.true_positives for r in per_image)
    total_fp = sum(r.false_positives for r in per_image)
    total_fn = sum(r.false_negatives for r in per_image)
    total_missed = sum(len(r.missed_words) for r in per_image)
    total_invented = sum(len(r.invented_words) for r in per_image)

    # Macro-average (mean of per-image metrics)
    valid = [r for r in per_image if r.error is None]
    mean_p = statistics.mean(r.precision for r in valid) if valid else 0.0
    mean_r = statistics.mean(r.recall for r in valid) if valid else 0.0
    mean_f1 = statistics.mean(r.f1 for r in valid) if valid else 0.0
    mean_cov = statistics.mean(r.coverage for r in valid) if valid else 0.0

    aggregate = AggregateScoringResult(
        engine=engine,
        image_count=len(per_image),
        total_true_positives=total_tp,
        total_false_positives=total_fp,
        total_false_negatives=total_fn,
        mean_precision=mean_p,
        mean_recall=mean_r,
        mean_f1=mean_f1,
        mean_coverage=mean_cov,
        total_missed_words=total_missed,
        total_invented_words=total_invented,
        per_image=per_image,
    )

    logger.info(
        "Aggregate [%s] — images=%d P=%.3f R=%.3f F1=%.3f",
        engine.value,
        len(per_image),
        mean_p,
        mean_r,
        mean_f1,
    )
    return aggregate


# =============================================================================
# Helpers
# =============================================================================


def _safe_divide(numerator: float, denominator: float) -> float:
    """Return numerator / denominator, or 0.0 if denominator is zero."""
    return numerator / denominator if denominator > 0 else 0.0
