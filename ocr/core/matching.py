"""Word-level matching between OCR output and annotation ground truth.

Implements the Strategy pattern for matching:
  - IoU-only
  - IoU + text similarity (default)
  - Text-only

Returns lists of MatchedWord covering TP, FP, and FN cases.
"""

from __future__ import annotations

import difflib
from dataclasses import dataclass
from typing import TYPE_CHECKING

from ocr.core.geometry import bbox_iou
from ocr.core.logger import get_logger
from ocr.core.models import (
    AnnotationWord,
    MatchedWord,
    MatchStatus,
    NormalizedWord,
)

if TYPE_CHECKING:
    from ocr.core.config import MatchingConfig

logger = get_logger(__name__)


# =============================================================================
# Text similarity
# =============================================================================


def text_similarity(a: str, b: str) -> float:
    """Character-level similarity using SequenceMatcher (Gestalt algorithm).

    Returns a value in [0, 1].  Both strings are lowercased and stripped
    before comparison.
    """
    a_clean = a.strip().lower()
    b_clean = b.strip().lower()
    if not a_clean and not b_clean:
        return 1.0
    if not a_clean or not b_clean:
        return 0.0
    return difflib.SequenceMatcher(None, a_clean, b_clean).ratio()


# =============================================================================
# Matching strategies
# =============================================================================


def _match_iou_only(
    ocr_word: NormalizedWord,
    annotation_word: AnnotationWord,
    iou_threshold: float,
) -> tuple[bool, float, float]:
    """Return (is_match, iou, text_sim)."""
    iou = bbox_iou(ocr_word.bbox, annotation_word.bbox)
    return iou >= iou_threshold, iou, 0.0


def _match_iou_and_text(
    ocr_word: NormalizedWord,
    annotation_word: AnnotationWord,
    iou_threshold: float,
    text_threshold: float,
) -> tuple[bool, float, float]:
    """Return (is_match, iou, text_sim).

    A match requires IoU ≥ threshold AND text similarity ≥ threshold.
    """
    iou = bbox_iou(ocr_word.bbox, annotation_word.bbox)
    if iou < iou_threshold:
        return False, iou, 0.0
    sim = text_similarity(ocr_word.text, annotation_word.text)
    return sim >= text_threshold, iou, sim


def _match_text_only(
    ocr_word: NormalizedWord,
    annotation_word: AnnotationWord,
    text_threshold: float,
) -> tuple[bool, float, float]:
    """Return (is_match, iou, text_sim)."""
    sim = text_similarity(ocr_word.text, annotation_word.text)
    iou = bbox_iou(ocr_word.bbox, annotation_word.bbox)
    return sim >= text_threshold, iou, sim


# =============================================================================
# Greedy bipartite matching
# =============================================================================


@dataclass
class _Candidate:
    ocr_idx: int
    ann_idx: int
    iou: float
    text_sim: float
    score: float  # composite score used for sorting


def match_words(
    ocr_words: list[NormalizedWord],
    annotation_words: list[AnnotationWord],
    cfg: "MatchingConfig",
) -> list[MatchedWord]:
    """Greedily match OCR words to annotation words.

    Algorithm:
      1. Compute a composite score for every (ocr_word, ann_word) pair.
      2. Sort all candidates by score descending.
      3. Greedily assign: once an OCR word or annotation word is claimed,
         skip further pairs involving it.
      4. Unmatched OCR words → FALSE_POSITIVE (invented words).
         Unmatched annotation words → FALSE_NEGATIVE (missed words).

    This is O(n·m) where n = |ocr_words| and m = |annotation_words|.
    For typical document pages this is fast enough; a more optimal
    Hungarian algorithm can replace it if needed for very long documents.

    Args:
        ocr_words: Words produced by the OCR engine (NormalizedWord list).
        annotation_words: Ground-truth words from Label Studio.
        cfg: Matching configuration (thresholds and strategy).

    Returns:
        List of MatchedWord covering all TPs, FPs, and FNs.
    """
    strategy = cfg.strategy
    iou_thr = cfg.iou_threshold
    txt_thr = cfg.text_similarity_threshold

    candidates: list[_Candidate] = []

    for oi, ow in enumerate(ocr_words):
        for ai, aw in enumerate(annotation_words):
            if strategy == "iou_only":
                matched, iou, sim = _match_iou_only(ow, aw, iou_thr)
            elif strategy == "text_only":
                matched, iou, sim = _match_text_only(ow, aw, txt_thr)
            else:  # iou_and_text (default)
                matched, iou, sim = _match_iou_and_text(ow, aw, iou_thr, txt_thr)

            if matched:
                # Composite score: equal weight to IoU and text similarity.
                score = (iou + sim) / 2.0
                candidates.append(
                    _Candidate(
                        ocr_idx=oi,
                        ann_idx=ai,
                        iou=iou,
                        text_sim=sim,
                        score=score,
                    )
                )

    # Sort by score descending — best matches first
    candidates.sort(key=lambda c: c.score, reverse=True)

    matched_ocr: set[int] = set()
    matched_ann: set[int] = set()
    results: list[MatchedWord] = []

    for cand in candidates:
        if cand.ocr_idx in matched_ocr or cand.ann_idx in matched_ann:
            continue
        matched_ocr.add(cand.ocr_idx)
        matched_ann.add(cand.ann_idx)
        results.append(
            MatchedWord(
                status=MatchStatus.TRUE_POSITIVE,
                ocr_word=ocr_words[cand.ocr_idx],
                annotation_word=annotation_words[cand.ann_idx],
                iou=cand.iou,
                text_similarity=cand.text_sim,
            )
        )

    # False positives: OCR words that were never matched
    for oi, ow in enumerate(ocr_words):
        if oi not in matched_ocr:
            results.append(
                MatchedWord(
                    status=MatchStatus.FALSE_POSITIVE,
                    ocr_word=ow,
                    annotation_word=None,
                    iou=0.0,
                    text_similarity=0.0,
                )
            )

    # False negatives: annotation words that were never matched
    for ai, aw in enumerate(annotation_words):
        if ai not in matched_ann:
            results.append(
                MatchedWord(
                    status=MatchStatus.FALSE_NEGATIVE,
                    ocr_word=None,
                    annotation_word=aw,
                    iou=0.0,
                    text_similarity=0.0,
                )
            )

    tp = sum(1 for r in results if r.status == MatchStatus.TRUE_POSITIVE)
    fp = sum(1 for r in results if r.status == MatchStatus.FALSE_POSITIVE)
    fn = sum(1 for r in results if r.status == MatchStatus.FALSE_NEGATIVE)
    logger.debug("Match result — TP=%d FP=%d FN=%d", tp, fp, fn)

    return results
