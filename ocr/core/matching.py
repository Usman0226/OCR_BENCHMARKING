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


def text_similarity(a: str, b: str) -> float:
    a_clean = a.strip().lower()
    b_clean = b.strip().lower()
    if not a_clean and not b_clean:
        return 1.0
    if not a_clean or not b_clean:
        return 0.0
    return difflib.SequenceMatcher(None, a_clean, b_clean).ratio()


def _match_iou_only(
    ocr_word: NormalizedWord,
    annotation_word: AnnotationWord,
    iou_threshold: float,
) -> tuple[bool, float, float]:
    iou = bbox_iou(ocr_word.bbox, annotation_word.bbox)
    return iou >= iou_threshold, iou, 0.0


def _match_iou_and_text(
    ocr_word: NormalizedWord,
    annotation_word: AnnotationWord,
    iou_threshold: float,
    text_threshold: float,
) -> tuple[bool, float, float]:
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
    sim = text_similarity(ocr_word.text, annotation_word.text)
    iou = bbox_iou(ocr_word.bbox, annotation_word.bbox)
    return sim >= text_threshold, iou, sim


@dataclass
class _Candidate:
    ocr_idx: int
    ann_idx: int
    iou: float
    text_sim: float
    score: float


def match_words(
    ocr_words: list[NormalizedWord],
    annotation_words: list[AnnotationWord],
    cfg: "MatchingConfig",
) -> list[MatchedWord]:
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
            else:
                matched, iou, sim = _match_iou_and_text(ow, aw, iou_thr, txt_thr)

            if matched:
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
    return results
