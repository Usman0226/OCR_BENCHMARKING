from __future__ import annotations

import pytest

from ocr.core.matching import match_words, text_similarity
from ocr.core.models import (
    AnnotationWord,
    BoundingBox,
    EngineType,
    MatchStatus,
    NormalizedWord,
)


class TestTextSimilarity:
    def test_identical_strings(self) -> None:
        assert text_similarity("hello", "hello") == pytest.approx(1.0)

    def test_case_insensitive(self) -> None:
        assert text_similarity("Hello", "HELLO") == pytest.approx(1.0)

    def test_completely_different(self) -> None:
        assert text_similarity("abc", "xyz") == pytest.approx(0.0)

    def test_partial_match(self) -> None:
        sim = text_similarity("hello", "helo")
        assert 0.8 < sim < 1.0

    def test_both_empty(self) -> None:
        assert text_similarity("", "") == pytest.approx(1.0)

    def test_one_empty(self) -> None:
        assert text_similarity("hello", "") == pytest.approx(0.0)

    def test_whitespace_stripped(self) -> None:
        assert text_similarity("  hello  ", "hello") == pytest.approx(1.0)


def _make_cfg(
    strategy: str = "iou_and_text",
    iou_threshold: float = 0.5,
    text_threshold: float = 0.8,
) -> "MockMatchingConfig":
    class MockMatchingConfig:
        def __init__(self) -> None:
            self.strategy = strategy
            self.iou_threshold = iou_threshold
            self.text_similarity_threshold = text_threshold

    return MockMatchingConfig()  # type: ignore[return-value]


def _word(
    text: str,
    x_min: float,
    y_min: float,
    x_max: float,
    y_max: float,
) -> NormalizedWord:
    return NormalizedWord(
        text=text,
        bbox=BoundingBox(x_min, y_min, x_max, y_max),
        confidence=0.9,
        page=1,
        image_name="test.jpg",
        engine=EngineType.TESSERACT,
    )


def _ann(
    text: str,
    x_min: float,
    y_min: float,
    x_max: float,
    y_max: float,
) -> AnnotationWord:
    return AnnotationWord(
        text=text,
        bbox=BoundingBox(x_min, y_min, x_max, y_max),
        label="text",
        annotation_id="ann-x",
        image_name="test.jpg",
    )


class TestMatchWords:
    def test_perfect_match(self) -> None:
        ocr = [_word("Hello", 0, 0, 100, 50)]
        ann = [_ann("Hello", 0, 0, 100, 50)]
        cfg = _make_cfg(strategy="iou_and_text")
        results = match_words(ocr, ann, cfg)
        tp = [r for r in results if r.status == MatchStatus.TRUE_POSITIVE]
        fp = [r for r in results if r.status == MatchStatus.FALSE_POSITIVE]
        fn = [r for r in results if r.status == MatchStatus.FALSE_NEGATIVE]
        assert len(tp) == 1
        assert len(fp) == 0
        assert len(fn) == 0

    def test_all_false_positives(self) -> None:
        ocr = [_word("Ghost", 0, 0, 50, 20), _word("Word", 60, 0, 100, 20)]
        ann: list[AnnotationWord] = []
        cfg = _make_cfg()
        results = match_words(ocr, ann, cfg)
        assert all(r.status == MatchStatus.FALSE_POSITIVE for r in results)
        assert len(results) == 2

    def test_all_false_negatives(self) -> None:
        ocr: list[NormalizedWord] = []
        ann = [_ann("Word1", 0, 0, 50, 20), _ann("Word2", 60, 0, 100, 20)]
        cfg = _make_cfg()
        results = match_words(ocr, ann, cfg)
        assert all(r.status == MatchStatus.FALSE_NEGATIVE for r in results)
        assert len(results) == 2

    def test_no_spatial_overlap_gives_fp_and_fn(self) -> None:
        ocr = [_word("Bonjour", 0, 0, 50, 20)]
        ann = [_ann("Bonjour", 500, 500, 600, 520)]
        cfg = _make_cfg(strategy="iou_and_text", iou_threshold=0.5)
        results = match_words(ocr, ann, cfg)
        statuses = {r.status for r in results}
        assert MatchStatus.TRUE_POSITIVE not in statuses

    def test_text_only_strategy_matches_by_text(self) -> None:
        ocr = [_word("Bonjour", 0, 0, 50, 20)]
        ann = [_ann("Bonjour", 500, 500, 600, 520)]
        cfg = _make_cfg(strategy="text_only", text_threshold=0.9)
        results = match_words(ocr, ann, cfg)
        tp = [r for r in results if r.status == MatchStatus.TRUE_POSITIVE]
        assert len(tp) == 1

    def test_iou_only_strategy(self) -> None:
        ocr = [_word("Foo", 0, 0, 100, 50)]
        ann = [_ann("Bar", 0, 0, 100, 50)]
        cfg = _make_cfg(strategy="iou_only", iou_threshold=0.5)
        results = match_words(ocr, ann, cfg)
        tp = [r for r in results if r.status == MatchStatus.TRUE_POSITIVE]
        assert len(tp) == 1

    def test_greedy_one_to_one_assignment(self) -> None:
        ocr = [
            _word("Hello", 0, 0, 100, 50),
            _word("World", 0, 0, 100, 50),
        ]
        ann = [_ann("Hello", 0, 0, 100, 50)]
        cfg = _make_cfg(strategy="iou_and_text")
        results = match_words(ocr, ann, cfg)
        tp = [r for r in results if r.status == MatchStatus.TRUE_POSITIVE]
        fp = [r for r in results if r.status == MatchStatus.FALSE_POSITIVE]
        assert len(tp) == 1
        assert len(fp) == 1

    def test_two_words_matched_correctly(
        self,
        ocr_word_hello: NormalizedWord,
        ocr_word_world: NormalizedWord,
        annotation_word_hello: AnnotationWord,
        annotation_word_world: AnnotationWord,
    ) -> None:
        cfg = _make_cfg(strategy="iou_and_text")
        results = match_words(
            [ocr_word_hello, ocr_word_world],
            [annotation_word_hello, annotation_word_world],
            cfg,
        )
        tp = [r for r in results if r.status == MatchStatus.TRUE_POSITIVE]
        assert len(tp) == 2

    def test_missed_word_is_false_negative(
        self,
        ocr_word_hello: NormalizedWord,
        annotation_word_hello: AnnotationWord,
        annotation_word_missing: AnnotationWord,
    ) -> None:
        cfg = _make_cfg()
        results = match_words(
            [ocr_word_hello],
            [annotation_word_hello, annotation_word_missing],
            cfg,
        )
        fn = [r for r in results if r.status == MatchStatus.FALSE_NEGATIVE]
        assert len(fn) == 1
        assert fn[0].annotation_word is not None
        assert fn[0].annotation_word.text == "Missed"
