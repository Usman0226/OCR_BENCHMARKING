from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class PathsConfig:
    images_dir: Path
    annotations_dir: Path
    ocr_output_dir: Path
    results_dir: Path
    reports_dir: Path
    logs_dir: Path
    paddle_model_dir: Path

    def ensure_dirs(self) -> None:
        for attr in vars(self).values():
            if isinstance(attr, Path):
                attr.mkdir(parents=True, exist_ok=True)


@dataclass
class LoggingConfig:
    level: str = "INFO"
    max_bytes: int = 10_485_760
    backup_count: int = 5
    files: dict[str, str] = field(
        default_factory=lambda: {
            "application": "application.log",
            "ocr": "ocr.log",
            "errors": "errors.log",
        }
    )


@dataclass
class PaddleEngineConfig:
    lang: str = "ch"
    use_angle_cls: bool = True
    use_gpu: bool = False
    det_db_thresh: float = 0.3
    det_db_box_thresh: float = 0.6
    rec_batch_num: int = 6
    enable_mkldnn: bool = False
    show_log: bool = False
    cls_thresh: float = 0.9


@dataclass
class TesseractEngineConfig:
    lang: str = "fra+eng"
    psm: int = 3
    oem: int = 3
    config_extra: str = ""
    dpi: int = 300


@dataclass
class EnginesConfig:
    paddle: PaddleEngineConfig = field(default_factory=PaddleEngineConfig)
    tesseract: TesseractEngineConfig = field(default_factory=TesseractEngineConfig)


@dataclass
class MatchingConfig:
    iou_threshold: float = 0.5
    text_similarity_threshold: float = 0.8
    strategy: str = "iou_and_text"


@dataclass
class ScoringConfig:
    metrics: list[str] = field(
        default_factory=lambda: [
            "precision",
            "recall",
            "f1",
            "coverage",
            "false_positives",
            "false_negatives",
            "missed_words",
            "invented_words",
        ]
    )


@dataclass
class ReportsConfig:
    formats: list[str] = field(default_factory=lambda: ["csv", "json", "markdown"])
    include_per_image: bool = True
    include_word_detail: bool = False
    markdown_template: str | None = None


@dataclass
class PreprocessingConfig:
    enabled: bool = True
    max_long_edge: int | None = 4096
    tesseract_grayscale: bool = True
    default_dpi: int = 300


@dataclass
class LabelStudioConfig:
    url: str = "http://labelstudio:8080"
    project_id: int = 1
    export_format: str = "JSON"
    result_type: str = "rectanglelabels"


@dataclass
class AppConfig:
    paths: PathsConfig
    logging: LoggingConfig
    engines: EnginesConfig
    matching: MatchingConfig
    scoring: ScoringConfig
    reports: ReportsConfig
    preprocessing: PreprocessingConfig
    label_studio: LabelStudioConfig


def _resolve_path(value: str, env_key: str | None = None) -> Path:
    if env_key:
        env_val = os.environ.get(env_key)
        if env_val:
            return Path(env_val)
    return Path(value)


def load_config(config_path: Path | None = None) -> AppConfig:
    if config_path is None:
        env_path = os.environ.get("CONFIG_PATH")
        if env_path:
            config_path = Path(env_path)
        else:
            config_path = Path(__file__).parents[2] / "configs" / "config.yaml"

    if not config_path.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {config_path}\n"
            "Set CONFIG_PATH environment variable or pass --config to the CLI."
        )

    raw: dict[str, Any] = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}

    p = raw.get("paths", {})
    paths = PathsConfig(
        images_dir=_resolve_path(p.get("images_dir", "./images"), "OCR_IMAGES_DIR"),
        annotations_dir=_resolve_path(
            p.get("annotations_dir", "./annotations"), "OCR_ANNOTATIONS_DIR"
        ),
        ocr_output_dir=_resolve_path(
            p.get("ocr_output_dir", "./ocr_output"), "OCR_OUTPUT_DIR"
        ),
        results_dir=_resolve_path(p.get("results_dir", "./"), "OCR_RESULTS_DIR"),
        reports_dir=_resolve_path(p.get("reports_dir", "./reports"), "OCR_REPORTS_DIR"),
        logs_dir=_resolve_path(p.get("logs_dir", "./logs"), "OCR_LOG_DIR"),
        paddle_model_dir=_resolve_path(
            p.get("paddle_model_dir", "./paddle-models"), "PADDLE_MODEL_DIR"
        ),
    )

    lc = raw.get("logging", {})
    logging_cfg = LoggingConfig(
        level=lc.get("level", "INFO"),
        max_bytes=int(lc.get("max_bytes", 10_485_760)),
        backup_count=int(lc.get("backup_count", 5)),
        files=lc.get(
            "files",
            {
                "application": "application.log",
                "ocr": "ocr.log",
                "errors": "errors.log",
            },
        ),
    )

    ec = raw.get("engines", {})

    paddle_raw = ec.get("paddle", {})
    paddle_cfg = PaddleEngineConfig(
        lang=paddle_raw.get("lang", "ch"),
        use_angle_cls=bool(paddle_raw.get("use_angle_cls", True)),
        use_gpu=bool(paddle_raw.get("use_gpu", False)),
        det_db_thresh=float(paddle_raw.get("det_db_thresh", 0.3)),
        det_db_box_thresh=float(paddle_raw.get("det_db_box_thresh", 0.6)),
        rec_batch_num=int(paddle_raw.get("rec_batch_num", 6)),
        enable_mkldnn=bool(paddle_raw.get("enable_mkldnn", False)),
        show_log=bool(paddle_raw.get("show_log", False)),
        cls_thresh=float(paddle_raw.get("cls_thresh", 0.9)),
    )

    tess_raw = ec.get("tesseract", {})
    tess_cfg = TesseractEngineConfig(
        lang=tess_raw.get("lang", "fra+eng"),
        psm=int(tess_raw.get("psm", 3)),
        oem=int(tess_raw.get("oem", 3)),
        config_extra=tess_raw.get("config_extra", ""),
        dpi=int(tess_raw.get("dpi", 300)),
    )

    mc = raw.get("matching", {})
    matching_cfg = MatchingConfig(
        iou_threshold=float(mc.get("iou_threshold", 0.5)),
        text_similarity_threshold=float(mc.get("text_similarity_threshold", 0.8)),
        strategy=mc.get("strategy", "iou_and_text"),
    )

    sc = raw.get("scoring", {})
    scoring_cfg = ScoringConfig(metrics=sc.get("metrics", ScoringConfig().metrics))

    rc = raw.get("reports", {})
    reports_cfg = ReportsConfig(
        formats=rc.get("formats", ["csv", "json", "markdown"]),
        include_per_image=bool(rc.get("include_per_image", True)),
        include_word_detail=bool(rc.get("include_word_detail", False)),
        markdown_template=rc.get("markdown_template"),
    )

    pp = raw.get("preprocessing", {})
    max_edge = pp.get("max_long_edge", 4096)
    preprocessing_cfg = PreprocessingConfig(
        enabled=bool(pp.get("enabled", True)),
        max_long_edge=int(max_edge) if max_edge is not None else None,
        tesseract_grayscale=bool(pp.get("tesseract_grayscale", True)),
        default_dpi=int(pp.get("default_dpi", 300)),
    )

    ls = raw.get("label_studio", {})
    ls_cfg = LabelStudioConfig(
        url=ls.get("url", "http://labelstudio:8080"),
        project_id=int(ls.get("project_id", 1)),
        export_format=ls.get("export_format", "JSON"),
        result_type=ls.get("result_type", "rectanglelabels"),
    )

    return AppConfig(
        paths=paths,
        logging=logging_cfg,
        engines=EnginesConfig(paddle=paddle_cfg, tesseract=tess_cfg),
        matching=matching_cfg,
        scoring=scoring_cfg,
        reports=reports_cfg,
        preprocessing=preprocessing_cfg,
        label_studio=ls_cfg,
    )
