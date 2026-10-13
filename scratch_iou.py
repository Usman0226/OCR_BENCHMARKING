import json
import statistics
from pathlib import Path
from ocr.core.config import load_config
from ocr.core.io import load_all_ocr_results, load_label_studio_annotations
from ocr.core.models import EngineType, MatchStatus
from ocr.core.scoring import score_all

cfg = load_config()
cfg.matching.strategy = 'iou_only'

annotations = load_label_studio_annotations(Path('annotations/annotations_attempt_5.json'), cfg.paths.images_dir)
tess_dirs = sorted(cfg.paths.ocr_output_dir.glob('ablation_tesseract_*'))
tess_dir = tess_dirs[-1] if tess_dirs else None

results = load_all_ocr_results(tess_dir, engine=EngineType.TESSERACT)
agg = score_all(results, annotations, cfg.matching, EngineType.TESSERACT)

ious = []
for img_res in agg.per_image:
    for mw in img_res.matched_words:
        if mw.status == MatchStatus.TRUE_POSITIVE:
            ious.append(mw.iou)

if ious:
    print(f"Tesseract Mean IoU (True Positives): {statistics.mean(ious):.4f}")
else:
    print("No true positives.")
