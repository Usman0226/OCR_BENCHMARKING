# OCR Word Position Benchmark — Results Summary

## 1. Summary

| Metric | PaddleOCR | Tesseract |
|---|---:|---:|
| Mean Coverage (%) | 92.0% | 72.7% |
| Total Missed Words | 1050 | 541 |
| Total Invented Boxes | 132 | 638 |
| Mean IoU | N/A | 61.9% |

## 2. Worst Performing Images

*Images with no ground-truth printed words were excluded from the coverage ranking.*

### PaddleOCR — Bottom 5 by Coverage

1. **`IMG_061.jpg` (93.3%)** — Missed several small or isolated annotated text regions.
2. **`IMG_060.jpg` (94.1%)** — Missed faint punctuation and small printed characters.
3. **`IMG_096.jpg` (94.1%)** — Missed some faint or isolated printed text.
4. **`IMG_086.jpg` (94.3%)** — Missed small isolated elements such as leader dots and dashes.
5. **`IMG_073.jpg` (99.5%)** — Dense multi-column layout resulted in some ground-truth words falling below the coverage threshold.

### Tesseract — Bottom 5 by Coverage

1. **`IMG_076.jpg` (0.0%)** — The detected boxes did not meet the 50% ground-truth coverage threshold.
2. **`IMG_078.jpg` (0.0%)** — Detected text was present, but the word-level boxes did not meet the required coverage threshold.
3. **`IMG_073.jpg` (47.9%)** — Dense multi-column layout resulted in substantial missed coverage.
4. **`IMG_069.jpg` (50.0%)** — Produced a large number of invented boxes, indicating many detections outside the annotated text regions.
5. **`IMG_061.jpg` (66.7%)** — Missed several small or isolated annotated text regions.

## 3. Key Observations

1. **PaddleOCR coverage behavior:** PaddleOCR's high coverage scores are influenced by its line-level bounding boxes, which can cover multiple word-level ground-truth boxes. Under the defined 50% coverage metric, those words are counted as found.

2. **Tesseract word-level behavior:** Tesseract produces word-level boxes. Its lower coverage on several images indicates that some detected word regions did not meet the 50% ground-truth coverage threshold. Its mean IoU provides an additional measure of box alignment.

3. **Invented boxes:** Tesseract produced substantially more invented boxes than PaddleOCR (638 vs. 132), indicating more detections that did not overlap any ground-truth word.

4. **Document layout affects performance:** Dense or complex layouts, such as `IMG_073.jpg`, affected both engines differently, with Tesseract showing a substantially larger coverage drop than PaddleOCR.
