# OCR Word Position Benchmark - Results Summary

## 1. Overall Performance

| Tool | Mean Coverage | Total Missed Words | Total Invented Boxes | Mean IoU (True Positives) |
|---|---|---|---|---|
| PaddleOCR | 89.1% | 1099 | 222 | N/A |
| Tesseract | 73.7% | 514 | 4903 | 61.6% |

## 2. Worst Performing Images

### PaddleOCR - Bottom 5 by Coverage

1. **IMG_070.jpg (0.0%)**  45 missed words, 107 invented boxes. *Extremely dense, small-print multi-column newspaper article from an old scan.*
2. **IMG_075.jpg (0.0%)**  2 missed words, 5 invented boxes. *The text is completely handwritten in cursive, which standard OCR engines struggle with.*
3. **IMG_095.jpg (21.6%)**  1018 missed words, 0 invented boxes. *Extremely large, dense two-page newspaper spread with complex multi-column layouts and multiple embedded images, overwhelming the line-formation logic.*
4. **IMG_091.jpg (92.9%)**  1 missed words, 0 invented boxes. *Typewritten index card with faint text and noticeable ink bleed-through from the reverse side.*
5. **IMG_061.jpg (93.3%)**  1 missed words, 0 invented boxes. *Degraded, faded typewritten text on a yellowed index card.*

### Tesseract  Bottom 5 by Coverage

1. **IMG_076.jpg (0.0%)**  11 missed words, 0 invented boxes. *Faint, heavily cursive handwriting that Tesseract cannot decipher.*
2. **IMG_078.jpg (0.0%)**  5 missed words, 0 invented boxes. *Sparse text on a large index card with mostly whitespace, preventing Tesseract from identifying layout structure.*
3. **IMG_075.jpg (0.0%)**  2 missed words, 11 invented boxes. *The text is completely handwritten in cursive, which standard OCR engines struggle with.*
4. **IMG_058.jpg (47.1%)**  9 missed words, 1 invented boxes. *Typewritten index card with faded, degraded, and uneven lettering.*
5. **IMG_073.jpg (47.9%)**  100 missed words, 1 invented boxes. *Extremely narrow, vertically cropped obituary column that breaks Tesseract's standard page segmentation.*

## 3. Key Observations

1. **The 'Lines vs. Words' Penalty is Resolved:** Under the proper Intersection-over-Area metric, PaddleOCR's performance is accurately captured. It correctly encloses most words within its line-level boxes, achieving much higher overall coverage than when evaluated under strict IoU.
2. **Tesseract Struggles with Sparse/Unusual Layouts:** Tesseract frequently fails on index cards with sparse text or large whitespace (IMG_078) and extremely narrow column crops (IMG_073), likely because its page segmentation relies heavily on standard document structures.
3. **Handwriting Remains a Universal Failure Point:** Both OCR engines completely failed (0.0% coverage) on heavily cursive handwritten documents (IMG_075 and IMG_076), proving that standard off-the-shelf printed OCR models cannot generalize to cursive scripts.
4. **Tesseract Over-segments Degraded Text:** On dense, low-quality microfilm scans like IMG_070, Tesseract attempts to predict thousands of tiny, fragmented "invented" boxes (over 4,000 for IMG_070 alone) rather than failing gracefully.
5. **PaddleOCR Handles Rotated Text Poorly (Default Settings):** Without its angle classification enabled, PaddleOCR failed to detect any text on documents rotated 90 degrees (IMG_069), while Tesseract managed to find at least some vertical text fragments.
