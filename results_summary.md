# OCR Word Position Benchmark — Results Summary

## 1. Summary

| Metric | PaddleOCR | Tesseract |
|---|---:|---:|
| Mean Coverage (%) | 92.0% | 72.7% |
| Total Missed Words | 1050 | 541 |
| Total Invented Boxes | 132 | 638 |
| Mean IoU | N/A | 61.9% |

## 2. Worst Performing Images

### PaddleOCR — Bottom 5 by Coverage

1. **`IMG_061.jpg` (93.3%)** — Missed some small or separated text.
2. **`IMG_060.jpg` (94.1%)** — Missed faint punctuation and very small letters.
3. **`IMG_096.jpg` (94.1%)** — Missed some faint or separated text.
4. **`IMG_086.jpg` (94.3%)** — Missed small items like dots and dashes.
5. **`IMG_070.jpg` (99.5%)** — Crowded text columns caused a few words to be missed.

### Tesseract — Bottom 5 by Coverage

1. **`IMG_076.jpg` (0.0%)** — The boxes it drew weren't close enough to the threshold(50%)
2. **`IMG_078.jpg` (0.0%)** — The boxes it drew weren't close enough to the threshold(50%)
3. **`IMG_070.jpg` (47.9%)** — Crowded text columns caused many words to be missed.
4. **`IMG_069.jpg` (50.0%)** — Drew 395 extra boxes around things that were not text (like dust or smudges).
5. **`IMG_061.jpg` (66.7%)** — Missed some small or separated text.

## 3. Key Observations

1. **PaddleOCR draws bigger boxes:** PaddleOCR draws boxes around whole lines of text. Because these big boxes often cover several smaller words from the ground truth, PaddleOCR gets a very high coverage score.

2. **Tesseract draws smaller boxes:** Tesseract tries to draw tight boxes around individual words. Because these boxes are smaller, they often don't overlap enough with the coverage to count as a "found" word. The Mean IoU metric helps show how tightly these boxes actually fit.

3. **Extra boxes:** Tesseract drew many more "invented" boxes than PaddleOCR (638 vs. 132). This means Tesseract often thought random marks on the page were text.

4. **Complex pages:** Pages with crowded columns (like `IMG_070.jpg`) made it harder for both tools to get the boxes right, but Tesseract struggled much more than PaddleOCR on these pages.
