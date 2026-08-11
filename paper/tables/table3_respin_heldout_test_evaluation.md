# Table 3: Comprehensive Evaluation Matrix on IISc_RESPIN_test_mr Benchmark Set (2,170 Utterances)

Source: Extracted directly from empirical held-out test evaluation log `dialect-normalisation/logs/eval_mr_indicbart_mt5.log`

> [!IMPORTANT]
> This table presents the official held-out test benchmark metrics evaluated on the `IISc_RESPIN_test_mr` dataset (2,170 utterances). (Note: This corrects an issue in content.tex where 5-fold CV training pool averages were previously inserted for Multi-Dialect IndicBART).

| Model Scope | Training Data | Dialect / Subset | Utts | IndicBART (244M) BLEU (↑) | IndicBART (244M) WER (↓) | mT5-Small (300M) BLEU (↑) | mT5-Small (300M) WER (↓) |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Single-Dialect** | Original | Southern Konkan (D1) | 559 | **57.80** | **26.60%** | 43.86 | 34.37% |
| | | Northern Konkan (D2) | 540 | **90.13** | **6.46%** | 79.46 | 11.95% |
| | | Varhadi (D4) | 516 | **83.59** | **10.19%** | 74.81 | 14.73% |
|---|---|---|---|---|---|---|---|
| **Single-Dialect** | Synthetically Expanded | Southern Konkan (D1) | 559 | 24.06 | 60.32% | **44.36** | **32.75%** |
| | | Northern Konkan (D2) | 540 | 65.41 | 28.70% | **79.76** | **12.61%** |
| | | Varhadi (D4) | 516 | 67.97 | 25.43% | **76.90** | **14.19%** |
|---|---|---|---|---|---|---|---|
| **Multi-Dialect** | Original | Southern Konkan (D1) | 559 | 26.08 | 52.98% | **40.94** | **35.34%** |
| | | Northern Konkan (D2) | 540 | 47.92 | 32.64% | **77.50** | **14.65%** |
| | | Standard Benchmark (D3) | 555 | 96.12 | 3.12% | **96.65** | **1.91%** |
| | | Varhadi (D4) | 516 | 73.04 | 26.96% | **73.47** | **16.16%** |
| | | **Overall Benchmark** | **2,170** | 62.98 | 30.13% | **72.18** | **17.23%** |
|---|---|---|---|---|---|---|---|
| **Multi-Dialect** | Synthetically Expanded | Southern Konkan (D1) | 559 | **52.15** | 36.50% | 43.88 | **34.96%** |
| | | Northern Konkan (D2) | 540 | 54.35 | 25.65% | **78.16** | **13.55%** |
| | | Standard Benchmark (D3) | 555 | 96.80 | 1.65% | **97.39** | **1.37%** |
| | | Varhadi (D4) | 516 | 69.96 | 21.04% | **74.74** | **15.57%** |
| | | **Overall Benchmark** | **2,170** | **76.50** | **16.23%** | 73.48 | 16.58% |