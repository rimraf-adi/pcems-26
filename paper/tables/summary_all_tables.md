# Master Summary: All Empirical Benchmark Tables for Paper

This document compiles all extracted data tables for the dialect normalization paper, extracted directly from the `dialect-normalisation` codebase, CV summaries, and evaluation logs.

---

# Table 1: RESPIN-S1.0 Marathi Source Corpus Statistics

Source ground truth: `dialect-normalisation/stats_mr.yaml` (IISc Bangalore RESPIN-S1.0 Clean Subset)

### Overall Corpus Summary

| Metric | Value | Description |
| :--- | :---: | :--- |
| **Total Utterances** | 809,934 | Clean recorded audio-transcript pairs |
| **Total Duration** | 1,026.06 hrs | 3,693,804.99 seconds (Avg 4.56 s / utterance) |
| **Unique Reference Texts** | 23,069 | Unique prompt sentences across domains |
| **Unique Speakers** | 2,644 | Native speakers across Maharashtra |
| **Unique Pincodes** | 234 | Geographic origin locations |

### Sub-Dialect Breakdown

| Code | Dialect Name | Utterances | % of Total | Duration (hrs) | Unique Speakers | Unique Prompts |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **D1** | Southern Konkan (Malvani) | 203,651 | 25.14% | 247.31 | 732 | 5,838 |
| **D2** | Northern Konkan (Ahirani) | 198,560 | 24.52% | 265.25 | 557 | 5,825 |
| **D3** | Standard Marathi (Puneri) | 208,850 | 25.79% | 255.79 | 608 | 6,077 |
| **D4** | Varhadi (Vidarbha) | 198,873 | 24.55% | 257.70 | 747 | 5,330 |

### Domain & Demographic Split

| Category | Division | Utterances | Share (%) |
| :--- | :--- | :---: | :---: |
| **Domain** | Agriculture | 394,136 | 48.66% |
| **Domain** | Banking | 415,798 | 51.34% |
| **Gender** | FEMALE | 412,137 | 50.89% |
| **Gender** | MALE | 396,399 | 48.94% |
| **Gender** | NA | 1,398 | 0.17% |

---

# Table 2: Parallel Dataset Composition, Synthetic Generation Yield, and Verification Filtering

Source: Dataset files in `dialect-normalisation/data/synthetic_parallel/` & pipeline logs

| Dialect Name | Original Clean Pairs | Raw Synthetic Generated | Clean Synthetic Verified | Corrupted / Rejected Pairs (%) | Total Clean Expanded Pairs |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Southern Konkan (D1) | 5,576 | 5,838 | 5,569 | 269 (4.61%) | **11,145** |
| Northern Konkan (D2) | 5,501 | 5,825 | 5,534 | 291 (5.00%) | **11,035** |
| Varhadi (D4) | 5,086 | 5,330 | 5,069 | 261 (4.90%) | **10,155** |
| **Total (All 3 Dialects)** | **16,163** | **16,993** | **16,172** | **821 (4.83%)** | **32,335** |

> [!NOTE]
> Rejection Rate Calculation: `(Raw Synthetic Generated - Clean Synthetic Verified) / Raw Synthetic Generated`.

---

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

---

# Table 4: Benchmark Performance Matrix on Original Datasets: Single-Dialect Models (5-Fold CV)

Source: `dialect-normalisation/models/*/cv_summary.json` (5-Fold Cross-Validation Test Split Averages)

| Dialect Name | Total Pairs | IndicBART Test Size | IndicBART BLEU | IndicBART chrF++ | mT5 Test Size | mT5-Small BLEU | mT5-Small chrF++ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Southern Konkan (D1) | 5,576 | 837 | **48.54** | **78.35** | 836 | 46.31 | 74.43 |
| Northern Konkan (D2) | 5,501 | 826 | **60.58** | **80.30** | 825 | 60.31 | 77.34 |
| Varhadi (D4) | 5,086 | 763 | 76.63 | 90.93 | 762 | **81.00** | **91.21** |

---

# Table 5: Benchmark Performance Matrix on Original Datasets: Multi-Dialect Dialectwise Breakdown (5-Fold CV)

Source: `dialect-normalisation/models/indicbart_combined/` & `mt5_combined_16k/` (5-Fold CV Averages)

| Evaluated Subset | Total Pairs | Test Set Size | IndicBART (244M) BLEU | IndicBART (244M) chrF++ | mT5-Small (300M) BLEU | mT5-Small (300M) chrF++ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Southern Konkan (D1) | 16,163 | 836/837 | 26.21 | 53.15 | **48.28** | **75.04** |
| Northern Konkan (D2) | 16,163 | 798/826 | 47.59 | 67.16 | **62.35** | **78.92** |
| Varhadi (D4) | 16,163 | 763/790 | 72.75 | 85.63 | **79.57** | **90.51** |
| **Overall Combined** | 16,163 | 2,424/2,426 | 48.17 | 68.58 | **63.29** | **81.37** |

---

# Table 6: Benchmark Performance Matrix on Synthetically Expanded Datasets: Single-Dialect Models (5-Fold CV)

Source: `dialect-normalisation/models/*_32k/cv_summary.json` (5-Fold CV Averages)

| Dialect Name | Total Pairs | IndicBART Test Size | IndicBART BLEU | IndicBART chrF++ | mT5 Test Size | mT5-Small BLEU | mT5-Small chrF++ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Southern Konkan (D1) | 11,145 | 1672 | 47.25 | 64.69 | 1671 | **65.10** | **82.88** |
| Northern Konkan (D2) | 11,035 | 1656 | 40.51 | 62.21 | 1655 | **62.07** | **79.29** |
| Varhadi (D4) | 10,155 | 1524 | 73.62 | 86.75 | 1523 | **78.89** | **90.59** |

---

# Table 7: Benchmark Performance Matrix on Synthetically Expanded Datasets: Multi-Dialect Dialectwise Breakdown (5-Fold CV)

Source: `dialect-normalisation/models/indicbart_combined_32k/` & `mt5_combined_32k/` (5-Fold CV Averages)

| Evaluated Subset | Total Pairs | Test Set Size | IndicBART (244M) BLEU | IndicBART (244M) chrF++ | mT5-Small (300M) BLEU | mT5-Small (300M) chrF++ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Southern Konkan (D1) | 32,335 | 1,672/1,710 | 52.06 | 72.15 | **66.62** | **83.57** |
| Northern Konkan (D2) | 32,335 | 1,627/1,656 | 49.08 | 69.90 | **62.72** | **79.63** |
| Varhadi (D4) | 32,335 | 1,513/1,524 | 70.27 | 84.27 | **78.73** | **90.46** |
| **Overall Combined** | 32,335 | 4,850/4,852 | 57.12 | 75.49 | **69.51** | **84.58** |

---

# Table 8: Impact of Synthetic Data Augmentation: Original vs. Synthetically Expanded Data BLEU Change (5-Fold CV)

Source: Computed from 5-fold cross-validation multi-dialect model summaries (`cv_summary.json`)

| Evaluated Subset | IndicBART Original | IndicBART Expanded | IndicBART Δ BLEU | mT5-Small Original | mT5-Small Expanded | mT5-Small Δ BLEU |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Southern Konkan (D1) | 26.21 | 52.06 | **+25.85** | 48.28 | **66.62** | **+18.34** |
| Northern Konkan (D2) | 47.59 | 49.08 | **+1.49** | 62.35 | **62.72** | **+0.37** |
| Varhadi (D4) | 72.75 | 70.27 | -2.48 | 79.57 | **78.73** | -0.84 |
| **Overall Combined** | 48.17 | 57.12 | **+8.95** | 63.29 | **69.51** | **+6.22** |

---

# Table 9: Impact of Multi-Tier Verification Engine: Raw Unverified vs. Filtered Clean Synthetic Data

Source: Verification engine ablation experiment summaries (`indicbart_raw_unverified_32k` vs `indicbart_combined_32k`, `mt5_raw_unverified_32k` vs `mt5_combined_32k`)

| Model Architecture | Pipeline State | Total Pairs | BLEU | chrF++ | Δ BLEU | Δ chrF++ |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| IndicBART (244M) | Raw Unverified Data | 19,914 | 43.09 | 64.54 | -14.03 | -10.95 |
| IndicBART (244M) | Filtered Clean Data | 32,335 | **57.12** | **75.49** | **+14.03** | **+10.95** |
|---|---|---|---|---|---|---|
| mT5-Small (300M) | Raw Unverified Data | 19,914 | 61.21 | 80.18 | -8.30 | -4.40 |
| mT5-Small (300M) | Filtered Clean Data | 32,335 | **69.51** | **84.58** | **+8.30** | **+4.40** |
