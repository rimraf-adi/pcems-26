# Table 7: Benchmark Performance Matrix on Synthetically Expanded Datasets: Multi-Dialect Dialectwise Breakdown (5-Fold CV)

Source: `dialect-normalisation/models/indicbart_combined_32k/` & `mt5_combined_32k/` (5-Fold CV Averages)

| Evaluated Subset | Total Pairs | Test Set Size | IndicBART (244M) BLEU | IndicBART (244M) chrF++ | mT5-Small (300M) BLEU | mT5-Small (300M) chrF++ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Southern Konkan (D1) | 32,335 | 1,672/1,710 | 52.06 | 72.15 | **66.62** | **83.57** |
| Northern Konkan (D2) | 32,335 | 1,627/1,656 | 49.08 | 69.90 | **62.72** | **79.63** |
| Varhadi (D4) | 32,335 | 1,513/1,524 | 70.27 | 84.27 | **78.73** | **90.46** |
| **Overall Combined** | 32,335 | 4,850/4,852 | 57.12 | 75.49 | **69.51** | **84.58** |