# Table 5: Benchmark Performance Matrix on Original Datasets: Multi-Dialect Dialectwise Breakdown (5-Fold CV)

Source: `dialect-normalisation/models/indicbart_combined/` & `mt5_combined_16k/` (5-Fold CV Averages)

| Evaluated Subset | Total Pairs | Test Set Size | IndicBART (244M) BLEU | IndicBART (244M) chrF++ | mT5-Small (300M) BLEU | mT5-Small (300M) chrF++ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Southern Konkan (D1) | 16,163 | 836/837 | 26.21 | 53.15 | **48.28** | **75.04** |
| Northern Konkan (D2) | 16,163 | 798/826 | 47.59 | 67.16 | **62.35** | **78.92** |
| Varhadi (D4) | 16,163 | 763/790 | 72.75 | 85.63 | **79.57** | **90.51** |
| **Overall Combined** | 16,163 | 2,424/2,426 | 48.17 | 68.58 | **63.29** | **81.37** |