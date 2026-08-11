# Table 8: Impact of Synthetic Data Augmentation: Original vs. Synthetically Expanded Data BLEU Change (5-Fold CV)

Source: Computed from 5-fold cross-validation multi-dialect model summaries (`cv_summary.json`)

| Evaluated Subset | IndicBART Original | IndicBART Expanded | IndicBART Δ BLEU | mT5-Small Original | mT5-Small Expanded | mT5-Small Δ BLEU |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Southern Konkan (D1) | 26.21 | 52.06 | **+25.85** | 48.28 | **66.62** | **+18.34** |
| Northern Konkan (D2) | 47.59 | 49.08 | **+1.49** | 62.35 | **62.72** | **+0.37** |
| Varhadi (D4) | 72.75 | 70.27 | -2.48 | 79.57 | **78.73** | -0.84 |
| **Overall Combined** | 48.17 | 57.12 | **+8.95** | 63.29 | **69.51** | **+6.22** |