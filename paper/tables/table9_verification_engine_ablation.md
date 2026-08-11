# Table 9: Impact of Multi-Tier Verification Engine: Raw Unverified vs. Filtered Clean Synthetic Data

Source: Verification engine ablation experiment summaries (`indicbart_raw_unverified_32k` vs `indicbart_combined_32k`, `mt5_raw_unverified_32k` vs `mt5_combined_32k`)

| Model Architecture | Pipeline State | Total Pairs | BLEU | chrF++ | Δ BLEU | Δ chrF++ |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| IndicBART (244M) | Raw Unverified Data | 19,914 | 43.09 | 64.54 | -14.03 | -10.95 |
| IndicBART (244M) | Filtered Clean Data | 32,335 | **57.12** | **75.49** | **+14.03** | **+10.95** |
|---|---|---|---|---|---|---|
| mT5-Small (300M) | Raw Unverified Data | 19,914 | 61.21 | 80.18 | -8.30 | -4.40 |
| mT5-Small (300M) | Filtered Clean Data | 32,335 | **69.51** | **84.58** | **+8.30** | **+4.40** |