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