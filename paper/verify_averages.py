import re

# Table 3: Single-dialect model performance on original datasets (5-Fold CV)
t3_ib_bleu = [48.54, 60.58, 76.63]
t3_ib_chrf = [78.35, 80.30, 90.93]
t3_mt5_bleu = [46.31, 60.31, 81.00]
t3_mt5_chrf = [74.43, 77.34, 91.21]

# Table 4: Multi-dialect model performance on original datasets (5-Fold CV)
t4_ib_bleu = [26.21, 47.59, 72.75]
t4_ib_chrf = [53.15, 67.16, 85.63]
t4_mt5_bleu = [48.28, 62.35, 79.57]
t4_mt5_chrf = [75.04, 78.92, 90.51]

# Table 5: Single-dialect model performance on expanded datasets (5-Fold CV)
t5_ib_bleu = [47.25, 40.51, 73.62]
t5_ib_chrf = [64.69, 62.21, 86.75]
t5_mt5_bleu = [65.10, 62.07, 78.89]
t5_mt5_chrf = [82.88, 79.29, 90.59]

# Table 6: Multi-dialect model performance on expanded datasets (5-Fold CV)
t6_ib_bleu = [52.06, 49.08, 70.27]
t6_ib_chrf = [72.15, 69.90, 84.27]
t6_mt5_bleu = [66.62, 62.72, 78.73]
t6_mt5_chrf = [83.57, 79.63, 90.46]

# Table 7: Held-Out Test Set (Single & Multi-Dialect)
t7_single_orig_ib = [57.80, 90.13, 83.59]
t7_single_orig_mt5 = [43.86, 79.46, 74.81]
t7_single_exp_ib = [24.06, 65.41, 67.97]
t7_single_exp_mt5 = [44.36, 79.76, 76.90]

t7_multi_orig_ib = [26.08, 47.92, 73.04]
t7_multi_orig_mt5 = [40.94, 77.50, 73.47]
t7_multi_exp_ib = [52.15, 54.35, 69.96]
t7_multi_exp_mt5 = [43.88, 78.16, 74.74]

# Table 8: Augmentation Impact (Held-out Test BLEU)
t8_ib_orig = [26.08, 47.92, 73.04]
t8_ib_exp = [52.15, 54.35, 69.96]
t8_mt5_orig = [40.94, 77.50, 73.47]
t8_mt5_exp = [43.88, 78.16, 74.74]

# Table 9: Verification Ablation
t9_ib_raw = (43.09, 64.54)
t9_ib_clean = (58.82, 75.49)
t9_mt5_raw = (61.21, 80.18)
t9_mt5_clean = (65.59, 84.58)

tables = {
    "Table 3 (Single Orig CV)": {
        "IB BLEU": t3_ib_bleu,
        "IB chrF++": t3_ib_chrf,
        "mT5 BLEU": t3_mt5_bleu,
        "mT5 chrF++": t3_mt5_chrf,
    },
    "Table 4 (Multi Orig CV)": {
        "IB BLEU": t4_ib_bleu,
        "IB chrF++": t4_ib_chrf,
        "mT5 BLEU": t4_mt5_bleu,
        "mT5 chrF++": t4_mt5_chrf,
    },
    "Table 5 (Single Exp CV)": {
        "IB BLEU": t5_ib_bleu,
        "IB chrF++": t5_ib_chrf,
        "mT5 BLEU": t5_mt5_bleu,
        "mT5 chrF++": t5_mt5_chrf,
    },
    "Table 6 (Multi Exp CV)": {
        "IB BLEU": t6_ib_bleu,
        "IB chrF++": t6_ib_chrf,
        "mT5 BLEU": t6_mt5_bleu,
        "mT5 chrF++": t6_mt5_chrf,
    },
    "Table 7 (Held-Out Test Set)": {
        "Single Orig IB BLEU": t7_single_orig_ib,
        "Single Orig mT5 BLEU": t7_single_orig_mt5,
        "Single Exp IB BLEU": t7_single_exp_ib,
        "Single Exp mT5 BLEU": t7_single_exp_mt5,
        "Multi Orig IB BLEU": t7_multi_orig_ib,
        "Multi Orig mT5 BLEU": t7_multi_orig_mt5,
        "Multi Exp IB BLEU": t7_multi_exp_ib,
        "Multi Exp mT5 BLEU": t7_multi_exp_mt5,
    }
}

print(f"{'='*80}")
print(f"{'TABLE':<30} | {'METRIC':<22} | {'SUM':<8} | {'EXACT AVG':<15} | {'ROUNDED (2dp)':<12}")
print(f"{'='*80}")

for tname, metrics in tables.items():
    for mname, vals in metrics.items():
        s = sum(vals)
        avg = s / len(vals)
        rnd = round(avg, 2)
        print(f"{tname:<30} | {mname:<22} | {s:<8.2f} | {avg:<15.6f} | {rnd:<12.2f}")
    print("-" * 80)
