import os
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

# Set publication style for matplotlib/seaborn with EXTRA LARGE, high-visibility fonts
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['font.size'] = 14
plt.rcParams['axes.edgecolor'] = '#777777'
plt.rcParams['axes.linewidth'] = 1.1
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['xtick.labelsize'] = 13
plt.rcParams['ytick.labelsize'] = 13
plt.rcParams['legend.fontsize'] = 13
plt.rcParams['grid.color'] = '#333333'
plt.rcParams['grid.linestyle'] = ':'
plt.rcParams['grid.linewidth'] = 1.4
plt.rcParams['grid.alpha'] = 0.85

OUTPUT_DIR = 'paper/figures'
INTERACTIVE_DIR = 'paper/figures/interactive'
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(INTERACTIVE_DIR, exist_ok=True)

print("=== Generating Figures with Extra Large Canvas Dimensions & Legible Fonts ===")

# ==============================================================================
# FIGURE 1: RESPIN-S1.0 Source Corpus Statistics (Table 1)
# ==============================================================================
def generate_fig1():
    print("Generating Figure 1: RESPIN Demographics & Domains...")
    domains = ['Banking', 'Agriculture']
    domain_counts = [415798, 394136]
    
    genders = ['Female', 'Male', 'N/A']
    gender_counts = [412137, 396399, 1398]
    
    dialects = ['Southern Konkan', 'Northern Konkan', 'Standard Marathi', 'Varhadi']
    dialect_hours = [247.31, 265.25, 255.79, 257.70]
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6.2), dpi=300)
    
    # 1. Domain Donut
    wedges, texts = axes[0].pie(domain_counts, labels=['Banking (51.3%)', 'Agriculture (48.7%)'], 
                               colors=['#2b5c8f', '#2ecc71'], wedgeprops=dict(width=0.45, edgecolor='w'),
                               textprops={'fontsize': 13.5, 'weight': 'bold'})
    axes[0].set_title("A. Domain Split (Utterances)", fontsize=16, fontweight='bold', pad=14)
    
    # 2. Gender Donut
    wedges, texts = axes[1].pie(gender_counts[:2], labels=['Female (51.0%)', 'Male (49.0%)'], 
                               colors=['#e74c3c', '#3498db'], wedgeprops=dict(width=0.45, edgecolor='w'),
                               textprops={'fontsize': 13.5, 'weight': 'bold'})
    axes[1].set_title("B. Gender Distribution", fontsize=16, fontweight='bold', pad=14)
    
    # 3. Dialect Hours Bar
    bar_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#9467bd']
    bars = axes[2].bar(dialects, dialect_hours, color=bar_colors, width=0.6, edgecolor='black', linewidth=1.0)
    axes[2].set_title("C. Audio Duration per Dialect (Hours)", fontsize=16, fontweight='bold', pad=14)
    axes[2].set_ylabel("Hours", fontsize=14.5, fontweight='bold')
    axes[2].set_xticks(range(len(dialects)))
    axes[2].set_xticklabels(dialects, rotation=20, ha='right', fontsize=12.5, fontweight='bold')
    axes[2].set_ylim(0, 310)
    
    for p in bars:
        h = p.get_height()
        axes[2].annotate(f"{h:.1f}h", (p.get_x() + p.get_width() / 2., h + 7),
                         ha='center', va='bottom', fontsize=13, color='black', fontweight='bold')
        
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig1_respin_demographics.png'), bbox_inches='tight')
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig1_respin_demographics.pdf'), bbox_inches='tight')
    plt.close()
    
    # Plotly Interactive
    df_dialect = pd.DataFrame({'Dialect': dialects, 'Hours': dialect_hours})
    fig_px = px.bar(df_dialect, x='Dialect', y='Hours', title='RESPIN Marathi Audio Duration by Dialect',
                    color='Dialect', color_discrete_sequence=bar_colors, text_auto='.1f')
    fig_px.update_layout(font=dict(size=16, family="Arial, sans-serif"))
    fig_px.write_html(os.path.join(INTERACTIVE_DIR, 'fig1_respin_demographics.html'))

# ==============================================================================
# FIGURE 2: Parallel Dataset Composition & Synthetic Yield (Table 2)
# ==============================================================================
def generate_fig2():
    print("Generating Figure 2: Parallel Dataset Composition & Synthetic Yield...")
    dialects = ['Southern Konkan', 'Northern Konkan', 'Varhadi']
    orig_pairs = [5576, 5501, 5086]
    synth_verified = [5569, 5534, 5069]
    rejected_pairs = [269, 291, 261]
    raw_synth = [5838, 5825, 5330]
    total_clean = [11145, 11035, 10155]
    rejection_pct = [4.61, 5.00, 4.90]
    
    df = pd.DataFrame({
        'Dialect': dialects,
        'Original Clean Pairs': orig_pairs,
        'Clean Synthetic Verified': synth_verified,
        'Corrupted / Rejected': rejected_pairs,
        'Raw Synthetic Generated': raw_synth,
        'Total Clean Expanded': total_clean
    })
    
    fig, ax = plt.subplots(figsize=(12, 6.8), dpi=300)
    x = np.arange(len(dialects))
    width = 0.35
    
    # Left bar: Original Clean Pairs
    rects1 = ax.bar(x - width/2, orig_pairs, width, label='Original Clean Pairs', color='#2c3e50', edgecolor='black', linewidth=1.0)
    
    # Right bar (Stacked): Clean Synthetic Verified + Corrupted/Rejected
    rects2 = ax.bar(x + width/2, synth_verified, width, label='Clean Synthetic Verified (Passed)', color='#27ae60', edgecolor='black', linewidth=1.0)
    rects3 = ax.bar(x + width/2, rejected_pairs, width, bottom=synth_verified, label='Corrupted / Rejected (Filtered)', color='#c0392b', edgecolor='black', linewidth=1.0)
    
    ax.set_ylabel('Parallel Sentence Pairs', fontsize=14.5, fontweight='bold')
    ax.set_title('Parallel Dataset Expansion, Synthetic Yield & Verification Filtering per Dialect', fontsize=15.5, fontweight='bold', pad=14)
    ax.set_xticks(x)
    ax.set_xticklabels(dialects, fontsize=13, fontweight='bold')
    ax.legend(loc='upper left', frameon=True, fontsize=12)
    
    for i in range(len(dialects)):
        # Annotate Original bar
        ax.text(x[i] - width/2, orig_pairs[i] + 120, f"{orig_pairs[i]:,}", ha='center', va='bottom', fontsize=11, fontweight='bold', color='#2c3e50')
        
        # Annotate Raw Synthetic bar total (verified + rejected)
        tot_raw = synth_verified[i] + rejected_pairs[i]
        ax.text(x[i] + width/2, tot_raw + 120, f"{tot_raw:,}\n({rejection_pct[i]:.2f}% rej)", ha='center', va='bottom', fontsize=11, fontweight='bold', color='#c0392b')
        
        # Overall annotation on top of cluster
        ax.annotate(f"Total Clean: {total_clean[i]:,}", (x[i], max(orig_pairs[i], tot_raw) + 1100), ha='center', fontsize=12, fontweight='bold', color='#1e8449',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='#e8f8f5', edgecolor='#27ae60', linewidth=1.2))
        
    ax.set_ylim(0, 7800)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig2_dataset_composition.png'), bbox_inches='tight')
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig2_dataset_composition.pdf'), bbox_inches='tight')
    plt.close()
    
    # Plotly Interactive
    fig_px = px.bar(df, x='Dialect', y=['Original Clean Pairs', 'Clean Synthetic Verified', 'Corrupted / Rejected'],
                    title='Parallel Dataset Expansion & Verification Filtering per Dialect', barmode='group', text_auto=True)
    fig_px.update_layout(font=dict(size=16))
    fig_px.write_html(os.path.join(INTERACTIVE_DIR, 'fig2_dataset_composition.html'))

# ==============================================================================
# FIGURE 3: Comprehensive RESPIN Benchmark Matrix (Table 3)
# ==============================================================================
def generate_fig3():
    print("Generating Figure 3: RESPIN Test Set Evaluation Matrix...")
    data = [
        {'Partition': 'Southern Konkan (Orig)', 'IndicBART': 57.80, 'mT5': 43.86},
        {'Partition': 'Northern Konkan (Orig)', 'IndicBART': 90.13, 'mT5': 79.46},
        {'Partition': 'Varhadi (Orig)', 'IndicBART': 83.59, 'mT5': 74.81},
        {'Partition': 'Southern Konkan (Exp)', 'IndicBART': 24.06, 'mT5': 44.36},
        {'Partition': 'Northern Konkan (Exp)', 'IndicBART': 65.41, 'mT5': 79.76},
        {'Partition': 'Varhadi (Exp)', 'IndicBART': 67.97, 'mT5': 76.90},
    ]
    df = pd.DataFrame(data)
    
    fig, ax = plt.subplots(figsize=(14, 7.0), dpi=300)
    x = np.arange(len(df['Partition']))
    width = 0.35
    
    rects1 = ax.bar(x - width/2, df['IndicBART'], width, label='IndicBART (244M)', color='#2ca02c', edgecolor='black', linewidth=1.0)
    rects2 = ax.bar(x + width/2, df['mT5'], width, label='mT5-Small (300M)', color='#d62728', edgecolor='black', linewidth=1.0)
    
    ax.set_ylabel('BLEU Score (sacreBLEU)', fontsize=14.5, fontweight='bold')
    ax.set_title('Evaluation Matrix: Single-Dialect vs. Multi-Dialect Models on RESPIN Test Benchmark', fontsize=16, fontweight='bold', pad=14)
    ax.set_xticks(x)
    ax.set_xticklabels(df['Partition'], rotation=20, ha='right', fontsize=12.5, fontweight='bold')
    ax.legend(loc='upper left', frameon=True, fontsize=12.5)
    ax.set_ylim(0, 116)
    
    ax.axvline(2.5, color='#222222', linestyle=':', alpha=0.9, linewidth=1.8)
    ax.text(1.3, 103, 'Original Training Set', ha='center', fontsize=12.5, fontweight='bold', bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='gray'))
    ax.text(4.1, 103, 'Synthetically Expanded Set', ha='center', fontsize=12.5, fontweight='bold', bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='gray'))
    
    for rect in rects1 + rects2:
        height = rect.get_height()
        ax.annotate(f'{height:.1f}', (rect.get_x() + rect.get_width() / 2., height + 1.2),
                    ha='center', va='bottom', fontsize=12, fontweight='bold', color='black')
        
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig3_respin_comprehensive.png'), bbox_inches='tight')
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig3_respin_comprehensive.pdf'), bbox_inches='tight')
    plt.close()
    
    # Interactive Plotly Chart
    df_long = pd.melt(df, id_vars=['Partition'], value_vars=['IndicBART', 'mT5'], var_name='Model', value_name='BLEU')
    fig_px = px.bar(df_long, x='Partition', y='BLEU', color='Model', barmode='group',
                    title='RESPIN Test Benchmark BLEU Comparison (IndicBART vs mT5-Small)', text_auto='.1f')
    fig_px.update_layout(font=dict(size=16))
    fig_px.write_html(os.path.join(INTERACTIVE_DIR, 'fig3_respin_comprehensive.html'))

# ==============================================================================
# FIGURE 4: Single-Dialect 5-Fold Cross-Validation Matrix (Table 4 & Table 6)
# ==============================================================================
# ==============================================================================
# FIGURE 4 OPTIONS: Dumbbell, Slope, Radar, and Grouped Bar
# ==============================================================================
def generate_fig4_dumbbell():
    print("Generating Figure 4 Option 1: Dumbbell / Cleveland Dot Plot...")
    dialects = ['Southern Konkan', 'Northern Konkan', 'Varhadi']
    y_pos = np.arange(len(dialects))
    
    ib_orig = [48.54, 60.58, 76.63]
    ib_exp  = [47.25, 40.51, 73.62]
    
    mt5_orig = [46.31, 60.31, 80.99]
    mt5_exp  = [65.10, 62.07, 78.89]
    
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 5.5), dpi=300)
    
    # Subplot A: IndicBART (244M)
    for i in range(len(dialects)):
        axes[0].plot([ib_orig[i], ib_exp[i]], [y_pos[i], y_pos[i]], color='gray', linewidth=2.5, alpha=0.7, zorder=1)
    axes[0].scatter(ib_orig, y_pos, color='#1f77b4', s=180, label='Original', zorder=2, edgecolors='black', linewidth=1.2)
    axes[0].scatter(ib_exp, y_pos, color='#00a896', s=180, marker='s', label='Synthetically Expanded', zorder=2, edgecolors='black', linewidth=1.2)
    
    for i in range(len(dialects)):
        axes[0].annotate(f'{ib_orig[i]:.1f}', (ib_orig[i], y_pos[i] + 0.18), ha='center', fontsize=11, fontweight='bold', color='#1f77b4')
        axes[0].annotate(f'{ib_exp[i]:.1f}', (ib_exp[i], y_pos[i] - 0.28), ha='center', fontsize=11, fontweight='bold', color='#00a896')
        
    axes[0].set_yticks(y_pos)
    axes[0].set_yticklabels(dialects, fontsize=13, fontweight='bold')
    axes[0].set_xlabel('5-Fold CV Test BLEU', fontsize=13.5, fontweight='bold')
    axes[0].set_title('A. IndicBART (244M) Single-Dialect Performance Shift', fontsize=14, fontweight='bold', pad=12)
    axes[0].set_xlim(35, 90)
    axes[0].grid(axis='x', linestyle='--', alpha=0.5)
    axes[0].legend(fontsize=12, loc='lower right')
    
    # Subplot B: mT5-Small (300M)
    for i in range(len(dialects)):
        axes[1].plot([mt5_orig[i], mt5_exp[i]], [y_pos[i], y_pos[i]], color='gray', linewidth=2.5, alpha=0.7, zorder=1)
    axes[1].scatter(mt5_orig, y_pos, color='#8b263e', s=180, label='Original', zorder=2, edgecolors='black', linewidth=1.2)
    axes[1].scatter(mt5_exp, y_pos, color='#ff6f00', s=180, marker='s', label='Synthetically Expanded', zorder=2, edgecolors='black', linewidth=1.2)
    
    for i in range(len(dialects)):
        axes[1].annotate(f'{mt5_orig[i]:.1f}', (mt5_orig[i], y_pos[i] + 0.18), ha='center', fontsize=11, fontweight='bold', color='#8b263e')
        axes[1].annotate(f'{mt5_exp[i]:.1f}', (mt5_exp[i], y_pos[i] - 0.28), ha='center', fontsize=11, fontweight='bold', color='#ff6f00')
        diff = mt5_exp[i] - mt5_orig[i]
        if diff > 10:
            axes[1].annotate(f'(+{diff:.1f})', (mt5_exp[i] + 3.8, y_pos[i]), va='center', fontsize=11, fontweight='bold', color='#007700')
        
    axes[1].set_yticks(y_pos)
    axes[1].set_yticklabels(dialects, fontsize=13, fontweight='bold')
    axes[1].set_xlabel('5-Fold CV Test BLEU', fontsize=13.5, fontweight='bold')
    axes[1].set_title('B. mT5-Small (300M) Single-Dialect Performance Shift', fontsize=14, fontweight='bold', pad=12)
    axes[1].set_xlim(35, 90)
    axes[1].grid(axis='x', linestyle='--', alpha=0.5)
    axes[1].legend(fontsize=12, loc='lower right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig4_single_dialect_cv.png'), bbox_inches='tight')
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig4_single_dialect_cv.pdf'), bbox_inches='tight')
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig4_dumbbell.png'), bbox_inches='tight')
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig4_dumbbell.pdf'), bbox_inches='tight')
    plt.close()

def generate_fig4_slope():
    print("Generating Figure 4 Option 2: Slope Chart...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=300)
    
    dialects = ['Southern Konkan', 'Northern Konkan', 'Varhadi']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    # IndicBART
    ib_orig = [48.54, 60.58, 76.63]
    ib_exp  = [47.25, 40.51, 73.62]
    
    for i in range(3):
        axes[0].plot([0, 1], [ib_orig[i], ib_exp[i]], marker='o', markersize=8, linewidth=2.5, label=dialects[i], color=colors[i])
        axes[0].annotate(f'{ib_orig[i]:.1f}', (-0.05, ib_orig[i]), ha='right', va='center', fontweight='bold', fontsize=11)
        axes[0].annotate(f'{ib_exp[i]:.1f}', (1.05, ib_exp[i]), ha='left', va='center', fontweight='bold', fontsize=11)
        
    axes[0].set_xticks([0, 1])
    axes[0].set_xticklabels(['Original', 'Synthetically Expanded'], fontsize=12, fontweight='bold')
    axes[0].set_ylabel('5-Fold CV BLEU', fontsize=13, fontweight='bold')
    axes[0].set_title('A. IndicBART (244M) Slope Chart', fontsize=14, fontweight='bold', pad=12)
    axes[0].legend(fontsize=11, loc='lower left')
    axes[0].set_ylim(35, 85)
    axes[0].grid(axis='y', linestyle='--', alpha=0.5)
    
    # mT5-Small
    mt5_orig = [46.31, 60.31, 80.99]
    mt5_exp  = [65.10, 62.07, 78.89]
    
    for i in range(3):
        axes[1].plot([0, 1], [mt5_orig[i], mt5_exp[i]], marker='s', markersize=8, linewidth=2.5, label=dialects[i], color=colors[i])
        axes[1].annotate(f'{mt5_orig[i]:.1f}', (-0.05, mt5_orig[i]), ha='right', va='center', fontweight='bold', fontsize=11)
        axes[1].annotate(f'{mt5_exp[i]:.1f}', (1.05, mt5_exp[i]), ha='left', va='center', fontweight='bold', fontsize=11)
        
    axes[1].set_xticks([0, 1])
    axes[1].set_xticklabels(['Original', 'Synthetically Expanded'], fontsize=12, fontweight='bold')
    axes[1].set_ylabel('5-Fold CV BLEU', fontsize=13, fontweight='bold')
    axes[1].set_title('B. mT5-Small (300M) Slope Chart', fontsize=14, fontweight='bold', pad=12)
    axes[1].legend(fontsize=11, loc='lower left')
    axes[1].set_ylim(35, 85)
    axes[1].grid(axis='y', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig4_slope.png'), bbox_inches='tight')
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig4_slope.pdf'), bbox_inches='tight')
    plt.close()

def generate_fig4_radar():
    print("Generating Figure 4 Option 3: Radar / Spider Chart...")
    categories = ['Southern Konkan', 'Northern Konkan', 'Varhadi']
    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.5), subplot_kw=dict(polar=True), dpi=300)
    
    # IndicBART
    ib_orig = [48.54, 60.58, 76.63]
    ib_exp  = [47.25, 40.51, 73.62]
    ib_orig_r = ib_orig + ib_orig[:1]
    ib_exp_r  = ib_exp + ib_exp[:1]
    
    axes[0].plot(angles, ib_orig_r, linewidth=2.5, linestyle='solid', label='Original', color='#1f77b4')
    axes[0].fill(angles, ib_orig_r, color='#1f77b4', alpha=0.2)
    axes[0].plot(angles, ib_exp_r, linewidth=2.5, linestyle='solid', label='Synthetically Expanded', color='#00a896')
    axes[0].fill(angles, ib_exp_r, color='#00a896', alpha=0.2)
    axes[0].set_xticks(angles[:-1])
    axes[0].set_xticklabels(categories, fontsize=12, fontweight='bold')
    axes[0].set_title('A. IndicBART Radar Profile', fontsize=14, fontweight='bold', pad=15)
    axes[0].legend(loc='lower left', bbox_to_anchor=(-0.1, -0.1), fontsize=11)
    
    # mT5-Small
    mt5_orig = [46.31, 60.31, 80.99]
    mt5_exp  = [65.10, 62.07, 78.89]
    mt5_orig_r = mt5_orig + mt5_orig[:1]
    mt5_exp_r  = mt5_exp + mt5_exp[:1]
    
    axes[1].plot(angles, mt5_orig_r, linewidth=2.5, linestyle='solid', label='Original', color='#8b263e')
    axes[1].fill(angles, mt5_orig_r, color='#8b263e', alpha=0.2)
    axes[1].plot(angles, mt5_exp_r, linewidth=2.5, linestyle='solid', label='Synthetically Expanded', color='#ff6f00')
    axes[1].fill(angles, mt5_exp_r, color='#ff6f00', alpha=0.2)
    axes[1].set_xticks(angles[:-1])
    axes[1].set_xticklabels(categories, fontsize=12, fontweight='bold')
    axes[1].set_title('B. mT5-Small Radar Profile', fontsize=14, fontweight='bold', pad=15)
    axes[1].legend(loc='lower left', bbox_to_anchor=(-0.1, -0.1), fontsize=11)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig4_radar.png'), bbox_inches='tight')
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig4_radar.pdf'), bbox_inches='tight')
    plt.close()

def generate_fig4():
    generate_fig4_dumbbell()

# ==============================================================================
# FIGURE 5: Multi-Dialect 5-Fold CV Heatmap (Table 5 & Table 7)
# ==============================================================================
def generate_fig5():
    print("Generating Figure 5: Multi-Dialect Cross-Validation Dual Heatmap...")
    matrix_bleu = np.array([
        [26.21, 47.59, 72.75],  # IndicBART Original
        [48.28, 62.35, 79.57],  # mT5-Small Original
        [52.06, 49.08, 70.27],  # IndicBART Expanded
        [66.62, 62.72, 78.73]   # mT5-Small Expanded
    ])
    
    matrix_delta = np.array([
        [+25.85, +1.49, -2.48],  # IndicBART Expansion Gain
        [+18.34, +0.37, -0.84]   # mT5-Small Expansion Gain
    ])
    
    y_labels_a = [
        'IndicBART (Original)',
        'mT5-Small (Original)',
        'IndicBART (Expanded)',
        'mT5-Small (Expanded)'
    ]
    y_labels_b = [
        'IndicBART (244M)',
        'mT5-Small (300M)'
    ]
    x_labels = ['Southern Konkan', 'Northern Konkan', 'Varhadi']
    
    fig, axes = plt.subplots(1, 2, figsize=(16.5, 6.2), dpi=300, gridspec_kw={'width_ratios': [1.2, 1]})
    
    # Subplot A: Absolute BLEU Heatmap
    sns.heatmap(matrix_bleu, annot=True, fmt=".2f", cmap="YlGnBu", xticklabels=x_labels, yticklabels=y_labels_a,
                cbar_kws={'label': '5-Fold CV BLEU Score'}, ax=axes[0], linewidths=2.0, linecolor='white',
                annot_kws={"size": 13, "weight": "bold"})
    axes[0].set_title("A. Multi-Dialect Performance Heatmap (BLEU)", fontsize=14.5, fontweight='bold', pad=12)
    axes[0].set_xticklabels(x_labels, rotation=0, ha='center', fontsize=12, fontweight='bold')
    axes[0].set_yticklabels(y_labels_a, rotation=0, fontsize=11.5, fontweight='bold')
    
    # Subplot B: Net BLEU Gain Heatmap (Expanded - Original)
    sns.heatmap(matrix_delta, annot=True, fmt="+.2f", cmap="PiYG", center=0, xticklabels=x_labels, yticklabels=y_labels_b,
                cbar_kws={'label': 'Net BLEU Gain (Δ BLEU)'}, ax=axes[1], linewidths=2.0, linecolor='white',
                annot_kws={"size": 13.5, "weight": "bold"})
    axes[1].set_title("B. Synthetic Expansion Net Impact (Δ BLEU)", fontsize=14.5, fontweight='bold', pad=12)
    axes[1].set_xticklabels(x_labels, rotation=0, ha='center', fontsize=12, fontweight='bold')
    axes[1].set_yticklabels(y_labels_b, rotation=0, fontsize=11.5, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig5_multidialect_heatmap.png'), bbox_inches='tight')
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig5_multidialect_heatmap.pdf'), bbox_inches='tight')
    plt.close()

# ==============================================================================
# FIGURE 6: Synthetic Data Augmentation Impact (Table 8)
# ==============================================================================
def generate_fig6():
    print("Generating Figure 6: Synthetic Augmentation Impact Diverging Bar...")
    subsets = ['Southern Konkan', 'Northern Konkan', 'Varhadi']
    delta_ib = [+25.85, +1.49, -2.48]
    delta_mt5 = [+18.34, +0.37, -0.84]
    
    x = np.arange(len(subsets))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(11, 6.5), dpi=300)
    
    rects1 = ax.bar(x - width/2, delta_ib, width, label='IndicBART Δ BLEU', color='#2b5c8f', edgecolor='black', linewidth=1.0)
    rects2 = ax.bar(x + width/2, delta_mt5, width, label='mT5-Small Δ BLEU', color='#27ae60', edgecolor='black', linewidth=1.0)
    
    ax.axhline(0, color='black', linewidth=1.8)
    ax.set_ylabel('Δ BLEU Score (Expanded vs Original 5-Fold CV)', fontsize=14.5, fontweight='bold')
    ax.set_title('Impact of LLM Synthetic Data Expansion across Dialects (Δ BLEU)', fontsize=16, fontweight='bold', pad=14)
    ax.set_xticks(x)
    ax.set_xticklabels(subsets, fontsize=13, fontweight='bold')
    ax.legend(loc='upper right', frameon=True, fontsize=13)
    ax.set_ylim(-15, 26)
    
    for rect in rects1 + rects2:
        val = rect.get_height()
        va = 'bottom' if val >= 0 else 'top'
        offset = 0.7 if val >= 0 else -2.0
        ax.annotate(f'{val:+.2f}', (rect.get_x() + rect.get_width() / 2., val + offset),
                    ha='center', va=va, fontsize=12.5, fontweight='bold', color='black')
        
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig6_augmentation_impact.png'), bbox_inches='tight')
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig6_augmentation_impact.pdf'), bbox_inches='tight')
    plt.close()

# ==============================================================================
# FIGURE 7: Verification Engine Ablation Study (Table 9)
# ==============================================================================
def generate_fig7():
    print("Generating Figure 7: Multi-Tier Verification Engine Ablation...")
    models = ['IndicBART (244M)', 'mT5-Small (300M)']
    raw_bleu = [43.09, 61.21]
    clean_bleu = [57.12, 69.51]
    
    raw_chrf = [64.54, 80.18]
    clean_chrf = [75.49, 84.58]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.5), dpi=300)
    x = np.arange(len(models))
    width = 0.35
    
    # Panel 1: BLEU Comparison
    axes[0].bar(x - width/2, raw_bleu, width, label='Raw Unverified Data (19,914 pairs)', color='#e74c3c', edgecolor='black', linewidth=1.0)
    axes[0].bar(x + width/2, clean_bleu, width, label='Filtered Clean Data (32,335 pairs)', color='#2ecc71', edgecolor='black', linewidth=1.0)
    axes[0].set_ylabel('BLEU Score', fontsize=14.5, fontweight='bold')
    axes[0].set_title('A. BLEU Score Recovery via Verification', fontsize=15.5, fontweight='bold', pad=12)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(models, fontsize=13, fontweight='bold')
    axes[0].legend(fontsize=12.5)
    axes[0].set_ylim(0, 92)
    
    for i in range(len(models)):
        axes[0].annotate(f'{raw_bleu[i]:.2f}', (x[i] - width/2, raw_bleu[i] + 2.0), ha='center', fontsize=12.5, fontweight='bold', color='black')
        axes[0].annotate(f'{clean_bleu[i]:.2f}\n(+{clean_bleu[i]-raw_bleu[i]:.2f})', (x[i] + width/2, clean_bleu[i] + 2.0), ha='center', fontsize=12.5, fontweight='bold', color='green')
        
    # Panel 2: chrF++ Comparison
    axes[1].bar(x - width/2, raw_chrf, width, label='Raw Unverified Data', color='#e74c3c', edgecolor='black', linewidth=1.0)
    axes[1].bar(x + width/2, clean_chrf, width, label='Filtered Clean Data', color='#2ecc71', edgecolor='black', linewidth=1.0)
    axes[1].set_ylabel('chrF++ Score', fontsize=14.5, fontweight='bold')
    axes[1].set_title('B. chrF++ Score Enhancement via Verification', fontsize=15.5, fontweight='bold', pad=12)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(models, fontsize=13, fontweight='bold')
    axes[1].legend(fontsize=12.5)
    axes[1].set_ylim(0, 102)
    
    for i in range(len(models)):
        axes[1].annotate(f'{raw_chrf[i]:.2f}', (x[i] - width/2, raw_chrf[i] + 2.0), ha='center', fontsize=12.5, fontweight='bold', color='black')
        axes[1].annotate(f'{clean_chrf[i]:.2f}\n(+{clean_chrf[i]-raw_chrf[i]:.2f})', (x[i] + width/2, clean_chrf[i] + 2.0), ha='center', fontsize=12.5, fontweight='bold', color='green')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig7_verification_ablation.png'), bbox_inches='tight')
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig7_verification_ablation.pdf'), bbox_inches='tight')
    plt.close()

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================
if __name__ == '__main__':
    generate_fig1()
    generate_fig2()
    generate_fig3()
    generate_fig4_dumbbell()
    generate_fig4_slope()
    generate_fig4_radar()
    generate_fig5()
    generate_fig6()
    generate_fig7()
    print("SUCCESS: All static & interactive figures generated successfully in paper/figures/!")
