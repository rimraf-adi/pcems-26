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
    
    fig = plt.figure(figsize=(15, 9.5), dpi=300)
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 0.9])
    
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[1, :])
    
    # 1. Domain Donut
    wedges, texts = ax0.pie(domain_counts, labels=['Banking (51.3%)', 'Agriculture (48.7%)'], 
                           colors=['#2b5c8f', '#2ecc71'], wedgeprops=dict(width=0.45, edgecolor='w', linewidth=2.0),
                           textprops={'fontsize': 14.5, 'weight': 'bold'})
    ax0.set_title("A. Domain Split (Utterances)", fontsize=16.5, fontweight='bold', pad=14)
    
    # 2. Gender Donut
    wedges, texts = ax1.pie(gender_counts[:2], labels=['Female (51.0%)', 'Male (49.0%)'], 
                           colors=['#e74c3c', '#3498db'], wedgeprops=dict(width=0.45, edgecolor='w', linewidth=2.0),
                           textprops={'fontsize': 14.5, 'weight': 'bold'})
    ax1.set_title("B. Gender Distribution", fontsize=16.5, fontweight='bold', pad=14)
    
    # 3. Dialect Hours Bar
    bar_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#9467bd']
    bars = ax2.bar(dialects, dialect_hours, color=bar_colors, width=0.5, edgecolor='black', linewidth=1.4)
    ax2.set_title("C. Audio Duration per Dialect (Hours)", fontsize=16.5, fontweight='bold', pad=14)
    ax2.set_ylabel("Audio Duration (Hours)", fontsize=15, fontweight='bold')
    ax2.set_xticks(range(len(dialects)))
    ax2.set_xticklabels(dialects, fontsize=14, fontweight='bold')
    ax2.set_ylim(0, 310)
    ax2.grid(axis='y', linestyle='--', alpha=0.6)
    
    for p in bars:
        h = p.get_height()
        ax2.annotate(f"{h:.1f}h", (p.get_x() + p.get_width() / 2., h + 7),
                     ha='center', va='bottom', fontsize=14, color='black', fontweight='bold')
        
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
def generate_fig4():
    print("Generating Figure 4: Single-Dialect Cross-Validation Vertical Grouped Bar Chart...")
    dialects = ['Southern Konkan', 'Northern Konkan', 'Varhadi']
    x = np.arange(len(dialects))
    width = 0.28
    
    ib_orig = [48.54, 60.58, 76.63]
    ib_exp  = [47.25, 40.51, 73.62]
    
    mt5_orig = [46.31, 60.31, 80.99]
    mt5_exp  = [65.10, 62.07, 78.89]
    
    fig, axes = plt.subplots(2, 1, figsize=(11.0, 7.8), dpi=300)
    
    # Subplot A: IndicBART (244M)
    rects1 = axes[0].bar(x - width/2, ib_orig, width, label='Original', color='#2b5c8f', edgecolor='black', linewidth=1.1)
    rects2 = axes[0].bar(x + width/2, ib_exp, width, label='Synthetically Expanded', color='#41b6c4', edgecolor='black', linewidth=1.1)
    
    axes[0].set_ylabel('5-Fold CV BLEU Score', fontsize=12, fontweight='bold')
    axes[0].set_title('A. IndicBART (244M) Single-Dialect CV BLEU', fontsize=13, fontweight='bold', pad=12)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(dialects, fontsize=11.5, fontweight='bold')
    axes[0].set_ylim(0, 98)
    axes[0].legend(loc='upper right', frameon=True, facecolor='white', edgecolor='#cccccc', fontsize=10.5)
    axes[0].grid(axis='y', linestyle='--', alpha=0.5)
    
    for rect in rects1:
        h = rect.get_height()
        axes[0].annotate(f'{h:.1f}', (rect.get_x() + rect.get_width()/2, h + 1.8), ha='center', va='bottom', fontsize=10.5, fontweight='bold')
    for rect in rects2:
        h = rect.get_height()
        axes[0].annotate(f'{h:.1f}', (rect.get_x() + rect.get_width()/2, h + 1.8), ha='center', va='bottom', fontsize=10.5, fontweight='bold')
        
    # Subplot B: mT5-Small (300M)
    rects3 = axes[1].bar(x - width/2, mt5_orig, width, label='Original', color='#8c2d19', edgecolor='black', linewidth=1.1)
    rects4 = axes[1].bar(x + width/2, mt5_exp, width, label='Synthetically Expanded', color='#fe9929', edgecolor='black', linewidth=1.1)
    
    axes[1].set_ylabel('5-Fold CV BLEU Score', fontsize=12, fontweight='bold')
    axes[1].set_title('B. mT5-Small (300M) Single-Dialect CV BLEU', fontsize=13, fontweight='bold', pad=12)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(dialects, fontsize=11.5, fontweight='bold')
    axes[1].set_ylim(0, 98)
    axes[1].legend(loc='upper left', frameon=True, facecolor='white', edgecolor='#cccccc', fontsize=10.5)
    axes[1].grid(axis='y', linestyle='--', alpha=0.5)
    
    for rect in rects3:
        h = rect.get_height()
        axes[1].annotate(f'{h:.1f}', (rect.get_x() + rect.get_width()/2, h + 1.8), ha='center', va='bottom', fontsize=10.5, fontweight='bold')
    for rect in rects4:
        h = rect.get_height()
        axes[1].annotate(f'{h:.1f}', (rect.get_x() + rect.get_width()/2, h + 1.8), ha='center', va='bottom', fontsize=10.5, fontweight='bold')
        
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig4_single_dialect_cv.png'), bbox_inches='tight')
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig4_single_dialect_cv.pdf'), bbox_inches='tight')
    plt.close()

def generate_fig4_dumbbell():
    generate_fig4()

def generate_fig4_slope():
    generate_fig4()

def generate_fig4_radar():
    generate_fig4()

# ==============================================================================
# FIGURE 5: Multi-Dialect 5-Fold CV Performance Matrix & Net Impact
# ==============================================================================
def generate_fig5():
    print("Generating Figure 5: Multi-Dialect Cross-Validation Vertical Grouped Bar & Net Impact Chart...")
    dialects = ['Southern Konkan', 'Northern Konkan', 'Varhadi']
    x = np.arange(len(dialects))
    width = 0.18
    
    ib_orig = [26.21, 47.59, 72.75]
    mt5_orig = [48.28, 62.35, 79.57]
    ib_exp  = [52.06, 49.08, 70.27]
    mt5_exp  = [66.62, 62.72, 78.73]
    
    ib_delta  = [25.85, 1.49, -2.48]
    mt5_delta = [18.34, 0.37, -0.84]
    
    fig, axes = plt.subplots(2, 1, figsize=(11.0, 8.2), dpi=300)
    
    # Subplot A: Absolute BLEU Scores across Multi-Dialect Models
    rects1 = axes[0].bar(x - 1.5*width, ib_orig, width, label='IndicBART (Original)', color='#2b5c8f', edgecolor='black', linewidth=1.1)
    rects2 = axes[0].bar(x - 0.5*width, mt5_orig, width, label='mT5-Small (Original)', color='#8c2d19', edgecolor='black', linewidth=1.1)
    rects3 = axes[0].bar(x + 0.5*width, ib_exp, width, label='IndicBART (Expanded)', color='#41b6c4', edgecolor='black', linewidth=1.1)
    rects4 = axes[0].bar(x + 1.5*width, mt5_exp, width, label='mT5-Small (Expanded)', color='#fe9929', edgecolor='black', linewidth=1.1)
    
    axes[0].set_ylabel('5-Fold CV BLEU Score', fontsize=12, fontweight='bold')
    axes[0].set_title('A. Multi-Dialect Performance Matrix (BLEU)', fontsize=13, fontweight='bold', pad=12)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(dialects, fontsize=11.5, fontweight='bold')
    axes[0].set_ylim(0, 98)
    axes[0].legend(loc='upper left', frameon=True, facecolor='white', edgecolor='#cccccc', fontsize=10, ncol=2)
    axes[0].grid(axis='y', linestyle='--', alpha=0.5)
    
    for rects in [rects1, rects2, rects3, rects4]:
        for rect in rects:
            h = rect.get_height()
            axes[0].annotate(f'{h:.1f}', (rect.get_x() + rect.get_width()/2, h + 1.8), ha='center', va='bottom', fontsize=10, fontweight='bold')
            
    # Subplot B: Net Synthetic Expansion Gain (Δ BLEU)
    w_b = 0.28
    rects_b1 = axes[1].bar(x - w_b/2, ib_delta, w_b, label='IndicBART Δ BLEU', color='#2ca02c', edgecolor='black', linewidth=1.1)
    rects_b2 = axes[1].bar(x + w_b/2, mt5_delta, w_b, label='mT5-Small Δ BLEU', color='#1f77b4', edgecolor='black', linewidth=1.1)
    
    axes[1].axhline(0, color='black', linewidth=1.2, linestyle='-')
    axes[1].set_ylabel('Net BLEU Gain (Δ BLEU)', fontsize=12, fontweight='bold')
    axes[1].set_title('B. Synthetic Expansion Net Impact Across Dialects', fontsize=13, fontweight='bold', pad=12)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(dialects, fontsize=11.5, fontweight='bold')
    axes[1].set_ylim(-6, 32)
    axes[1].legend(loc='upper right', frameon=True, facecolor='white', edgecolor='#cccccc', fontsize=10.5)
    axes[1].grid(axis='y', linestyle='--', alpha=0.5)
    
    for rect in rects_b1:
        h = rect.get_height()
        va = 'bottom' if h >= 0 else 'top'
        offset = 1.0 if h >= 0 else -1.8
        axes[1].annotate(f'{h:+.2f}', (rect.get_x() + rect.get_width()/2, h + offset), ha='center', va=va, fontsize=10.5, fontweight='bold')
        
    for rect in rects_b2:
        h = rect.get_height()
        va = 'bottom' if h >= 0 else 'top'
        offset = 1.0 if h >= 0 else -1.8
        axes[1].annotate(f'{h:+.2f}', (rect.get_x() + rect.get_width()/2, h + offset), ha='center', va=va, fontsize=10.5, fontweight='bold')
        
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig5_multidialect_heatmap.png'), bbox_inches='tight')
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig5_multidialect_heatmap.pdf'), bbox_inches='tight')
    plt.close()

# ==============================================================================
# FIGURE 6: Synthetic Data Augmentation Impact (Table 8)
# ==============================================================================
def generate_fig6():
    print("Generating Figure 6: Synthetic Augmentation Impact Diverging Bar...")
    metrics = [
        'IndicBART - SK', 'IndicBART - NK', 'IndicBART - VH',
        'mT5-Small - SK', 'mT5-Small - NK', 'mT5-Small - VH'
    ]
    deltas = [25.85, 1.49, -2.48, 18.34, 0.37, -0.84]
    colors = ['#2ca02c' if d > 0 else '#d62728' for d in deltas]
    
    fig, ax = plt.subplots(figsize=(12, 4.6), dpi=300)
    y_pos = np.arange(len(metrics))
    
    bars = ax.barh(y_pos, deltas, color=colors, edgecolor='black', linewidth=1.1, height=0.6)
    ax.axvline(0, color='black', linewidth=1.2, linestyle='-')
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(metrics, fontsize=12, fontweight='bold')
    ax.invert_yaxis()
    ax.set_xlabel('Net BLEU Change (Δ BLEU)', fontsize=13, fontweight='bold')
    ax.set_title('Impact of Synthetic Data Augmentation Across Model Architectures and Dialects', fontsize=14, fontweight='bold', pad=15)
    ax.set_xlim(-6, 30)
    ax.grid(axis='x', linestyle='--', alpha=0.5)
    
    for bar in bars:
        w = bar.get_width()
        ha = 'left' if w >= 0 else 'right'
        offset = 0.5 if w >= 0 else -0.5
        ax.annotate(f'{w:+.2f}', (w + offset, bar.get_y() + bar.get_height()/2),
                    ha=ha, va='center', fontsize=11.5, fontweight='bold', color='black')
        
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
    
    fig, axes = plt.subplots(2, 1, figsize=(11.0, 7.8), dpi=300)
    x = np.arange(len(models))
    width = 0.30
    
    # Panel 1: BLEU Comparison
    axes[0].bar(x - width/2, raw_bleu, width, label='Raw Unverified Data (19,914 pairs)', color='#e74c3c', edgecolor='black', linewidth=1.2)
    axes[0].bar(x + width/2, clean_bleu, width, label='Filtered Clean Data (32,335 pairs)', color='#2ecc71', edgecolor='black', linewidth=1.2)
    axes[0].set_ylabel('BLEU Score', fontsize=12, fontweight='bold')
    axes[0].set_title('A. BLEU Score Recovery via Verification', fontsize=13, fontweight='bold', pad=12)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(models, fontsize=11.5, fontweight='bold')
    axes[0].legend(fontsize=10.5, loc='upper left', frameon=True, facecolor='white', edgecolor='#cccccc')
    axes[0].set_ylim(0, 85)
    axes[0].grid(axis='y', linestyle='--', alpha=0.5)
    
    for i in range(len(models)):
        axes[0].annotate(f'{raw_bleu[i]:.2f}', (x[i] - width/2, raw_bleu[i] + 1.8), ha='center', fontsize=11, fontweight='bold', color='black')
        axes[0].annotate(f'{clean_bleu[i]:.2f} (+{clean_bleu[i]-raw_bleu[i]:.2f})', (x[i] + width/2, clean_bleu[i] + 1.8), ha='center', fontsize=11, fontweight='bold', color='#006600')
        
    # Panel 2: chrF++ Comparison
    axes[1].bar(x - width/2, raw_chrf, width, label='Raw Unverified Data', color='#e74c3c', edgecolor='black', linewidth=1.2)
    axes[1].bar(x + width/2, clean_chrf, width, label='Filtered Clean Data', color='#2ecc71', edgecolor='black', linewidth=1.2)
    axes[1].set_ylabel('chrF++ Score', fontsize=12, fontweight='bold')
    axes[1].set_title('B. chrF++ Score Enhancement via Verification', fontsize=13, fontweight='bold', pad=12)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(models, fontsize=11.5, fontweight='bold')
    axes[1].legend(fontsize=10.5, loc='upper left', frameon=True, facecolor='white', edgecolor='#cccccc')
    axes[1].set_ylim(0, 98)
    axes[1].grid(axis='y', linestyle='--', alpha=0.5)
    
    for i in range(len(models)):
        axes[1].annotate(f'{raw_chrf[i]:.2f}', (x[i] - width/2, raw_chrf[i] + 1.8), ha='center', fontsize=11, fontweight='bold', color='black')
        axes[1].annotate(f'{clean_chrf[i]:.2f} (+{clean_chrf[i]-raw_chrf[i]:.2f})', (x[i] + width/2, clean_chrf[i] + 1.8), ha='center', fontsize=11, fontweight='bold', color='#006600')

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
