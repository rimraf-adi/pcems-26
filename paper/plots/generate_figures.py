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
    total_clean = [11145, 11035, 10155]
    
    total_orig = sum(orig_pairs)
    total_synth = sum(synth_verified)
    overall_total = sum(total_clean)
    
    df = pd.DataFrame({
        'Dialect': dialects,
        'Original Clean Pairs': orig_pairs,
        'Clean Synthetic Verified': synth_verified,
        'Total Clean Expanded': total_clean
    })
    
    fig, ax = plt.subplots(figsize=(12, 6.8), dpi=300)
    x = np.arange(len(dialects))
    width = 0.35
    
    # Left bar: Original Clean Pairs
    rects1 = ax.bar(x - width/2, orig_pairs, width, label='Original Clean Pairs', color='#2c3e50', edgecolor='black', linewidth=1.0)
    
    # Right bar: Clean Synthetic Verified
    rects2 = ax.bar(x + width/2, synth_verified, width, label='Clean Synthetic Verified', color='#27ae60', edgecolor='black', linewidth=1.0)
    
    ax.set_ylabel('Parallel Sentence Pairs', fontsize=14.5, fontweight='bold')
    ax.set_title('Parallel Dataset Expansion & Synthetic Yield per Dialect', fontsize=15.5, fontweight='bold', pad=14)
    ax.set_xticks(x)
    ax.set_xticklabels(dialects, fontsize=13, fontweight='bold')
    ax.legend(loc='upper left', frameon=True, fontsize=12)
    
    for i in range(len(dialects)):
        # Annotate Original bar
        ax.text(x[i] - width/2, orig_pairs[i] + 120, f"{orig_pairs[i]:,}", ha='center', va='bottom', fontsize=11, fontweight='bold', color='#2c3e50')
        
        # Annotate Synthetic Verified bar
        ax.text(x[i] + width/2, synth_verified[i] + 120, f"{synth_verified[i]:,}", ha='center', va='bottom', fontsize=11, fontweight='bold', color='#1e8449')
        
        # Overall annotation on top of cluster
        ax.annotate(f"Total Clean: {total_clean[i]:,}", (x[i], max(orig_pairs[i], synth_verified[i]) + 1000), ha='center', fontsize=12, fontweight='bold', color='#1e8449',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='#e8f8f5', edgecolor='#27ae60', linewidth=1.2))
        
    # Overall Dataset Summary Box
    summary_text = (
        f"Overall Dataset Totals\n"
        f"• Total Original Clean: {total_orig:,}\n"
        f"• Total Synthetic Verified: {total_synth:,}\n"
        f"• Overall Clean Dataset: {overall_total:,}"
    )
    ax.text(
        0.98, 0.96, summary_text,
        transform=ax.transAxes,
        fontsize=11.5, fontweight='bold',
        va='top', ha='right', multialignment='left',
        color='#2c3e50',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='#f8f9f9', edgecolor='#2c3e50', linewidth=1.5)
    )
    
    ax.set_ylim(0, 8200)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig2_dataset_composition.png'), bbox_inches='tight')
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig2_dataset_composition.pdf'), bbox_inches='tight')
    plt.close()
    
    # Plotly Interactive
    fig_px = px.bar(df, x='Dialect', y=['Original Clean Pairs', 'Clean Synthetic Verified'],
                    title='Parallel Dataset Expansion per Dialect', barmode='group', text_auto=True)
    fig_px.update_layout(font=dict(size=16))
    fig_px.write_html(os.path.join(INTERACTIVE_DIR, 'fig2_dataset_composition.html'))

# ==============================================================================
# FIGURE 3: Comprehensive RESPIN Benchmark Matrix (Table 3)
# ==============================================================================
def generate_fig3():
    print("Generating Figure 3: RESPIN Test Set Evaluation Matrix (Single vs Multi-Dialect)...")
    
    # Panel A: Single-Dialect Models on Test Set
    single_dialects = ['Southern Konkan\n(559 utts)', 'Northern Konkan\n(540 utts)', 'Varhadi\n(516 utts)']
    ib_single_orig = [57.80, 90.13, 83.59]
    ib_single_exp  = [24.06, 65.41, 67.97]
    mt5_single_orig = [43.86, 79.46, 74.81]
    mt5_single_exp  = [44.36, 79.76, 76.90]
    
    # Panel B: Multi-Dialect Models on Test Set
    multi_subsets = ['Southern Konkan\n(559 utts)', 'Northern Konkan\n(540 utts)', 'Varhadi\n(516 utts)', 'Standard Benchmark\n(555 utts)', 'Overall Benchmark\n(2,170 utts)']
    ib_multi_orig = [26.08, 47.92, 73.04, 96.12, 62.98]
    ib_multi_exp  = [52.15, 54.35, 69.96, 96.80, 76.50]
    mt5_multi_orig = [40.94, 77.50, 73.47, 96.65, 72.18]
    mt5_multi_exp  = [43.88, 78.16, 74.74, 97.39, 73.48]
    
    fig, axes = plt.subplots(2, 1, figsize=(13.0, 9.5), dpi=300)
    width = 0.18
    
    # --- SUBPLOT A: Single-Dialect Models on Test Set ---
    x_a = np.arange(len(single_dialects))
    r_a1 = axes[0].bar(x_a - 1.5*width, ib_single_orig, width, label='IndicBART (Original)', color='#2b5c8f', edgecolor='black', linewidth=1.1)
    r_a2 = axes[0].bar(x_a - 0.5*width, ib_single_exp, width, label='IndicBART (Synthetically Expanded)', color='#41b6c4', edgecolor='black', linewidth=1.1)
    r_a3 = axes[0].bar(x_a + 0.5*width, mt5_single_orig, width, label='mT5-Small (Original)', color='#8c2d19', edgecolor='black', linewidth=1.1)
    r_a4 = axes[0].bar(x_a + 1.5*width, mt5_single_exp, width, label='mT5-Small (Synthetically Expanded)', color='#fe9929', edgecolor='black', linewidth=1.1)
    
    axes[0].set_ylabel('Test Set BLEU Score', fontsize=13, fontweight='bold')
    axes[0].set_title('A. Single-Dialect Models Evaluated on Held-Out RESPIN Test Benchmark', fontsize=14.5, fontweight='bold', pad=12)
    axes[0].set_xticks(x_a)
    axes[0].set_xticklabels(single_dialects, fontsize=12, fontweight='bold')
    axes[0].set_ylim(0, 120)
    axes[0].legend(loc='upper left', frameon=True, facecolor='white', edgecolor='#cccccc', fontsize=10.5, ncol=2)
    axes[0].grid(axis='y', linestyle='--', alpha=0.5)
    
    for rects in [r_a1, r_a2, r_a3, r_a4]:
        for rect in rects:
            h = rect.get_height()
            axes[0].annotate(f'{h:.1f}', (rect.get_x() + rect.get_width()/2, h + 1.8), ha='center', va='bottom', fontsize=10.5, fontweight='bold')
            
    # --- SUBPLOT B: Multi-Dialect Models on Test Set ---
    x_b = np.arange(len(multi_subsets))
    r_b1 = axes[1].bar(x_b - 1.5*width, ib_multi_orig, width, label='IndicBART (Original)', color='#2b5c8f', edgecolor='black', linewidth=1.1)
    r_b2 = axes[1].bar(x_b - 0.5*width, ib_multi_exp, width, label='IndicBART (Synthetically Expanded)', color='#41b6c4', edgecolor='black', linewidth=1.1)
    r_b3 = axes[1].bar(x_b + 0.5*width, mt5_multi_orig, width, label='mT5-Small (Original)', color='#8c2d19', edgecolor='black', linewidth=1.1)
    r_b4 = axes[1].bar(x_b + 1.5*width, mt5_multi_exp, width, label='mT5-Small (Synthetically Expanded)', color='#fe9929', edgecolor='black', linewidth=1.1)
    
    axes[1].set_ylabel('Test Set BLEU Score', fontsize=13, fontweight='bold')
    axes[1].set_title('B. Multi-Dialect Models Evaluated on Held-Out RESPIN Test Benchmark', fontsize=14.5, fontweight='bold', pad=12)
    axes[1].set_xticks(x_b)
    axes[1].set_xticklabels(multi_subsets, fontsize=11.5, fontweight='bold')
    axes[1].set_ylim(0, 125)
    axes[1].legend(loc='upper left', frameon=True, facecolor='white', edgecolor='#cccccc', fontsize=10.5, ncol=2)
    axes[1].grid(axis='y', linestyle='--', alpha=0.5)
    
    for rects in [r_b1, r_b2, r_b3, r_b4]:
        for rect in rects:
            h = rect.get_height()
            axes[1].annotate(f'{h:.1f}', (rect.get_x() + rect.get_width()/2, h + 1.8), ha='center', va='bottom', fontsize=10, fontweight='bold')
            
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig3_respin_comprehensive.png'), bbox_inches='tight')
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig3_respin_comprehensive.pdf'), bbox_inches='tight')
    plt.close()
    
    # Interactive Plotly Chart
    df_single = pd.DataFrame({
        'Dialect': single_dialects * 4,
        'Model & Setup': ['IndicBART (Orig)']*3 + ['IndicBART (Exp)']*3 + ['mT5-Small (Orig)']*3 + ['mT5-Small (Exp)']*3,
        'BLEU': ib_single_orig + ib_single_exp + mt5_single_orig + mt5_single_exp,
        'Type': ['Single-Dialect'] * 12
    })
    df_multi = pd.DataFrame({
        'Dialect': multi_subsets * 4,
        'Model & Setup': ['IndicBART (Orig)']*5 + ['IndicBART (Exp)']*5 + ['mT5-Small (Orig)']*5 + ['mT5-Small (Exp)']*5,
        'BLEU': ib_multi_orig + ib_multi_exp + mt5_multi_orig + mt5_multi_exp,
        'Type': ['Multi-Dialect'] * 20
    })
    df_comb = pd.concat([df_single, df_multi])
    fig_px = px.bar(df_comb, x='Dialect', y='BLEU', color='Model & Setup', facet_col='Type', barmode='group',
                    title='RESPIN Test Benchmark BLEU (Single vs Multi-Dialect Models)', text_auto='.1f')
    fig_px.update_layout(font=dict(size=14))
    fig_px.write_html(os.path.join(INTERACTIVE_DIR, 'fig3_respin_comprehensive.html'))

# ==============================================================================
# FIGURE 4: Single-Dialect 5-Fold Cross-Validation (Table 4 & Table 6)
# ==============================================================================
def generate_fig4():
    print("Generating Figure 4: Single-Dialect Cross-Validation Vertical Grouped Bar Chart...")
    dialects = ['Southern Konkan', 'Northern Konkan', 'Varhadi']
    x = np.arange(len(dialects))
    width = 0.28
    
    ib_orig = [48.54, 60.58, 76.63]
    ib_exp  = [47.25, 40.51, 73.62]
    
    mt5_orig = [46.31, 60.31, 81.00]
    mt5_exp  = [65.10, 62.07, 78.89]
    
    fig, axes = plt.subplots(2, 1, figsize=(11.0, 7.8), dpi=300)
    
    # Subplot A: IndicBART (244M)
    rects1 = axes[0].bar(x - width/2, ib_orig, width, label='Original Data (Table 4)', color='#2b5c8f', edgecolor='black', linewidth=1.1)
    rects2 = axes[0].bar(x + width/2, ib_exp, width, label='Synthetically Expanded Data (Table 6)', color='#41b6c4', edgecolor='black', linewidth=1.1)
    
    axes[0].set_ylabel('5-Fold CV BLEU Score', fontsize=12.5, fontweight='bold')
    axes[0].set_title('A. IndicBART (244M) Single-Dialect CV BLEU', fontsize=13.5, fontweight='bold', pad=12)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(dialects, fontsize=12, fontweight='bold')
    axes[0].set_ylim(0, 98)
    axes[0].legend(loc='upper right', frameon=True, facecolor='white', edgecolor='#cccccc', fontsize=11)
    axes[0].grid(axis='y', linestyle='--', alpha=0.5)
    
    for rect in rects1:
        h = rect.get_height()
        axes[0].annotate(f'{h:.1f}', (rect.get_x() + rect.get_width()/2, h + 1.8), ha='center', va='bottom', fontsize=11, fontweight='bold')
    for rect in rects2:
        h = rect.get_height()
        axes[0].annotate(f'{h:.1f}', (rect.get_x() + rect.get_width()/2, h + 1.8), ha='center', va='bottom', fontsize=11, fontweight='bold')
        
    # Subplot B: mT5-Small (300M)
    rects3 = axes[1].bar(x - width/2, mt5_orig, width, label='Original Data (Table 4)', color='#8c2d19', edgecolor='black', linewidth=1.1)
    rects4 = axes[1].bar(x + width/2, mt5_exp, width, label='Synthetically Expanded Data (Table 6)', color='#fe9929', edgecolor='black', linewidth=1.1)
    
    axes[1].set_ylabel('5-Fold CV BLEU Score', fontsize=12.5, fontweight='bold')
    axes[1].set_title('B. mT5-Small (300M) Single-Dialect CV BLEU', fontsize=13.5, fontweight='bold', pad=12)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(dialects, fontsize=12, fontweight='bold')
    axes[1].set_ylim(0, 98)
    axes[1].legend(loc='upper left', frameon=True, facecolor='white', edgecolor='#cccccc', fontsize=11)
    axes[1].grid(axis='y', linestyle='--', alpha=0.5)
    
    for rect in rects3:
        h = rect.get_height()
        axes[1].annotate(f'{h:.1f}', (rect.get_x() + rect.get_width()/2, h + 1.8), ha='center', va='bottom', fontsize=11, fontweight='bold')
    for rect in rects4:
        h = rect.get_height()
        axes[1].annotate(f'{h:.1f}', (rect.get_x() + rect.get_width()/2, h + 1.8), ha='center', va='bottom', fontsize=11, fontweight='bold')
        
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
# FIGURE 5: Multi-Dialect 5-Fold CV Performance Matrix & Net Impact (Table 5 & 7 & 8)
# ==============================================================================
def generate_fig5():
    print("Generating Figure 5: Multi-Dialect Cross-Validation Vertical Grouped Bar & Net Impact Chart...")
    subsets = ['Southern Konkan', 'Northern Konkan', 'Varhadi', 'Overall Combined']
    x = np.arange(len(subsets))
    width = 0.18
    
    ib_orig = [26.21, 47.59, 72.75, 48.17]
    mt5_orig = [48.28, 62.35, 79.57, 63.29]
    ib_exp  = [52.06, 49.08, 70.27, 57.12]
    mt5_exp  = [66.62, 62.72, 78.73, 69.51]
    
    ib_delta  = [25.85, 1.49, -2.48, 8.95]
    mt5_delta = [18.34, 0.37, -0.84, 6.22]
    
    fig, axes = plt.subplots(2, 1, figsize=(12.0, 8.5), dpi=300)
    
    # Subplot A: Absolute BLEU Scores across Multi-Dialect Models
    rects1 = axes[0].bar(x - 1.5*width, ib_orig, width, label='IndicBART (Original)', color='#2b5c8f', edgecolor='black', linewidth=1.1)
    rects2 = axes[0].bar(x - 0.5*width, mt5_orig, width, label='mT5-Small (Original)', color='#8c2d19', edgecolor='black', linewidth=1.1)
    rects3 = axes[0].bar(x + 0.5*width, ib_exp, width, label='IndicBART (Expanded)', color='#41b6c4', edgecolor='black', linewidth=1.1)
    rects4 = axes[0].bar(x + 1.5*width, mt5_exp, width, label='mT5-Small (Expanded)', color='#fe9929', edgecolor='black', linewidth=1.1)
    
    axes[0].set_ylabel('5-Fold CV BLEU Score', fontsize=12.5, fontweight='bold')
    axes[0].set_title('A. Multi-Dialect Performance Matrix (5-Fold CV BLEU)', fontsize=13.5, fontweight='bold', pad=12)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(subsets, fontsize=12, fontweight='bold')
    axes[0].set_ylim(0, 98)
    axes[0].legend(loc='upper left', frameon=True, facecolor='white', edgecolor='#cccccc', fontsize=10.5, ncol=2)
    axes[0].grid(axis='y', linestyle='--', alpha=0.5)
    
    for rects in [rects1, rects2, rects3, rects4]:
        for rect in rects:
            h = rect.get_height()
            axes[0].annotate(f'{h:.1f}', (rect.get_x() + rect.get_width()/2, h + 1.8), ha='center', va='bottom', fontsize=10.5, fontweight='bold')
            
    # Subplot B: Net Synthetic Expansion Gain (Δ BLEU)
    w_b = 0.28
    rects_b1 = axes[1].bar(x - w_b/2, ib_delta, w_b, label='IndicBART Δ BLEU', color='#2ca02c', edgecolor='black', linewidth=1.1)
    rects_b2 = axes[1].bar(x + w_b/2, mt5_delta, w_b, label='mT5-Small Δ BLEU', color='#1f77b4', edgecolor='black', linewidth=1.1)
    
    axes[1].axhline(0, color='black', linewidth=1.2, linestyle='-')
    axes[1].set_ylabel('Net BLEU Gain (Δ BLEU)', fontsize=12.5, fontweight='bold')
    axes[1].set_title('B. Synthetic Expansion Net Impact Across Dialects & Overall (Table 8)', fontsize=13.5, fontweight='bold', pad=12)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(subsets, fontsize=12, fontweight='bold')
    axes[1].set_ylim(-6, 32)
    axes[1].legend(loc='upper right', frameon=True, facecolor='white', edgecolor='#cccccc', fontsize=11)
    axes[1].grid(axis='y', linestyle='--', alpha=0.5)
    
    for rect in rects_b1:
        h = rect.get_height()
        va = 'bottom' if h >= 0 else 'top'
        offset = 1.0 if h >= 0 else -1.8
        axes[1].annotate(f'{h:+.2f}', (rect.get_x() + rect.get_width()/2, h + offset), ha='center', va=va, fontsize=11, fontweight='bold')
        
    for rect in rects_b2:
        h = rect.get_height()
        va = 'bottom' if h >= 0 else 'top'
        offset = 1.0 if h >= 0 else -1.8
        axes[1].annotate(f'{h:+.2f}', (rect.get_x() + rect.get_width()/2, h + offset), ha='center', va=va, fontsize=11, fontweight='bold')
        
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig5_multidialect_heatmap.png'), bbox_inches='tight')
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig5_multidialect_heatmap.pdf'), bbox_inches='tight')
    plt.close()

# ==============================================================================
# FIGURE 6: Synthetic Data Augmentation Impact (Table 8)
# ==============================================================================
def generate_fig6():
    print("Generating Figure 6: Synthetic Augmentation Impact Diverging Bar (Table 8)...")
    metrics = [
        'IndicBART - Southern Konkan', 'IndicBART - Northern Konkan', 'IndicBART - Varhadi', 'IndicBART - Overall Combined',
        'mT5-Small - Southern Konkan', 'mT5-Small - Northern Konkan', 'mT5-Small - Varhadi', 'mT5-Small - Overall Combined'
    ]
    deltas = [25.85, 1.49, -2.48, 8.95, 18.34, 0.37, -0.84, 6.22]
    colors = ['#2ca02c' if d > 0 else '#d62728' for d in deltas]
    
    fig, ax = plt.subplots(figsize=(13, 6.2), dpi=300)
    y_pos = np.arange(len(metrics))
    
    bars = ax.barh(y_pos, deltas, color=colors, edgecolor='black', linewidth=1.1, height=0.65)
    ax.axvline(0, color='black', linewidth=1.2, linestyle='-')
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(metrics, fontsize=12, fontweight='bold')
    ax.invert_yaxis()
    ax.set_xlabel('Net BLEU Change (Δ BLEU)', fontsize=13.5, fontweight='bold')
    ax.set_title('Impact of Synthetic Data Augmentation Across Models and Dialects (Table 8)', fontsize=14.5, fontweight='bold', pad=15)
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
    axes[0].bar(x - width/2, raw_bleu, width, label='Raw Unverified Data', color='#e74c3c', edgecolor='black', linewidth=1.2, zorder=3)
    axes[0].bar(x + width/2, clean_bleu, width, label='Filtered Clean Data', color='#2ecc71', edgecolor='black', linewidth=1.2, zorder=3)
    axes[0].set_ylabel('BLEU Score', fontsize=12.5, fontweight='bold')
    axes[0].set_title('A. BLEU Score Recovery via Verification', fontsize=13.5, fontweight='bold', pad=12)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(models, fontsize=12, fontweight='bold')
    axes[0].legend(fontsize=11, loc='upper left', frameon=True, facecolor='white', edgecolor='#cccccc')
    axes[0].set_ylim(0, 85)
    axes[0].grid(axis='y', linestyle='--', alpha=0.4, zorder=0)
    
    for i in range(len(models)):
        axes[0].annotate(f'{raw_bleu[i]:.2f}', (x[i] - width/2, raw_bleu[i] + 2.0), ha='center', fontsize=11.5, fontweight='bold', color='black',
                         bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.9), zorder=4)
        axes[0].annotate(f'{clean_bleu[i]:.2f}', (x[i] + width/2, clean_bleu[i] + 2.0), ha='center', fontsize=11.5, fontweight='bold', color='#006600',
                         bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.9), zorder=4)
        
    # Panel 2: chrF++ Comparison
    axes[1].bar(x - width/2, raw_chrf, width, label='Raw Unverified Data', color='#e74c3c', edgecolor='black', linewidth=1.2, zorder=3)
    axes[1].bar(x + width/2, clean_chrf, width, label='Filtered Clean Data', color='#2ecc71', edgecolor='black', linewidth=1.2, zorder=3)
    axes[1].set_ylabel('chrF++ Score', fontsize=12.5, fontweight='bold')
    axes[1].set_title('B. chrF++ Score Enhancement via Verification', fontsize=13.5, fontweight='bold', pad=12)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(models, fontsize=12, fontweight='bold')
    axes[1].legend(fontsize=11, loc='upper left', frameon=True, facecolor='white', edgecolor='#cccccc')
    axes[1].set_ylim(0, 98)
    axes[1].grid(axis='y', linestyle='--', alpha=0.4, zorder=0)
    
    for i in range(len(models)):
        axes[1].annotate(f'{raw_chrf[i]:.2f}', (x[i] - width/2, raw_chrf[i] + 2.0), ha='center', fontsize=11.5, fontweight='bold', color='black',
                         bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.9), zorder=4)
        axes[1].annotate(f'{clean_chrf[i]:.2f}', (x[i] + width/2, clean_chrf[i] + 2.0), ha='center', fontsize=11.5, fontweight='bold', color='#006600',
                         bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.9), zorder=4)

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
