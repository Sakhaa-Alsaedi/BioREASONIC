import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import pi
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).parent.parent
DATA_FILE = BASE_DIR / 'data' / 'processed' / 'cares_taxonomy_scores.csv'
OUTPUT_FILE = BASE_DIR / 'plots' / 'cares_taxonomy_radar_pastel.png'

# Load data
df = pd.read_csv(DATA_FILE)

# CARES taxonomy categories (actual names)
categories = ['C (Causal)', 'R (Risk)', 'E (Evidence)', 'S (Structure)']
# Map to CSV columns
col_map = {'C (Causal)': 'C_Causal', 'R (Risk)': 'R_Risk', 'E (Evidence)': 'M_Mechanism', 'S (Structure)': 'S_Structure'}

# Find best model (highest average)
best_idx = df['Avg_SCRM'].idxmax()
best_model = df.loc[best_idx, 'model']
best_strategy = df.loc[best_idx, 'strategy']
best_row = df.loc[best_idx]

# Get best technique (structured-cot) averages across all models
best_technique = 'structured-cot'
technique_df = df[df['strategy'] == best_technique]

# Also get comparison strategies
strategies_to_plot = ['zero-shot', 'few-shot', 'cot', 'structured-cot']

# Very light pastel colors
pastel_colors = {
    'zero-shot': '#FFFDE7',      # Very light yellow
    'few-shot': '#E8F5E9',       # Very light green
    'cot': '#E3F2FD',            # Very light blue
    'structured-cot': '#FCE4EC', # Very light pink
}

edge_colors = {
    'zero-shot': '#FFD54F',      # Soft yellow
    'few-shot': '#81C784',       # Soft green
    'cot': '#64B5F6',            # Soft blue
    'structured-cot': '#F48FB1', # Soft pink
}

# Create figure with single plot
fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(polar=True))
fig.patch.set_facecolor('white')

# Number of categories
N = 4
angles = [n / float(N) * 2 * pi for n in range(N)]
angles += angles[:1]  # Complete the loop

# Category labels for radar
cat_labels = ['Causal', 'Risk', 'Mechanism', 'Structure']

# ============ Strategy Comparison (Average across models) ============
for strategy in strategies_to_plot:
    strategy_data = df[df['strategy'] == strategy]
    if len(strategy_data) > 0:
        # Average across all models for this strategy
        avg_values = [
            strategy_data['C_Causal'].mean(),
            strategy_data['R_Risk'].mean(),
            strategy_data['M_Mechanism'].mean(),
            strategy_data['S_Structure'].mean()
        ]
        avg_values += avg_values[:1]

        ax.plot(angles, avg_values, 'o-', linewidth=2.5, label=strategy.replace('-', ' ').title(),
                color=edge_colors[strategy], markersize=7)
        ax.fill(angles, avg_values, alpha=0.3, color=pastel_colors[strategy])

ax.set_xticks(angles[:-1])
ax.set_xticklabels(cat_labels, fontsize=14, fontweight='bold')
ax.set_ylim(0, 0.8)
ax.set_yticks([0.2, 0.4, 0.6])
ax.set_yticklabels(['0.2', '0.4', '0.6'], fontsize=10, color='gray')
ax.set_title('CARES Taxonomy: Strategy Comparison\n(Average Across All Models)',
              fontsize=15, fontweight='bold', pad=25, color='#2C3E50')
ax.grid(True, alpha=0.3)

# Legend at the bottom
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.08), fontsize=12,
          frameon=True, fancybox=True, shadow=False, framealpha=0.9,
          edgecolor='lightgray', ncol=4)

plt.tight_layout()
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(OUTPUT_FILE, dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

print("Radar plot saved!")
print(f"\nBest Model: {best_model} with {best_strategy}")
print(f"Scores: C={best_row['C_Causal']:.2f}, R={best_row['R_Risk']:.2f}, M={best_row['M_Mechanism']:.2f}, S={best_row['S_Structure']:.2f}")
print(f"Average: {best_row['Avg_SCRM']:.3f}")
