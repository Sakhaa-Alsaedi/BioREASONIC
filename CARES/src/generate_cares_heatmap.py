import pandas as pd
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).parent.parent
DATA_FILE = BASE_DIR / 'data' / 'processed' / 'cares_taxonomy_scores.csv'
OUTPUT_FILE = BASE_DIR / 'plots' / 'cares_heatmap.png'
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_csv(DATA_FILE)

strategies = ['zero-shot', 'few-shot', 'cot', 'structured-cot']
categories = ['C_Causal', 'R_Risk', 'M_Mechanism', 'S_Structure']
cat_labels = ['Causal', 'Risk', 'Mechanism', 'Structure']
strategy_labels = ['Zero Shot', 'Few Shot', 'CoT', 'Structured CoT']

# Calculate average for each category and strategy
heatmap_data = []
for strategy in strategies:
    strategy_data = df[df['strategy'] == strategy]
    row = [strategy_data[cat].mean() for cat in categories]
    heatmap_data.append(row)

heatmap_df = pd.DataFrame(heatmap_data, index=strategy_labels, columns=cat_labels)

# Create heatmap
fig, ax = plt.subplots(figsize=(10, 6))
fig.patch.set_facecolor('white')

# Use a light pastel colormap
cmap = sns.light_palette("#F8BBD9", as_cmap=True)

sns.heatmap(heatmap_df, annot=True, fmt='.3f', cmap=cmap,
            linewidths=2, linecolor='white',
            cbar_kws={'label': 'Score'},
            annot_kws={'fontsize': 12, 'fontweight': 'bold'},
            ax=ax)

ax.set_title('CARES Taxonomy: Strategy × Category Heatmap\n(Average Across All Models)',
             fontsize=14, fontweight='bold', pad=15)
ax.set_xlabel('Category', fontsize=12, fontweight='bold')
ax.set_ylabel('Strategy', fontsize=12, fontweight='bold')

plt.tight_layout()
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(OUTPUT_FILE,
            dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

print("Heatmap saved!")
