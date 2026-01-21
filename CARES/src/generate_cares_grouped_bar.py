import pandas as pd
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).parent.parent
DATA_FILE = BASE_DIR / 'data' / 'processed' / 'cares_taxonomy_scores.csv'
OUTPUT_FILE = BASE_DIR / 'plots' / 'cares_grouped_bar.png'
import numpy as np
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv(DATA_FILE)

strategies = ['zero-shot', 'few-shot', 'cot', 'structured-cot']
categories = ['C_Causal', 'R_Risk', 'M_Mechanism', 'S_Structure']
cat_labels = ['Causal', 'Risk', 'Mechanism', 'Structure']

# Very light pastel colors
pastel_colors = {
    'zero-shot': '#FFFDE7',
    'few-shot': '#E8F5E9',
    'cot': '#E3F2FD',
    'structured-cot': '#FCE4EC',
}

edge_colors = {
    'zero-shot': '#FFD54F',
    'few-shot': '#81C784',
    'cot': '#64B5F6',
    'structured-cot': '#F48FB1',
}

# Calculate average for each category and strategy
data = {}
for strategy in strategies:
    strategy_data = df[df['strategy'] == strategy]
    data[strategy] = [strategy_data[cat].mean() for cat in categories]

# Create grouped bar chart
fig, ax = plt.subplots(figsize=(12, 6))
fig.patch.set_facecolor('white')

x = np.arange(len(categories))
width = 0.2
multiplier = 0

for strategy in strategies:
    offset = width * multiplier
    bars = ax.bar(x + offset, data[strategy], width,
                  label=strategy.replace('-', ' ').title(),
                  color=pastel_colors[strategy],
                  edgecolor=edge_colors[strategy],
                  linewidth=1.5)
    multiplier += 1

ax.set_xticks(x + width * 1.5)
ax.set_xticklabels(cat_labels, fontsize=12, fontweight='bold')
ax.set_ylabel('Score', fontsize=12, fontweight='bold')
ax.set_title('CARES Taxonomy: Score by Category and Strategy\n(Average Across All Models)',
             fontsize=14, fontweight='bold', pad=15)
ax.set_ylim(0, 0.8)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=4, fontsize=11,
          frameon=True, fancybox=True, edgecolor='lightgray')
ax.grid(axis='y', alpha=0.3)
ax.set_axisbelow(True)

plt.tight_layout()
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(OUTPUT_FILE,
            dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

print("Grouped bar chart saved!")
