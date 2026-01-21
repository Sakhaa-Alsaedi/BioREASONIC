import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).parent.parent
DATA_FILE = BASE_DIR / 'data' / 'processed' / 'cares_taxonomy_scores.csv'
OUTPUT_FILE = BASE_DIR / 'plots' / 'cares_bar_chart.png'

# Load data
df = pd.read_csv(DATA_FILE)

strategies = ['zero-shot', 'few-shot', 'cot', 'structured-cot']

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

# Calculate average across all models for each strategy
avg_scores = []
for strategy in strategies:
    strategy_data = df[df['strategy'] == strategy]
    avg_scores.append(strategy_data['Avg_SCRM'].mean())

# Create bar chart
fig, ax = plt.subplots(figsize=(10, 6))
fig.patch.set_facecolor('white')

bars = ax.bar(range(len(strategies)), avg_scores,
              color=[pastel_colors[s] for s in strategies],
              edgecolor=[edge_colors[s] for s in strategies],
              linewidth=2)

# Add value labels on bars
for bar, score in zip(bars, avg_scores):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f'{score:.3f}', ha='center', va='bottom', fontsize=12, fontweight='bold')

ax.set_xticks(range(len(strategies)))
ax.set_xticklabels([s.replace('-', ' ').title() for s in strategies], fontsize=12, fontweight='bold')
ax.set_ylabel('Average CARES Score', fontsize=12, fontweight='bold')
ax.set_title('CARES Taxonomy: Strategy Comparison\n(Average Across All Models)',
             fontsize=14, fontweight='bold', pad=15)
ax.set_ylim(0, 0.8)
ax.grid(axis='y', alpha=0.3)
ax.set_axisbelow(True)

plt.tight_layout()
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(OUTPUT_FILE, dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

print("Bar chart saved!")
