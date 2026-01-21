import pandas as pd
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).parent.parent
DATA_FILE = BASE_DIR / 'data' / 'processed' / 'cares_taxonomy_scores.csv'
OUTPUT_DIR = BASE_DIR / 'plots'
import numpy as np
import matplotlib.pyplot as plt
from math import pi

# Load data
df = pd.read_csv(DATA_FILE)

# Focus on structured-cot (best strategy) - show individual models
strategy = 'structured-cot'
sdf = df[df['strategy'] == strategy].sort_values('Avg_SCRM', ascending=False)

categories = ['C_Causal', 'R_Risk', 'M_Mechanism', 'S_Structure']
cat_labels = ['Causal', 'Risk', 'Mechanism', 'Structure']

# Pastel colors for each model
model_colors = {
    'GPT-4o-Mini': ('#E8F5E9', '#66BB6A'),
    'DeepSeek-V3': ('#E3F2FD', '#42A5F5'),
    'Llama-3.1-8B': ('#FFF3E0', '#FFA726'),
    'GPT-4.1': ('#FCE4EC', '#EC407A'),
    'GPT-4.1-Mini': ('#F3E5F5', '#AB47BC'),
    'Claude-3-Haiku': ('#E0F7FA', '#26C6DA'),
    'Qwen-2.5-7B': ('#FFF8E1', '#FFCA28'),
    'GPT-4o': ('#FFEBEE', '#EF5350'),
}

# Create radar plot for individual models
fig, ax = plt.subplots(figsize=(12, 10), subplot_kw=dict(polar=True))
fig.patch.set_facecolor('white')

N = 4
angles = [n / float(N) * 2 * pi for n in range(N)]
angles += angles[:1]

for _, row in sdf.iterrows():
    model = row['model']
    values = [row[cat] for cat in categories]
    values += values[:1]

    fill_color, edge_color = model_colors.get(model, ('#E0E0E0', '#757575'))

    ax.plot(angles, values, 'o-', linewidth=2, markersize=6,
            label=f"{model} ({row['Avg_SCRM']:.3f})",
            color=edge_color)
    ax.fill(angles, values, alpha=0.15, color=fill_color)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(cat_labels, fontsize=14, fontweight='bold')
ax.set_ylim(0, 0.85)
ax.set_yticks([0.2, 0.4, 0.6, 0.8])
ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8'], fontsize=10, color='gray')
ax.set_title('CARES Taxonomy: Individual Model Performance\n(Structured-CoT Strategy)',
             fontsize=15, fontweight='bold', pad=25)
ax.grid(True, alpha=0.3)

ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.08), ncol=4, fontsize=10,
          frameon=True, fancybox=True, edgecolor='lightgray', title='Model (Avg Score)')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'cares_models_radar.png',
            dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print("Individual models radar saved!")

# Create grouped bar showing variance across models for each category
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.patch.set_facecolor('white')

strategies = ['zero-shot', 'few-shot', 'cot', 'structured-cot']
strategy_labels = ['Zero Shot', 'Few Shot', 'CoT', 'Structured CoT']
colors = ['#FFFDE7', '#E8F5E9', '#E3F2FD', '#FCE4EC']
edge_cols = ['#FFD54F', '#81C784', '#64B5F6', '#F48FB1']

for idx, (strat, strat_label, color, edge) in enumerate(zip(strategies, strategy_labels, colors, edge_cols)):
    ax = axes[idx // 2, idx % 2]
    sdf = df[df['strategy'] == strat].sort_values('Avg_SCRM', ascending=True)

    models = sdf['model'].tolist()
    scores = sdf['Avg_SCRM'].tolist()

    bars = ax.barh(range(len(models)), scores, color=color, edgecolor=edge, linewidth=2)

    for bar, score in zip(bars, scores):
        ax.text(score + 0.01, bar.get_y() + bar.get_height()/2, f'{score:.3f}',
                va='center', fontsize=10, fontweight='bold')

    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models, fontsize=10)
    ax.set_xlim(0, 0.85)
    ax.set_xlabel('Average CARES Score', fontsize=11)
    ax.set_title(strat_label, fontsize=13, fontweight='bold', color=edge)
    ax.grid(axis='x', alpha=0.3)
    ax.set_axisbelow(True)

plt.suptitle('CARES Taxonomy: Model Variance by Strategy', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'cares_models_variance.png',
            dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print("Model variance plot saved!")
