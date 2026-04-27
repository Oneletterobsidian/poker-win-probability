"""
plot_results.py
Generates comparison charts for the four models.
Output: r2_comparison.png, mae_comparison.png, inference_comparison.png
"""

import matplotlib.pyplot as plt
import numpy as np

# ── Data ──────────────────────────────────────────────────────────────────────
models = ['Linear\nRegression', 'Random\nForest', 'Gradient\nBoosting', 'MLP']
mse    = [0.02131, 0.01362, 0.01443, 0.01323]
mae    = [0.10243, 0.06591, 0.07285, 0.06853]
r2     = [0.5757,  0.7288,  0.7127,  0.7367]
speed  = [0.0001,  0.0039,  0.0032,  0.0010]

colors = ['#4C72B0', '#55A868', '#C44E52', '#8172B2']

def save_bar(values, ylabel, title, filename, highlight_max=True):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(models, values, color=colors, width=0.5, edgecolor='white')

    # Highlight best bar
    if highlight_max:
        best = values.index(max(values))
    else:
        best = values.index(min(values))
    bars[best].set_edgecolor('black')
    bars[best].set_linewidth(2)

    # Value labels on bars
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + max(values)*0.01,
                f'{val:.4f}', ha='center', va='bottom', fontsize=10)

    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_ylim(0, max(values) * 1.18)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"Saved: {filename}")

# ── Figure 1: R² ──────────────────────────────────────────────────────────────
save_bar(r2, 'R²', 'Figure 1: R² Comparison Across Models', 'r2_comparison.png', highlight_max=True)

# ── Figure 2: MAE ─────────────────────────────────────────────────────────────
save_bar(mae, 'MAE', 'Figure 2: MAE Comparison Across Models', 'mae_comparison.png', highlight_max=False)

# ── Figure 3: Inference Time ──────────────────────────────────────────────────
save_bar(speed, 'Inference Time (ms/sample)', 'Figure 3: Inference Time Comparison Across Models', 'inference_comparison.png', highlight_max=False)

# ── Figure 4: R² vs Dataset Size ─────────────────────────────────────────────
models_short = ['Linear Regression', 'Random Forest', 'Gradient Boosting', 'MLP']
r2_500   = [0.5220, 0.7162, 0.7099, -0.2565]
r2_50k   = [0.5757, 0.7288, 0.7127,  0.7367]

x = np.arange(len(models_short))
width = 0.35

fig, ax = plt.subplots(figsize=(8, 5))
bars1 = ax.bar(x - width/2, r2_500, width, label='500 samples',   color='#4C72B0', alpha=0.8)
bars2 = ax.bar(x + width/2, r2_50k, width, label='50,000 samples', color='#55A868', alpha=0.8)

ax.set_ylabel('R²', fontsize=12)
ax.set_title('Figure 4: R² vs Dataset Size', fontsize=13, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(models_short, fontsize=10)
ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
ax.legend(fontsize=10)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig('r2_vs_dataset_size.png', dpi=150)
plt.close()
print("Saved: r2_vs_dataset_size.png")

print("\nAll figures saved!")