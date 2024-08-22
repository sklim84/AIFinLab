import matplotlib.pyplot as plt
import numpy as np

# Data for plotting
sizes = ['10k', '20k', '30k']
models = ['Identity', 'CopulaGAN', 'CopulaGAN-FNet', 'CTGAN', 'CTGAN-FNet']

data_auc = {
    '10k': [0.953, 0.679, 0.518, 0.733, 0.767],
    '20k': [0.953, 0.564, 0.644, 0.406, 0.802],
    '30k': [0.953, 0.536, 0.729, 0.782, 0.725]
}
data_f1m = {
    '10k': [0.959, 0.666, 0.483, 0.710, 0.765],
    '20k': [0.959, 0.564, 0.638, 0.369, 0.780],
    '30k': [0.959, 0.530, 0.712, 0.780, 0.704]
}
data_f1w = {
    '10k': [0.952, 0.646, 0.402, 0.696, 0.745],
    '20k': [0.952, 0.514, 0.611, 0.281, 0.768],
    '30k': [0.952, 0.472, 0.696, 0.762, 0.677]
}

# Define pastel colors
colors = ['#ff9999', '#66b3ff', '#3366cc', '#ffd700', '#e6ac00']

fig_width, fig_height = (7, 5)
label_font_size = 16
tick_font_size = 16
legend_font_size = 14

# Plotting AUC by Dataset Size
plt.figure(figsize=(fig_width, fig_height))
x = np.arange(len(sizes))
width = 0.15

for i, model in enumerate(models):
    auc_values = [data_auc[size][i] for size in sizes]
    plt.bar(x + i * width, auc_values, width, label=model, color=colors[i % len(colors)])

plt.ylabel('AUROC', fontsize=20)
plt.yticks(size=tick_font_size)
plt.xticks(x + width, sizes, size=tick_font_size)
plt.xlabel('Synthetic Data Size', fontsize=label_font_size)
plt.legend(loc='lower right', fontsize=legend_font_size)
plt.grid(True)
plt.tight_layout()
plt.savefig('./_results/cd_size_fds_auc.pdf', format='pdf', bbox_inches='tight')
plt.show()

# Plotting F1-M by Dataset Size
plt.figure(figsize=(fig_width, fig_height))
for i, model in enumerate(models):
    f1m_values = [data_f1m[size][i] for size in sizes]
    plt.bar(x + i * width, f1m_values, width, label=model, color=colors[i % len(colors)])

plt.ylabel('F1-Macro', fontsize=20)
plt.yticks(size=tick_font_size)
plt.xticks(x + width, sizes, size=tick_font_size)
plt.xlabel('Synthetic Data Size', fontsize=label_font_size)
plt.legend(loc='lower right', fontsize=legend_font_size)
plt.grid(True)
plt.tight_layout()
plt.savefig('./_results/cd_size_fds_f1m.pdf', format='pdf', bbox_inches='tight')
plt.show()

# Plotting F1-W by Dataset Size
plt.figure(figsize=(fig_width, fig_height))
for i, model in enumerate(models):
    f1w_values = [data_f1w[size][i] for size in sizes]
    plt.bar(x + i * width, f1w_values, width, label=model, color=colors[i % len(colors)])

plt.ylabel('F1-Weighted', fontsize=20)
plt.yticks(size=tick_font_size)
plt.xticks(x + width, sizes, size=tick_font_size)
plt.xlabel('Synthetic Data Size', fontsize=label_font_size)
plt.legend(loc='lower right', fontsize=legend_font_size)
plt.grid(True)
plt.tight_layout()
plt.savefig('./_results/cd_size_fds_f1w.pdf', format='pdf', bbox_inches='tight')
plt.show()