import matplotlib.pyplot as plt
import numpy as np

# Data for plotting
sizes = ['10k', '20k', '30k']
models = ['Identity', 'CopulaGAN', 'CopulaGAN-FNet', 'CTGAN', 'CTGAN-FNet']

# Extracting data for AUC, F1-M, and F1-W
data_auc = {
    '10k': [0.703, 0.593, 0.646, 0.547, 0.574],
    '20k': [0.703, 0.615, 0.576, 0.573, 0.583],
    '30k': [0.703, 0.609, 0.637, 0.589, 0.599]
}
data_f1m = {
    '10k': [0.745, 0.505, 0.622, 0.549, 0.595],
    '20k': [0.745, 0.511, 0.590, 0.578, 0.591],
    '30k': [0.745, 0.515, 0.637, 0.575, 0.589]
}
data_f1w = {
    '10k': [0.556, 0.261, 0.376, 0.239, 0.304],
    '20k': [0.556, 0.276, 0.300, 0.279, 0.307],
    '30k': [0.556, 0.275, 0.390, 0.298, 0.316]
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
plt.savefig('./_results/hf_size_fds_auc.pdf', format='pdf', bbox_inches='tight')
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
plt.savefig('./_results/hf_size_fds_f1m.pdf', format='pdf', bbox_inches='tight')
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
plt.savefig('./_results/hf_size_fds_f1w.pdf', format='pdf', bbox_inches='tight')
plt.show()