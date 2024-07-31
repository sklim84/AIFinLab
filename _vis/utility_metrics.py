import pandas as pd

# Extract data from the provided table image
data = {
    'Model': ['Copula', 'CTGAN', 'CTGAN w/ FNetSyn', 'Copula w/ FNetSyn',
              'Copula', 'CTGAN', 'CTGAN w/ FNetSyn', 'Copula w/ FNetSyn',
              'Copula', 'CTGAN', 'CTGAN w/ FNetSyn', 'Copula w/ FNetSyn'],
    'Dataset Size': ['10k', '10k', '10k', '10k', '20k', '20k', '20k', '20k', '30k', '30k', '30k', '30k'],
    'Pair-wise Correlation': [0.968, 0.975, 0.967, 0.969, 0.969, 0.977, 0.968, 0.971, 0.968, 0.976, 0.965, 0.965],
    'Statistics': [0.882, 0.842, 0.807, 0.861, 0.886, 0.847, 0.854, 0.807, 0.882, 0.846, 0.849, 0.797],
    'PMSE-based Score': [0.547, 0.574, 0.457, 0.505, 0.542, 0.575, 0.554, 0.567, 0.542, 0.578, 0.569, 0.578]
}

df = pd.DataFrame(data)
print(df)

import matplotlib.pyplot as plt

# Data for plotting
sizes = ['10k', '20k', '30k']
models = ['Copula', 'CTGAN', 'CTGAN w/ FNetSyn', 'Copula w/ FNetSyn']

# Extracting data
pairwise_corr = df.pivot(index='Dataset Size', columns='Model', values='Pair-wise Correlation')

# Plotting Pair-wise Correlation
plt.figure(figsize=(6, 4))
for model in models:
    plt.plot(sizes, pairwise_corr[model], marker='o', label=model)
# plt.title('Pair-wise Correlation by Dataset Size')
plt.xlabel('Synthetic Data Size')
plt.ylabel('Pair-wise Correlation')
plt.ylim(0.96, 0.98)  # Setting y-axis range for AUC
plt.legend(loc='upper right')
plt.grid(True)
plt.tight_layout()
plt.savefig('./_results/size_pair.pdf', format='pdf', bbox_inches='tight')
plt.show()

# Plotting Statistics
statistics = df.pivot(index='Dataset Size', columns='Model', values='Statistics')
plt.figure(figsize=(6, 4))
for model in models:
    plt.plot(sizes, statistics[model], marker='o', label=model)
# plt.title('Statistics by Dataset Size')
plt.xlabel('Synthetic Data Size')
plt.ylabel('Statistics')
plt.ylim(0.75, 0.9)  # Setting y-axis range for F1-Macro
plt.legend(loc='upper right')
plt.grid(True)
plt.tight_layout()
plt.savefig('./_results/size_stat.pdf', format='pdf', bbox_inches='tight')
plt.show()


# Plotting PMSE-based Score
pmse_score = df.pivot(index='Dataset Size', columns='Model', values='PMSE-based Score')
plt.figure(figsize=(6, 4))
for model in models:
    plt.plot(sizes, pmse_score[model], marker='o', label=model)
# plt.title('PMSE-based Score by Dataset Size')
plt.xlabel('Synthetic Data Size')
plt.ylabel('pMSE Score')
plt.ylim(0.4, 0.6)  # Setting y-axis range for F1-Weighted
plt.legend(loc='upper right')
plt.grid(True)
plt.tight_layout()
plt.savefig('./_results/size_pmse.pdf', format='pdf', bbox_inches='tight')
plt.show()

