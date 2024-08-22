import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from scipy.stats import gaussian_kde
import pandas as pd
import torch
import time
import seaborn as sns

target_data='hf'
target_model='ctgan'
n_gen_samples = 10000
file_names = [f'{target_data}_training.csv', f'{target_data}_{target_model}_base_{n_gen_samples}_3.csv',
              f'{target_data}_{target_model}_syn_{n_gen_samples}_3.csv']
labels = ['Identity', target_model.upper(), f'{target_model.upper()}-FNet']


n_samples = 10000
def calculate_distribution(file_name):
    start_time = time.time()
    data = pd.read_csv(f'../../results/{file_name}')
    print(f'##### Data loading time: {time.time() - start_time:.2f} seconds')

    # 노드 생성
    start_time = time.time()
    if target_data == 'hf':
        wd_ac_col_name = 'wd_ac_sn'
    elif target_data == 'cd':
        wd_ac_col_name = 'pay_ac_sn'

    data['WD_NODE'] = data['wd_fc_sn'].astype(str) + '-' + data[wd_ac_col_name].astype(str)
    data['DPS_NODE'] = data['dps_fc_sn'].astype(str) + '-' + data['dps_ac_sn'].astype(str)
    data = data.drop(columns=['dps_fc_sn', 'wd_fc_sn', wd_ac_col_name, 'dps_ac_sn'])
    print(f'##### Data preparing time: {time.time() - start_time:.2f} seconds')

    # 네트워크 생성
    start_time = time.time()
    G = nx.from_pandas_edgelist(
        data,
        source='WD_NODE',
        target='DPS_NODE',
        edge_attr=True,
        create_using=nx.MultiDiGraph()
    )
    in_degree_values = np.array(list(nx.in_degree_centrality(G).values())) * (n_samples - 1)
    out_degree_values = np.array(list(nx.out_degree_centrality(G).values())) * (n_samples - 1)

    in_kde = gaussian_kde(in_degree_values)
    out_kde = gaussian_kde(out_degree_values)
    print(f'##### Network analysis time: {time.time() - start_time:.2f} seconds')

    return in_degree_values, out_degree_values

kdes = []
for file_name in file_names:
    in_values, out_values = calculate_distribution(file_name)
    kdes.append([in_values, out_values])

plt.figure(1)
for i in range(len(file_names)):
    sns.kdeplot(kdes[i][0], label=labels[i], shade=True)
plt.title('In-degree centrality')
plt.legend()
plt.yscale('log') 
plt.savefig('./analysis_3.png')

plt.figure(2)
for i in range(len(file_names)):
    sns.kdeplot(kdes[i][1], label=labels[i], shade=True)
plt.title('Out-degree centrality')
plt.legend()
plt.yscale('log')
plt.savefig('./analysis_4.png')

plt.show()

# kdes = []
# for file_name in file_names:
#     in_kde, out_kde = calculate_distribution(file_name)
#     kdes.append([in_kde, out_kde])
#
# plt.figure(1)
# x = np.linspace(0.1, 20, 200)
# for i in range(0, len(file_names)):
#     plt.plot(x, kdes[i][0](x), label=labels[i])
# plt.title('In-degree centrality')
# plt.legend()
# plt.savefig('./analysis_3.png')
#
# plt.figure(2)
# for i in range(0, len(file_names)):
#     plt.plot(x, kdes[i][1](x), label=labels[i])
# plt.title('Out-degree centrality')
# plt.legend()
# plt.savefig('./analysis_4.png')
#
# plt.show()
