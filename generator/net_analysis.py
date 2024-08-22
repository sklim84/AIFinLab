import random

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

'''
1. Data Preparation
'''
import pandas as pd
import torch
import time

n_samples=10000
n_gen_samples_arr = [10000, 20000, 30000]
file_names = ['hf_training.csv', 'hf_ctgan_base_10000_3.csv', 'hf_ctgan_syn_10000_3.csv']
labels = ['base', 'ctgan', 'ctgan w/FNet']

# FIXME GPU
cuda_device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(f'##### Using device: {cuda_device}')

def calculate_distribution(file_name):
    start_time = time.time()
    data = pd.read_csv(f'./results/{file_name}')
    # data = pd.read_csv(f'/home/dtestbed/workspace/AIFinLab/results/hf_ctgan_syn_100000_2.csv')
    print(f'##### Data loading time: {time.time() - start_time:.2f} seconds')

    # 노드 생성
    start_time = time.time()
    data['WD_NODE'] = data['wd_fc_sn'].astype(str) + '-' + data['wd_ac_sn'].astype(str)
    data['DPS_NODE'] = data['dps_fc_sn'].astype(str) + '-' + data['dps_ac_sn'].astype(str)
    data = data.drop(columns=['dps_fc_sn', 'wd_fc_sn', 'wd_ac_sn', 'dps_ac_sn'])
    print(f'##### Data preparing time: {time.time() - start_time:.2f} seconds')

    '''
    2. Network Analysis
    '''

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
    # if file_name.endswith('_3.csv'):
    # in_kde.set_bandwidth(bw_method=in_kde.factor * 6)
    # out_kde.set_bandwidth(bw_method=out_kde.factor * 6)
    # else:
    #     in_kde.set_bandwidth(bw_method=in_kde.factor * 4)
    #     out_kde.set_bandwidth(bw_method=out_kde.factor * 4)
    print(f'##### Network analysis time: {time.time() - start_time:.2f} seconds')

    return in_kde, out_kde

kdes = []
for file_name in file_names:
    in_kde, out_kde = calculate_distribution(file_name)
    kdes.append([in_kde, out_kde])

plt.figure(1)
x = np.linspace(0.1, 20, 200)
for i in range(0, len(file_names)):
    plt.plot(x, kdes[i][0](x), label=labels[i])
plt.title('In-degree centrality')
plt.legend()
plt.savefig('./analysis_3.png')

plt.figure(2)
for i in range(0, len(file_names)):
    plt.plot(x, kdes[i][1](x), label=labels[i])
plt.title('Out-degree centrality')
plt.legend()
plt.savefig('./analysis_4.png')

plt.show()
