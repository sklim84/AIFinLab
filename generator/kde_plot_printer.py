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

n_samples = 10000
file_name = 'hf_training.csv'
# file_names = ['hf_ctgan_syn_10000_3.csv']

# FIXME GPU
cuda_device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(f'##### Using device: {cuda_device}')

def calculate_distribution(file_name, bandwidth):
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
    in_degree_values = np.array(list(nx.in_degree_centrality(G).values())) * n_samples
    out_degree_values = np.array(list(nx.out_degree_centrality(G).values())) * n_samples

    in_kde = gaussian_kde(in_degree_values)
    out_kde = gaussian_kde(out_degree_values)

    # if file_name.endswith('_3.csv'):
    # in_kde.set_bandwidth(bw_method=in_kde.factor * 6)
    # out_kde.set_bandwidth(bw_method=out_kde.factor * 6)
    # else:
    in_kde.set_bandwidth(bw_method=in_kde.factor * bandwidth)
    out_kde.set_bandwidth(bw_method=out_kde.factor * bandwidth)
    print(f'##### Network analysis time: {time.time() - start_time:.2f} seconds')

    return in_kde, out_kde

kdes = []
for bandwidth in range(1, 5):
    in_kde, out_kde = calculate_distribution(file_name, bandwidth)
    kdes.append([in_kde, out_kde])

plt.figure(1)
x = np.linspace(0, 30, 100)
for i in range(1, 5):
    plt.plot(x, kdes[i-1][0](x), label='smoothing=' + str(i))
plt.title('In-degree centrality')
plt.legend()
plt.savefig('./kde_in.png')

plt.figure(2)
for i in range(1, 5):
    plt.plot(x, kdes[i-1][1](x), label='smoothing=' + str(i))
plt.title('Out-degree centrality')
plt.legend()
plt.savefig('./kde_out.png')

plt.show()
