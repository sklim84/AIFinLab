import random

'''
1. Data Preparation
'''
import pandas as pd
import torch
import time

exclude_dummy_handling = True

n_samples = 10000
n_gen_samples_arr = [10000, 20000, 30000]
# n_gen_samples_arr = [100000]

# FIXME GPU
# cuda_device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
cuda_device = 'cpu'
print(f'##### Using device: {cuda_device}')

# 예제 데이터 로드
start_time = time.time()
# data = pd.read_csv(f'/home/dtestbed/workspace/AIFinLab/datasets/CD_TRNS_TRAN_net.csv')
# data = data.sample(n_samples)
# data.to_csv(f'./results/cd_ctgan_training.csv', index=False)
# data = pd.read_csv(f'/home/dtestbed/workspace/AIFinLab/results/hf_training.csv')
data = pd.read_csv(f'./results/cd_training.csv')
print(f'##### Data loading time: {time.time() - start_time:.2f} seconds')

# 노드 생성
# start_time = time.time()
# data['WD_NODE'] = data['wd_fc_sn'].astype(str) + '-' + data['wd_ac_sn'].astype(str)
# data['DPS_NODE'] = data['dps_fc_sn'].astype(str) + '-' + data['dps_ac_sn'].astype(str)
# data = data.drop(columns=['dps_fc_sn', 'wd_fc_sn'])
# print(f'##### Data preparing time: {time.time() - start_time:.2f} seconds')

'''
2. Network Analysis
'''
# import networkx as nx
# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.stats import gaussian_kde
#
# # 네트워크 생성
# start_time = time.time()
# G = nx.from_pandas_edgelist(
#     data,
#     source='WD_NODE',
#     target='DPS_NODE',
#     edge_attr=True,
#     create_using=nx.MultiDiGraph()
# )
# in_degree_values = np.array(list(nx.in_degree_centrality(G).values())) * (n_samples - 1)
# out_degree_values = np.array(list(nx.out_degree_centrality(G).values())) * (n_samples - 1)
#
# def degree_to_distribution(degree_values):
#     # 네트워크 집중도 분포 분석
#     kde = gaussian_kde(degree_values)
#
#     # 네트워크 집중도 그래프 확인
#     x = np.linspace(0, 20, 100)
#     plt.plot(x, kde(x))
#     plt.xlabel('Edges')
#     plt.ylabel('Density')
#     plt.show()
#
#     # 네트워크 집중도 분포 분석
#     dist = []
#     for i in range(100):
#         dist.append(kde(i))
#         if sum(dist) >= 1:
#             # make overall percentage as 1
#             dist[i] = dist[i] - (sum(dist) - 1)
#             break
#         # complement limitation
#         elif kde(i) < 0.01:
#             dist[i] = 0.01
#
#     return dist
#
# in_dist = degree_to_distribution(in_degree_values)
# out_dist = degree_to_distribution(out_degree_values)
# print(f'##### Network analysis time: {time.time() - start_time:.2f} seconds')

if exclude_dummy_handling is False:
    data = pd.get_dummies(data, columns=['ff_sp_ai'], dummy_na=True)

'''
3. Prepare CT-GAN
'''
from sdv.single_table import CTGANSynthesizer
from sdv.single_table import TVAESynthesizer
from sdv.single_table import CopulaGANSynthesizer
from sdv.metadata import SingleTableMetadata
import pprint

# 메타데이터 정의
start_time = time.time()
metadata = SingleTableMetadata()
metadata.detect_from_dataframe(data=data)
metadata.remove_primary_key()
metadata.update_column(
    column_name='tran_dt',
    sdtype='datetime',
    datetime_format='%Y-%m-%d'
)
# metadata.update_columns(
#     column_names=['WD_NODE', 'DPS_NODE'],
#     sdtype='id')
metadata.update_columns(
    # hf
    # column_name='tran_amt',
    # cd
    column_names=['tran_amt', 'fee'],
    sdtype='numerical'
)
metadata.update_columns(
    # hf
    # column_names=['ff_sp_ai', 'fnd_type', 'md_type', 'tran_tmrg'],
    # cd
    column_names=['ff_sp_ai', 'tran_tmrg', 'tran_code'],
    # column_names=['tran_code', 'tran_tmrg'],
    sdtype='categorical'
)
pprint.pprint(metadata.to_dict())

models = []
# CTGANSynthesizer 모델 초기화 및 데이터 학습
models.append(CTGANSynthesizer(metadata, cuda=cuda_device))
models.append(CopulaGANSynthesizer(metadata, cuda=cuda_device))

model_name = ['ctgan', 'copula']
# model_name = ['tvae']
print(f"##### Model preparing time: {time.time() - start_time:.2f} seconds")

i = 0
for model in models:
    start_time = time.time()
    model.fit(data)
    print(f"##### Model training time: {time.time() - start_time:.2f} seconds")

    for n_gen_samples in n_gen_samples_arr:
        '''
        6. Training CT-GAN
        '''
        # 합성 데이터 생성
        start_time = time.time()
        synthetic_data = model.sample(n_gen_samples)
        print(f"##### Total synthetic data generation time: {time.time() - start_time:.2f} seconds")

        print(synthetic_data.head())
        # synthetic_data.to_csv(f'./results/hf_{model_name[i]}_base_{n_gen_samples}_3.csv', index=False)
        synthetic_data.to_csv(f'./results/cd_{model_name[i]}_base_{n_gen_samples}_3.csv', index=False)

        # dummy_handling = ''
        # if exclude_dummy_handling is True:
        #     dummy_handling = 'wo_dummy'
        #
        # synthetic_data.to_csv(f'./results/hf_{model_name[i]}_base_{n_gen_samples}{dummy_handling}.csv', index=False)

    i = i + 1
