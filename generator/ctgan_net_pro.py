import random

'''
1. Data Preparation
'''
import pandas as pd
import torch
import time

n_samples=100000
n_gen_samples=100000
print(f'##### number of samples: {n_samples}')

# FIXME GPU
cuda_device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(f'##### Using device: {cuda_device}')

# 예제 데이터 로드
start_time = time.time()
# data = pd.read_csv(f'/home/dtestbed/workspace/AIFinLab/datasets/HF_TRNS_TRAN.csv')
# data = data.sample(n_samples)
# data.to_csv(f'./results/hf_ctgan_training.csv', index=False)
data = pd.read_csv(f'./results/hf_ctgan_training.csv')
print(f'##### Data loading time: {time.time() - start_time:.2f} seconds')

# 노드 생성
start_time = time.time()
data['WD_NODE'] = data['wd_fc_sn'].astype(str) + '-' + data['wd_ac_sn'].astype(str)
data['DPS_NODE'] = data['dps_fc_sn'].astype(str) + '-' + data['dps_ac_sn'].astype(str)
data = data.drop(columns=['dps_fc_sn', 'wd_fc_sn'])
print(f'##### Data preparing time: {time.time() - start_time:.2f} seconds')

'''
2. Network Analysis
'''
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

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

def degree_to_distribution(degree_values):
    # 네트워크 집중도 분포 분석
    kde = gaussian_kde(degree_values)

    # 네트워크 집중도 그래프 확인
    x = np.linspace(0, 20, 100)
    plt.plot(x, kde(x))
    plt.xlabel('Edges')
    plt.ylabel('Density')
    plt.show()

    # 네트워크 집중도 분포 분석
    dist = []
    for i in range(100):
        dist.append(kde(i))
        if sum(dist) >= 1:
            # make overall percentage as 1
            dist[i] = dist[i] - (sum(dist) - 1)
            break
        # complement limitation
        elif kde(i) < 0.01:
            dist[i] = 0.01

    return dist

in_dist = degree_to_distribution(in_degree_values)
out_dist = degree_to_distribution(out_degree_values)
print(f'##### Network analysis time: {time.time() - start_time:.2f} seconds')

'''
3. Prepare CT-GAN
'''
from sdv.single_table import CTGANSynthesizer
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
    datetime_format='%Y%m%d'
)
metadata.update_columns(
    column_names=['WD_NODE', 'DPS_NODE'],
    sdtype='id')
metadata.update_column(
    column_name='tran_amt',
    sdtype='numerical'
)
metadata.update_columns(
    column_names=['ff_sp_ai', 'fnd_type', 'md_type', 'tran_tmrg'],
    sdtype='categorical'
)
pprint.pprint(metadata.to_dict())

# CTGANSynthesizer 모델 초기화 및 데이터 학습
model = CTGANSynthesizer(metadata, cuda=cuda_device)

model.auto_assign_transformers(data)
pprint.pprint(model.get_transformers())
print(f"##### Model preparing time: {time.time() - start_time:.2f} seconds")

'''
4. Custom CT-GAN
'''
from rdt.transformers.pii import AnonymizedFaker

class CustomTransformer(AnonymizedFaker):
    INPUT_SDTYPE = 'text'
    SUPPORTED_INPUT_SDTYPES = ['text']

    def __init__(self, dist):
        super().__init__(
            function_name='numerify',
            function_kwargs={'text': '###-################'}
        )
        self.dist = dist
        self.generated = [[] for _ in range(len(dist))]

    @classmethod
    def get_supported_sdtypes(cls):
        return list({cls.INPUT_SDTYPE})

    def _function(self):
        # degree = 1 의 갯수를 이미 초과한 경우
        if len(self.generated[0]) >= self.dist[0] * n_gen_samples and self.dist[0] > 1:
            for i in range(1, len(self.dist)):
                if len(self.generated[i]) < self.dist[i] * n_gen_samples:
                    gen_item = self.generated[i - 1][0]
                    break
        else:
            gen_item = super()._function()

        # check if it was generated before
        is_generated_before = False
        position = (-1, -1)
        for i, row in enumerate(self.generated):
            for j, item in enumerate(row):
                if item == gen_item:
                    is_generated_before = True
                    position = (i, j)
                    break
            if is_generated_before:
                break

        if is_generated_before:
            self.generated[position[0] + 1].append(gen_item)
            del self.generated[position[0]][position[1]]
        else:
            self.generated[0].append(gen_item)
        return gen_item

'''
5. Training CT-GAN
'''
start_time = time.time()
model.update_transformers(column_name_to_transformer={
    'WD_NODE': CustomTransformer(out_dist),
    'DPS_NODE': CustomTransformer(in_dist)
})

model.fit(data)
# model.save(f'./results/hf_ctgan.pkl')
print(f"##### Model training time: {time.time() - start_time:.2f} seconds")

'''
6. Training CT-GAN
'''
# 합성 데이터 생성
start_time = time.time()
synthetic_data = model.sample(n_gen_samples)
print(f"##### Total synthetic data generation time: {time.time() - start_time:.2f} seconds")

synthetic_data[['dps_ac_sn', 'dps_fc_sn']] = synthetic_data['DPS_NODE'].str.split('-', expand=True)
synthetic_data[['wd_ac_sn', 'wd_fc_sn']] = synthetic_data['WD_NODE'].str.split('-', expand=True)

print(synthetic_data.head())
synthetic_data.to_csv(f'./results/hf_ctgan_syn_{n_gen_samples}.csv', index=False)
