'''
1. Data Preparation
'''
import pandas as pd
import torch
import time

# CUDA 메모리 해제
torch.cuda.empty_cache()

n_samples=10000
n_gen_samples_arr = [10000]

# FIXME GPU
cuda_device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(f'##### Using device: {cuda_device}')

# 예제 데이터 로드
start_time = time.time()
data = pd.read_csv(f'/home/dtestbed/workspace/AIFinLab/results/hf_training.csv')

# 'NaN' 값을 '00'으로 변경
data['ff_sp_ai'] = data['ff_sp_ai'].fillna('00')
# '01', '02', 'SP'를 '01'로, '00'을 유지
data['ff_sp_ai'] = data['ff_sp_ai'].replace({'01': '01', '02': '01', 'SP': '01', '00': '00'})
print(data.head(5))
print(f'real data의 갯수: {data.shape[0]}')
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
in_degree_values = np.array(list(nx.in_degree_centrality(G).values())) * n_samples
out_degree_values = np.array(list(nx.out_degree_centrality(G).values())) * n_samples

def degree_to_distribution(degree_values):
    # 네트워크 집중도 분포 분석
    kde = gaussian_kde(degree_values)
    # kde.set_bandwidth(bw_method=kde.factor * 2)
    # 자동 밴드폭 최적화를 위해 bw_method 조정
    kde.set_bandwidth(bw_method='silverman')  # 'scott', 'silverman' or a scalar value

    # 네트워크 집중도 그래프 확인
    x = np.linspace(start=0, stop=500, num=100)
    plt.plot(x, kde(x))
    plt.xlabel('Edges')
    plt.ylabel('Density')
    # plt.show()

    density = kde(x)
    # KDE를 사용하여 분포를 계산
    dist = density / density.sum()  # 전체 합이 1이 되도록 정규화

    # 최소 확률 제한
    dist = np.clip(dist, 0.1, None)
    dist /= dist.sum()  # 다시 전체 합이 1이 되도록 정규화

    print(f'First 10 degrees are {dist[:10]}')
    sorted_dist = sorted(dist, reverse=True)
    print(f'정렬된 첫 10개의 값: {sorted_dist[:10]}')
    return dist.tolist()

in_dist = degree_to_distribution(in_degree_values)
out_dist = degree_to_distribution(out_degree_values)
print(f'##### Network analysis time: {time.time() - start_time:.2f} seconds')


# 3. Prepare CT-GAN

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
metadata.update_columns(
    column_names=['WD_NODE', 'DPS_NODE'],
    sdtype='text')
metadata.update_column(
    column_name='tran_amt',
    sdtype='numerical'
)
metadata.update_columns(
    # column_names=['ff_sp_ai', 'tran_code', 'tran_tmrg'],
    column_names=['ff_sp_ai', 'fnd_type', 'md_type', 'tran_tmrg'],
    sdtype='categorical'
)
pprint.pprint(metadata.to_dict())


# 4. Custom CT-GAN

from rdt.transformers.pii import AnonymizedFaker

class CustomTransformer(AnonymizedFaker):
    INPUT_SDTYPE = 'pii'
    SUPPORTED_INPUT_SDTYPES = ['pii', 'text']

    def __init__(self, dist, n_gen_samples):
        super().__init__(
            function_name='numerify',
            function_kwargs={'text': '###-################'}
        )
        self.dist = dist
        self.generated = [[] for _ in range(len(dist))]
        self.n_gen_samples = n_gen_samples
        print(f'dist[0] should be {self.dist[0] * self.n_gen_samples}')

    @classmethod
    def get_supported_sdtypes(cls):
        return list({cls.INPUT_SDTYPE})

    def _function(self):
        # degree = 1 의 갯수를 이미 초과한 경우
        if len(self.generated[0]) >= self.dist[0] * self.n_gen_samples:
            for i in range(1, len(self.dist)):
                if len(self.generated[i]) < self.dist[i] * self.n_gen_samples:
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

# CTGANSynthesizer 모델 초기화 및 데이터 학습
models = []
# CTGANSynthesizer 모델 초기화 및 데이터 학습
# models.append(CTGANSynthesizer(metadata, batch_size=250, cuda=cuda_device))
models.append(CTGANSynthesizer(metadata, cuda=cuda_device))
# models.append(TVAESynthesizer(metadata, cuda=cuda_device))
# models.append(CopulaGANSynthesizer(metadata, cuda=cuda_device))

model_name = ['ctgan']
print(f"##### Model preparing time: {time.time() - start_time:.2f} seconds")

i = 0
for model_item in models:
    start_time = time.time()

    for n_gen_samples in n_gen_samples_arr:
        
        # 5. Training CT-GAN
        
        start_time = time.time()
        model = model_item
        model.auto_assign_transformers(data)
        model.update_transformers(column_name_to_transformer={
            'WD_NODE': CustomTransformer(out_dist, n_gen_samples),
            'DPS_NODE': CustomTransformer(in_dist, n_gen_samples)
        })

        model.fit(data)
        # model.save(f'./results/hf_ctgan.pkl')
        print(f"##### Model training time: {time.time() - start_time:.2f} seconds")

        
        # 6. Training CT-GAN
        
        # 합성 데이터 생성
        start_time = time.time()
        synthetic_data = model.sample(n_gen_samples)
        print(f"##### Total synthetic data generation time: {time.time() - start_time:.2f} seconds")

        synthetic_data[['dps_fc_sn', 'dps_ac_sn']] = synthetic_data['DPS_NODE'].str.split('-', expand=True)
        synthetic_data[['wd_fc_sn', 'wd_ac_sn']] = synthetic_data['WD_NODE'].str.split('-', expand=True)

        print(synthetic_data.head())
        synthetic_data.to_csv(f'./results/hf_{model_name[i]}_syn_{n_gen_samples}KG_10k_1.csv', index=False)

    i = i + 1

