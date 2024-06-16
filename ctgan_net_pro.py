import pandas as pd
from sdv.single_table import CTGANSynthesizer
import networkx as nx
from sdv.metadata import SingleTableMetadata
import random
import torch
import time

n_samples=1000000
print(f'##### number of samples: {n_samples}')

# FIXME GPU
cuda_device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(f'##### Using device: {cuda_device}')

# 예제 데이터 로드
start_time = time.time()
data = pd.read_csv(f'./datasets/hf_sample_{n_samples}.csv')
print(f'##### Data loading time: {time.time() - start_time:.2f} seconds')

# 노드 생성
data['WD_NODE'] = data['WD_FC_SN'].astype(str) + '-' + data['WD_AC_SN'].astype(str)
data['DPS_NODE'] = data['DPS_FC_SN'].astype(str) + '-' + data['DPS_AC_SN'].astype(str)

# 네트워크 생성
G = nx.from_pandas_edgelist(
    data,
    source='WD_NODE',
    target='DPS_NODE',
    edge_attr=True,
    create_using=nx.DiGraph()
)

# Centrality 계산
start_time = time.time()
centrality = nx.degree_centrality(G)
print(f'##### Centrality calculation time: {time.time() - start_time:.2f} seconds')

# Centrality 값을 데이터에 추가
data['WD_NODE_CENTRALITY'] = data['WD_NODE'].map(centrality)
data['DPS_NODE_CENTRALITY'] = data['DPS_NODE'].map(centrality)

# 메타데이터 정의
metadata = SingleTableMetadata()
metadata.detect_from_dataframe(data=data)

# CTGANSynthesizer 모델 초기화 및 데이터 학습
start_time = time.time()
model = CTGANSynthesizer(metadata)
model.fit(data)
print(f"##### Model training time: {time.time() - start_time:.2f} seconds")

# 학습된 모델 저장
model.save(f'./results/hf_ctgan_{n_samples}.pkl')

# Custom Provider 정의
class NetworkCustomProvider:
    def __init__(self, network, centrality):
        self.network = network
        self.centrality = centrality

    def provide(self, row):
        wd_node = row['WD_NODE']
        dps_node = row['DPS_NODE']

        # 네트워크 특성을 반영하여 거래금액 등 생성
        if self.network.has_edge(wd_node, dps_node):
            edge_data = self.network.get_edge_data(wd_node, dps_node)
            row['TRAN_AMT'] = edge_data['TRAN_AMT']
            row['TRAN_DT'] = edge_data['TRAN_DT']
            row['TRAN_TMRG'] = edge_data['TRAN_TMRG']
            row['MD_TYPE'] = edge_data['MD_TYPE']
            row['FND_TYPE'] = edge_data['FND_TYPE']
            row['FF_SP_AI'] = edge_data['FF_SP_AI']

            # Centrality 값을 반영하여 출금 및 입금 계좌 일련번호 무작위 생성
            row['WD_FC_SN'] = random.randint(100, 999)
            row['WD_AC_SN'] = random.randint(1000000000000000, 9999999999999999)
            row['DPS_FC_SN'] = random.randint(100, 999)
            row['DPS_AC_SN'] = random.randint(1000000000000000, 9999999999999999)

            # Centrality 값을 반영
            row['WD_NODE_CENTRALITY'] = self.centrality.get(wd_node, 0)
            row['DPS_NODE_CENTRALITY'] = self.centrality.get(dps_node, 0)
        return row

# Custom Provider 생성
provider = NetworkCustomProvider(G, centrality)

# CTGANSynthesizer sampling function
def sample_with_provider(model, num_rows, provider):
    synthetic_data = model.sample(num_rows)
    synthetic_data['WD_NODE'] = synthetic_data['WD_FC_SN'].astype(str) + '-' + synthetic_data['WD_AC_SN'].astype(str)
    synthetic_data['DPS_NODE'] = synthetic_data['DPS_FC_SN'].astype(str) + '-' + synthetic_data['DPS_AC_SN'].astype(str)

    for index, row in synthetic_data.iterrows():
        provided_data = provider.provide(row)
        for column, value in provided_data.items():
            synthetic_data.at[index, column] = value
    return synthetic_data

# 합성 데이터 생성
start_time = time.time()
n_gen_samples=100
synthetic_data = sample_with_provider(model, n_gen_samples, provider)
print(f"##### Total synthetic data generation time: {time.time() - start_time:.2f} seconds")

print(synthetic_data.head())
synthetic_data.to_csv(f'./results/hf_ctgan_syn_{n_samples}_to_{n_gen_samples}.csv', index=False)
