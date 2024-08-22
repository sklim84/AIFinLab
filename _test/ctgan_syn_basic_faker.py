'''
1. Data Preparation
'''
import pandas as pd
import torch
import time

n_samples=10000
n_gen_samples_arr = [10000]

# FIXME GPU
cuda_device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(f'##### Using device: {cuda_device}')

# 예제 데이터 로드
start_time = time.time()
data = pd.read_csv(f'/home/dtestbed/workspace/AIFinLab/results/hf_ctgan_training.csv')
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
    kde.set_bandwidth(bw_method=kde.factor * 2)

    # 네트워크 집중도 그래프 확인
    x = np.linspace(0, 20, 100)
    plt.plot(x, kde(x))
    plt.xlabel('Edges')
    plt.ylabel('Density')
    # plt.show()

    # 네트워크 집중도 분포 분석
    dist = []
    for i in range(1, 500):
        dist.append(kde(i)[0])
        if sum(dist) >= 1:
            # make overall percentage as 1
            dist[i-1] = dist[i-1] - (sum(dist) - 1)
            break
        # complement limitation
        elif kde(i)[0] < 0.001:
            dist[i-1] = 0.001

    print(f'first 5 degress are {dist[:5]}')
    return dist

in_dist = degree_to_distribution(in_degree_values)
out_dist = degree_to_distribution(out_degree_values)
print(f'##### Network analysis time: {time.time() - start_time:.2f} seconds')

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
    datetime_format='%Y%m%d'
)
metadata.update_columns(
    column_names=['WD_NODE', 'DPS_NODE'],
    sdtype='id',
    #pii=False
)
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

'''
4. Custom CT-GAN
'''
from rdt.transformers import BaseTransformer
class CustomTransformer(BaseTransformer):
    INPUT_SDTYPE = 'text'  # Specify the input sdtype as 'text'
    OUTPUT_SDTYPE = 'text'  # Specify the output sdtype as 'text'

    def __init__(self, dist, n_gen_samples, original_data=None):
        super().__init__()
        self.dist = dist
        self.generated = [[] for _ in range(len(dist))]
        self.n_gen_samples = n_gen_samples
        self.original_data = original_data

    def fit(self, data):
        # Fit logic if needed
        pass

    def transform(self, data):
        # Transform logic if needed (usually for fitting the model)
        return data

    def reverse_transform(self, data):
        # Generate synthetic data based on the given distribution
        transformed_data = []
        for _ in data:
            gen_item = self.generate_item()
            while not self.is_valid(gen_item):
                gen_item = self.generate_item()
            transformed_data.append(gen_item)
        return transformed_data

    def generate_item(self):
        # Placeholder for generating data item logic
        # This should create data consistent with the intended structure
        return {"ff_sp_ai": self.generate_ff_sp_ai()}

    def generate_ff_sp_ai(self):
        # Generate 'ff_sp_ai' field based on distribution
        # Example logic to maintain a ratio or pattern
        if random.random() < 0.1:  # Example ratio for anomaly
            return '01'  # Indicating an anomaly
        else:
            return '00'  # Indicating normal transaction

    def is_valid(self, gen_item):
        # Validate the generated item against the original data's characteristics
        is_anomaly = gen_item.get('ff_sp_ai') == '01'

        if is_anomaly:
            # Calculate the anomaly rate in the original and generated data
            orig_anomaly_rate = self.calculate_anomaly_rate(self.original_data)
            gen_anomaly_rate = self.calculate_anomaly_rate(self.generated_data_snapshot())

            # Ensure the anomaly rate in generated data is within acceptable bounds
            if not (0.9 * orig_anomaly_rate <= gen_anomaly_rate <= 1.1 * orig_anomaly_rate):
                return False

        # Additional integrity and consistency checks can be added here
        return True

    def calculate_anomaly_rate(self, data):
        # Calculate the rate of '01' in 'ff_sp_ai'
        return (data['ff_sp_ai'] == '01').mean()

    def generated_data_snapshot(self):
        # Return a snapshot of generated data as a DataFrame
        # Combine all generated items into a DataFrame for analysis
        return pd.DataFrame([item for sublist in self.generated for item in sublist], columns=self.original_data.columns)

    def track_generated_item(self, gen_item):
        # Track generated items to maintain distribution
        for i, row in enumerate(self.generated):
            if len(row) < self.dist[i] * self.n_gen_samples:
                self.generated[i].append(gen_item)
                return

# CTGANSynthesizer 모델 초기화 및 데이터 학습
models = []
# CTGANSynthesizer 모델 초기화 및 데이터 학습
models.append(CTGANSynthesizer(metadata, cuda=cuda_device))
# models.append(TVAESynthesizer(metadata, cuda=cuda_device))
# models.append(CopulaGANSynthesizer(metadata, cuda=cuda_device))

model_name = ['ctgan']
print(f"##### Model preparing time: {time.time() - start_time:.2f} seconds")

i = 0
for model_item in models:
    start_time = time.time()

    for n_gen_samples in n_gen_samples_arr:
        '''
        5. Training CT-GAN
        '''
        start_time = time.time()
        model = model_item
        model.auto_assign_transformers(data)
        model.update_transformers(column_name_to_transformer={
            'WD_NODE': CustomTransformer(out_dist, n_gen_samples),
            'DPS_NODE': CustomTransformer(in_dist, n_gen_samples)
        })
        print("good")

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

        synthetic_data[['dps_fc_sn', 'dps_ac_sn']] = synthetic_data['DPS_NODE'].str.split('-', expand=True)
        synthetic_data[['wd_fc_sn', 'wd_ac_sn']] = synthetic_data['WD_NODE'].str.split('-', expand=True)

        print(synthetic_data.head())
        synthetic_data.to_csv(f'./results/hf_{model_name[i]}_syn_{n_gen_samples}KG_5.csv', index=False)

    i = i + 1


'''


            
'''