import random

'''
1. Data Preparation
'''
import pandas as pd
import torch
import time
from sdv.single_table import CTGANSynthesizer
from sdv.single_table import TVAESynthesizer
from sdv.single_table import CopulaGANSynthesizer
from sdv.metadata import SingleTableMetadata
import pprint

n_samples = 100000
n_gen_samples_arr = [100000, 150000, 300000]

# FIXME GPU
cuda_device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(f'##### Using device: {cuda_device}')

# 예제 데이터 로드
start_time = time.time()
data = pd.read_csv(f'/home/dtestbed/workspace/AIFinLab/results/hf_ctgan_training.csv')
print(f'##### Data loading time: {time.time() - start_time:.2f} seconds')

'''
2. Network Analysis
'''
data = pd.get_dummies(data, columns=['ff_sp_ai'], dummy_na=True)

'''
3. Prepare CT-GAN
'''
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
# metadata.update_columns(
#     column_names=['WD_NODE', 'DPS_NODE'],
#     sdtype='id')
metadata.update_column(
    column_name='tran_amt',
    sdtype='numerical'
)
metadata.update_columns(
    # column_names=['ff_sp_ai', 'fnd_type', 'md_type', 'tran_tmrg'],
    column_names=['fnd_type', 'md_type', 'tran_tmrg'],
    sdtype='categorical'
)
pprint.pprint(metadata.to_dict())

models = []
# CTGANSynthesizer 모델 초기화 및 데이터 학습
# models.append(CTGANSynthesizer(metadata, cuda=cuda_device))
models.append(TVAESynthesizer(metadata, cuda=cuda_device))
# models.append(CopulaGANSynthesizer(metadata, cuda=cuda_device))

# model_name = ['ctgan', 'tvae', 'copula']
model_name = ['tvae']
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
        synthetic_data.to_csv(f'./results/hf_{model_name[i]}_base_{n_gen_samples}.csv', index=False)

    i = i + 1
