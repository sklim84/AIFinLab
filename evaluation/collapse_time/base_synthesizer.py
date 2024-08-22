import random

'''
1. Data Preparation
'''
import pandas as pd
import torch
import time

exclude_dummy_handling = True

n_samples = 10000
n_gen_samples_arr = [10000]

# FIXME GPUo
cuda_device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
# cuda_device = 'cpu'
print(f'##### Using device: {cuda_device}')

# 데이터 로드
start_time = time.time()
data = pd.read_csv(f'../../results/hf_training.csv')
print(f'##### Data loading time: {time.time() - start_time:.2f} seconds')

if exclude_dummy_handling is False:
    data = pd.get_dummies(data, columns=['ff_sp_ai'], dummy_na=True)

'''
3. Prepare CT-GAN
'''
from sdv.single_table import CTGANSynthesizer
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

metadata.update_column(
    column_name='tran_amt',
    sdtype='numerical'
)
metadata.update_columns(
    column_names=['ff_sp_ai', 'fnd_type', 'md_type', 'tran_tmrg'],
    sdtype='categorical'
)
pprint.pprint(metadata.to_dict())

models = []
# CTGANSynthesizer 모델 초기화 및 데이터 학습
models.append(CTGANSynthesizer(metadata, cuda=cuda_device))
models.append(CopulaGANSynthesizer(metadata, cuda=cuda_device))

model_name = ['ctgan', 'copula']
print(f"##### Model preparing time: {time.time() - start_time:.2f} seconds")

i = 0
for model in models:


    start_time = time.time()
    model.fit(data)
    print(f"##### {model} training time: {time.time() - start_time:.2f} seconds")

    for n_gen_samples in n_gen_samples_arr:
        '''
        6. Training CT-GAN
        '''
        # 합성 데이터 생성
        start_time = time.time()
        synthetic_data = model.sample(n_gen_samples)
        print(f"##### {model} {n_gen_samples} synthetic data generation time: {time.time() - start_time:.2f} seconds")

        print(synthetic_data.head())
        synthetic_data.to_csv(f'./results/hf_{model_name[i]}_base_{n_gen_samples}_3.csv', index=False)

    i = i + 1
