'''
1. Data Preparation
'''
import pandas as pd
import time
import torch

n_samples = 10000
n_gen_samples_arr = [10000]

# FIXME GPU
cuda_device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
# cuda_device = 'cpu'
print(f'###### Using device: {cuda_device}')

# 예제 데이터 로드
start_time = time.time()
data = pd.read_csv(f'/home/dtestbed/workspace/AIFinLab/results/hf_training.csv')
if 'WD_NODE' in data.columns:
    data = data.drop(columns=['WD_NODE', 'DPS_NODE'])


# 'NaN' 값을 '00'으로 변경
data['ff_sp_ai'] = data['ff_sp_ai'].fillna('00')
# '01', '02', 'SP'를 '01'로, '00'을 유지
data['ff_sp_ai'] = data['ff_sp_ai'].replace({'01': '01', '02': '01', 'SP': '01', '00': '00'})

print(data.head(5))
print(f'real data의 갯수: {data.shape[0]}')

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
metadata.update_column(
    column_name='tran_amt',
    sdtype='numerical'
)
metadata.update_columns(
    column_names=['ff_sp_ai', 'fnd_type', 'md_type', 'tran_tmrg'],
    # column_names=['tran_code', 'tran_tmrg'],
    sdtype='categorical'
)
pprint.pprint(metadata.to_dict())

models = []
# CTGANSynthesizer 모델 초기화 및 데이터 학습
models.append(CTGANSynthesizer(metadata, cuda=cuda_device))
# models.append(CopulaGANSynthesizer(metadata, cuda=cuda_device))

model_name = ['ctgan']
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
        synthetic_data.to_csv(f'./results/hf_{model_name[i]}_base_{n_gen_samples}KG_base_1.csv', index=False)

        # dummy_handling = ''
        # if exclude_dummy_handling is True:
        #     dummy_handling = 'wo_dummy'
        #
        # synthetic_data.to_csv(f'./results/hf_{model_name[i]}_base_{n_gen_samples}{dummy_handling}.csv', index=False)

    i = i + 1
