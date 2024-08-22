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
data = pd.read_csv(f'/home/dtestbed/workspace/AIFinLab/results/hf_ctgan_training.csv')
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
metadata = SingleTableMetadata()
metadata.detect_from_dataframe(data=data)
metadata.remove_primary_key()

metadata.update_column(
    column_name='tran_dt',
    sdtype='datetime',
    datetime_format='%Y%m%d'
)
metadata.update_column(
    column_name='tran_amt',
    sdtype='numerical'
)
metadata.update_columns(
    column_names=['ff_sp_ai', 'fnd_type', 'md_type', 'tran_tmrg'],
    # column_names=['tran_code', 'tran_tmrg'],
    sdtype='categorical'
)
metadata.update_columns(
    column_names=['wd_fc_sn', 'dps_fc_sn'],
    sdtype= "id",
    regex_format= "[0-9]{3}"
)
metadata.update_columns(
    column_names=['wd_ac_sn', 'dps_ac_sn'],
    sdtype= "id",
    regex_format= "[0-9]{16}"
)
pprint.pprint(metadata.to_dict())

'''
4. CTGANSynthesizer 모델 초기화 및 데이터 학습
'''
from sdv.constraints import create_custom_constraint_class
synthesizer = CTGANSynthesizer(metadata, cuda=cuda_device)
# models = []
# models.append(CTGANSynthesizer(metadata, cuda=cuda_device))
print(f'###### Using device: {cuda_device}')
# models.append(CopulaGANSynthesizer(metadata, cuda=cuda_device))
model_name = ['ctgan']
# model_name = ['tvae']

'''
# 'tran_amt' 열에 대해 양수 값만 허용하는 제약 조건 정의
positive_constraint = {
    'constraint_class': 'Positive',
    'constraint_parameters': {
        'column_name': 'tran_amt',
        'strict_boundaries': True # False로 설정하면 값이 0도 허용가능
    }
}
synthesizer.add_constraints(constraints=[positive_constraint])
'''
'''
# Categorical 컬럼 제약 조건 정의
allowed_md_type_values = [1, 2, 3, 4, 5, 6, 7, 8]
allowed_fnd_type_values = [
    0, 1, 2, 3, 4, 11, 12, 13, 14, 15, 16, 17, 18, 19, 
    20, 21, 22, 23, 24, 25, 26, 27, 28, 30
]
allowed_ff_sp_ai_values = ['00', '01', '02', 'SP']
allowed_tran_tmrg_values = [0, 3, 6, 9, 12, 15, 18, 21]

# 유효성 검사 함수 정의
def is_valid_md_type(data):
    return data['md_type'].isin(allowed_md_type_values)
def is_valid_fnd_type(data):
    return data['fnd_type'].isin(allowed_fnd_type_values)
def is_valid_ff_sp_ai(data):
    return data['ff_sp_ai'].isin(allowed_ff_sp_ai_values)
def is_valid_tran_tmrg(data):
    return data['tran_tmrg'].isin(allowed_tran_tmrg_values)
# CustomConstraint 정의
md_type_constraint_calss = create_custom_constraint_class(
    is_valid_fn=is_valid_md_type
)
fnd_type_constraint_calss = create_custom_constraint_class(
    is_valid_fn=is_valid_fnd_type
)
ff_sp_ai_constraint_calss = create_custom_constraint_class(
    is_valid_fn=is_valid_ff_sp_ai
)
tran_tmrg_constraint_calss = create_custom_constraint_class(
    is_valid_fn=is_valid_tran_tmrg
)

md_type_constraint = {
    'constraint_class': 'md_type_constraint_calss',
    'constraint_parameters': {
        'column_names': ['md_type']
    }
}
fnd_type_constraint = {
    'constraint_class': 'fnd_type_constraint_calss',
    'constraint_parameters': {
        'column_names': ['fnd_type']
    }
}
ff_sp_ai_constraint = {
    'constraint_class': 'ff_sp_ai_constraint_calss',
    'constraint_parameters': {
        'column_names': ['ff_sp_ai']
    }
}
tran_tmrg_constraint = {
    'constraint_class': 'tran_tmrg_constraint_calss',
    'constraint_parameters': {
        'column_names': ['tran_tmrg']
    }
}

# 제약 조건을 synthesizer에 추가
synthesizer.add_constraints([md_type_constraint, fnd_type_constraint, ff_sp_ai_constraint, tran_tmrg_constraint])
'''

# 데이터 전처리 준비
preprocessed_data = synthesizer.preprocess(data)
# Transformer 설정
from rdt.transformers import (
    UnixTimestampEncoder, OrderedLabelEncoder, FloatFormatter, RegexGenerator
)
transformers = {
    'tran_dt': UnixTimestampEncoder(datetime_format='%Y%m%d', enforce_min_max_values=True),
    'tran_tmrg': OrderedLabelEncoder(order=[0, 3, 6, 9, 12, 15, 18, 21]),
    # 'wd_fc_sn': RegexGenerator(regex_format='^\d{3}$'),  # 3자리 숫자
    # 'dps_fc_sn': RegexGenerator(regex_format='^\d{3}$'),  # 3자리 숫자
    # 'wd_ac_sn': RegexGenerator(regex_format=r'^\d{16}$'),  # 16자리 숫자
    # 'dps_ac_sn': RegexGenerator(regex_format=r'^\d{16}$'),  # 16자리 숫자
    'tran_amt': FloatFormatter(learn_rounding_scheme=True, enforce_min_max_values=True, missing_value_replacement=0.00),
    'md_type': OrderedLabelEncoder(order=[1, 2, 3, 4, 5, 6, 7, 8]),
    'fnd_type': OrderedLabelEncoder(order=[0, 1, 2, 3, 4, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 30]),
    # 'ff_sp_ai': OrderedLabelEncoder(order=['00', '01', '02', 'SP'])
}

# Transformer를 synthesizer에 추가
synthesizer.update_transformers(transformers)

print(f"##### Model preparing time: {time.time() - start_time:.2f} seconds")

i = 0

start_time = time.time()
print(f'파라메터: {synthesizer.get_parameters()}')
# synthesizer.fit_processed_data(preprocessed_data)
synthesizer.fit(data)
print(f"##### Model training time: {time.time() - start_time:.2f} seconds")
print(f'트랜스포머: {synthesizer.get_transformers()}')
synthesizer.save(
    filepath='kg_synthesizer2.pkl'
)

for n_gen_samples in n_gen_samples_arr:
    '''

6. Training CT-GAN

'''
    # 합성 데이터 생성
    start_time = time.time()
    synthetic_data = synthesizer.sample(n_gen_samples)
    print(f"##### Total synthetic data generation time: {time.time() - start_time:.2f} seconds")

    print(synthetic_data.head())
    synthetic_data.to_csv(f'./results/hf_{model_name[i]}_base_{n_gen_samples}_KG_3.csv', index=False)

    # dummy_handling = ''
    # if exclude_dummy_handling is True:
    #     dummy_handling = 'wo_dummy'
    #
    # synthetic_data.to_csv(f'./results/hf_{model_name[i]}_base_{n_gen_samples}{dummy_handling}.csv', index=False)

i = i + 1

