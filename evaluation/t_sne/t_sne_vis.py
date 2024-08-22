import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

# 원본 데이터 로드
original_data = pd.read_csv('../results/hf_ctgan_training.csv')

# 합성 데이터 로드
synthetic_data = pd.read_csv('../results/hf_ctgan_syn_100000.csv')
# 필요한 열 선택
columns = ['tran_dt', 'tran_tmrg', 'tran_amt', 'md_type', 'fnd_type', 'ff_sp_ai']

# 원본 데이터와 합성 데이터의 공통된 컬럼 선택
original_data = original_data[columns]
synthetic_data = synthetic_data[columns]

# 날짜 열을 숫자로 변환 (예: 연도, 월, 일로 분리하여 각각 숫자로 변환)
original_data['tran_dt'] = pd.to_datetime(original_data['tran_dt'], format='%Y%m%d').map(lambda x: x.toordinal())
synthetic_data['tran_dt'] = pd.to_datetime(synthetic_data['tran_dt'], format='%Y%m%d').map(lambda x: x.toordinal())

# 범주형 데이터를 숫자로 인코딩
for column in ['md_type', 'fnd_type', 'ff_sp_ai']:
    original_data[column] = original_data[column].astype('category').cat.codes
    synthetic_data[column] = synthetic_data[column].astype('category').cat.codes

# 데이터 정규화
scaler = StandardScaler()
original_data_scaled = scaler.fit_transform(original_data)
synthetic_data_scaled = scaler.transform(synthetic_data)

# t-SNE 적용
tsne = TSNE(n_components=2, random_state=42)
original_data_tsne = tsne.fit_transform(original_data_scaled)
synthetic_data_tsne = tsne.fit_transform(synthetic_data_scaled)

# 시각화
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.scatter(original_data_tsne[:, 0], original_data_tsne[:, 1], s=5, color='blue', label='Original Data')
plt.title('t-SNE Visualization of Original Data')
plt.xlabel('t-SNE 1')
plt.ylabel('t-SNE 2')
plt.legend()

plt.subplot(1, 2, 2)
plt.scatter(synthetic_data_tsne[:, 0], synthetic_data_tsne[:, 1], s=5, color='red', label='Synthetic Data')
plt.title('t-SNE Visualization of Synthetic Data')
plt.xlabel('t-SNE 1')
plt.ylabel('t-SNE 2')
plt.legend()

plt.tight_layout()
plt.show()