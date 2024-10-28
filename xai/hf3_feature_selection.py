import pandas as pd
import numpy as np
from datetime import datetime
pd.set_option('display.max_columns', None)

time_stamp = datetime.now()

def mark_time_stamp(step_name):
    global time_stamp
    ms_difference = (datetime.now() - time_stamp).total_seconds()
    print(f"{step_name} : {ms_difference:.2f} sec")
    time_stamp = datetime.now()

import platform
is_server = (platform.system() != 'Windows')

data = pd.read_csv('./HF_TRNS_TRAN_augmented.csv',
                   parse_dates=['tran_dtm'],
                   dtype={'wd_acnt': 'string', 'dps_acnt': 'string'})

df = pd.DataFrame(data)
mark_time_stamp('1. Load Data')

# 중강 피처 간의 상관계수를 재계산하여 제거
def remove_high_corr_columns(data, threshold=0.3):
    # 상관계수 계산
    corr_matrix = data.corr().abs()

    # 상관계수 행렬의 상삼각 행렬만을 선택 (자기 자신과의 상관계수를 제외하고 중복을 피하기 위해)
    upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    # 상관계수 threshold(기본 0.1)보다 큰 열들을 순차적으로 제거
    while upper_triangle.max().max() > threshold:
        # 상관계수가 가장 높은 열 선택
        max_corr_col = upper_triangle.max().idxmax()  # 최대값을 가진 열
        max_corr_row = upper_triangle[max_corr_col].idxmax()  # 그 열에서 최대값을 가지는 행

        # 열 중 하나를 선택하여 제거 (경우, 열 우선 삭제)
        data = data.drop(columns=[max_corr_col])

        # 상관계수 행렬 업데이트
        corr_matrix = data.corr().abs()
        upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    return data

filtered_data = remove_high_corr_columns(data.loc[:, 'nx_wd_1':'accl_dps_180D_dist_9'], threshold=0.8)
original_data = data.loc[:, 'tran_dtm':'is_weekend']
encoded_data = data.loc[:, 'md_type_1':'tran_hour']

final_data = pd.concat([original_data, filtered_data, encoded_data], axis=1)
mark_time_stamp('2. Feature Selection by Correlation')

final_data.to_csv('./HF_TRNS_TRAN_selected.csv', float_format='%.4f', index=False)

