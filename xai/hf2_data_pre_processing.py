import numpy as np
import pandas as pd
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

data = pd.read_csv('./hf_copula_base_30000_3.csv')
df = pd.DataFrame(data)
mark_time_stamp('1. Load data')

if not is_server:
    df = df.sample(10000)
    mark_time_stamp('1-1. Sample 10k data')
    df.to_csv('./HF_TRNS_TRAN_10k.csv', index=False)

print('total:', len(df), ", suspicious:", len(df[df['ff_sp_ai'].notnull()]), ', suspicious_ml:', len(df[df['ff_sp_ai'] == 'SP']))
# (HF) >> total: 9947307 , suspicious: 10752 , suspicious_ml: 2498
# (CD) >> total: 379135 , suspicious: 6010 , suspicious_ml: 38

# 금융결제원 분석 용 코드
df['ff_sp_ai'] = df['ff_sp_ai'].apply(lambda x: 1 if pd.notna(x) else x)
df['ff_sp_ai'] = df['ff_sp_ai'].replace(np.NaN, 0)
mark_time_stamp('2-1. Data Pre-processing (ff_sp_ai)')

# 계좌번호로 연결
df['wd_acnt'] = df['wd_fc_sn'].astype(str) + '-' + df['wd_ac_sn'].astype(str)
df['dps_acnt'] = df['dps_fc_sn'].astype(str) + '-' + df['dps_ac_sn'].astype(str)
df = df.drop(['wd_fc_sn', 'wd_ac_sn', 'dps_fc_sn', 'dps_ac_sn'], axis=1)
mark_time_stamp('2-2. Data Pre-processing (account)')

# 날짜 및 시간열을 변경 - tran_tmng 통해 날짜 피처 엔지니어링에 필요하며 추후 삭제
df['tran_dt'] = df['tran_dt'].astype(str)
df['tran_tmng'] = df['tran_tmng'].astype(str).apply(lambda x: x.zfill(2))

# tran_dtm 의 sec는 idx 역할 (동시간대 6개거래까지 처리 가능)
df['tran_dtm'] = df['tran_dt'] + ' ' + df['tran_tmng'] + ':00'
df['tran_dtm_sec'] = df.groupby('tran_dtm').cumcount().astype(str).apply(lambda x: x.zfill(2))
df['tran_dtm'] = pd.to_datetime(df['tran_dtm'] + df['tran_dtm_sec'], format='%Y%m%d %H:%M:%S', errors='coerce')

df = df.drop(['tran_dt', 'tran_dtm_sec'], axis=1)
df = df.set_index('tran_dtm').sort_index()
is_weekend = df.index.weekday >= 5
df['is_weekend'] = np.where(is_weekend, '1', '0')
mark_time_stamp('2-2. Data Pre-processing (tran_dt)')

# 네트워크 기반 피처 엔지니어링
import networkx as nx

G = nx.DiGraph()
G.add_edges_from(
    (row.wd_acnt, row.dps_acnt) for row in df.itertuples(index=False)
)
mark_time_stamp('2-3.0. Data Pre-processing (network constructed)')

centralities = [
    nx.degree_centrality(G),
    nx.betweenness_centrality(G),
    nx.closeness_centrality(G),
    nx.eigenvector_centrality_numpy(G, tol=1e-3)
]

for i in range(len(centralities)):
    idx = i + 1
    df['nx_wd_' + str(idx)] = df['wd_acnt'].map(centralities[i])
    df['nx_dps_' + str(idx)] = df['dps_acnt'].map(centralities[i])
    mark_time_stamp('2-3-' + str(idx) + '. Data Pre-processing (network features)')

# 통계적 피처 엔지니어링
periods = ['30D', '90D', '180D']
account_types = ['wd', 'dps']

# 통계 추가를 위해 필요한 열 값의 범위 변경/변경
df['tran_tmng'] = df['tran_tmng'].replace({'03': '00', '06': '00', '12': '09', '15': '09', '21': '18'})
mark_time_stamp('2-4-0. Data Pre-processing (append statistical attribute)')

# 입금/출금 관련 작업 수행
for account_type in account_types:
    account = account_type + '_acnt'

    for target_account in df[account].unique():
        # 계좌별로 데이터 필터링
        df_filtered = df[df[account] == target_account].copy()

        # 주중/주말 별로 데이터 필터링
        df_filtered_weekday = df_filtered[df_filtered['is_weekend'] == '0'].copy()
        df_filtered_weekend = df_filtered[df_filtered['is_weekend'] == '1'].copy()

        # 시간대별로 분류된 데이터 필터링
        df_filtered_night = df_filtered[df_filtered['tran_tmng'] == '00'].copy()
        df_filtered_day = df_filtered[df_filtered['tran_tmng'] == '09'].copy()
        df_filtered_evening = df_filtered[df_filtered['tran_tmng'] == '18'].copy()

        # 3개월 (90일) 윈도우 롤링
        for period in periods:
            # 거래계좌 기준 과거 1개월 동안, 출금계좌로서의 거래건수
            df.loc[df[account] == target_account, f'accl_{account_type}_{period}_cnt_1'] = df_filtered[
                'tran_amt'].rolling(period).count()
            # 거래계좌 기준 과거 1개월 동안, 출금계좌로서의 거래금액
            df.loc[df[account] == target_account, f'accl_{account_type}_{period}_amt_1'] = df_filtered[
                'tran_amt'].rolling(period).sum()

            # 거래계좌 기준 과거 1개월 등록, 출금계좌로서의 거래건수 00시~09시
            if df_filtered_night.empty:
                df.loc[df[account] == target_account, f'accl_{account_type}_{period}_cnt_2'] = 0
                df.loc[df[account] == target_account, f'accl_{account_type}_{period}_amt_2'] = 0
            else:
                df.loc[df[account] == target_account, f'accl_{account_type}_{period}_cnt_2'] = df_filtered_night[
                    'tran_amt'].rolling(period).count()
                df.loc[df[account] == target_account, f'accl_{account_type}_{period}_amt_2'] = df_filtered_night[
                    'tran_amt'].rolling(period).sum()

            # 거래계좌 기준 과거 1개월 등록, 출금계좌로서의 거래건수 09시~18시
            if df_filtered_day.empty:
                df.loc[df[account] == target_account, f'accl_{account_type}_{period}_cnt_3'] = 0
                df.loc[df[account] == target_account, f'accl_{account_type}_{period}_amt_3'] = 0
            else:
                df.loc[df[account] == target_account, f'accl_{account_type}_{period}_cnt_3'] = df_filtered_day[
                    'tran_amt'].rolling(period).count()
                df.loc[df[account] == target_account, f'accl_{account_type}_{period}_amt_3'] = df_filtered_day[
                    'tran_amt'].rolling(period).sum()

            # 거래계좌 기준 과거 1개월 등록, 출금계좌로서의 거래건수 18시~24시
            if df_filtered_evening.empty:
                df.loc[df[account] == target_account, f'accl_{account_type}_{period}_cnt_4'] = 0
                df.loc[df[account] == target_account, f'accl_{account_type}_{period}_amt_4'] = 0
            else:
                df.loc[df[account] == target_account, f'accl_{account_type}_{period}_cnt_4'] = df_filtered_evening[
                    'tran_amt'].rolling(period).count()
                df.loc[df[account] == target_account, f'accl_{account_type}_{period}_amt_4'] = df_filtered_evening[
                    'tran_amt'].rolling(period).sum()

            # 거래계좌 기준 과거 1개월 등록, 출금계좌로서의 거래건수 주중
            if df_filtered_weekday.empty:
                df.loc[df[account] == target_account, f'accl_{account_type}_{period}_cnt_5'] = 0
                df.loc[df[account] == target_account, f'accl_{account_type}_{period}_amt_5'] = 0
            else:
                df.loc[df[account] == target_account, f'accl_{account_type}_{period}_cnt_5'] = df_filtered_weekday[
                    'tran_amt'].rolling(period).count()
                df.loc[df[account] == target_account, f'accl_{account_type}_{period}_amt_5'] = df_filtered_weekday[
                    'tran_amt'].rolling(period).sum()
            # 거래계좌 기준 과거 1개월 동안, 출금계좌로서의 거래건수 : 거래요일 주말
            if df_filtered_weekend.empty:
                df.loc[df[account] == target_account, f'accl_{account_type}_{period}_cnt_6'] = 0
                df.loc[df[account] == target_account, f'accl_{account_type}_{period}_amt_6'] = 0
            else:
                df.loc[df[account] == target_account, f'accl_{account_type}_{period}_cnt_6'] = df_filtered_weekend[
                    'tran_amt'].rolling(period).count()
                df.loc[df[account] == target_account, f'accl_{account_type}_{period}_amt_6'] = df_filtered_weekend[
                    'tran_amt'].rolling(period).sum()

            # 거래계좌 기준 과거 1개월 동안, 출금계좌로서의 거래금액 평균
            df.loc[df[account] == target_account, f'accl_{account_type}_{period}_dist_1'] = df_filtered[
                'tran_amt'].rolling(period).mean()

            # 거래계좌 기준 과거 1개월 동안, 출금계좌로서의 거래금액 최소
            df.loc[df[account] == target_account, f'accl_{account_type}_{period}_dist_2'] = df_filtered[
                'tran_amt'].rolling(period).min()

            # 거래계좌 기준 과거 1개월 동안, 출금계좌로서의 거래금액 최대
            df.loc[df[account] == target_account, f'accl_{account_type}_{period}_dist_3'] = df_filtered[
                'tran_amt'].rolling(period).max()

            # 거래계좌 기준 과거 1개월 동안, 출금계좌로서의 거래금액 분산
            df.loc[df[account] == target_account, f'accl_{account_type}_{period}_dist_4'] = df_filtered[
                'tran_amt'].rolling(period).var(ddof=0)

            # 거래계좌 기준 과거 1개월 동안, 출금계좌로서의 거래금액 표준편차
            df.loc[df[account] == target_account, f'accl_{account_type}_{period}_dist_5'] = df_filtered[
                'tran_amt'].rolling(period).std(ddof=0)

            # 거래계좌 기준 과거 1개월 동안, 출금계좌로서의 거래금액 1사분위수
            df.loc[df[account] == target_account, f'accl_{account_type}_{period}_dist_6'] = df_filtered[
                'tran_amt'].rolling(period).quantile(0.25)

            # 거래계좌 기준 과거 1개월 동안, 출금계좌로서의 거래금액 중앙값
            df.loc[df[account] == target_account, f'accl_{account_type}_{period}_dist_7'] = df_filtered[
                'tran_amt'].rolling(period).quantile(0.5)

            # 거래계좌 기준 과거 1개월 동안, 출금계좌로서의 거래금액 3사분위수
            df.loc[df[account] == target_account, f'accl_{account_type}_{period}_dist_8'] = df_filtered[
                'tran_amt'].rolling(period).quantile(0.75)

            # 거래계좌 기준 과거 1개월 동안, 출금계좌로서의 거래금액 표준오차
            df.loc[df[account] == target_account, f'accl_{account_type}_{period}_dist_9'] = df_filtered[
                'tran_amt'].rolling(period).sem(ddof=0)

df = df.drop(['tran_tmng'], axis=1)

df.loc[:, 'nx_wd_1':'accl_dps_180D_dist_9'] = df.loc[:, 'nx_wd_1':'accl_dps_180D_dist_9'].fillna(0)
df.loc[:, 'nx_wd_1':'accl_dps_180D_dist_9'] = df.loc[:, 'nx_wd_1':'accl_dps_180D_dist_9'].round(4)
df.loc[:, 'nx_wd_1':'accl_dps_180D_dist_9'] = df.loc[:, 'nx_wd_1':'accl_dps_180D_dist_9'].applymap(lambda x: int(x) if x.is_integer() else x)
mark_time_stamp('2-4-1. Data Pre-processing (statistics)')

# 자료구분, 매체구분을 원핫인코딩
df = pd.get_dummies(df, columns=['md_type', 'fnd_type'])

bool_columns = df.select_dtypes(include='bool').columns
df[bool_columns] = df[bool_columns].astype(int)
mark_time_stamp('2-5. Data Pre-processing (one-hot-encoding categorical)')

# 7월 이전 데이터 제거 - 6개월 정량성을 위해
date_threshold = pd.to_datetime('2023-06-30')
df = df[df.index > '2023-06-30']
mark_time_stamp('2-6. Data Pre-processing (remove before July)')
print(len(df))

df['tran_nm'] = df.index.month
df['tran_dt'] = df.index.day
df['tran_hour'] = df.index.hour

df.to_csv('./HF_TRNS_TRAN_augmented.csv', float_format='%.4f', index=True)

