import warnings

import pandas as pd
import numpy as np

warnings.filterwarnings('ignore')

data_name = 'CD_TRNS_TRAN'

# "tran_dt","tran_tmrg","atm_sn","hdl_fc_sn","wd_fc_sn","pay_ac_sn","dps_fc_sn","dps_ac_sn","tran_code","tran_amt","fee","ff_sp_ai"

# Generate 'WD_NODE' and 'DPS_NODE' columns
df_cd_trns_tran = pd.read_csv(f'./{data_name}.csv')
df_cd_trns_tran = df_cd_trns_tran[df_cd_trns_tran['dps_fc_sn'].notnull()]

print(df_cd_trns_tran.head())

df_cd_trns_tran['tran_dt'] = pd.to_datetime(df_cd_trns_tran['tran_dt'], format='ISO8601')
df_cd_trns_tran = df_cd_trns_tran.astype(
    {'wd_fc_sn': 'str', 'pay_ac_sn': 'str', 'dps_fc_sn': 'str', 'dps_ac_sn': 'str', 'tran_amt': 'float'})
df_cd_trns_tran['WD_NODE'] = df_cd_trns_tran[['wd_fc_sn', 'pay_ac_sn']].apply('-'.join, axis=1)
df_cd_trns_tran['DPS_NODE'] = df_cd_trns_tran[['dps_fc_sn', 'dps_ac_sn']].apply('-'.join, axis=1)
df_cd_trns_tran.to_csv(f'{data_name}_ST.csv', index=False)

# Default information : 'TR' - # transactions, 'AL' - 01+02+SP, 'AC' - # accounts
df_cd_trns_tran = pd.read_csv(f'./{data_name}_ST.csv', dtype=object)
count_tran = len(df_cd_trns_tran)
anomaly_count = df_cd_trns_tran['ff_sp_ai'].value_counts()
count_01 = anomaly_count.get('01', 0)
count_02 = anomaly_count.get('02', 0)
count_SP = anomaly_count.get('SP', 0)

# 입출금 계좌 리스트업 : 중복 제거
src_acc = df_cd_trns_tran['WD_NODE'].to_numpy()
tar_acc = df_cd_trns_tran['DPS_NODE'].to_numpy()
total_acc = np.concatenate((src_acc, tar_acc))
total_acc = np.unique(total_acc)

print('##### Original Data Information')
print(f'# \'AC\':\t{total_acc.shape[0]}')
print(f'# \'TR\':\t{count_tran}')
print(f'# \'AL\':\t{count_01 + count_02 + count_SP}\t|\t{(count_01 + count_02 + count_SP) * 100 / count_tran} %')
print(f'# \'01\':\t{count_01}\t|\t{count_01 * 100 / count_tran} %')
print(f'# \'02\':\t{count_02}\t|\t{count_02 * 100 / count_tran} %')
print(f'# \'SP\':\t{count_SP}\t|\t{count_SP * 100 / count_tran} %')

# Extract anomaly transactions
anomaly_labels = ['01', '02', 'SP']
# anomaly_labels = ['SP']

df_anomaly_tran_h0 = df_cd_trns_tran.loc[df_cd_trns_tran['ff_sp_ai'].isin(anomaly_labels)]
df_anomaly_tran_h0.to_csv(f'{data_name}_{"_".join(anomaly_labels)}_H0.csv', index=False)
count_tran_h0 = len(df_anomaly_tran_h0)

# 의심거래 내 계좌 리스트업 : 입금계좌만, 중복 제거
tar_acc_h0 = df_anomaly_tran_h0['DPS_NODE'].to_numpy()
anomaly_acc_h0 = np.unique(tar_acc_h0)

print('##### H0 Data Information')
print(f'# \'AC\':\t{anomaly_acc_h0.shape[0]}')
print(f'# \'TR\':\t{count_tran_h0}')
print(f'# \'SP\':\t{count_SP * 100 / count_tran_h0} %')

# Extract 1-hop transactions
df_anomaly_tran_h1 = df_cd_trns_tran.loc[df_cd_trns_tran['DPS_NODE'].isin(anomaly_acc_h0)]
df_anomaly_tran_h1.to_csv(f'{data_name}_{"_".join(anomaly_labels)}_H1.csv', index=False)
count_tran_h1 = len(df_anomaly_tran_h1)

# 거래 내 계좌 리스트업
src_acc_h1 = df_cd_trns_tran.loc[df_cd_trns_tran['DPS_NODE'].isin(anomaly_acc_h0)]['WD_NODE'].to_numpy()
anomaly_acc_h1 = np.concatenate((anomaly_acc_h0, src_acc_h1))
anomaly_acc_h1 = np.unique(anomaly_acc_h1)

anomaly_count_h1 = df_anomaly_tran_h1['ff_sp_ai'].value_counts()
count_01_h1 = anomaly_count_h1.get('01', 0)
count_02_h1 = anomaly_count_h1.get('02', 0)
count_SP_h1 = anomaly_count_h1.get('SP', 0)

print('##### H1 Data Information')
print(f'# \'AC\':\t{anomaly_acc_h1.shape[0]}')
print(f'# \'TO\':\t{count_tran_h1}')
print(f'# \'AL\':\t{(count_01_h1 + count_02_h1 + count_SP_h1) * 100 / count_tran_h1} %')
print(f'# \'01\':\t{count_01_h1 * 100 / count_tran_h1} %')
print(f'# \'02\':\t{count_02_h1 * 100 / count_tran_h1} %')
print(f'# \'SP\':\t{count_SP_h1 * 100 / count_tran_h1} %')

# Extract 2-hop transactions
df_anomaly_tran_h2 = df_cd_trns_tran.loc[df_cd_trns_tran['DPS_NODE'].isin(anomaly_acc_h1)]
df_anomaly_tran_h2.to_csv(f'{data_name}_{"_".join(anomaly_labels)}_H2.csv', index=False)
count_tran_h2 = len(df_anomaly_tran_h2)

# 거래 내 계좌 리스트업
src_acc_h2 = df_cd_trns_tran.loc[df_cd_trns_tran['DPS_NODE'].isin(anomaly_acc_h1)]['WD_NODE'].to_numpy()
anomaly_acc_h2 = np.concatenate((anomaly_acc_h1, src_acc_h2))
anomaly_acc_h2 = np.unique(anomaly_acc_h2)

anomaly_count_h2 = df_anomaly_tran_h2['ff_sp_ai'].value_counts()
count_01_h2 = anomaly_count_h2.get('01', 0)
count_02_h2 = anomaly_count_h2.get('02', 0)
count_SP_h2 = anomaly_count_h2.get('SP', 0)

print('##### H2 Data Information')
print(f'# \'AC\':\t{anomaly_acc_h2.shape[0]}')
print(f'# \'TO\':\t{count_tran_h2}')
print(f'# \'AL\':\t{(count_01_h2 + count_02_h2 + count_SP_h2) * 100 / count_tran_h2} %')
print(f'# \'01\':\t{count_01_h2 * 100 / count_tran_h2} %')
print(f'# \'02\':\t{count_02_h2 * 100 / count_tran_h2} %')
print(f'# \'SP\':\t{count_SP_h2 * 100 / count_tran_h2} %')

df_hf_training = df_anomaly_tran_h2[:10000]
df_hf_training.to_csv(f'../results/cd_training.csv', index=False)
count_tran_tr = len(df_hf_training)

anomaly_count_tr = df_hf_training['ff_sp_ai'].value_counts()
count_01_tr = anomaly_count_tr.get('01', 0)
count_02_tr = anomaly_count_tr.get('02', 0)
count_SP_tr = anomaly_count_tr.get('SP', 0)

##### Training Data Information
# TO:	10000
# ALL:	8.03 %
# 01:	3.19 %
# 02:	2.81 %
# SP:	2.03 %
print('##### Training Data Information')
print(f'# TO:\t{count_tran_tr}')
print(f'# ALL:\t{(count_01_tr + count_02_tr + count_SP_tr)}, {(count_01_tr + count_02_tr + count_SP_tr) * 100 / count_tran_tr} %')
print(f'# 01:\t{count_01_tr * 100 / count_tran_tr} %')
print(f'# 02:\t{count_02_tr * 100 / count_tran_tr} %')
print(f'# SP:\t{count_SP_tr * 100 / count_tran_tr} %')