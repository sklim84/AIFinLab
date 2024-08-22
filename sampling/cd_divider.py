import pandas as pd
import numpy as np

data_name = 'CD_TRNS_TRAN'

df_hf_trns_tran = pd.read_csv(f'./{data_name}.csv')
df_hf_trns_tran['tran_dt'] = pd.to_datetime(df_hf_trns_tran['tran_dt'], format='ISO8601')
df_hf_trns_tran.sort_values(by='tran_dt', ascending=True, inplace=True)

# Month
for month in np.unique(df_hf_trns_tran['tran_dt'].dt.month.values):
    print(month)
    df_hf_trns_tran[df_hf_trns_tran['tran_dt'].dt.month == month].to_csv(f'./{data_name}_M{str(month).zfill(2)}.csv', index=False)

# Quarter
for quarter in np.unique(df_hf_trns_tran['tran_dt'].dt.quarter.values):
    print(quarter)
    df_hf_trns_tran[df_hf_trns_tran['tran_dt'].dt.quarter == quarter].to_csv(f'./{data_name}_Q{str(quarter)}.csv', index=False)