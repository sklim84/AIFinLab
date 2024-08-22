import os
import sys

parent_path = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(parent_path)

import pandas as pd
from sklearn.model_selection import train_test_split
from evaluation.mle.config import get_config

if __name__ == "__main__":
    args = get_config()
    print(args)

    # hf_ctgan_training
    # hf_ctgan_base_100000 hf_ctgan_base_150000 hf_ctgan_base_300000 hf_ctgan_syn_100000 hf_ctgan_syn_150000 hf_ctgan_syn_300000
    # hf_copula_base_100000 hf_copula_base_150000 hf_copula_base_300000 hf_copula_syn_100000 hf_copula_syn_150000 hf_copula_syn_300000
    # hf_tvae_base_100000 hf_tvae_base_150000 hf_tvae_base_300000 hf_tvae_syn_100000 hf_tvae_syn_150000 hf_tvae_syn_300000

    data_name = 'hf_ctgan_training'

    df_hf_trns_tran = pd.read_csv(f'../results/{data_name}.csv')
    if (data_name != 'hf_ctgan_training') and ('base' not in data_name):
        df_hf_trns_tran.drop(columns=['WD_NODE', 'DPS_NODE'], inplace=True)
    df_hf_trns_tran['ff_sp_ai'] = df_hf_trns_tran['ff_sp_ai'].replace('SP', '03')
    # df_hf_trns_tran['ff_sp_ai'] = df_hf_trns_tran['ff_sp_ai'].replace(pd.NA, '00')

    df_hf_trns_tran = pd.get_dummies(df_hf_trns_tran, columns=['ff_sp_ai'], dummy_na=True)
    print(df_hf_trns_tran.columns)
    print(df_hf_trns_tran['ff_sp_ai_nan'])
    total_ff_sp_ai_value_counts = df_hf_trns_tran['ff_sp_ai_nan'].value_counts()
    print(f'total ff_sp_ai value counts: {total_ff_sp_ai_value_counts}')

    df_train_data, df_eval_data = train_test_split(df_hf_trns_tran, test_size=0.3, shuffle=False)
    df_valid_data, df_test_data = train_test_split(df_eval_data, test_size=0.5, shuffle=False)

    test_ff_sp_ai_value_counts = df_test_data['ff_sp_ai_nan'].value_counts()
    print(f'test ff_sp_ai value counts: {test_ff_sp_ai_value_counts}')

    print(f'train data: {len(df_train_data)}')
    print(f'valid data: {len(df_valid_data)}')
    print(f'test data: {len(df_test_data)}')


    # total_ff_sp_ai_value_counts = df_hf_trns_tran['ff_sp_ai'].value_counts()
    # print(f'total ff_sp_ai value counts: {total_ff_sp_ai_value_counts}')
    #
    # df_train_data, df_eval_data = train_test_split(df_hf_trns_tran, test_size=0.3, shuffle=False)
    # df_valid_data, df_test_data = train_test_split(df_eval_data, test_size=0.5, shuffle=False)
    #
    # test_ff_sp_ai_value_counts = df_test_data['ff_sp_ai'].value_counts()
    # print(f'test ff_sp_ai value counts: {test_ff_sp_ai_value_counts}')
    #
    # print(f'train data: {len(df_train_data)}')
    # print(f'valid data: {len(df_valid_data)}')
    # print(f'test data: {len(df_test_data)}')
