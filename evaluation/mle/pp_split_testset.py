import os
import sys

parent_path = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(parent_path)

import pandas as pd
from sklearn.model_selection import train_test_split

# hf_ctgan_training (10만개), hf_training (1만개), cd_training (1만개)
origin_data_name = 'cd_training'

df = pd.read_csv(f'../results/{origin_data_name}.csv')
df_train_data, df_eval_data = train_test_split(df, test_size=0.3, shuffle=False)
df_valid_data, df_test_data = train_test_split(df_eval_data, test_size=0.5, shuffle=False)

df_test_data.to_csv(f'../results/{origin_data_name}_testset.csv', index=False)
