import pandas as pd
from sklearn.model_selection import train_test_split

df_data = pd.read_csv('./Base.csv')
print(f"Total records in base.csv: {len(df_data)}")

train_data, valid_data = train_test_split(df_data, test_size=0.3, random_state=42)
print(f"Train dataset records: {len(train_data)}")
print(f"Validation dataset records: {len(valid_data)}")

train_data.reset_index(drop=True, inplace=True)
valid_data.reset_index(drop=True, inplace=True)

train_data.to_csv('./Base_train.csv')
valid_data.to_csv('./Base_valid.csv')
