import os.path
import pandas as pd
from datetime import datetime


def exists_metrics(save_path, args, arg_names):
    if not os.path.exists(save_path):
        return False

    df_results = pd.read_csv(save_path)

    for index, result in df_results.iterrows():
        existence_flag = True
        for arg_name in arg_names:
            result_item = result[arg_name]
            args_item = vars(args).get(arg_name)

            if result_item != args_item:
                existence_flag = False
                break

        if existence_flag == True:
            break

    return existence_flag


def save_metrics(save_path, args, arg_names, metrics, metric_names):
    columns = ['timestamp']
    values = [datetime.now().strftime('%Y-%m-%d %H:%M:%S')]
    for arg_name in arg_names:
        arg_value = vars(args).get(arg_name)
        columns.append(arg_name)
        values.append(arg_value)
    columns.extend(metric_names)
    values.extend(metrics)

    if os.path.exists(save_path):
        df_results = pd.read_csv(save_path)
    else:
        df_results = pd.DataFrame(columns=columns)

    df_results.loc[len(df_results)] = values
    # df_results.sort_values(by='mse', ascending=True, inplace=True)
    print(df_results)
    df_results.to_csv(save_path, index=False)
