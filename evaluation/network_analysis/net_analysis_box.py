import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import torch
import time
import seaborn as sns

target_data = 'cd'
target_model = 'copula'
n_gen_samples = 30000

for target_data in ['hf', 'cd']:
    for target_model in ['ctgan', 'copula']:
        for n_gen_samples in [10000, 20000, 30000]:
            file_names = [f'{target_data}_training.csv', f'{target_data}_{target_model}_base_{n_gen_samples}_3.csv',
                          f'{target_data}_{target_model}_syn_{n_gen_samples}_3.csv']
            label_target = target_model.upper()
            label_our = f'{target_model.upper()}-FNet'
            if target_model == 'copula':
                label_target = 'CopulaGAN'
                label_our = 'CopulaGAN-FNet'
            labels = ['Identity', label_target, label_our]

            n_samples = 10000


            def calculate_distribution(file_name):
                start_time = time.time()
                data = pd.read_csv(f'../../results/{file_name}')
                print(f'##### Data loading time: {time.time() - start_time:.2f} seconds')

                # 노드 생성
                start_time = time.time()
                if target_data == 'hf':
                    wd_ac_col_name = 'wd_ac_sn'
                elif target_data == 'cd':
                    wd_ac_col_name = 'pay_ac_sn'

                data['WD_NODE'] = data['wd_fc_sn'].astype(str) + '-' + data[wd_ac_col_name].astype(str)
                data['DPS_NODE'] = data['dps_fc_sn'].astype(str) + '-' + data['dps_ac_sn'].astype(str)
                data = data.drop(columns=['dps_fc_sn', 'wd_fc_sn', wd_ac_col_name, 'dps_ac_sn'])
                print(f'##### Data preparing time: {time.time() - start_time:.2f} seconds')

                # 네트워크 생성
                start_time = time.time()
                G = nx.from_pandas_edgelist(
                    data,
                    source='WD_NODE',
                    target='DPS_NODE',
                    edge_attr=True,
                    create_using=nx.MultiDiGraph()
                )

                in_degree_values = np.array(list(nx.in_degree_centrality(G).values())) * (n_samples - 1)
                out_degree_values = np.array(list(nx.out_degree_centrality(G).values())) * (n_samples - 1)

                print(f'##### Network analysis time: {time.time() - start_time:.2f} seconds')

                return in_degree_values, out_degree_values


            in_degrees = []
            out_degrees = []
            for file_name in file_names:
                in_values, out_values = calculate_distribution(file_name)
                in_degrees.append(in_values)
                out_degrees.append(out_values)

            # Box plot for in-degree centrality
            plt.figure(figsize=(5.5, 3))
            sns.boxplot(data=in_degrees)
            plt.xticks(ticks=np.arange(len(labels)), labels=labels, size=12)
            plt.ylabel('In-degree Centrality', fontsize=14)
            plt.yscale('log')
            plt.tight_layout()
            plt.savefig(f'./results/{target_data}_{target_model}_{n_gen_samples}_in_deg_boxplot.png')

            # Box plot for out-degree centrality
            plt.figure(figsize=(5.5, 3))
            sns.boxplot(data=out_degrees)
            plt.xticks(ticks=np.arange(len(labels)), labels=labels, size=12)
            plt.ylabel('Out-degree Centrality', fontsize=14)
            plt.yscale('log')
            plt.tight_layout()
            plt.savefig(f'./results/{target_data}_{target_model}_{n_gen_samples}_out_deg_boxplot.png')

            plt.show()
