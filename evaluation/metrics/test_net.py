import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import numpy as np


# 네트워크 생성
def create_network(data):
    ## 노드 생성
    if 'WD_NODE' not in data.columns:
        data['WD_NODE'] = data['wd_fc_sn'].astype(str) + '-' + data['wd_ac_sn'].astype(str)
    if 'DPS_NODE' not in data.columns:
        data['DPS_NODE'] = data['dps_fc_sn'].astype(str) + '-' + data['dps_ac_sn'].astype(str)

    ## 네트워크 생성
    G = nx.from_pandas_edgelist(
        data,
        source='WD_NODE',
        target='DPS_NODE',
        edge_attr=True,
        create_using=nx.DiGraph()
    )
    return G

# Centrality 계산
def calculate_centrality_measures(G):
    # Degree Centrality
    degree_centrality = nx.degree_centrality(G)
    # Closeness Centrality
    closeness_centrality = nx.closeness_centrality(G)
    # Betweenness Centrality
    # betweenness_centrality = nx.betweenness_centrality(G)
    # Eigenvector Centrality
    # eigenvector_centrality = nx.eigenvector_centrality(G, max_iter=500, tol=1e-06)
    return {
        'degree_centrality': degree_centrality,
        'closeness_centrality': closeness_centrality,
        # 'betweenness_centrality': betweenness_centrality,
        # 'eigenvector_centrality': eigenvector_centrality
    }

# 요약 통계 출력 함수
def calculate_centrality_statistics(centrality):
    values = list(centrality.values())
    return {
        'mean': np.mean(values),
        'median': np.median(values),
        'std_dev': np.std(values),
        'max': np.max(values),
        'min': np.min(values)
    }
'''
# 시각화 함수 (필용 시)
def plot_combined_centrality_distribution(original_centrality, synthetic_centralities, centrality_type):
    plt.figure(figsize=(10, 6))
    # Plot original data centrality distribution
    original_values = np.array(list(original_centrality.values()))

    # Check if original_values has more than one unique value
    if len(np.unique(original_values)) > 1:
        original_kde = gaussian_kde(original_values)
        x = np.linspace(min(original_values), max(original_values), 100)
        plt.plot(x, original_kde(x), label='Original', color='blue')
    else:
        plt.hist(original_values, bins=20, alpha=0.5, label='Original', color='blue')
        print(f"Original data for {centrality_type} has too few unique values to plot KDE.")

    # Plot synthetic data centrality distributions
    colors = plt.cm.get_cmap('tab10', len(synthetic_centralities)).colors
    for i, (name, synthetic_centrality) in enumerate(synthetic_centralities.items()):
        synthetic_values = np.array(list(synthetic_centrality.values()))
        if len(np.unique(synthetic_values)) > 1:
            synthetic_kde = gaussian_kde(synthetic_values)
            plt.plot(x, synthetic_kde(x), label=f'Synthetic ({name})', color=colors[i])
        else:
            plt.hist(synthetic_values, bins=20, alpha=0.5, label=f'Synthetic ({name})', color=colors[i])
            print(f"Synthetic data {name} for {centrality_type} has too few unique values to plot KDE.")

'''
'''

    original_kde = gaussian_kde(original_values)
    x = np.linspace(min(original_values), max(original_values), 100)
    plt.plot(x, original_kde(x), label='Original', color='blue')

    # Plot synthetic data centrality distributions
    colors = plt.cm.get_cmap('tab10', len(synthetic_centralities)).colors
    for i, (name, synthetic_centrality) in enumerate(synthetic_centralities.items()):
        synthetic_values = np.array(list(synthetic_centrality.values()))
        synthetic_kde = gaussian_kde(synthetic_values)
        plt.plot(x, synthetic_kde(x), label=f'Synthetic ({name})', color=colors[i])
'''
'''

    plt.xlabel('Centrality')
    plt.ylabel('Density')
    plt.title(f'{centrality_type.capitalize()} Distribution')
    plt.legend()
    plt.show()
'''
def main():
    # 데이터 로드
    original_data = pd.read_csv(f'/home/dtestbed/workspace/AIFinLab/results/hf_ctgan_training.csv')
    BASE_PATH = '/home/dtestbed/workspace/AIFinLab/results'
    synthetic_data_files = {
        'syn_data_ctgan_base_100k': f'{BASE_PATH}/hf_ctgan_base_100000.csv',
        'syn_data_ctgan_base_150k': f'{BASE_PATH}/hf_ctgan_base_150000.csv',
        # 'syn_data_ctgan_base_300k': f'{BASE_PATH}/hf_ctgan_base_300000.csv',
        # 'syn_data_ctgan_syn_100k': f'{BASE_PATH}/hf_ctgan_syn_100000.csv',
        # 'syn_data_ctgan_syn_150k': f'{BASE_PATH}/hf_ctgan_syn_150000.csv',
        # 'syn_data_ctgan_syn_300k': f'{BASE_PATH}/hf_ctgan_syn_300000.csv',
        'syn_data_ctgan_syn_100k': f'{BASE_PATH}/hf_ctgan_syn_100000_2.csv',
        'syn_data_ctgan_syn_150k': f'{BASE_PATH}/hf_ctgan_syn_150000_2.csv',
        # 'syn_data_ctgan_syn_300k': f'{BASE_PATH}/hf_ctgan_syn_300000_2.csv',
        # 'syn_data_copula_base_100k': f'{BASE_PATH}/hf_copula_base_100000.csv',
        # 'syn_data_copula_base_150k': f'{BASE_PATH}/hf_copula_base_150000.csv',
        # 'syn_data_copula_base_300k': f'{BASE_PATH}/hf_copula_base_300000.csv',
        # # 'syn_data_copula_syn_100k': f'{BASE_PATH}/hf_copula_syn_100000.csv',
        # # 'syn_data_copula_syn_150k': f'{BASE_PATH}/hf_copula_syn_150000.csv',
        # # 'syn_data_copula_syn_300k': f'{BASE_PATH}/hf_copula_syn_300000.csv',
        # 'syn_data_copula_syn_100k': f'{BASE_PATH}/hf_copula_syn_10000_2.csv',
        # 'syn_data_copula_syn_150k': f'{BASE_PATH}/hf_copula_syn_150000_2.csv',
        # 'syn_data_copula_syn_300k': f'{BASE_PATH}/hf_copula_syn_300000_2.csv',
        # ## 'syn_data_tvae_base_100k': f'{BASE_PATH}/hf_tvae_base_100000.csv',
        # ## 'syn_data_tvae_base_150k': f'{BASE_PATH}/hf_tvae_base_150000.csv',
        # ## 'syn_data_tvae_base_300k': f'{BASE_PATH}/hf_tvae_base_300000.csv',
        # ## 'syn_data_tvae_syn_100k': f'{BASE_PATH}/hf_tvae_syn_100000.csv',
        # ## 'syn_data_tvae_syn_150k': f'{BASE_PATH}/hf_tvae_syn_150000.csv',
        # ## 'syn_data_tvae_syn_300k': f'{BASE_PATH}/hf_tvae_syn_300000.csv'
    }
    print("1")
    # 원본 데이터 네트워크 생성 및 평가
    G_original = create_network(original_data)
    print("2")
    original_centrality_measures = calculate_centrality_measures(G_original)
    print("3")

    # 결과 저장을 위한 데이터프레임 초기화
    summary_df = pd.DataFrame(columns=['Data', 'Centrality Type', 'Mean', 'Median', 'Std Dev', 'Max', 'Min'])
    print("4")

    # 원본 데이터 중심성 요약 통계 계산 및 저장
    for centrality_type, centrality in original_centrality_measures.items():
        stats = calculate_centrality_statistics(centrality)
        summary_df = pd.concat([summary_df, pd.DataFrame({
            'Data': ['Original'],
            'Centrality Type': [centrality_type],
            'Mean': [stats['mean']],
            'Median': [stats['median']],
            'Std Dev': [stats['std_dev']],
            'Max': [stats['max']],
            'Min': [stats['min']]
        })])

    # 합성 데이터 평가 및 요약 통계 계산
    synthetic_centrality_measures = {}
    for index, (name, file_path) in enumerate(synthetic_data_files.items()):
        print("★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★")
        print(f"{index}  {name}\n")

        synthetic_data = pd.read_csv(file_path)
        G_synthetic = create_network(synthetic_data)
        centrality_measures = calculate_centrality_measures(G_synthetic)
        synthetic_centrality_measures[name] = centrality_measures

        for centrality_type, centrality in centrality_measures.items():
            stats = calculate_centrality_statistics(centrality)
            summary_df = pd.concat([summary_df, pd.DataFrame({
                'Data': [f'Synthetic ({name})'],
                'Centrality Type': [centrality_type],
                'Mean': [stats['mean']],
                'Median': [stats['median']],
                'Std Dev': [stats['std_dev']],
                'Max': [stats['max']],
                'Min': [stats['min']]
            })])
        print("★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★")


    # 요약 통계 데이터프레임 출력
    print(summary_df)
    try:
        summary_df.to_csv(f'/home/dtestbed/workspace/AIFinLab/evaluation/centrality_summary_2.csv', index=False)
        print("CSV 파일이 성공적으로 저장되었습니다.")
    except Exception as e:
        print(f'파일저장 중 오류 발생: {e}')

'''
    # 중앙성 분포 비교 시각화
    for centrality_type in original_centrality_measures.keys():
        plot_combined_centrality_distribution(
            original_centrality_measures[centrality_type],
            {name: measures[centrality_type] for name, measures in synthetic_centrality_measures.items()},
            centrality_type
        )

'''
if __name__ == "__main__":
    main()