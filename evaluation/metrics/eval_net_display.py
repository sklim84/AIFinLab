import pandas as pd
import matplotlib.pyplot as plt

# summary_df 파일 로드
summary_df = pd.read_csv(f'/home/dtestbed/workspace/AIFinLab/evaluation/centrality_summary2.csv')

def plot_centrality_comparison(summary_df, centrality_type, centrality_name):
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    # 중심성 지표에 대한 통계값 필터링
    metrics = ['Mean', 'Median', 'Std Dev', 'Max', 'Min']
    centrality_data = summary_df[summary_df['Centrality Type'] == centrality_type]

    # 막대 그래프 생성
    for metric in metrics:
        ax.bar(centrality_data['Data'], centrality_data[metric], label=metric)

    ax.set_title(f'Comparison of {centrality_name} Centrality Statistics')
    ax.set_xlabel('Data Name')
    ax.set_ylabel(f'{centrality_name} Centrality Value')
    ax.legend()
    plt.xticks(rotation=45)
    plt.show()

# 중심성 지표별로 그래프 생성
plot_centrality_comparison(summary_df, 'degree_centrality', 'Degree')
plot_centrality_comparison(summary_df, 'closeness_centrality', 'Closeness')
# 필요에 따라 다른 중심성 지표에 대해서도 생성 가능

# 시각화된 그래프 저장
def save_plot(centrality_type, centrality_name):
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    metrics = ['Mean', 'Median', 'Std Dev', 'Max', 'Min']
    centrality_data = summary_df[summary_df['Centrality Type'] == centrality_type]

    for metric in metrics:
        ax.bar(centrality_data['Data'], centrality_data[metric], label=metric)

    ax.set_title(f'Comparison of {centrality_name} Centrality Statistics')
    ax.set_xlabel('Data Name')
    ax.set_ylabel(f'{centrality_name} Centrality Value')
    ax.legend()
    plt.xticks(rotation=45)
    plt.savefig(f'{centrality_name}_comparison.png')

# 시각화된 그래프 저장 호출
save_plot('degree_centrality', 'Degree')
save_plot('closeness_centrality', 'Closeness')

# 결과 파일 저장
# summary_df.to_csv('/mnt/data/centrality_measures_summary.csv', index=False)
