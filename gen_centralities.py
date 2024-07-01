import networkx as nx
import numpy as np
import pandas as pd
import scipy.sparse
import scipy.sparse.csgraph
from scipy.stats import chi2

def extract_value_from_dict(row, centralities, column_name):
    key = row[column_name]
    value = centralities.get(key, 0)
    return value


def compute_closeness_centrality(G):
    A = nx.adj_matrix(G).tolil()
    D = scipy.sparse.csgraph.floyd_warshall(A, directed=True, unweighted=False)
    n = D.shape[0]
    closeness_centrality = {}

    for r in range(0, n):
        print(f'### compute closeness centrality... {r}/{n}')
        cc = 0.0
        possible_paths = list(enumerate(D[r, :]))
        shortest_paths = dict(filter(lambda x: not x[1] == np.inf, possible_paths))

        total = sum(shortest_paths.values())
        n_shortest_paths = len(shortest_paths) - 1.0

        if total > 0.0 and n > 1:
            s = n_shortest_paths / (n - 1)
            cc = (n_shortest_paths / total) * s

        closeness_centrality[r] = cc

    return closeness_centrality


# 샘플 데이터 크기
n_samples = 1000
# 데이터 로드
df = pd.read_csv(f'./datasets/hf_sample_{n_samples}.csv')

# Source와 Target 생성
df['Source'] = df['WD_FC_SN'].astype(str) + '_' + df['WD_AC_SN'].astype(str)
df['Target'] = df['DPS_FC_SN'].astype(str) + '_' + df['DPS_AC_SN'].astype(str)
df.drop(columns=['WD_FC_SN', 'WD_AC_SN', 'DPS_FC_SN', 'DPS_AC_SN'], inplace=True)

# 네트워크 그래프 생성
G = nx.from_pandas_edgelist(df, source='Source', target='Target',
                            edge_attr=['TRAN_DT', 'TRAN_TMRG', 'TRAN_AMT', 'MD_TYPE', 'FND_TYPE', 'FF_SP_AI'],
                            create_using=nx.DiGraph())

print(G.number_of_nodes())
print(G.number_of_edges())

# Eigenvector Centrality
print('##### Eigenvector Centrality')
eigenvector_centralities = nx.eigenvector_centrality(G, tol=1e-03)
df['Source_EC'] = df.apply(extract_value_from_dict, axis=1,
                           args=(eigenvector_centralities, 'Source'))
df['Target_EC'] = df.apply(extract_value_from_dict, axis=1,
                           args=(eigenvector_centralities, 'Target'))

# Degree Centrality
print('##### Degree Centrality')
degree_centralities = nx.degree_centrality(G)
df['Source_DC'] = df.apply(extract_value_from_dict, axis=1,
                           args=(degree_centralities, 'Source'))
df['Target_DC'] = df.apply(extract_value_from_dict, axis=1,
                           args=(degree_centralities, 'Target'))

# # Closeness Centrality
print('##### Closeness Centrality')
closeness_centralities = nx.closeness_centrality(G)
df['Source_CC'] = df.apply(extract_value_from_dict, axis=1,
                           args=(closeness_centralities, 'Source'))
df['Target_CC'] = df.apply(extract_value_from_dict, axis=1,
                           args=(closeness_centralities, 'Target'))

# Betweenness Centrality
print('##### Betweenness Centrality')
betweenness_centralities = nx.betweenness_centrality(G, k=10)
df['Source_BC'] = df.apply(extract_value_from_dict, axis=1,
                           args=(betweenness_centralities, 'Source'))
df['Target_BC'] = df.apply(extract_value_from_dict, axis=1,
                           args=(betweenness_centralities, 'Target'))

# 마할라노비스 거리 계산 함수
def mahalanobis_distance(x, mean, cov):
    diff = x - mean
    return np.sqrt(diff.dot(np.linalg.inv(cov)).dot(diff))

# 중앙성 지표로 CCI 계산
def calculate_composite_centrality_index(df, node_type):
    centralities = ['EC', 'DC', 'CC', 'BC']
    X = df[[f'{node_type}_{c}' for c in centralities]]
    mean = X.mean()
    cov = X.cov()

    distances = X.apply(lambda row: mahalanobis_distance(row, mean, cov), axis=1)
    p_values = distances.apply(lambda d: 1 - chi2.cdf(d ** 2, df=len(centralities)))

    return distances, p_values

# Source (출금계좌)에 대한 마할라노비스 거리 및 CCI 계산
df['Source_mahalanobis'], df['Source_p_value'] = calculate_composite_centrality_index(df, 'Source')
df['Source_CCI'] = 1 - df['Source_p_value']

# Target (입금계좌)에 대한 마할라노비스 거리 및 CCI 계산
df['Target_mahalanobis'], df['Target_p_value'] = calculate_composite_centrality_index(df, 'Target')
df['Target_CCI'] = 1 - df['Target_p_value']

# 모든 컬럼 출력 설정
pd.set_option('display.max_columns', None)

# 결과 출력
print(df.head())
df.to_csv(f'./datasets/hf_sample_{n_samples}_cen_feat.csv', index=False)
