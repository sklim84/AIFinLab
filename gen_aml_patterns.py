import pandas as pd
import networkx as nx
import numpy as np
from scipy.stats import chi2

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

# 각 패턴에 대한 지표를 계산합니다.
pattern_counts = {
    'fan_out': {node: 0 for node in G.nodes()},
    'fan_in': {node: 0 for node in G.nodes()},
    'gather_scatter': {node: 0 for node in G.nodes()},
    'simple_cycle': {node: 0 for node in G.nodes()},
    # 'random': {node: 0 for node in G.nodes()},
    'bipartite': {node: 0 for node in G.nodes()},
    'stack': {node: 0 for node in G.nodes()}
}

# Fan-out: 한 노드에서 여러 노드로 나가는 엣지
for node in G.nodes():
    out_edges = G.out_edges(node, data=True)
    if len(out_edges) >= 2:
        pattern_counts['fan_out'][node] = len(out_edges)

# Fan-in: 여러 노드에서 한 노드로 들어오는 엣지
for node in G.nodes():
    in_edges = G.in_edges(node, data=True)
    if len(in_edges) >= 2:
        pattern_counts['fan_in'][node] = len(in_edges)

# Gather-scatter는 Fan-in과 Fan-out을 조합하여 계산
for node in G.nodes():
    in_edges = G.in_edges(node, data=True)
    out_edges = G.out_edges(node, data=True)
    if len(in_edges) >= 2 and len(out_edges) >= 2:
        pattern_counts['gather_scatter'][node] = min(len(in_edges), len(out_edges))

# Simple cycles
simple_cycles = list(nx.simple_cycles(G))
unique_cycles = set(tuple(sorted(cycle)) for cycle in simple_cycles)  # 중복 제거
for cycle in unique_cycles:
    for node in cycle:
        pattern_counts['simple_cycle'][node] += 1

# Random patterns: 임의의 노드로 연결된 엣지
# for node in G.nodes():
#     out_edges = G.out_edges(node, data=True)
#     pattern_counts['random'][node] = len(out_edges)

# Bipartite patterns: Source와 Target이 분리된 이분 그래프 형태
source_nodes = set(df['Source'])
target_nodes = set(df['Target'])
for u, v, d in G.edges(data=True):
    if u in source_nodes and v in target_nodes:
        pattern_counts['bipartite'][u] += 1
        pattern_counts['bipartite'][v] += 1

# Stack patterns: Bipartite의 확장으로 여러 계층의 이분 그래프
for u, v, d in G.edges(data=True):
    if u in source_nodes and v in target_nodes:
        for u2, v2, d2 in G.edges(data=True):
            if v == u2 and v2 in target_nodes:
                pattern_counts['stack'][u] += 1
                pattern_counts['stack'][v] += 1
                pattern_counts['stack'][v2] += 1

# 원본 데이터프레임에 패턴 카운트 추가 (Source 노드에 대한 지표)
df['Source_fan_out'] = df['Source'].map(pattern_counts['fan_out'])
df['Source_fan_in'] = df['Source'].map(pattern_counts['fan_in'])
df['Source_gather_scatter'] = df['Source'].map(pattern_counts['gather_scatter'])
df['Source_simple_cycle'] = df['Source'].map(pattern_counts['simple_cycle'])
# df['source_random'] = df['Source'].map(pattern_counts['random'])
df['Source_bipartite'] = df['Source'].map(pattern_counts['bipartite'])
df['Source_stack'] = df['Source'].map(pattern_counts['stack'])

# 원본 데이터프레임에 패턴 카운트 추가 (Target 노드에 대한 지표)
df['Target_fan_out'] = df['Target'].map(pattern_counts['fan_out'])
df['Target_fan_in'] = df['Target'].map(pattern_counts['fan_in'])
df['Target_gather_scatter'] = df['Target'].map(pattern_counts['gather_scatter'])
df['Target_simple_cycle'] = df['Target'].map(pattern_counts['simple_cycle'])
# df['target_random'] = df['Target'].map(pattern_counts['random'])
df['Target_bipartite'] = df['Target'].map(pattern_counts['bipartite'])
df['Target_stack'] = df['Target'].map(pattern_counts['stack'])

# 마할라노비스 거리 계산 함수
def mahalanobis_distance(x, mean, cov):
    diff = x - mean
    return np.sqrt(diff.dot(np.linalg.inv(cov)).dot(diff))

# Source와 Target에 대해 각각 마할라노비스 거리 계산
def calculate_mahalanobis(df, node_type):
    patterns = ['fan_out', 'fan_in', 'gather_scatter', 'simple_cycle', 'bipartite', 'stack']
    X = df[[f'{node_type}_{p}' for p in patterns]]
    mean = X.mean()
    cov = X.cov()

    distances = X.apply(lambda row: mahalanobis_distance(row, mean, cov), axis=1)
    p_values = distances.apply(lambda d: 1 - chi2.cdf(d ** 2, df=len(patterns)))

    return distances, p_values

# CAI (Composite Anti-Money Laundering Index)
# Source (출금계좌)에 대한 마할라노비스 거리 및 CAI 계산
df['Source_mahalanobis'], df['Source_p_value'] = calculate_mahalanobis(df, 'Source')
df['Source_CAI'] = 1 - df['Source_p_value']

# Target (입금계좌)에 대한 마할라노비스 거리 및 CAI 계산
df['Target_mahalanobis'], df['Target_p_value'] = calculate_mahalanobis(df, 'Target')
df['Target_CAI'] = 1 - df['Target_p_value']

# 모든 컬럼 출력 설정
pd.set_option('display.max_columns', None)

# 결과 출력
print(df.head())
df.to_csv(f'./datasets/hf_sample_{n_samples}_aml_feat.csv', index=False)

