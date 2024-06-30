import networkx as nx
import pandas as pd
from typing import Dict, Any


def load_data(file_path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(file_path)
    except Exception as e:
        print(e)
        raise


def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    df['Source'] = df['WD_FC_SN'].astype(str) + '_' + df['WD_AC_SN'].astype(str)
    df['Target'] = df['DPS_FC_SN'].astype(str) + '_' + df['DPS_AC_SN'].astype(str)
    return df.drop(columns=['WD_FC_SN', 'WD_AC_SN', 'DPS_FC_SN', 'DPS_AC_SN'])


def create_graph(df: pd.DataFrame) -> nx.DiGraph:
    return nx.from_pandas_edgelist(
        df,
        source='Source',
        target='Target',
        edge_attr=['TRAN_DT', 'TRAN_TMRG', 'TRAN_AMT', 'MD_TYPE', 'FND_TYPE', 'FF_SP_AI'],
        create_using=nx.DiGraph()
    )


def calculate_basic_features(graph: nx.DiGraph) -> Dict[str, Any]:
    return {
        'number_of_nodes': graph.number_of_nodes(),
        'number_of_edges': graph.number_of_edges(),
        'density': nx.density(graph),
        'average_degree': sum(dict(graph.degree()).values()) / graph.number_of_nodes()
    }


def calculate_connectivity_features(graph: nx.DiGraph) -> Dict[str, Any]:
    features = {}

    if nx.is_strongly_connected(graph):
        # 그래프가 강하게 연결된 경우
        features['diameter'] = nx.diameter(graph)
        features['average_path_length'] = nx.average_shortest_path_length(graph)
    else:
        # 그래프가 강하게 연결되어 있지 않은 경우, 가장 큰 강연결 컴포넌트 사용
        largest_scc = max(nx.strongly_connected_components(graph), key=len)
        subgraph = graph.subgraph(largest_scc)
        features['diameter'] = nx.diameter(subgraph)
        features['average_path_length'] = nx.average_shortest_path_length(subgraph)
    return features


def get_graph_features(graph: nx.DiGraph) -> Dict[str, Any]:
    features = calculate_basic_features(graph)
    features.update(calculate_connectivity_features(graph))
    return features


def main():
    try:
        n_samples = 10000
        df = load_data(f'./datasets/hf_sample_{n_samples}.csv')
        df = prepare_data(df)

        G = create_graph(df)

        # 그래프 특징 계산 및 출력
        features = get_graph_features(G)
        for feature, value in features.items():
            print(f"{feature}: {value}")

    except Exception as e:
        print(e)


if __name__ == "__main__":
    main()
