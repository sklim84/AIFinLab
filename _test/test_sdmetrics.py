import json
import numpy as np
import pandas as pd
from sdmetrics.reports.single_table import QualityReport, DiagnosticReport
from sdmetrics.single_table import CorrelationSimilarity, LogisticDetection, KSComplement, TVComplement
from sdmetrics.multi_table import CardinalityShapeSimilarity
from sdmetrics.single_table import NewRowSynthesis


# from sdmetrics.single_column import StatisticSimilarity
# from sdmetrics.column_pairs import CardinalityBoundaryAdherence


# SDMetrics에서 제공하는 기본 지표 정의 - Diagnostic Report와 Quality Report
def evaluate_basic_metrics(original_data, synthetic_data, metadata, data_name):
    # Diagnostic Report 생성 및 저장
    diagnostic = DiagnosticReport()
    diagnostic.generate(original_data, synthetic_data, metadata, False)
    validity_details = diagnostic.get_details('Data Validity')
    diagnostic_score = diagnostic.get_score()
    diagnostic_properties = diagnostic.get_properties()
    diagnostic.save(filepath=f'diagnostic_report_{data_name}.pkl')
    # 결과 출력
    print('==================================')
    print(f'Diagnostic Report for {data_name}')
    print(f'Score: {diagnostic_score}')
    print(f'Properties: {diagnostic_properties}')
    print(f'Validity Details: {validity_details}')
    print('==================================\n')

    # Quality Report 생성 및 저장
    quality_report = QualityReport()
    quality_report.generate(original_data, synthetic_data, metadata, False)
    overall_quality_score = quality_report.get_score()
    property_scores = quality_report.get_properties()
    column_shapes_details = quality_report.get_details('Column Shapes')
    quality_report.save(filepath=f'quality_report_{data_name}.pkl')
    # 결과 출력
    print('==================================')
    print(f'Quality Report for {data_name}')
    print(f'Overall Quality Score: {overall_quality_score}')
    print(f'Property Scores: {property_scores}')
    print(f'Column Shapes Details: {column_shapes_details}')
    print('==================================\n')

    '''
    # New Row Synthesis 점수 계산 및 출력
    new_row_synthesis_score = NewRowSynthesis.compute(
        real_data=original_data,
        synthetic_data=synthetic_data,
        metadata=metadata
    )
    print('==================================')
    print(f'New Row Synthesis Score for {data_name}: {new_row_synthesis_score}')
    print('==================================\n')
    print('\n')
    '''


# 박대영 대리 논문 3가지 지표 정의 - Pair-wise Correlation Score, pMSE-based Score, Statistics (STT) Score + ID컬럼 분포 평가(cardinality_similarity 지표 활용)
def evaluate_park_metrics(original_data, synthetic_data, metadata, data_name):
    # 데이터프레임의 열을 알파벳 순서로 정렬
    original_data = original_data.reindex(sorted(original_data.columns), axis=1)
    synthetic_data = synthetic_data.reindex(sorted(synthetic_data.columns), axis=1)

    # Pair-wise Correlation Score 산출 - CorrelationSimilarity 활용함
    correlation_similarity_score = CorrelationSimilarity.compute(
        real_data=original_data,
        synthetic_data=synthetic_data,
        coefficient='Spearman'
    )
    print('==================================')
    print(f'Pair-wise Correlation Score for {data_name}: {correlation_similarity_score:.4f}')
    print('==================================\n')

    # pMSE-based Score 산출 - LogisticDetection 활용함
    logistic_detection_score = LogisticDetection.compute(
        real_data=original_data,
        synthetic_data=synthetic_data,
        metadata=metadata
    )
    print('==================================')
    print(f'pMSE-based Score for {data_name}: {logistic_detection_score:.4f}')
    print('==================================\n')

    # Statistics (STT) Score 산출 - 연속형 (KS 지표) / 이산형 (TVD 지표) 활용
    stt_score = calculate_stt(original_data, synthetic_data, metadata)
    print('==================================')
    print(f'Statistics (STT) Score for {data_name}: {stt_score:.4f}')
    print('==================================\n')

    # 추가적으로 ID 컬럼에 대해서 평가해봄 - cardinality_similarity 지표 활용
    id_scores = evaluate_id_columns_with_cardinality_similarity(original_data, synthetic_data, metadata)
    print('==================================')
    print(f'ID Column Scores for {data_name}:')
    for (relationship, score) in id_scores.items():
        parent, child = relationship
        print(f'  {parent} -> {child}: {score["score"]:.4f}')
    print('==================================\n')


# Statistics (STT) Score 산출 세부내용 정의 - 연속형 (KS 지표 : KSComplement) / 이산형 (TVD 지표 : TVComplement) 활용
def calculate_stt(original_data, synthetic_data, metadata):
    ## 이산형과 연속형 컬럼 구분
    numerical_cols = [col for col, col_metadata in metadata['columns'].items() if col_metadata['sdtype'] == 'numerical']
    categorical_cols = [col for col, col_metadata in metadata['columns'].items() if
                        col_metadata['sdtype'] == 'categorical']

    ## 연속형 함수 분포 평가
    numerical_scores = []
    for col in numerical_cols:
        col_metadata = {'columns': {col: metadata['columns'][col]}}
        if not original_data[col].dropna().empty and not synthetic_data[col].dropna().empty:
            try:
                score = KSComplement.compute(
                    real_data=original_data[[col]],
                    synthetic_data=synthetic_data[[col]],
                    metadata=col_metadata
                )
                if not np.isnan(score):  # NaN 값 확인 및 처리
                    numerical_scores.append(score)
            except Exception as e:
                print(f'Error in KSComplement for {col}: {e}')

    ## 이산형 함수 분포 평가
    categorical_scores = []
    for col in categorical_cols:
        col_metadata = {'columns': {col: metadata['columns'][col]}}
        if not original_data[col].dropna().empty and not synthetic_data[col].dropna().empty:
            try:
                score = TVComplement.compute(
                    real_data=original_data[[col]],
                    synthetic_data=synthetic_data[[col]],
                    metadata=col_metadata
                )
                if not np.isnan(score):  # NaN 값 확인 및 처리
                    categorical_scores.append(score)
            except Exception as e:
                print(f'Error in TVComplement for {col}: {e}')

    ## 이산형 점수와 연속형 점수 평균을 통해 STT 값 계산
    avg_numerical_score = np.mean(numerical_scores) if numerical_scores else float('nan')
    avg_categorical_score = np.mean(categorical_scores) if categorical_scores else float('nan')
    stt_score = np.nanmean([avg_numerical_score, avg_categorical_score])
    return stt_score


# ID 값 분포 세부내용 정의
def evaluate_id_columns_with_cardinality_similarity(original_data, synthetic_data, metadata):
    real_data = {
        'transactions': original_data
    }
    synthetic_data_dict = {
        'transactions': synthetic_data
    }
    # 메타데이터에 필요한 부모-자식 관계 정의
    multi_table_metadata = {
        'tables': {
            'transactions': metadata
        },
        'relationships': [
            {
                'parent_table_name': 'transactions',
                'parent_primary_key': 'wd_fc_sn',
                'child_table_name': 'transactions',
                'child_foreign_key': 'dps_fc_sn'
            },
            {
                'parent_table_name': 'transactions',
                'parent_primary_key': 'wd_ac_sn',
                'child_table_name': 'transactions',
                'child_foreign_key': 'dps_ac_sn'
            }
        ]
    }
    scores = CardinalityShapeSimilarity.compute_breakdown(
        real_data=real_data,
        synthetic_data=synthetic_data_dict,
        metadata=multi_table_metadata
    )
    return scores


# 박대영 대리 논문 3가지 지표 정의 - Pair-wise Correlation Score, pMSE-based Score, Statistics (STT) Score + ID컬럼 분포 평가(cardinality_similarity 지표 활용)
def evaluate_ml_binary(original_data, synthetic_data, metadata, data_name):
    ######################### Beta Evaluation ######################################
    # 데이터프레임의 열을 알파벳 순서로 정렬
    original_data = original_data.reindex(sorted(original_data.columns), axis=1)
    synthetic_data = synthetic_data.reindex(sorted(synthetic_data.columns), axis=1)
    # original_data['ff_sp_ai'] = original_data['ff_sp_ai'].apply(lambda x: 0 if pd.isnull(x) else 1)
    # synthetic_data['ff_sp_ai'] = synthetic_data['ff_sp_ai'].apply(lambda x: 0 if pd.isnull(x) else 1)

    # Binary Classification 평가 추가
    from sdmetrics.single_table import BinaryAdaBoostClassifier, BinaryDecisionTreeClassifier, BinaryLogisticRegression, \
        BinaryMLPClassifier

    binary_classification_scores = {}
    binary_target = 'ff_sp_ai'  # 'ff_sp_ai' 사용 (이진 분류 대상 컬럼)
    binary_classification_scores['AdaBoost'] = BinaryAdaBoostClassifier.compute(
        test_data=original_data,
        train_data=synthetic_data,
        target=binary_target,
        metadata=metadata
    )
    binary_classification_scores['DecisionTree'] = BinaryDecisionTreeClassifier.compute(
        test_data=original_data,
        train_data=synthetic_data,
        target=binary_target,
        metadata=metadata
    )
    binary_classification_scores['LogisiticRegression'] = BinaryLogisticRegression.compute(
        test_data=original_data,
        train_data=synthetic_data,
        target=binary_target,
        metadata=metadata
    )
    binary_classification_scores['MLPClassifier'] = BinaryMLPClassifier.compute(
        test_data=original_data,
        train_data=synthetic_data,
        target=binary_target,
        metadata=metadata
    )
    # 평가 결과 출력
    print('==================================')
    print(f'Binary Classification Scores for {data_name}')
    for method, score in binary_classification_scores.items():
        print(f"  {method}: {score:.4f}")
    print('==================================\n')


def evaluate_ml_multi(original_data, synthetic_data, metadata, data_name):
    ######################### Beta Evaluation ######################################
    # 데이터프레임의 열을 알파벳 순서로 정렬
    original_data = original_data.reindex(sorted(original_data.columns), axis=1)
    synthetic_data = synthetic_data.reindex(sorted(synthetic_data.columns), axis=1)
    # original_data['ff_sp_ai'] = original_data['ff_sp_ai'].fillna('null')
    # synthetic_data['ff_sp_ai'] = synthetic_data['ff_sp_ai'].fillna('null')

    # Multiclass Classification 평가 추가
    from sdmetrics.single_table import MulticlassDecisionTreeClassifier, MulticlassMLPClassifier

    multiclass_classification_scores = {}
    multiclass_target = 'ff_sp_ai'  # 다중 클래스 분류 대상 컬럼
    multiclass_classification_scores['DecisionTree'] = MulticlassDecisionTreeClassifier.compute(
        test_data=original_data,
        train_data=synthetic_data,
        target=multiclass_target,
        metadata=metadata
    )
    multiclass_classification_scores['MLP'] = MulticlassMLPClassifier.compute(
        test_data=original_data,
        train_data=synthetic_data,
        target=multiclass_target,
        metadata=metadata
    )
    # 평가 결과 출력
    print('==================================')
    print(f'Multiclass Classification Scores for {data_name}')
    for method, score in multiclass_classification_scores.items():
        print(f"  {method}: {score:.4f}")
    print('==================================\n')
    '''
    # Regression 평가 추가
    from sdmetrics.single_table import LinearRegression, MLPRegressor
    regression_scores = {}
    regression_target = 'c'  # 회귀 대상 컬럼
    regression_scores['LinearRegression'] = LinearRegression.compute(
        test_data=original_data,
        train_data=synthetic_data,
        target=regression_target,
        metadata=metadata
    )
    regression_scores['MLPRegressor'] = MLPRegressor.compute(
        test_data=original_data,
        train_data=synthetic_data,
        target=regression_target,
        metadata=metadata
    )
    # 평가 결과 출력
    print('==================================')
    print(f'Regression Scores for {data_name}')
    for method, score in regression_scores.items():
        print(f"  {method}: {score:.4f}")
    print('==================================\n')
    '''
    ######################### Beta Evaluation ######################################


def main():
    # 데이터 로드
    original_data = pd.read_csv(f'/home/dtestbed/workspace/AIFinLab/results/hf_training.csv')
    if 'WD_NODE' in original_data.columns:
        original_data = original_data.drop(columns=['WD_NODE', 'DPS_NODE'])
    # 'NaN' 값을 '00'으로 변경
    original_data['ff_sp_ai'] = original_data['ff_sp_ai'].fillna('00')
    # '01', '02', 'SP'를 '01'로, '00'을 유지
    original_data['ff_sp_ai'] = original_data['ff_sp_ai'].replace({'01': 1, '02': 1, 'SP': 1, '00':0})
    print(original_data.head(5))

    # 총 레코드 수 계산
    total_records = len(original_data)

    # 'aa' 컬럼에서 0과 1의 개수 계산
    count_0 = original_data['ff_sp_ai'].value_counts().get(0, 0)
    count_1 = original_data['ff_sp_ai'].value_counts().get(1, 0)

    print(f"총 레코드 수: {total_records}")
    print(f"'ff_sp_ai' 컬럼에서 0의 개수: {count_0}")
    print(f"'ff_sp_ai' 컬럼에서 1의 개수: {count_1}")



    BASE_PATH = '/home/dtestbed/workspace/AIFinLab/syn_test/results'
    synthetic_data_files = {
        # 'hf_ctgan_base_10000KG_1.csv': f'{BASE_PATH}/hf_ctgan_base_10000KG_1.csv',
        ### 'hf_ctgan_base_10000_KG_2.csv': f'{BASE_PATH}/hf_ctgan_base_10000_KG_2.csv',
        # 'hf_ctgan_base_10000KG1.csv': f'{BASE_PATH}/hf_ctgan_base_10000KG1.csv',
        'hf_ctgan_syn_10000KG_1.csv': f'{BASE_PATH}/hf_ctgan_syn_10000KG_10k_1.csv',
        # 'hf_ctgan_syn_10000KG_1_0001.csv': f'{BASE_PATH}/hf_ctgan_syn_10000KG_1_0001.csv',
        ### 'hf_ctgan_syn_10000KG_2.csv': f'{BASE_PATH}/hf_ctgan_syn_10000KG_2.csv',
        # 'hf_ctgan_syn_10000KG_3.csv': f'{BASE_PATH}/hf_ctgan_syn_10000KG_3.csv',
        # 'hf_ctgan_syn_10000KG_3-1.csv': f'{BASE_PATH}/hf_ctgan_syn_10000KG_3_1.csv',
        # 'hf_ctgan_syn_10000KG_3-2.csv': f'{BASE_PATH}/hf_ctgan_syn_10000KG_3_2.csv',
        ### 'hf_ctgan_syn_10000KG_4.csv': f'{BASE_PATH}/hf_ctgan_syn_10000KG_4.csv',
    }


    with open('hb_metadata.json') as f:
        metadata = json.load(f)

    # 합성 데이터에 대한 평가
    for index, (name, file_path) in enumerate(synthetic_data_files.items()):
        synthetic_data = pd.read_csv(file_path)
        print(synthetic_data.head(5))
        if 'WD_NODE' in synthetic_data.columns:
            synthetic_data = synthetic_data.drop(columns=['WD_NODE', 'DPS_NODE'])
        print(synthetic_data.head(5))
        print("★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★")
        print(f"{index}  {name}\n- column 리스트:  {synthetic_data.columns}")
        print(f"- 레코드 수: {len(synthetic_data)}\n")
        evaluate_basic_metrics(original_data, synthetic_data, metadata, name)
        evaluate_park_metrics(original_data, synthetic_data, metadata, name)
        evaluate_ml_binary(original_data, synthetic_data, metadata, name)
        evaluate_ml_multi(original_data, synthetic_data, metadata, name)
        print("★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★")


if __name__ == "__main__":
    main()

'''

## 로드된 결과 check
print(f"[1. original data]\n- column 리스트: {original_data.columns}")
print(f"- 레코드 수: {len(original_data)}")



## ctgan 합성데이터 로드
syn_data_ctgan_base_100k = pd.read_csv(f'/home/dtestbed/workspace/AIFinLab/results/hf_ctgan_base_100000.csv')
syn_data_ctgan_base_150k = pd.read_csv(f'/home/dtestbed/workspace/AIFinLab/results/hf_ctgan_base_150000.csv')
syn_data_ctgan_base_300k = pd.read_csv(f'/home/dtestbed/workspace/AIFinLab/results/hf_ctgan_base_300000.csv')
syn_data_ctgan_syn_100k = pd.read_csv(f'/home/dtestbed/workspace/AIFinLab/results/hf_ctgan_syn_100000.csv').drop(columns=['WD_NODE', 'DPS_NODE']) # 합성데이터 생성에 추가된 컬럼 삭제
syn_data_ctgan_syn_150k = pd.read_csv(f'/home/dtestbed/workspace/AIFinLab/results/hf_ctgan_syn_150000.csv').drop(columns=['WD_NODE', 'DPS_NODE']) # 합성데이터 생성에 추가된 컬럼 삭제
syn_data_ctgan_syn_300k = pd.read_csv(f'/home/dtestbed/workspace/AIFinLab/results/hf_ctgan_syn_300000.csv').drop(columns=['WD_NODE', 'DPS_NODE']) # 합성데이터 생성에 추가된 컬럼 삭제
## copula 합성데이터 로드
syn_data_copula_base_100k = pd.read_csv(f'/home/dtestbed/workspace/AIFinLab/results/hf_copula_base_100000.csv')
syn_data_copula_base_150k = pd.read_csv(f'/home/dtestbed/workspace/AIFinLab/results/hf_copula_base_150000.csv')
syn_data_copula_base_300k = pd.read_csv(f'/home/dtestbed/workspace/AIFinLab/results/hf_copula_base_300000.csv')
syn_data_copula_syn_100k = pd.read_csv(f'/home/dtestbed/workspace/AIFinLab/results/hf_copula_syn_100000.csv').drop(columns=['WD_NODE', 'DPS_NODE']) # 합성데이터 생성에 추가된 컬럼 삭제
syn_data_copula_syn_150k = pd.read_csv(f'/home/dtestbed/workspace/AIFinLab/results/hf_copula_syn_150000.csv').drop(columns=['WD_NODE', 'DPS_NODE']) # 합성데이터 생성에 추가된 컬럼 삭제
syn_data_copula_syn_300k = pd.read_csv(f'/home/dtestbed/workspace/AIFinLab/results/hf_copula_syn_300000.csv').drop(columns=['WD_NODE', 'DPS_NODE']) # 합성데이터 생성에 추가된 컬럼 삭제
## tvae 합성데이터 로드
syn_data_tvae_base_100k = pd.read_csv(f'/home/dtestbed/workspace/AIFinLab/results/hf_tvae_base_100000.csv')
syn_data_tvae_base_150k = pd.read_csv(f'/home/dtestbed/workspace/AIFinLab/results/hf_tvae_base_150000.csv')
syn_data_tvae_base_300k = pd.read_csv(f'/home/dtestbed/workspace/AIFinLab/results/hf_tvae_base_300000.csv')
syn_data_tvae_syn_100k = pd.read_csv(f'/home/dtestbed/workspace/AIFinLab/results/hf_tvae_syn_100000.csv').drop(columns=['WD_NODE', 'DPS_NODE']) # 합성데이터 생성에 추가된 컬럼 삭제
syn_data_tvae_syn_150k = pd.read_csv(f'/home/dtestbed/workspace/AIFinLab/results/hf_tvae_syn_150000.csv').drop(columns=['WD_NODE', 'DPS_NODE']) # 합성데이터 생성에 추가된 컬럼 삭제
syn_data_tvae_syn_300k = pd.read_csv(f'/home/dtestbed/workspace/AIFinLab/results/hf_tvae_syn_300000.csv').drop(columns=['WD_NODE', 'DPS_NODE']) # 합성데이터 생성에 추가된 컬럼 삭제
## check
print(f"[1. original data]\n- column 리스트: {real_data.columns}")
print(f"- 레코드 수: {len(real_data)}")

# 메타데이터 로드
import json
with open('hb_metadata.json') as f:
    metadata = json.load(f)
'''

'''
# Diagnostic - 데이터 validate 및 structure 확인
from sdmetrics.reports.single_table import DiagnosticReport
diagnostic = DiagnosticReport()
print('base 모델')
diagnostic.generate(original_data, syn_data_ctgan_base_100k, metadata)
print('graph 모델')
diagnostic.generate(original_data, syn_data_ctgan_graph_100k, metadata)
## validity_details = diagnostic.get_details('Data Validity')
## print(validity_details)

# Quality - Column Shape, Column Pair Trends 확인
print('base 모델')
quality_report = QualityReport()
quality_report.generate(original_data, syn_data_ctgan_base_100k, metadata)
## 전체 품질 점수 출력
overall_score = quality_report.get_score()
print(f'Overall Quality Score: {overall_score}')
## 세부 점수 확인
property_scores = quality_report.get_properties()
print(property_scores)
## 위험도 측정
from sdmetrics.single_table import NewRowSynthesis
new_row_synthesis_score = NewRowSynthesis.compute(
    real_data=original_data,
    synthetic_data=syn_data_ctgan_base_100k,
    metadata=metadata
)
print(f'NewRowSynthesis Score: {new_row_synthesis_score}')


print('graph 모델')
quality_report = QualityReport()
quality_report.generate(original_data, syn_data_ctgan_graph_100k, metadata)
## 전체 품질 점수 출력
overall_score = quality_report.get_score()
print(f'Overall Quality Score: {overall_score}')
## 세부 점수 확인
property_scores = quality_report.get_properties()
print(property_scores)
## 위험도 측정
from sdmetrics.single_table import NewRowSynthesis
new_row_synthesis_score = NewRowSynthesis.compute(
    real_data=original_data,
    synthetic_data=syn_data_ctgan_graph_100k,
    metadata=metadata
)
print(f'NewRowSynthesis Score: {new_row_synthesis_score}')

## Column Shapes - 세부 점수 확인
print(quality_report.get_details('Column Shapes'))

from sdmetrics.visualization import get_column_plot
fig = get_column_plot(
    real_data=original_data,
    synthetic_data=syn_data_ctgan_base_100k,
    column_name='tran_amt',
)
fig.show()

column_pair_trends_fig = quality_report.get_visualization('Column Pair Trends')
column_pair_trends_fig.show()
from sdmetrics.visualization import get_column_pair_plot
fig = get_column_pair_plot(
    real_data=original_data,
    synthetic_data=syn_data_ctgan_base_100k,
    column_names=['md_type', 'ff_sp_ai'],
)

fig.show()

'''

'''
# QualityReport 객체 생성
print("QualityReport 객체 생성(1)")
report = QualityReport()

# 품질 보고서 생성
print("품질 보고서 생성(1)")
result = report.generate(original_data, syn_data_ctgan_base_100k, metadata)

# 속성 간 상관 관계 (PC) 평균 계산
pairwise_corr_details = report.get_details('Column Pair Trends')
pairwise_corr_mean = pairwise_corr_details['Score'].mean()

# 통계 (STT) 평균 계산
statistics_details = report.get_details('Column Shapes')
statistics_mean = statistics_details['Score'].mean()

# pMSE 기반 점수 (역 pMSE)
pmse_score = report.get_properties().loc[report.get_properties()['Property'] == 'Column Shapes', 'Score'].values[0]

# 결과 출력
print("[CTGAN 모델 합성 데이터 품질 보고서]")
print(f"속성 간 상관 관계 (PC) 평균: {pairwise_corr_mean}")
print(f"통계 (STT) 평균: {statistics_mean}")
print(f"pMSE 기반 점수 (역 pMSE): {pmse_score}")



# QualityReport 객체 생성
print("QualityReport 객체 생성(2)")
report = QualityReport()

# 품질 보고서 생성
print("품질 보고서 생성(2)")
result = report.generate(original_data, syn_data_ctgan_graph_100k, metadata)

# 속성 간 상관 관계 (PC) 평균 계산
pairwise_corr_details = report.get_details('Column Pair Trends')
pairwise_corr_mean = pairwise_corr_details['Score'].mean()

# 통계 (STT) 평균 계산
statistics_details = report.get_details('Column Shapes')
statistics_mean = statistics_details['Score'].mean()

# pMSE 기반 점수 (역 pMSE)
pmse_score = report.get_properties().loc[report.get_properties()['Property'] == 'Column Shapes', 'Score'].values[0]

# 결과 출력
print("[CTGAN-Graph 모델 합성 데이터 품질 보고서]")
print(f"속성 간 상관 관계 (PC) 평균: {pairwise_corr_mean}")
print(f"통계 (STT) 평균: {statistics_mean}")
print(f"pMSE 기반 점수 (역 pMSE): {pmse_score}")

----------------------

    # StatisticSimilarity 평가 ('tran_amt' 컬럼에 대해서만 - numerical!)
    column = 'tran_amt'
    real_column = original_data[column]
    synthetic_column = synthetic_data[column]
    mean_score = StatisticSimilarity.compute(real_data=real_column, synthetic_data=synthetic_column, statistic='mean')
    median_score = StatisticSimilarity.compute(real_data=real_column, synthetic_data=synthetic_column, statistic='median')
    std_score = StatisticSimilarity.compute(real_data=real_column, synthetic_data=synthetic_column, statistic='std')
    statistic_similarity_scores = {
        'mean': mean_score,
        'median': median_score,
        'std': std_score
    }
    ## StatisticSimilarity 평가 결과 출력
    print('==================================')
    print(f'Statistic Similarity Scores for {data_name}')
    print(f"  Mean: {statistic_similarity_scores['mean']}")
    print(f"  Median: {statistic_similarity_scores['median']}")
    print(f"  Std: {statistic_similarity_scores['std']}")
    print('==================================\n')

    # CardinalityShapeSimilarity 평가 - id 컬럼 기준 (입/출금 기반)
    real_data = {
        'transactions': original_data
    }
    synthetic_data_dict = {
        'transactions': synthetic_data
    }
    multi_table_metadata = {
        'tables': {
            'transactions': metadata
        },
        'relationships': [
            {
                'parent_table_name': 'transactions',
                'parent_primary_key': 'wd_fc_sn',
                'child_table_name': 'transactions',
                'child_foreign_key': 'dps_fc_sn'
            },
            {
                'parent_table_name': 'transactions',
                'parent_primary_key': 'wd_ac_sn',
                'child_table_name': 'transactions',
                'child_foreign_key': 'dps_ac_sn'
            }
        ]
    }
    cardinality_scores = CardinalityShapeSimilarity.compute_breakdown(
        real_data=real_data,
        synthetic_data=synthetic_data_dict,
        metadata=multi_table_metadata
    )
    for (parent_child, score) in cardinality_scores.items():
        parent, child = parent_child
        print(f"    {parent} -> {child}: {score['score']:.4f}")
    print('==================================\n')

    # CardinalityBoundaryAdherence 평가 추가
    id_columns = ['wd_fc_sn', 'wd_ac_sn', 'dps_fc_sn', 'dps_ac_sn']
    boundary_adherence_scores = {}
    for primary_key in id_columns:
        for foreign_key in id_columns:
            if primary_key != foreign_key:
                real_data_tuple = (original_data[primary_key], original_data[foreign_key])
                synthetic_data_tuple = (synthetic_data[primary_key], synthetic_data[foreign_key])
                score = CardinalityBoundaryAdherence.compute(
                    real_data=real_data_tuple,
                    synthetic_data=synthetic_data_tuple
                )
                boundary_adherence_scores[(primary_key, foreign_key)] = score

    # 평가 결과 출력
    print("  Cardinality Boundary Adherence Scores:")
    for (primary_key, foreign_key), score in boundary_adherence_scores.items():
        print(f"    {primary_key} -> {foreign_key}: {score:.4f}")
    print('==================================\n')

    # LogisticDetection 평가 추가
    detection_score = LogisticDetection.compute(
        real_data=original_data,
        synthetic_data=synthetic_data,
        metadata=metadata
    )

    # 평가 결과 출력
    print('==================================')
    print(f'Logistic Detection Score for {data_name}: {detection_score:.4f}')
    print('==================================\n')

    ######################### Beta Evaluation ######################################
    # 데이터프레임의 열을 알파벳 순서로 정렬
    original_data = original_data.reindex(sorted(original_data.columns), axis=1)
    synthetic_data = synthetic_data.reindex(sorted(synthetic_data.columns), axis=1)

    # Binary Classification 평가 추가
    from sdmetrics.single_table import BinaryAdaBoostClassifier, BinaryDecisionTreeClassifier
    binary_classification_scores = {}
    binary_target = 'ff_sp_ai'  # 'ff_sp_ai' 사용 (이진 분류 대상 컬럼)
    binary_classification_scores['AdaBoost'] = BinaryAdaBoostClassifier.compute(
        test_data=original_data,
        train_data=synthetic_data,
        target=binary_target,
        metadata=metadata
    )
    binary_classification_scores['DecisionTree'] = BinaryDecisionTreeClassifier.compute(
        test_data=original_data,
        train_data=synthetic_data,
        target=binary_target,
        metadata=metadata
    )
    # 평가 결과 출력
    print('==================================')
    print(f'Binary Classification Scores for {data_name}')
    for method, score in binary_classification_scores.items():
        print(f"  {method}: {score:.4f}")
    print('==================================\n')

    # Multiclass Classification 평가 추가
    from sdmetrics.single_table import MulticlassDecisionTreeClassifier, MulticlassMLPClassifier
    multiclass_classification_scores = {}
    multiclass_target = 'fnd_type'  # 다중 클래스 분류 대상 컬럼
    multiclass_classification_scores['DecisionTree'] = MulticlassDecisionTreeClassifier.compute(
        test_data=original_data,
        train_data=synthetic_data,
        target=multiclass_target,
        metadata=metadata
    )
    multiclass_classification_scores['MLP'] = MulticlassMLPClassifier.compute(
        test_data=original_data,
        train_data=synthetic_data,
        target=multiclass_target,
        metadata=metadata
    )
    # 평가 결과 출력
    print('==================================')
    print(f'Multiclass Classification Scores for {data_name}')
    for method, score in multiclass_classification_scores.items():
        print(f"  {method}: {score:.4f}")
    print('==================================\n')

    # Regression 평가 추가
    from sdmetrics.single_table import LinearRegression, MLPRegressor
    regression_scores = {}
    regression_target = 'tran_amt'  # 회귀 대상 컬럼
    regression_scores['LinearRegression'] = LinearRegression.compute(
        test_data=original_data,
        train_data=synthetic_data,
        target=regression_target,
        metadata=metadata
    )
    regression_scores['MLPRegressor'] = MLPRegressor.compute(
        test_data=original_data,
        train_data=synthetic_data,
        target=regression_target,
        metadata=metadata
    )
    # 평가 결과 출력
    print('==================================')
    print(f'Regression Scores for {data_name}')
    for method, score in regression_scores.items():
        print(f"  {method}: {score:.4f}")
    print('==================================\n')
    ######################### Beta Evaluation ######################################




    # New Row Synthesis 점수 계산 및 출력
    new_row_synthesis_score = NewRowSynthesis.compute(
        real_data=original_data,
        synthetic_data=synthetic_data,
        metadata=metadata
    )
    print(f'New Row Synthesis Score for {data_name}: {new_row_synthesis_score}')
    print('\n')

'''