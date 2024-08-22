import pandas as pd
from syntheval import SynthEval

# 데이터 로드
# original_data = pd.read_csv("C:\\Users\\fuser03\\PycharmProjects\\AIFinLab\\results\\cd_training.csv")
original_data = pd.read_csv("C:\\Users\\fuser03\\PycharmProjects\\AIFinLab\\results\\hf_training.csv")
# 'tran_dt'를 datetime 형식으로 변환 후 타임스탬프 (초 단위)로 변환
# original_data['tran_dt'] = pd.to_datetime(original_data['tran_dt']).astype('int64') / 10**9
# original_data['tran_dt'] = original_data['tran_dt'].str.replace('-', '')
# original_data['tran_dt'] = pd.to_datetime(original_data['tran_dt'], format='%Y%m%d').original_data.strftime('%Y-%m-%d')

print("★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★")
print(f"- column 리스트:  {original_data.columns}")
print(f"- 레코드 수: {len(original_data)}\n")
print(original_data.head(5))

if 'WD_NODE' in original_data.columns:
    original_data = original_data.drop(columns=['WD_NODE', 'DPS_NODE', 'tran_dt'])
BASE_PATH = 'C:\\Users\\fuser03\\PycharmProjects\\AIFinLab\\results'
'''
synthetic_data_files = {
    'syn_data_ctgan_base_10k': f'{BASE_PATH}\\cd_ctgan_base_10000_3.csv',
    'syn_data_ctgan_base_20k': f'{BASE_PATH}\\cd_ctgan_base_20000_3.csv',
    'syn_data_ctgan_base_30k': f'{BASE_PATH}\\cd_ctgan_base_30000_3.csv',
    'syn_data_ctgan_syn_10k': f'{BASE_PATH}\\cd_ctgan_syn_10000_3.csv',
    'syn_data_ctgan_syn_20k': f'{BASE_PATH}\\cd_ctgan_syn_20000_3.csv',
    'syn_data_ctgan_syn_30k': f'{BASE_PATH}\\cd_ctgan_syn_30000_3.csv',
    'syn_data_copula_base_10k': f'{BASE_PATH}\\cd_copula_base_10000_3.csv',
    'syn_data_copula_base_20k': f'{BASE_PATH}\\cd_copula_base_20000_3.csv',
    'syn_data_copula_base_30k': f'{BASE_PATH}\\cd_copula_base_30000_3.csv',
    'syn_data_copula_syn_10k': f'{BASE_PATH}\\cd_copula_syn_10000_3.csv',
    'syn_data_copula_syn_20k': f'{BASE_PATH}\\cd_copula_syn_20000_3.csv',
    'syn_data_copula_syn_30k': f'{BASE_PATH}\\cd_copula_syn_30000_3.csv',
    # 'syn_data_copula_syn_100k_2': f'{BASE_PATH}/hf_copula_syn_100000_2.csv',
    # 'syn_data_copula_syn_150k_2': f'{BASE_PATH}/hf_copula_syn_150000_2.csv',
    # 'syn_data_copula_syn_300k_2': f'{BASE_PATH}/hf_copula_syn_300000_2.csv',
    ## 'syn_data_tvae_base_100k': f'{BASE_PATH}/hf_tvae_base_100000.csv',
    ## 'syn_data_tvae_base_150k': f'{BASE_PATH}/hf_tvae_base_150000.csv',
    ## 'syn_data_tvae_base_300k': f'{BASE_PATH}/hf_tvae_base_300000.csv',
    ## 'syn_data_tvae_syn_100k': f'{BASE_PATH}/hf_tvae_syn_100000.csv',
    ## 'syn_data_tvae_syn_150k': f'{BASE_PATH}/hf_tvae_syn_150000.csv',
    ## 'syn_data_tvae_syn_300k': f'{BASE_PATH}/hf_tvae_syn_300000.csv'
}
'''
synthetic_data_files = {
    'syn_data_ctgan_base_10k': f'{BASE_PATH}\\hf_ctgan_base_10000_3.csv',
    'syn_data_ctgan_base_20k': f'{BASE_PATH}\\hf_ctgan_base_20000_3.csv',
    'syn_data_ctgan_base_30k': f'{BASE_PATH}\\hf_ctgan_base_30000_3.csv',
    'syn_data_ctgan_syn_10k': f'{BASE_PATH}\\hf_ctgan_syn_10000_3.csv',
    'syn_data_ctgan_syn_20k': f'{BASE_PATH}\\hf_ctgan_syn_20000_3.csv',
    'syn_data_ctgan_syn_30k': f'{BASE_PATH}\\hf_ctgan_syn_30000_3.csv',
    'syn_data_copula_base_10k': f'{BASE_PATH}\\hf_copula_base_10000_3.csv',
    'syn_data_copula_base_20k': f'{BASE_PATH}\\hf_copula_base_20000_3.csv',
    'syn_data_copula_base_30k': f'{BASE_PATH}\\hf_copula_base_30000_3.csv',
    'syn_data_copula_syn_10k': f'{BASE_PATH}\\hf_copula_syn_10000_3.csv',
    'syn_data_copula_syn_20k': f'{BASE_PATH}\\hf_copula_syn_20000_3.csv',
    'syn_data_copula_syn_30k': f'{BASE_PATH}\\hf_copula_syn_30000_3.csv',
    # 'syn_data_copula_syn_100k_2': f'{BASE_PATH}/hf_copula_syn_100000_2.csv',
    # 'syn_data_copula_syn_150k_2': f'{BASE_PATH}/hf_copula_syn_150000_2.csv',
    # 'syn_data_copula_syn_300k_2': f'{BASE_PATH}/hf_copula_syn_300000_2.csv',
    ## 'syn_data_tvae_base_100k': f'{BASE_PATH}/hf_tvae_base_100000.csv',
    ## 'syn_data_tvae_base_150k': f'{BASE_PATH}/hf_tvae_base_150000.csv',
    ## 'syn_data_tvae_base_300k': f'{BASE_PATH}/hf_tvae_base_300000.csv',
    ## 'syn_data_tvae_syn_100k': f'{BASE_PATH}/hf_tvae_syn_100000.csv',
    ## 'syn_data_tvae_syn_150k': f'{BASE_PATH}/hf_tvae_syn_150000.csv',
    ## 'syn_data_tvae_syn_300k': f'{BASE_PATH}/hf_tvae_syn_300000.csv'
}
'''
with open('hb_metadata.json') as f:
    # with open('cd_metadata.json') as f:
    metadata = json.load(f)
'''
# 합성 데이터에 대한 평가
for index, (name, file_path) in enumerate(synthetic_data_files.items()):
    synthetic_data = pd.read_csv(file_path)
    if 'WD_NODE' in synthetic_data.columns:
        synthetic_data = synthetic_data.drop(columns=['WD_NODE', 'DPS_NODE', 'tran_dt'])
    # synthetic_data['tran_dt'] = synthetic_data['tran_dt'].str.replace('-', '')
    # synthetic_data['tran_dt'] = pd.to_datetime(synthetic_data['tran_dt']).astype('int64') / 10 ** 9
    print("★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★")
    print(f"{index}  {name}\n- column 리스트:  {synthetic_data.columns}")
    print(f"- 레코드 수: {len(synthetic_data)}\n")
    print(synthetic_data.head(5))
    target_column = 'ff_sp_ai'  # column to use as target for classification metrics and coloration of PCA plot.
    categorical_columns = ['tran_tmrg', 'md_type', 'fnd_type', 'ff_sp_ai']

    ### First SynthEval object is created then run with the "full_eval" presets file.
    S = SynthEval(original_data, cat_cols=categorical_columns)
    results = S.evaluate(synthetic_data, target_column, presets_file="fast_eval")  # The _ is for Jupyter purposes only, to avoid printing the results dictionary as well
    print("★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★")
    print(results)
    print("FIN\n★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★")







