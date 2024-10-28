import itertools
import pandas as pd
import numpy as np
import math
from datetime import datetime
import shap

pd.set_option("display.max_columns", None)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, f1_score
from imblearn.over_sampling import SMOTE

from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

time_stamp = datetime.now()

def mark_time_stamp(step_name):
    global time_stamp
    ms_difference = (datetime.now() - time_stamp).total_seconds()
    print(f"{step_name} : {ms_difference:.2f} sec")
    time_stamp = datetime.now()

data = pd.read_csv('./HF_TRNS_TRAN_selected.csv')
df = pd.DataFrame(data)

# Remove non-numeric columns
df = df.drop(['tran_dtm', 'wd_acnt', 'dps_acnt'], axis=1)
mark_time_stamp('1. Load Data')

# 2. 입력 값 타겟 변수 분할 (ff_sp_ai는 타겟 변수를 가정)
y = df['ff_sp_ai']  # 타겟 변수
X = df.drop('ff_sp_ai', axis=1)

# 데이터 스케일링 (필요시)
scaler = StandardScaler()
numeric_cols = X.select_dtypes(include=['float64', 'int64'])
X_scaled = scaler.fit_transform(numeric_cols)
X[numeric_cols.columns] = X_scaled

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
mark_time_stamp('2. Split Data')

# 3. Basic Model
model = XGBClassifier(eval_metric="logloss", learning_rate=0.2, max_depth=3, n_estimators=100, subsample=0.8)
model.fit(X_train, y_train)

# 전체 예측값 (모든 특징을 포함한 상태에서의 예측)
base_predictions = model.predict_proba(X_test)[:, 1]

# 4. 특정한 집단을 포함한 혹은 제외한 예측을 계산하는 함수
def get_predictions_with_group(X, group_cols, model):
    X_modified = X.copy()
    X_modified.loc[:, ~X_modified.columns.isin(group_cols)] = np.nan
    return model.predict_proba(X_modified)[:, 1]

def get_predictions_without_group(X, group_cols, model):
    X_modified = X.copy()
    X_modified.loc[:, X_modified.columns.isin(group_cols)] = np.nan
    return model.predict_proba(X_modified)[:, 1]


def detail_shap(X_test, cols, model):
    n_features = len(cols)
    shapley_values = np.zeros(n_features)
    subset_cache = {}

    # 모든 피처의 조합에 대해 샤플리 값을 계산
    for i in range(n_features):
        # 피처 i를 포함하지 않고 포함한 모든 조합에 대해 기여도 계산
        for subset_size in range(n_features):
            # 피처 i를 제외한 서브셋 생성
            subsets = list(itertools.combinations([j for j in range(n_features) if j != i], subset_size))

            for subset in subsets:
                # 서브셋에 대한 예측값 계산
                subset_key = tuple(sorted(subset))  # 정렬된 튜플로 캐시 키 생성

                if subset_key not in subset_cache:
                    subset_value = get_predictions_without_group(X_test, subset, model)
                    subset_cache[subset_key] = subset_value
                else:
                    subset_value = subset_cache[subset_key]

                # 서브셋 + 피처 i에 대한 예측값 계산
                subset_with_i = list(subset) + [i]
                subset_with_i_key = tuple(sorted(subset_with_i))

                if subset_with_i_key not in subset_cache:
                    subset_with_i_value = get_predictions_without_group(X_test, subset_with_i, model)
                    subset_cache[subset_with_i_key] = subset_with_i_value
                else:
                    subset_with_i_value = subset_cache[subset_with_i_key]

                # 피처 i의 기여도는 해당 서브셋에 예측값 차이에 가중치를 곱한 값
                weight = math.factorial(len(subset)) * math.factorial(
                    n_features - len(subset) - 1) / math.factorial(n_features)
                shapley_values[i] += weight * (subset_with_i_value - subset_value)

    return shapley_values

def group_shap(X_test, grouping, model):
    n_group = len(grouping)
    group_names = list(grouping.keys())
    group_list = list(grouping.values())
    contributions = {group: 0 for group in grouping.keys()}
    subset_cache = {}

    # 각 그룹에 대해 계산 수행
    for i in range(n_group):
        # 피처 i를 제외한 나머지 그룹을 이용해 조합 가능한 피처 셋
        for subset_size in range(n_group):
            subsets = list(itertools.combinations([j for j in range(n_group) if j != i], subset_size))

            for subset in subsets:
                # 서브셋에 대한 예측값 계산
                subset_key = tuple(sorted(subset))
                subset_indices = [index for group in subset for index in group_list[group]]

                if subset_key not in subset_cache:
                    subset_value = get_predictions_with_group(X_test, subset_indices, model)
                    subset_cache[subset_key] = subset_value
                else:
                    subset_value = subset_cache[subset_key]

                # 서브셋 + 피처 i에 대한 예측값 계산
                subset_with_i = list(subset) + [i]
                subset_with_i_key = tuple(sorted(subset_with_i))

                if subset_with_i_key not in subset_cache:
                    if subset_indices:
                        subset_with_i_indices = subset_indices + group_list[i]
                    else:
                        subset_with_i_indices = group_list[i]
                    subset_with_i_value = get_predictions_with_group(X_test, subset_with_i_indices, model)
                    subset_cache[subset_with_i_key] = subset_with_i_value
                else:
                    subset_with_i_value = subset_cache[subset_with_i_key]

                # 가중치 계산
                weight = (math.factorial(len(subset)) * math.factorial(
                    n_group - len(subset) - 1) / math.factorial(n_group))
                contributions[group_names[i]] += weight * (subset_with_i_value - subset_value)

    return contributions


group1 = {
    'date': ['tran_mm', 'tran_dt', 'tran_hour', 'is_weekend'],
    'amount': ['tran_amt'],
    'withdrawal_network': [col for col in data.columns if col.startswith('nx_wd')],
    'deposit_network': [col for col in data.columns if col.startswith('nx_dps')],
    'withdrawal_statistics': [col for col in data.columns if col.startswith('accl_wd')],
    'deposit_statistics': [col for col in data.columns if col.startswith('accl_dps')],
    'method_type': df.columns[df.columns.get_loc('md_type_1'):df.columns.get_loc('md_type_7') + 1],
    'fund_type': df.columns[df.columns.get_loc('fnd_type_0'):df.columns.get_loc('fnd_type_12') + 1]
}

group_contributions = group_shap(X_test, group1, model)
mean_group_contributions = {key: np.sum(np.abs(value)) for key, value in group_contributions.items()}

# 결과 출력
mark_time_stamp('3. Analyze group contributions : ')

for group, contribution in mean_group_contributions.items():
    print(f" - {group}: {contribution:.4f}")

group_list = list(group1.values())
for i in range(len(group1)):
    print(f"--- [{list(group1.keys())[i]}] ---")
    shap_values = detail_shap(X_test, group_list[i], model)
    mean_shap_values = {key: np.sum(np.abs(value)) for key, value in shap_values.items()}

    for feature, contribution in mean_shap_values.items():
        print(f" - {feature}: {contribution:.4f}")

# Tree SHAP
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)
mean_shap_values = np.mean(np.abs(shap_values), axis=0)

mark_time_stamp('4. Tree SHAP results : ')
feature_names = [f'{col}' for col in X_test.columns]
for i in range(len(mean_shap_values)):
    if mean_shap_values[i] > 0.01:
        print(f' - {feature_names[i]}: {mean_shap_values[i]}')

# Kernel SHAP
k_explainer = shap.KernelExplainer(lambda x: model.predict_proba(x), np.array(X_train.mean(axis=0)).reshape(1, -1))
k_shap_values = k_explainer.shap_values(X_test)

k_mean_shap_values = np.mean(np.abs(k_shap_values), axis=0)

mark_time_stamp('5. Kernel SHAP results : ')
for i in range(len(k_mean_shap_values)):
    if k_mean_shap_values[i] > 0.01:
        print(f' - {feature_names[i]}: {k_mean_shap_values[i]}')
