import pandas as pd
import numpy as np
from datetime import datetime
pd.set_option('display.max_columns', None)

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

X_train_smote = X_train
y_train_smote = y_train

# # 3. SMOTE 적용 (훈련 데이터에 대해서만 적용, smote는 1대 5 정도까지면 유효함.)
# # Smote 처리 전 데이터에 임시 분리
#
# X_train['wd_fc_sn'] = X_train['wd_acnt'].str.split('-').str[0]
# X_train['wd_ac_sn'] = X_train['wd_acnt'].str.split('-').str[1]
# X_train['dps_fc_sn'] = X_train['dps_acnt'].str.split('-').str[0]
# X_train['dps_ac_sn'] = X_train['dps_acnt'].str.split('-').str[1]
# X_train = X_train.drop(['tran_dtm', 'wd_acnt', 'dps_acnt'], axis=1)
#
# # smote 적용 (랜덤상태 고정)
# smote = SMOTE(random_state=42)
# X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
#
# X_train_smote['tran_mm'] = X_train_smote['tran_mm'].round().astype(int)
# X_train_smote['tran_dd'] = X_train_smote['tran_dd'].round().astype(int)
# X_train_smote['tran_hour'] = X_train_smote['tran_hour'].round().astype(int)
# X_train_smote['is_weekend'] = X_train_smote['is_weekend'].round().astype(int)
#
# X_train_smote['wd_fc_sn'] = X_train_smote['wd_fc_sn'].round().astype(int)
# X_train_smote['wd_ac_sn'] = X_train_smote['wd_ac_sn'].round().astype(int)
# X_train_smote['dps_fc_sn'] = X_train_smote['dps_fc_sn'].round().astype(int)
# X_train_smote['dps_ac_sn'] = X_train_smote['dps_ac_sn'].round().astype(int)
#
# mark_time_stamp('3. Smote Data')

# 4. 모델 설정 (모델 하이퍼파라미터 튜닝)
models = {
    'XGBoost': XGBClassifier(eval_metric="logloss"),
    'RandomForest': RandomForestClassifier(random_state=42),
    'CatBoost': CatBoostClassifier(silent=True, random_state=42)
}

X_train_smote['wd_acnt'] = X_train_smote['wd_fc_sn'].astype(str) + '-' + X_train_smote['wd_ac_sn'].astype(str)
X_train_smote['dps_acnt'] = X_train_smote['dps_fc_sn'].astype(str) + '-' + X_train_smote['dps_ac_sn'].astype(str)

df = df.drop(['tran_mm', 'tran_dd', 'tran_hour', 'wd_fc_sn', 'wd_ac_sn', 'dps_fc_sn', 'dps_ac_sn'], axis=1)

# 그리드 서치로 하이퍼파라미터 튜닝
param_grids = {
    'XGBoost': {
        'n_estimators': [100, 200],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.2],
        'subsample': [0.8, 1.0]
    },
    'RandomForest': {
        'n_estimators': [100, 200],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5, 10]
    },
    'CatBoost': {
        'iterations': [100, 200],
        'depth': [4, 6, 10],
        'learning_rate': [0.01, 0.1, 0.2]
    }
}

eval_list = []
for model_name, model in models.items():
    print(f"--- {model_name} ---")

    # 그리드 서치를 통한 하이퍼파라미터 튜닝
    grid_search = GridSearchCV(estimator=model, param_grid=param_grids[model_name], cv=3, scoring='roc_auc', verbose=1, n_jobs=-1)
    grid_search.fit(X_train_smote, y_train_smote)

    print("3. Training & Optimizing:", grid_search.best_params_)

    # 최적의 모델로 예측
    best_model = grid_search.best_estimator_

    y_pred = best_model.predict(X_test)
    y_pred_proba = best_model.predict_proba(X_test)[:, 1]
    mark_time_stamp('4. Predict')

    # 5. AUC와 F1-Score 측정
    eval_list.append({
        'model name': model_name,
        'auc': roc_auc_score(y_test, y_pred_proba),
        'f1-macro': f1_score(y_test, y_pred, average='macro'),
        'f1-weighted': f1_score(y_test, y_pred, average='weighted')
    })

print(f"Evaluation Results: {pd.DataFrame(eval_list)}")
