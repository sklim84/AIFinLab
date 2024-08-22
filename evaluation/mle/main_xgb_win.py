import os
import sys

parent_path = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(parent_path)

import pandas as pd
from imblearn.metrics import classification_report_imbalanced

from sklearn.metrics import roc_auc_score, average_precision_score, make_scorer
from sklearn.metrics import f1_score

from utils import save_metrics, exists_metrics, weighted_f1
from config import get_config

from sklearn.model_selection import train_test_split, GridSearchCV
import xgboost as xgb

if __name__ == "__main__":
    args = get_config()
    print(args)

    arg_names = ['origin_data_name', 'data_name', 'n_estimators', 'learning_rate', 'max_depth', 'gamma', 'reg_lambda']

    # cd
    data_names = [
                  # 'cd_training',
                  # 'cd_copula_base_10000_3',
                  # 'cd_copula_base_20000_3',
                  # 'cd_copula_base_30000_3',
                  # 'cd_ctgan_base_10000_3',
                  # 'cd_ctgan_base_20000_3',
                  # 'cd_ctgan_base_30000_3',
                  # 'cd_copula_syn_10000_3',
                  'cd_copula_syn_20000_3',
                  'cd_copula_syn_30000_3',
                  'cd_ctgan_syn_10000_3',
                  'cd_ctgan_syn_20000_3',
                  'cd_ctgan_syn_30000_3']
    # hf
    # data_names = [
    #     'hf_ctgan_syn_10000KG_10k_1', 'hf_ctgan_syn_10000KG_10k_000001',
    #     'hf_ctgan_syn_10000KG_basic_0001', 'hf_ctgan_syn_10000KG_basic_01',
    #     'hf_ctgan_base_10000KG_base_1'
    # ]

    # hf_ctgan_training (10만개), hf_training (1만개), cd_training 1만개
    # origin_data_name = 'hf_training'
    origin_data_name = 'cd_training'

    args.metric_save_path = '_results/results_xgb.csv'

    for data_name in data_names:
        args.data_name = data_name

        if exists_metrics(args.metric_save_path, args, arg_names):
            print(f'There exist experiment results! - {args}')
            continue

        df_hf_trns_tran = pd.read_csv(os.path.join(args.data_path, f'{data_name}.csv'), dtype={'ff_sp_ai': 'str'})
        if 'WD_NODE' in df_hf_trns_tran.columns or 'DPS_NODE' in df_hf_trns_tran.columns:
            df_hf_trns_tran.drop(columns=['WD_NODE', 'DPS_NODE'], inplace=True)

        df_hf_trns_tran['ff_sp_ai'] = df_hf_trns_tran['ff_sp_ai'].replace('SP', '1')
        df_hf_trns_tran['ff_sp_ai'] = df_hf_trns_tran['ff_sp_ai'].replace('01', '1')
        df_hf_trns_tran['ff_sp_ai'] = df_hf_trns_tran['ff_sp_ai'].replace('02', '1')
        df_hf_trns_tran['ff_sp_ai'] = df_hf_trns_tran['ff_sp_ai'].replace(pd.NA, '0')

        # cd
        column_type = {
            'tran_dt': 'category',
            'tran_tmrg': 'category',
            'atm_sn': 'category',
            'hdl_fc_sn':'category',
            'wd_fc_sn': 'category',
            'pay_ac_sn': 'category',
            'dps_fc_sn': 'category',
            'dps_ac_sn': 'category',
            'tran_code': 'category',
            'tran_amt': 'float',
            'fee': 'float',
            'ff_sp_ai': 'int'
        }

        # hf
        # column_type = {
        #     'tran_dt': 'category',
        #     'tran_tmrg': 'category',
        #     'wd_fc_sn': 'category',
        #     'wd_ac_sn': 'category',
        #     'dps_fc_sn': 'category',
        #     'dps_ac_sn': 'category',
        #     'tran_amt': 'float',
        #     'md_type': 'category',
        #     'fnd_type': 'category',
        #     'ff_sp_ai': 'int'
        # }

        df_hf_trns_tran = df_hf_trns_tran.astype(column_type)

        df_train_data, df_eval_data = train_test_split(df_hf_trns_tran, test_size=0.3)
        df_valid_data, df_test_data = train_test_split(df_eval_data, test_size=0.5)

        # 테스트 데이터셋은 원천 데이터로 설정(pp_split_testset.py에서 작업)
        if data_name != origin_data_name:
            print('##### load original testsets')

            df_test_data = pd.read_csv(os.path.join(args.data_path, f'{origin_data_name}_testset.csv'))
            if 'WD_NODE' in df_test_data.columns or 'DPS_NODE' in df_test_data.columns:
                df_test_data.drop(columns=['WD_NODE', 'DPS_NODE'], inplace=True)
            df_test_data['ff_sp_ai'] = df_test_data['ff_sp_ai'].replace('SP', '1')
            df_test_data['ff_sp_ai'] = df_test_data['ff_sp_ai'].replace('01', '1')
            df_test_data['ff_sp_ai'] = df_test_data['ff_sp_ai'].replace('02', '1')
            df_test_data['ff_sp_ai'] = df_test_data['ff_sp_ai'].replace(pd.NA, '0')
            df_test_data = df_test_data.astype(column_type)

        # train dataset
        train_labels = df_train_data['ff_sp_ai']
        df_train_data.drop(columns=['ff_sp_ai'], inplace=True)

        valid_labels = df_valid_data['ff_sp_ai']
        df_valid_data.drop(columns=['ff_sp_ai'], inplace=True)

        # test dataset
        test_labels = df_test_data['ff_sp_ai']
        df_test_data.drop(columns=['ff_sp_ai'], inplace=True)

        # 컬럼 순서를 동일하게 정렬
        columns_order = df_train_data.columns.tolist()
        df_test_data = df_test_data[columns_order]

        # model = xgb.XGBClassifier(
        #     # tree_method='gpu_hist',
        #     n_estimators=args.n_estimators,
        #     learning_rate=args.learning_rate,
        #     max_depth=args.max_depth,
        #     gamma=args.gamma,
        #     reg_lambda=args.reg_lambda,
        #     enable_categorical=True,
        #     max_cat_to_onehot=1,
        #     n_jobs=args.n_jobs,
        #     random_state=args.seed
        # )
        #
        # model.fit(df_train_data, train_labels, eval_set=[(df_valid_data, valid_labels)])
        # pred_labels = model.predict(df_test_data)

        # param_grid = {
        #     'n_estimators': [100, 500, 1000, 1500, 2000],
        #     'learning_rate': [0.01, 0.1, 0.001],
        #     'max_depth': [2, 4, 6, 8, 10],
        #     'reg_lambda': [0.0, 0.1, 1.0, 5.0, 10.0]
        # }

        param_grid = {
            'n_estimators': [100, 500],
            'learning_rate': [0.01, 0.001],
            'max_depth': [2, 4],
            'reg_lambda': [0.0, 1.0]
        }

        model = xgb.XGBClassifier(
            tree_method='hist',
            enable_categorical=True,
            max_cat_to_onehot=1,
            n_jobs=args.n_jobs,
            random_state=args.seed
        )

        scorer = make_scorer(roc_auc_score, needs_proba=True, multi_class='ovr')

        grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, scoring=scorer, verbose=3)
        grid_search.fit(df_train_data, train_labels, eval_set=[(df_valid_data, valid_labels)])

        print(f"Best parameters found for {data_name}: {grid_search.best_params_}")
        print(f"Best ROC-AUC score for {data_name}: {grid_search.best_score_}")

        best_model = grid_search.best_estimator_
        pred_labels = best_model.predict(df_test_data)

        im_metrics = classification_report_imbalanced(test_labels, pred_labels, digits=5, output_dict=True)

        auc, ap, f1_macro, f1_weighted, auc_ovr = None, None, None, None, None
        try:
            auc = roc_auc_score(test_labels, pred_labels)
        except:
            print('roc-auc metric exception!')
        try:
            ap = average_precision_score(test_labels, pred_labels)
        except:
            print('ap metric exception!')
        try:
            f1_macro = f1_score(test_labels, pred_labels, average='macro')
        except:
            print('f1_macro metric exception!')
        try:
            f1_weighted = weighted_f1(test_labels, pred_labels)
        except:
            print('f1_weighted metric exception!')
        try:
            auc_ovr = roc_auc_score(test_labels, pred_labels, multi_class='ovr')
        except:
            print('auc_ovr metric exception!')

        metric_names = ['0_pre', '0_rec', '0_spe', '0_f1', '0_geo', '0_iba',
                        '1_pre', '1_rec', '1_spe', '1_f1', '1_geo', '1_iba',
                        'avg_pre', 'avg_rec', 'avg_spe', 'avg_f1', 'avg_geo', 'avg_iba',
                        'auc', 'ap', 'f1_macro', 'f1_weighted', 'auc_ovr']

        metrics = [im_metrics['0']['pre'], im_metrics['0']['rec'], im_metrics['0']['spe'], im_metrics['0']['f1'],
                   im_metrics['0']['geo'], im_metrics['0']['iba'],
                   im_metrics['1']['pre'], im_metrics['1']['rec'], im_metrics['1']['spe'], im_metrics['1']['f1'],
                   im_metrics['1']['geo'], im_metrics['1']['iba'],
                   im_metrics['avg_pre'], im_metrics['avg_rec'], im_metrics['avg_spe'], im_metrics['avg_f1'],
                   im_metrics['avg_geo'], im_metrics['avg_iba'],
                   auc, ap, f1_macro, f1_weighted, auc_ovr]
        save_metrics(args.metric_save_path, args, arg_names, metrics, metric_names)
