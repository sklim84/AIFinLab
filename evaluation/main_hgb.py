import sys
import os

parent_path = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(parent_path)

import pandas as pd
from imblearn.metrics import classification_report_imbalanced
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score

from utils import save_metrics, exists_metrics
from config import get_config

if __name__ == "__main__":
    args = get_config()

    arg_names = ['data_name', 'max_iter', 'learning_rate', 'max_depth', 'max_leaf_nodes', 'l2_regularization']
    if exists_metrics(args.metric_save_path, args, arg_names):
        print(f'There exist experiment results! - {args}')
        sys.exit()

    data_name = args.data_name
    cat_features = ['md_type', 'fnd_type']
    df_hf_trns_tran = pd.read_csv(f'../datasets/{data_name}.csv')
    if '_ST_' in data_name:
        df_hf_trns_tran.drop(columns=['Source', 'Target'], inplace=True)
    df_hf_trns_tran['ff_sp_ai'] = df_hf_trns_tran['ff_sp_ai'].replace('SP', '03')
    df_hf_trns_tran['ff_sp_ai'] = df_hf_trns_tran['ff_sp_ai'].replace(pd.NA, '00')

    # df_hf_trns_tran['tran_dt'] = df_hf_trns_tran['tran_dt'].str.replace('-', '', regex=False).astype(int)

    df_train_data, df_test_data = train_test_split(df_hf_trns_tran, test_size=0.15, shuffle=False)
    print(f'train data: {len(df_train_data)}')
    print(f'test data: {len(df_test_data)}')

    cat_features = ['tran_tmrg', 'wd_fc_sn', 'dps_fc_sn', 'md_type', 'fnd_type']
    # train dataset
    train_labels = df_train_data['ff_sp_ai'].to_numpy()
    df_train_data.drop(columns=['ff_sp_ai'], inplace=True)

    # test dataset
    test_labels = df_test_data['ff_sp_ai'].to_numpy()
    df_test_data.drop(columns=['ff_sp_ai'], inplace=True)



    model = HistGradientBoostingClassifier(
        loss=args.loss,
        learning_rate=args.learning_rate,
        max_iter=args.max_iter,
        max_leaf_nodes=args.max_leaf_nodes,
        l2_regularization=args.l2_regularization,
        max_depth=args.max_depth,
        early_stopping=True,
        validation_fraction=0.215,
        random_state=args.seed,
        categorical_features=cat_features,
        verbose=1
    )
    model.fit(df_train_data, train_labels)
    pred_labels = model.predict(df_test_data)

    im_metrics = classification_report_imbalanced(test_labels, pred_labels, digits=5, output_dict=True)

    auc, ap = None, None
    try:
        auc = roc_auc_score(test_labels, pred_labels)
        ap = average_precision_score(test_labels, pred_labels)
    except:
        print('roc-auc or ap metric exception!')

    metric_names = ['00_pre', '00_rec', '00_spe', '00_f1', '00_geo', '00_iba',
                    '01_pre', '01_rec', '01_spe', '01_f1', '01_geo', '01_iba',
                    '02_pre', '02_rec', '02_spe', '02_f1', '02_geo', '02_iba',
                    '03_pre', '03_rec', '03_spe', '03_f1', '03_geo', '03_iba',
                    'avg_pre', 'avg_rec', 'avg_spe', 'avg_f1', 'avg_geo', 'avg_iba',
                    'auc', 'ap']

    metrics = [im_metrics['00']['pre'], im_metrics['00']['rec'], im_metrics['00']['spe'], im_metrics['00']['f1'],
               im_metrics['00']['geo'], im_metrics['00']['iba'],
               im_metrics['01']['pre'], im_metrics['01']['rec'], im_metrics['01']['spe'], im_metrics['01']['f1'],
               im_metrics['01']['geo'], im_metrics['01']['iba'],
               im_metrics['02']['pre'], im_metrics['02']['rec'], im_metrics['02']['spe'], im_metrics['02']['f1'],
               im_metrics['02']['geo'], im_metrics['02']['iba'],
               im_metrics['03']['pre'], im_metrics['03']['rec'], im_metrics['03']['spe'], im_metrics['03']['f1'],
               im_metrics['03']['geo'], im_metrics['03']['iba'],
               im_metrics['avg_pre'], im_metrics['avg_rec'], im_metrics['avg_spe'], im_metrics['avg_f1'],
               im_metrics['avg_geo'], im_metrics['avg_iba'],
               auc, ap]
    save_metrics(args.metric_save_path, args, arg_names, metrics, metric_names)
