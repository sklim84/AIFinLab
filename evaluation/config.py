import argparse

parser = argparse.ArgumentParser()

# Commons
parser.add_argument('--data_name', type=str, default='hf_sample_10000')
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--seed', type=int, default=2023)
parser.add_argument('--metric_save_path', type=str, default='./_results/result_default.csv')
parser.add_argument('--learning_rate', type=float, default=0.1)  # ALL
parser.add_argument('--boosting_type', type=str, default='Plain')  # CatBoost : Ordered, Plain, LGBM: gbdt
parser.add_argument('--max_depth', type=int, default=25)  # HGBoost, LGBM, XGBoost
parser.add_argument('--n_estimators', type=int, default=100)  # LGBM, XGBoost
parser.add_argument('--reg_lambda', type=float, default=0.0)  # LGBM, XGBoost
parser.add_argument('--n_jobs', type=int, default=4)  # LGBM, XGBoost

# CatBoost
parser.add_argument('--iterations', type=int, default=500)
parser.add_argument('--depth', type=int, default=4)
parser.add_argument('--l2_leaf_reg', type=float, default=3.0)
parser.add_argument('--max_ctr_complexity', type=int, default=4)
# On GPU loss MultiClass can't be used with ordered boosting
parser.add_argument('--od_type', type=str, default='Iter')

# HGBoost
parser.add_argument('--loss', type=str, default='log_loss')
parser.add_argument('--max_iter', type=int, default=100)
parser.add_argument('--max_leaf_nodes', type=int, default=31)
parser.add_argument('--l2_regularization', type=float, default=0.0)

# LGBM
parser.add_argument('--num_leaves', type=int, default=31)

# XGBoost
parser.add_argument('--gamma', type=float, default=0.0)


def get_config():
    return parser.parse_args()


def get_params(model):
    pp = 0
    for p in list(model.parameters()):
        nn = 1
        for s in list(p.size()):
            nn = nn * s
        pp += nn
    return pp
