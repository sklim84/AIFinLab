import copy
import ctgan
import logging
import multiprocessing as mp
import numpy as np
import os
import pandas as pd
import pathlib
import yaml

from contextlib import redirect_stdout, redirect_stderr
from dataclasses import dataclass, field
from numpy import random
from sklearn import preprocessing
from sklearn import utils
from typing import Dict, Union
import datetime
from random_search import RandomValueTrial, suggest_callable_hyperparams

CATEGORICAL_FEATURES = [
    'source',
    'payment_type',
    'device_os',
    'housing_status',
    'employment_status',
    'month'
]

BOOLEAN_FEATURES = [
    'email_is_free',
    'fraud_bool',
    'foreign_request',
    'keep_alive_session',
    'phone_home_valid',
    'phone_mobile_valid',
    'has_other_cards',
]

N_RUNS = 4

TARGET_FPR = 0.05

timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
args = {
    'run_dir': f'./run_dir_{timestamp}',
    'config_path': 'config-rs.yml',
    'n_procs': 4,
    'devices': ["cuda:0", "cuda:1", "cuda:2", "cuda:3"],
    'log_level': "DEBUG",
    'seed': 42,
    'dry_run': False,
    'dev_run': False,
    'n_trials': 100,
}

@dataclass
class RunConfig:
    """Dataclass with required information to train a model."""
    model_id: int

    train_df: pd.DataFrame = field(repr=False)
    val_df: pd.DataFrame = field(repr=False)
    discrete_columns: list

    model_run_dir: str

    config: Dict

    seed: Union[int, None]


def configure_logging(log_arg):
    received_level = getattr(logging, log_arg.upper(), None)

    logging_level = received_level if received_level else logging.INFO

    logging.basicConfig(
        format='[ %(levelname)s ] %(asctime)s (%(process)s-%(processName)s) %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        level=logging_level
    )

    if not received_level:
        logging.warning('Unknown logging level %s: Setting logging to INFO', log_arg.upper())

def create_run_dir(run_dir):
    if run_dir.exists():
        logging.error('Run Directory already exists: \'%s\'', run_dir)
        exit(1)

    os.mkdir(run_dir)

    logging.info('Run results stored at: \'%s\'', run_dir)

def read_configurations(config_path):
    logging.info('Reading configurations from %s', config_path)

    with open(config_path, 'r') as f:
        configs = yaml.safe_load(f)

    return configs['data'], configs['sweep_params']

def load_data(data_config):

    logging.info('Loading train dataset from \'%s\'', data_config['train'])
    logging.info('Loading validation dataset from \'%s\'', data_config['validation'])

    train_df = pd.read_csv(data_config['train'], index_col=0)
    val_df = pd.read_csv(data_config['validation'], index_col=0)

    print(train_df.head())
    print(val_df.head())

    if 'keep' in data_config:
        train_df = train_df[data_config['keep']]
        val_df = val_df[data_config['keep']]
    elif 'remove' in data_config:
        train_df = train_df.drop(columns=data_config['remove'])
        val_df = val_df.drop(columns=data_config['remove'])

    discrete_columns = [f for f in CATEGORICAL_FEATURES + BOOLEAN_FEATURES if f in train_df.columns]

    logging.info('Train Dataset: %s Features, %s Rows', len(train_df.columns), len(train_df))
    logging.info('Validation Dataset: %s Features, %s Rows', len(val_df.columns), len(val_df))
    logging.debug('Train Features: %s', list(train_df.columns))
    logging.debug('Validation features: %s', list(val_df.columns))
    logging.debug('Discrete columns: %s', discrete_columns)

    return train_df, val_df, discrete_columns

# Just path functions

def pad_int(model_id, zfill=3):
    return str(model_id).zfill(zfill)

def model_run_dir(run_dir, model_id):
    return run_dir / pad_int(model_id)

def config_path(model_run_dir, model_id):
    return model_run_dir / f'config-{pad_int(model_id)}.yml'


def model_path(model_run_dir, model_id):
    return model_run_dir / f'model-{pad_int(model_id)}.pkl'


def train_dataset_path(model_run_dir, model_id):
    return model_run_dir / f'train-dataset-{pad_int(model_id)}.csv'


def synthetic_dataset_path(model_run_dir, model_id):
    return model_run_dir / f'synthetic-dataset-{pad_int(model_id)}.csv'


def model_evaluation_path(model_run_dir, model_id):
    return model_run_dir / f'evaluation-{pad_int(model_id)}.csv'


def stdout_path(model_run_dir, model_id):
    return model_run_dir / f'stdout-{pad_int(model_id)}.log'


def stderr_path(model_run_dir, model_id):
    return model_run_dir/ f'stderr-{pad_int(model_id)}.log'

def build_run_configs(
        run_dir: str,
        datasets_config: dict,
        data_sweep_params: dict,
        model_sweep_params: dict,
        devices: list,
        n_trials: int,
        seed: int,
) -> list:    
    train_df, val_df, discrete_columns = load_data(datasets_config)

    run_configs = []
    
    random.seed(seed)
    seeds = random.randint(n_trials*1000, size=n_trials)
    for i, seed  in enumerate(seeds, start=1):
        # Method to random sample configurations
        print(f'##### data_sweep_params: {data_sweep_params}')
        configs_data =  suggest_callable_hyperparams(RandomValueTrial(seed=seed), data_sweep_params)
        configs_model = suggest_callable_hyperparams(RandomValueTrial(seed=seed), model_sweep_params['kwargs'])
        configs_model['generator_dim'] = eval(configs_model['generator_dim'])
        configs_model['discriminator_dim'] = eval(configs_model['discriminator_dim'])
        configs_model['generator_decay'] = eval(configs_model['generator_decay'])
        configs_model['discriminator_decay'] = eval(configs_model['discriminator_decay'])
        configs_model['cuda'] = devices[i % len(devices)]
        config = {"data": configs_data, "model": configs_model}
        run_configs.append(
            RunConfig(
                model_id=i,
                train_df=train_df,
                val_df=val_df,
                discrete_columns=discrete_columns,
                model_run_dir=model_run_dir(run_dir, i),
                config=config,
                seed=seed,
            )
        )


    return run_configs

def subsample_with_prevalence(df, prevalence, seed):
    if prevalence:
        fraud = df[df['fraud_bool'] == 1]
        non_fraud = df[df['fraud_bool'] == 0]

        fraud_proportion, non_fraud_proportion = prevalence
        non_fraud_instances = (len(fraud) * non_fraud_proportion) // fraud_proportion
        if non_fraud_instances >= len(non_fraud):
            logging.warning(
                'Unable to subsample dataframe: Expected more than %s negative examples but got %s',
                non_fraud_instances,
                len(fraud)
            )
            non_fraud_sample = non_fraud
        else:
            non_fraud_sample = non_fraud.sample(n=non_fraud_instances, random_state=seed)
        return utils.shuffle(pd.concat((fraud, non_fraud_sample)), random_state=seed)
    else:
        return df


def apply_config_to_data(df, data_config, model_id, seed):

    if 'prevalence' in data_config:
        df = subsample_with_prevalence(df, eval(data_config['prevalence']), seed)

    if logging.root.isEnabledFor(logging.DEBUG):
        logging.debug(
            'Model %s: Dataset with %s Examples (%s fraud, %s non fraud)',
            pad_int(model_id),
            len(df),
            len(df[df['fraud_bool'] == 1]),
            len(df[df['fraud_bool'] == 0])
        )

    return df


def preprocess_categorical(train_df, val_df, discrete_columns):

    categorical_columns = np.intersect1d(discrete_columns, CATEGORICAL_FEATURES)

    for column in categorical_columns:
        train_unique = train_df[column].unique()
        val_unique = val_df[column].unique()
        nans = np.setdiff1d(val_unique, train_unique)
        val_df.loc[val_df[column].isin(nans), [column]] = np.nan

    train_dummy = pd.get_dummies(train_df, columns=categorical_columns, dummy_na=True)
    val_dummy = pd.get_dummies(val_df, columns=categorical_columns, dummy_na=True)

    for unseen_column in np.setdiff1d(train_dummy.columns, val_dummy.columns):
        val_dummy[unseen_column] = 0

    return train_dummy, val_dummy

def split(train_df, val_df, target):
    train_x = train_df.drop(columns=[target])
    train_y = train_df[target]
    val_x = val_df.drop(columns=[target])
    val_y = val_df[target]
    return train_x, train_y, val_x, val_y

def preprocess_and_split(train_df, val_df, discrete_columns, target):
    train_dummy_df, val_dummy_df = preprocess_categorical(train_df, val_df, discrete_columns)
    return split(train_dummy_df, val_dummy_df, target)


def class_index(model, class_value):
    return np.argwhere(model.classes_ == class_value)[0]


def prediction_probabilities(model, x):
    return model.predict_proba(x)[:, class_index(model, 1)]


def ordinal_encode(train_df, val_df, categorical_features):
    for f in categorical_features:
        enc = preprocessing.OrdinalEncoder()
        train_df[f] = enc.fit_transform(train_df[[f]])
        val_df[f] = enc.fit_transform(val_df[[f]])
    return train_df, val_df


def compile_results(
        real_fprs,
        real_tprs,
        real_thresholds,
        synthetic_train_fprs,
        synthetic_train_tprs,
        synthetic_train_thresholds,
        synthetic_val_fprs,
        synthetic_val_tprs,
        synthetic_val_thresholds,
        synthetic_both_fprs,
        synthetic_both_tprs,
        synthetic_both_thresholds):

    records = []
    for i, results in enumerate(zip(real_fprs, real_tprs, real_thresholds), start=1):
        for fpr, tpr, threshold in zip(*results):
            records.append((i, fpr, tpr, threshold, 'real'))

    for j, results in enumerate(zip(synthetic_train_fprs, synthetic_train_tprs, synthetic_train_thresholds), start=i+1):
        for fpr, tpr, threshold in zip(*results):
            records.append((j, fpr, tpr, threshold, 'synthetic-train'))

    for k, results in enumerate(zip(synthetic_val_fprs, synthetic_val_tprs, synthetic_val_thresholds), start=j+1):
        for fpr, tpr, threshold in zip(*results):
            records.append((k, fpr, tpr, threshold, 'synthetic-val'))

    for n, results in enumerate(zip(synthetic_both_fprs, synthetic_both_tprs, synthetic_both_thresholds), start=k+1):
        for fpr, tpr, threshold in zip(*results):
            records.append((n, fpr, tpr, threshold, 'synthetic-both'))

    return pd.DataFrame.from_records(records, columns=['run_id', 'fpr', 'tpr', 'threshold', 'discrimination'])


def summarize_results(
        real_fprs,
        real_tprs,
        real_thresholds,
        synthetic_train_fprs,
        synthetic_train_tprs,
        synthetic_train_thresholds,
        synthetic_val_fprs,
        synthetic_val_tprs,
        synthetic_val_thresholds,
        synthetic_both_fprs,
        synthetic_both_tprs,
        synthetic_both_thresholds):

    def compute_avg_tpr_and_threshold(fprs, tprs, thresholds):
        avg_tpr = 0
        avg_threshold = 0

        for run_fprs, run_tprs, run_thresholds in zip(fprs, tprs, thresholds):
            target_fpr_index = np.argwhere(run_fprs <= TARGET_FPR).ravel()[-1]
            avg_tpr += run_tprs[target_fpr_index] / N_RUNS
            avg_threshold += run_thresholds[target_fpr_index] / N_RUNS

        return avg_tpr, avg_threshold

    avg_real_tpr, avg_real_threshold = \
        compute_avg_tpr_and_threshold(real_fprs, real_tprs, real_thresholds)

    avg_synthetic_train_tpr, avg_synthetic_train_threshold = \
        compute_avg_tpr_and_threshold(synthetic_train_fprs, synthetic_train_tprs, synthetic_train_thresholds)

    avg_synthetic_val_tpr, avg_synthetic_val_threshold = \
        compute_avg_tpr_and_threshold(synthetic_val_fprs, synthetic_val_tprs, synthetic_val_thresholds)

    avg_synthetic_both_tpr, avg_synthetic_both_threshold = \
        compute_avg_tpr_and_threshold(synthetic_both_fprs, synthetic_both_tprs, synthetic_both_thresholds)

    return (
        avg_real_tpr,
        avg_real_threshold,
        avg_synthetic_train_tpr,
        avg_synthetic_train_threshold,
        avg_synthetic_val_tpr,
        avg_synthetic_val_threshold,
        avg_synthetic_both_tpr,
        avg_synthetic_both_threshold
    )


def run_instance(run_config: RunConfig):

    model_id = run_config.model_id

    train_df = run_config.train_df
    val_df = run_config.val_df
    discrete_columns = run_config.discrete_columns

    model_run_dir = pathlib.Path(run_config.model_run_dir)

    config = run_config.config
    config_save_path = config_path(model_run_dir, model_id)
    model_save_path = model_path(model_run_dir, model_id)
    model_evaluation_save_path = model_evaluation_path(model_run_dir, model_id)
    train_data_save_path = train_dataset_path(model_run_dir, model_id)
    synthetic_data_save_path = synthetic_dataset_path(model_run_dir, model_id)

    run_stdout_path = stdout_path(model_run_dir, model_id)
    run_stderr_path = stderr_path(model_run_dir, model_id)

    seed = run_config.seed

    logging.info('Model %s: Training started', pad_int(model_id))
    logging.debug('Model %s: Config %s', pad_int(model_id), config)
    logging.debug('Model %s: Saved config to \'%s\'', pad_int(model_id), config_save_path)
    logging.debug('Model %s: Stdout redirected to \'%s\'', pad_int(model_id), run_stdout_path)
    logging.debug('Model %s: Stderr redirected to \'%s\'', pad_int(model_id), run_stderr_path)

    data_config = config['data']
    model_config = config['model']


    df = pd.concat((train_df, val_df))
    df = utils.shuffle(df)

    df = apply_config_to_data(df, data_config, model_id, seed)

    discrete_columns = copy.copy(discrete_columns)


    os.mkdir(model_run_dir)

    df.to_csv(train_data_save_path)
    logging.debug('Model %s: Training data saved to %s', pad_int(model_id), train_data_save_path)

    with open(config_save_path, 'w') as fd:
        yaml.safe_dump(config, stream=fd, default_flow_style=False)

    model = ctgan.CTGANSynthesizer(**model_config)

    with open(run_stdout_path, 'w') as out_fd, open(run_stderr_path, 'w') as err_fd:
        with redirect_stdout(out_fd), redirect_stderr(err_fd):
            model.fit(df, discrete_columns)

    model.save(model_save_path)

    logging.info('Model %s: Saved model to \'%s\'', pad_int(model_id), model_save_path)

    synthetic_df = model.sample(len(df))
    synthetic_df.to_csv(synthetic_data_save_path)

    logging.info('Model %s: Saved synthetic dataset to \'%s\'', pad_int(model_id), synthetic_data_save_path)

    return model_id, synthetic_df

def run_experiment():
    run_dir = pathlib.Path(args['run_dir'])

    config_path = args['config_path']
    n_procs = args['n_procs']
    devices = args['devices']
    dry_run = args['dry_run']
    seed = args['seed']
    n_trials = args['n_trials']

    configure_logging(args['log_level'])

    if seed:
        seed = int(seed)
        logging.info('Using seed value: %s', seed)

    create_run_dir(run_dir)
    
    datasets_config, sweep_params = read_configurations(config_path)

    data_sweep_params = sweep_params['data']
    model_sweep_params = sweep_params['model']

    run_configs = build_run_configs(
        run_dir,
        datasets_config,
        data_sweep_params,
        model_sweep_params,
        devices,
        n_trials,
        seed
    )
    
    dfs = []
    with mp.Pool(n_procs) as pool:
        for synthetic_df in pool.imap_unordered(run_instance, run_configs):
            if not dry_run:
                dfs.append(synthetic_df)

    logging.info('Finished Successfully')


if __name__ == '__main__':
    run_experiment()
