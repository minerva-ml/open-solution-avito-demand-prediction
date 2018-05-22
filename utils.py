import glob
import hashlib
import logging
import os
import random
import sys

import numpy as np
import pandas as pd
import yaml
from attrdict import AttrDict
from deepsense import neptune
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


def read_yaml(filepath):
    with open(filepath) as f:
        config = yaml.load(f)
    return AttrDict(config)


def init_logger():
    logger = logging.getLogger('talking-data')
    logger.setLevel(logging.INFO)
    message_format = logging.Formatter(fmt='%(asctime)s %(name)s >>> %(message)s',
                                       datefmt='%Y-%m-%d %H-%M-%S')

    # console handler for validation info
    ch_va = logging.StreamHandler(sys.stdout)
    ch_va.setLevel(logging.INFO)

    ch_va.setFormatter(fmt=message_format)

    # add the handlers to the logger
    logger.addHandler(ch_va)

    return logger


def stratified_train_valid_split(meta_train, target_column, target_bins, valid_size, random_state=1234):
    y = meta_train[target_column].values
    bins = np.linspace(0, y.shape[0], target_bins)
    y_binned = np.digitize(y, bins)

    return train_test_split(meta_train, test_size=valid_size, stratify=y_binned, random_state=random_state)


def get_logger():
    return logging.getLogger('avito')


def create_submission(meta, predictions):
    submission = pd.DataFrame({'item_id': meta['item_id'].tolist(),
                               'deal_probability': predictions
                               })
    return submission


def read_params(ctx):
    if ctx.params.__class__.__name__ == 'OfflineContextParams':
        try:
            neptune_config = read_yaml('neptune.yaml')
        except FileNotFoundError:
            neptune_config = read_yaml('../neptune.yaml')
        params = neptune_config.parameters
    else:
        params = ctx.params
    return params


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def rmse_row(y_true, y_pred):
    return np.sqrt((y_true - y_pred) ** 2)


def save_evaluation_predictions(experiment_dir, y_true, y_pred, raw_data):
    raw_data['y_pred'] = y_pred
    raw_data['score'] = rmse_row(y_true, y_pred)

    raw_data.sort_values('score', ascending=True, inplace=True)

    filepath = os.path.join(experiment_dir, 'evaluation_predictions.csv')
    raw_data.to_csv(filepath, index=None)


def data_hash_channel_send(ctx, name, data):
    hash_channel = ctx.create_channel(name=name, channel_type=neptune.ChannelType.TEXT)
    data_hash = create_data_hash(data)
    hash_channel.send(y=data_hash)


def create_data_hash(data):
    if isinstance(data, pd.DataFrame):
        data_hash = hashlib.sha256(data.to_json().encode()).hexdigest()
    else:
        raise NotImplementedError('only pandas.DataFrame and pandas.Series are supported')
    return str(data_hash)


def safe_eval(obj):
    try:
        return eval(obj)
    except Exception:
        return obj


def root_mean_squared_error(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def pandas_concat_inputs(inputs, axis=1):
    return pd.concat(inputs, axis=axis)


def pandas_subset_columns(inputs, cols):
    return inputs[0][cols]
