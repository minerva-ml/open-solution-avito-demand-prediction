import os
import shutil

import click
import pandas as pd
from deepsense import neptune

import pipeline_config as cfg
from pipelines import PIPELINES
from preprocessing import translate
from utils import init_logger, read_params, create_submission, set_seed, save_evaluation_predictions, \
    stratified_train_valid_split, data_hash_channel_send, root_mean_squared_error

set_seed(1234)
logger = init_logger()
ctx = neptune.Context()
params = read_params(ctx)


@click.group()
def action():
    pass


@action.command()
def translate_to_english():
    filepath_train_en = params.train_en_filepath
    filepath_test_en = params.test_en_filepath
    if not os.path.isfile(filepath_train_en):
        logger.info('translating train')
        translated_df = translate(filepath=params.train_filepath, column_to_translate=cfg.FEATURES_TO_TRANSLATE)
        translated_df.to_csv(filepath_train_en)
    if not os.path.isfile(filepath_test_en):
        logger.info('translating test')
        translated_df = translate(filepath=params.test_filepath, column_to_translate=cfg.FEATURES_TO_TRANSLATE)
        translated_df.to_csv(filepath_test_en)


@action.command()
@click.option('-p', '--pipeline_name', help='pipeline to be trained', required=True)
@click.option('-d', '--dev_mode', help='if true only a small sample of data will be used', is_flag=True, required=False)
def train(pipeline_name, dev_mode):
    _train(pipeline_name, dev_mode)


def _train(pipeline_name, dev_mode):
    if params.use_english:
        train_filepath = params.train_en_filepath
    else:
        train_filepath = params.train_filepath

    if bool(params.overwrite) and os.path.isdir(params.experiment_dir):
        shutil.rmtree(params.experiment_dir)

    logger.info('reading data in')
    if dev_mode:
        meta_train = pd.read_csv(train_filepath,
                                 usecols=cfg.FEATURE_COLUMNS + cfg.TARGET_COLUMNS + cfg.ITEM_ID_COLUMN,
                                 dtype=cfg.COLUMN_TYPES['train'],
                                 nrows=cfg.DEV_SAMPLE_SIZE)
    else:
        meta_train = pd.read_csv(train_filepath,
                                 usecols=cfg.FEATURE_COLUMNS + cfg.TARGET_COLUMNS + cfg.ITEM_ID_COLUMN,
                                 dtype=cfg.COLUMN_TYPES['train'])

    meta_train_split, meta_valid_split = stratified_train_valid_split(meta_train,
                                                                      target_column=cfg.TARGET_COLUMNS,
                                                                      target_bins=params.target_bins,
                                                                      valid_size=params.validation_size,
                                                                      random_state=1234)

    data_hash_channel_send(ctx, 'Training Data Hash', meta_train_split)
    data_hash_channel_send(ctx, 'Validation Data Hash', meta_valid_split)

    logger.info('Target distribution in train: {}'.format(meta_train_split[cfg.TARGET_COLUMNS].mean()))
    logger.info('Target distribution in valid: {}'.format(meta_valid_split[cfg.TARGET_COLUMNS].mean()))

    logger.info('shuffling data')
    meta_train_split = meta_train_split.sample(frac=1)
    meta_valid_split = meta_valid_split.sample(frac=1)

    data = {'input': {'X': meta_train_split[cfg.FEATURE_COLUMNS],
                      'y': meta_train_split[cfg.TARGET_COLUMNS],
                      'X_valid': meta_valid_split[cfg.FEATURE_COLUMNS],
                      'y_valid': meta_valid_split[cfg.TARGET_COLUMNS],
                      },
            'specs': {'is_train': True}
            }

    pipeline = PIPELINES[pipeline_name]['train'](cfg.SOLUTION_CONFIG)
    pipeline.clean_cache()
    pipeline.fit_transform(data)
    pipeline.clean_cache()


@action.command()
@click.option('-p', '--pipeline_name', help='pipeline to be trained', required=True)
@click.option('-d', '--dev_mode', help='if true only a small sample of data will be used', is_flag=True, required=False)
def evaluate(pipeline_name, dev_mode):
    _evaluate(pipeline_name, dev_mode)


def _evaluate(pipeline_name, dev_mode):
    logger.info('reading data in')
    if params.use_english:
        train_filepath = params.train_en_filepath
    else:
        train_filepath = params.train_filepath

    if dev_mode:
        meta_train = pd.read_csv(train_filepath,
                                 usecols=cfg.FEATURE_COLUMNS + cfg.TARGET_COLUMNS + cfg.ITEM_ID_COLUMN,
                                 dtype=cfg.COLUMN_TYPES['train'],
                                 nrows=cfg.DEV_SAMPLE_SIZE)
    else:
        meta_train = pd.read_csv(train_filepath,
                                 usecols=cfg.FEATURE_COLUMNS + cfg.TARGET_COLUMNS + cfg.ITEM_ID_COLUMN,
                                 dtype=cfg.COLUMN_TYPES['train'])

    _, meta_valid_split = stratified_train_valid_split(meta_train,
                                                       target_column=cfg.TARGET_COLUMNS,
                                                       target_bins=params.target_bins,
                                                       valid_size=params.validation_size,
                                                       random_state=1234)

    data_hash_channel_send(ctx, 'Evaluation Data Hash', meta_valid_split)

    logger.info('Target distribution in valid: {}'.format(meta_valid_split[cfg.TARGET_COLUMNS].mean()))

    data = {'input': {'X': meta_valid_split[cfg.FEATURE_COLUMNS],
                      'y': None,
                      },
            'specs': {'is_train': True}
            }
    pipeline = PIPELINES[pipeline_name]['inference'](cfg.SOLUTION_CONFIG)
    pipeline.clean_cache()
    output = pipeline.transform(data)
    pipeline.clean_cache()
    y_pred = output['y_pred']
    y_true = meta_valid_split[cfg.TARGET_COLUMNS].values.reshape(-1)

    logger.info('Saving evaluation predictions')
    save_evaluation_predictions(params.experiment_dir, y_true, y_pred, meta_valid_split)

    logger.info('Calculating RMSE')
    score = root_mean_squared_error(y_true, y_pred)
    logger.info('RMSE score on validation is {}'.format(score))
    ctx.channel_send('RMSE', 0, score)


@action.command()
@click.option('-p', '--pipeline_name', help='pipeline to be trained', required=True)
@click.option('-d', '--dev_mode', help='if true only a small sample of data will be used', is_flag=True, required=False)
def predict(pipeline_name, dev_mode):
    _predict(pipeline_name, dev_mode)


def _predict(pipeline_name, dev_mode):
    if params.use_english:
        test_filepath = params.test_en_filepath
    else:
        test_filepath = params.test_filepath

    logger.info('reading data in')
    if dev_mode:
        meta_test = pd.read_csv(test_filepath,
                                usecols=cfg.FEATURE_COLUMNS + cfg.ITEM_ID_COLUMN,
                                dtype=cfg.COLUMN_TYPES['inference'],
                                nrows=cfg.DEV_SAMPLE_SIZE)
    else:
        meta_test = pd.read_csv(test_filepath,
                                usecols=cfg.FEATURE_COLUMNS + cfg.ITEM_ID_COLUMN,
                                dtype=cfg.COLUMN_TYPES['inference'])

    data_hash_channel_send(ctx, 'Test Data Hash', meta_test)

    data = {'input': {'X': meta_test[cfg.FEATURE_COLUMNS],
                      'y': None,
                      },
            'specs': {'is_train': False}
            }

    pipeline = PIPELINES[pipeline_name]['inference'](cfg.SOLUTION_CONFIG)
    pipeline.clean_cache()
    output = pipeline.transform(data)
    pipeline.clean_cache()
    y_pred = output['y_pred']

    logger.info('creating submission test')
    submission = create_submission(meta_test, y_pred)
    submission_filepath = os.path.join(params.experiment_dir, 'submission.csv')
    submission.to_csv(submission_filepath, index=None, encoding='utf-8')
    logger.info('submission saved to {}'.format(submission_filepath))
    logger.info('submission head \n\n{}'.format(submission.head()))


@action.command()
@click.option('-p', '--pipeline_name', help='pipeline to be trained', required=True)
@click.option('-d', '--dev_mode', help='if true only a small sample of data will be used', is_flag=True, required=False)
def train_evaluate_predict(pipeline_name, dev_mode):
    logger.info('TRAINING')
    _train(pipeline_name, dev_mode)
    logger.info('EVALUATION')
    _evaluate(pipeline_name, dev_mode)
    logger.info('PREDICTION')
    _predict(pipeline_name, dev_mode)


@action.command()
@click.option('-p', '--pipeline_name', help='pipeline to be trained', required=True)
@click.option('-d', '--dev_mode', help='if true only a small sample of data will be used', is_flag=True, required=False)
def evaluate_predict(pipeline_name, dev_mode):
    logger.info('EVALUATION')
    _evaluate(pipeline_name, dev_mode)
    logger.info('PREDICTION')
    _predict(pipeline_name, dev_mode)


@action.command()
@click.option('-p', '--pipeline_name', help='pipeline to be trained', required=True)
@click.option('-d', '--dev_mode', help='if true only a small sample of data will be used', is_flag=True, required=False)
def train_evaluate(pipeline_name, dev_mode):
    logger.info('TRAINING')
    _train(pipeline_name, dev_mode)
    logger.info('EVALUATION')
    _evaluate(pipeline_name, dev_mode)


if __name__ == "__main__":
    action()
