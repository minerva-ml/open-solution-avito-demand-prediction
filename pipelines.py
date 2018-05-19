from functools import partial

import feature_extraction as fe
from hyperparameter_tuning import RandomSearchOptimizer, NeptuneMonitor, SaveResults
from steps.adapters import to_numpy_label_inputs, identity_inputs
from steps.base import Step, Dummy
from models import LightGBMLowMemory as LightGBM
from postprocessing import Clipper
from utils import root_mean_squared_error, pandas_concat_inputs


def main(config, train_mode):
    if train_mode:
        features, features_valid = feature_extraction(config, train_mode,
                                                      save_output=True, cache_output=True, load_saved_output=True)
        light_gbm = classifier_lgbm((features, features_valid), config, train_mode)
    else:
        features = feature_extraction(config, train_mode, cache_output=True)
        light_gbm = classifier_lgbm(features, config, train_mode)

    clipper = Step(name='clipper',
                   transformer=Clipper(**config.clipper),
                   input_steps=[light_gbm],
                   adapter={'prediction': ([(light_gbm.name, 'prediction')]),
                            },
                   cache_dirpath=config.env.cache_dirpath)

    output = Step(name='output',
                  transformer=Dummy(),
                  input_steps=[clipper],
                  adapter={'y_pred': ([(clipper.name, 'clipped_prediction')]),
                           },
                  cache_dirpath=config.env.cache_dirpath)
    return output


def feature_extraction(config, train_mode, **kwargs):
    if train_mode:

        missing, missing_valid = _is_missing_features(config, train_mode, **kwargs)
        feature_by_type_split, feature_by_type_split_valid = _feature_by_type_splits(config, train_mode)

        dataframe_features_train, dataframe_features_valid = dataframe_features(
            (feature_by_type_split, feature_by_type_split_valid), config, train_mode, **kwargs)
        categorical, timestamp, text_features, numerical, group_by, target_encoder = dataframe_features_train
        categorical_valid, timestamp_valid, text_features_valid, numerical_valid, group_by_valid, \
        target_encoder_valid = dataframe_features_valid

        feature_combiner, feature_combiner_valid = _join_features(numerical_features=[numerical,
                                                                                      target_encoder,
                                                                                      group_by,
                                                                                      text_features],
                                                                  numerical_features_valid=[numerical_valid,
                                                                                            target_encoder_valid,
                                                                                            group_by_valid,
                                                                                            text_features_valid],
                                                                  categorical_features=[timestamp,
                                                                                        missing,
                                                                                        categorical,
                                                                                        target_encoder],
                                                                  categorical_features_valid=[timestamp_valid,
                                                                                              missing_valid,
                                                                                              categorical_valid,
                                                                                              target_encoder_valid],
                                                                  config=config, train_mode=train_mode, **kwargs)
        return feature_combiner, feature_combiner_valid
    else:
        missing = _is_missing_features(config, train_mode, **kwargs)
        feature_by_type_split = _feature_by_type_splits(config, train_mode)

        categorical, timestamp, text_features, numerical, group_by, target_encoder = dataframe_features(
            feature_by_type_split, config, train_mode, **kwargs)

        feature_combiner = _join_features(numerical_features=[numerical, target_encoder, group_by, text_features],
                                          numerical_features_valid=[],
                                          categorical_features=[timestamp, missing, categorical, target_encoder],
                                          categorical_features_valid=[],
                                          config=config, train_mode=train_mode, **kwargs)
        return feature_combiner


def dataframe_features(dispatchers, config, train_mode, **kwargs):
    if train_mode:
        feature_by_type_split, feature_by_type_split_valid = dispatchers

        encoded_categorical, encoded_categorical_valid = _encode_categorical(
            (feature_by_type_split, feature_by_type_split_valid),
            config, train_mode, **kwargs)

        timestamp_features, timestamp_features_valid = _timestamp_features(
            (feature_by_type_split, feature_by_type_split_valid),
            config, train_mode, **kwargs)

        text_features, text_features_valid = _text_features(
            (feature_by_type_split, feature_by_type_split_valid),
            config, train_mode, **kwargs)

        numerical_features, numerical_features_valid = _numerical_features(
            (feature_by_type_split, feature_by_type_split_valid),
            config, train_mode, **kwargs)

        groupby_aggregation, groupby_aggregation_valid = _groupby_aggregations(
            (feature_by_type_split, feature_by_type_split_valid), (timestamp_features, timestamp_features_valid),
            config, train_mode, **kwargs)
        target_encoder, target_encoder_valid = _target_encoders((feature_by_type_split, feature_by_type_split_valid),
                                                                config, train_mode, **kwargs)
        train_features = (encoded_categorical,
                          timestamp_features,
                          text_features,
                          numerical_features,
                          groupby_aggregation,
                          target_encoder)
        valid_features = (encoded_categorical_valid,
                          timestamp_features_valid,
                          text_features_valid,
                          numerical_features_valid,
                          groupby_aggregation_valid,
                          target_encoder_valid)
        return train_features, valid_features
    else:
        feature_by_type_split = dispatchers

        encoded_categorical = _encode_categorical(feature_by_type_split, config, train_mode, **kwargs)
        timestamp_features = _timestamp_features(feature_by_type_split, config, train_mode, **kwargs)
        text_features = _text_features(feature_by_type_split, config, train_mode, **kwargs)
        numerical_features = _numerical_features(feature_by_type_split, config, train_mode, **kwargs)
        groupby_aggregation = _groupby_aggregations(feature_by_type_split, timestamp_features,
                                                    config, train_mode, **kwargs)
        target_encoder = _target_encoders(feature_by_type_split, config, train_mode, **kwargs)

        train_features = (encoded_categorical,
                          timestamp_features,
                          text_features,
                          numerical_features,
                          groupby_aggregation,
                          target_encoder)
        return train_features


def classifier_lgbm(features, config, train_mode, **kwargs):
    if train_mode:
        features_train, features_valid = features
        if config.random_search.light_gbm.n_runs:
            transformer = RandomSearchOptimizer(LightGBM, config.light_gbm,
                                                train_input_keys=[],
                                                valid_input_keys=['X_valid', 'y_valid'],
                                                score_func=root_mean_squared_error,
                                                maximize=False,
                                                n_runs=config.random_search.light_gbm.n_runs,
                                                callbacks=[NeptuneMonitor(
                                                    **config.random_search.light_gbm.callbacks.neptune_monitor),
                                                    SaveResults(
                                                        **config.random_search.light_gbm.callbacks.save_results),
                                                ]
                                                )
        else:
            transformer = LightGBM(**config.light_gbm)

        light_gbm = Step(name='light_gbm',
                         transformer=transformer,
                         input_data=['input'],
                         input_steps=[features_train, features_valid],
                         adapter={'X': ([(features_train.name, 'features')]),
                                  'y': ([('input', 'y')], to_numpy_label_inputs),
                                  'feature_names': ([(features_train.name, 'feature_names')]),
                                  'categorical_features': ([(features_train.name, 'categorical_features')]),
                                  'X_valid': ([(features_valid.name, 'features')]),
                                  'y_valid': ([('input', 'y_valid')], to_numpy_label_inputs),
                                  },
                         cache_dirpath=config.env.cache_dirpath,
                         **kwargs)
    else:
        light_gbm = Step(name='light_gbm',
                         transformer=LightGBM(**config.light_gbm),
                         input_steps=[features],
                         adapter={'X': ([(features.name, 'features')]),
                                  },
                         cache_dirpath=config.env.cache_dirpath,
                         **kwargs)
    return light_gbm


def _feature_by_type_splits(config, train_mode):
    if train_mode:
        feature_by_type_split = Step(name='feature_by_type_split',
                                     transformer=fe.DataFrameByTypeSplitter(**config.dataframe_by_type_splitter),
                                     input_data=['input'],
                                     adapter={'X': ([('input', 'X')]),
                                              },
                                     cache_dirpath=config.env.cache_dirpath)

        feature_by_type_split_valid = Step(name='feature_by_type_split_valid',
                                           transformer=feature_by_type_split,
                                           input_data=['input'],
                                           adapter={'X': ([('input', 'X_valid')]),
                                                    },
                                           cache_dirpath=config.env.cache_dirpath)

        return feature_by_type_split, feature_by_type_split_valid

    else:
        feature_by_type_split = Step(name='feature_by_type_split',
                                     transformer=fe.DataFrameByTypeSplitter(**config.dataframe_by_type_splitter),
                                     input_data=['input'],
                                     adapter={'X': ([('input', 'X')]),
                                              },
                                     cache_dirpath=config.env.cache_dirpath)

        return feature_by_type_split


def _is_missing_features(config, train_mode, **kwargs):
    if train_mode:
        is_missing = Step(name='is_missing',
                          transformer=fe.IsMissing(**config.is_missing),
                          input_data=['input'],
                          adapter={'X': ([('input', 'X')])},
                          cache_dirpath=config.env.cache_dirpath, **kwargs)

        is_missing_valid = Step(name='is_missing_valid',
                                transformer=is_missing,
                                input_data=['input'],
                                adapter={'X': ([('input', 'X_valid')])},
                                cache_dirpath=config.env.cache_dirpath, **kwargs)

        return is_missing, is_missing_valid

    else:
        is_missing = Step(name='is_missing',
                          transformer=fe.IsMissing(**config.is_missing),
                          input_data=['input'],
                          adapter={'X': ([('input', 'X')])},
                          cache_dirpath=config.env.cache_dirpath, **kwargs)

        return is_missing


def _encode_categorical(dispatchers, config, train_mode, **kwargs):
    if train_mode:
        feature_by_type_split, feature_by_type_split_valid = dispatchers
        categorical_encoder = Step(name='categorical_encoder',
                                   transformer=fe.HashingCategoricalEncoder(**config.categorical_encoder),
                                   input_steps=[feature_by_type_split],
                                   adapter={
                                       'categorical_features': ([(feature_by_type_split.name, 'categorical_features')])
                                   },
                                   cache_dirpath=config.env.cache_dirpath,
                                   **kwargs)

        categorical_encoder_valid = Step(name='categorical_encoder_valid',
                                         transformer=categorical_encoder,
                                         input_steps=[feature_by_type_split_valid],
                                         adapter={'categorical_features': (
                                             [(feature_by_type_split_valid.name, 'categorical_features')])
                                         },
                                         cache_dirpath=config.env.cache_dirpath,
                                         **kwargs)

        return categorical_encoder, categorical_encoder_valid

    else:
        feature_by_type_split = dispatchers
        categorical_encoder = Step(name='categorical_encoder',
                                   transformer=fe.HashingCategoricalEncoder(**config.categorical_encoder),
                                   input_steps=[feature_by_type_split],
                                   adapter={
                                       'categorical_features': ([(feature_by_type_split.name, 'categorical_features')])
                                   },
                                   cache_dirpath=config.env.cache_dirpath,
                                   **kwargs)

        return categorical_encoder


def _timestamp_features(dispatchers, config, train_mode, **kwargs):
    if train_mode:
        feature_by_type_split, feature_by_type_split_valid = dispatchers
        timestamp_features = Step(name='timestamp_features',
                                  transformer=fe.DateFeatures(**config.date_features),
                                  input_steps=[feature_by_type_split],
                                  adapter={
                                      'timestamp_features': ([(feature_by_type_split.name, 'timestamp_features')])
                                  },
                                  cache_dirpath=config.env.cache_dirpath,
                                  **kwargs)

        timestamp_features_valid = Step(name='timestamp_features_valid',
                                        transformer=timestamp_features,
                                        input_steps=[feature_by_type_split_valid],
                                        adapter={'timestamp_features': (
                                            [(feature_by_type_split_valid.name, 'timestamp_features')])
                                        },
                                        cache_dirpath=config.env.cache_dirpath,
                                        **kwargs)

        return timestamp_features, timestamp_features_valid

    else:
        feature_by_type_split = dispatchers
        timestamp_features = Step(name='timestamp_features',
                                  transformer=fe.DateFeatures(**config.date_features),
                                  input_steps=[feature_by_type_split],
                                  adapter={
                                      'timestamp_features': ([(feature_by_type_split.name, 'timestamp_features')])
                                  },
                                  cache_dirpath=config.env.cache_dirpath,
                                  **kwargs)

        return timestamp_features


def _text_features(dispatchers, config, train_mode, **kwargs):
    if train_mode:
        feature_by_type_split, feature_by_type_split_valid = dispatchers
        text_features = Step(name='text_features',
                             transformer=fe.TextCounter(**config.text_counter),
                             input_steps=[feature_by_type_split],
                             adapter={'X': ([(feature_by_type_split.name, 'categorical_features')])},
                             cache_dirpath=config.env.cache_dirpath,
                             **kwargs)

        text_features_valid = Step(name='text_features_valid',
                                   transformer=text_features,
                                   input_steps=[feature_by_type_split_valid],
                                   adapter={'X': ([(feature_by_type_split_valid.name, 'categorical_features')])},
                                   cache_dirpath=config.env.cache_dirpath,
                                   **kwargs)

        return text_features, text_features_valid

    else:
        feature_by_type_split = dispatchers
        text_features = Step(name='text_features',
                             transformer=fe.TextCounter(**config.text_counter),
                             input_steps=[feature_by_type_split],
                             adapter={'X': ([(feature_by_type_split.name, 'categorical_features')])},
                             cache_dirpath=config.env.cache_dirpath,
                             **kwargs)

        return text_features


def _numerical_features(dispatchers, config, train_mode, **kwargs):
    if train_mode:
        feature_by_type_split, feature_by_type_split_valid = dispatchers
        numerical_features = Step(name='numerical_features',
                                  transformer=Dummy(),
                                  input_steps=[feature_by_type_split],
                                  adapter={
                                      'numerical_features': ([(feature_by_type_split.name, 'numerical_features')])
                                  },
                                  cache_dirpath=config.env.cache_dirpath,
                                  **kwargs)

        numerical_features_valid = Step(name='numerical_features_valid',
                                        transformer=numerical_features,
                                        input_steps=[feature_by_type_split_valid],
                                        adapter={'numerical_features': (
                                            [(feature_by_type_split_valid.name, 'numerical_features')])
                                        },
                                        cache_dirpath=config.env.cache_dirpath,
                                        **kwargs)

        return numerical_features, numerical_features_valid

    else:
        feature_by_type_split = dispatchers
        numerical_features = Step(name='numerical_features',
                                  transformer=Dummy(),
                                  input_steps=[feature_by_type_split],
                                  adapter={
                                      'numerical_features': ([(feature_by_type_split.name, 'numerical_features')])
                                  },
                                  cache_dirpath=config.env.cache_dirpath,
                                  **kwargs)

        return numerical_features


def _target_encoders(dispatchers, config, train_mode, **kwargs):
    if train_mode:
        feature_by_type_split, feature_by_type_split_valid = dispatchers
        target_encoder = Step(name='target_encoder',
                              transformer=fe.TargetEncoderNSplits(**config.target_encoder),
                              input_data=['input'],
                              input_steps=[feature_by_type_split],
                              adapter={
                                  'categorical_features': ([(feature_by_type_split.name, 'categorical_features')]),
                                  'target': ([('input', 'y')])
                              },
                              cache_dirpath=config.env.cache_dirpath,
                              **kwargs)

        target_encoder_valid = Step(name='target_encoder_valid',
                                    transformer=target_encoder,
                                    input_data=['input'],
                                    input_steps=[feature_by_type_split_valid],
                                    adapter={'categorical_features': (
                                        [(feature_by_type_split_valid.name, 'categorical_features')]),
                                        'target': ([('input', 'y_valid')])
                                    },
                                    cache_dirpath=config.env.cache_dirpath,
                                    **kwargs)

        return target_encoder, target_encoder_valid

    else:
        feature_by_type_split = dispatchers
        target_encoder = Step(name='target_encoder',
                              transformer=fe.TargetEncoderNSplits(**config.target_encoder),
                              input_data=['input'],
                              input_steps=[feature_by_type_split],
                              adapter={
                                  'categorical_features': ([(feature_by_type_split.name, 'categorical_features')]),
                                  'target': ([('input', 'y')])
                              },
                              cache_dirpath=config.env.cache_dirpath,
                              **kwargs)

        return target_encoder


def _groupby_aggregations(dispatchers, additional_features, config, train_mode, **kwargs):
    if train_mode:
        feature_by_type_split, feature_by_type_split_valid = dispatchers
        added_feature, added_feature_valid = additional_features
        groupby_aggregations = Step(name='groupby_aggregations',
                                    transformer=fe.GroupbyAggregations(**config.groupby_aggregation),
                                    input_steps=[feature_by_type_split, added_feature],
                                    adapter={
                                        'categorical_features': ([(feature_by_type_split.name, 'categorical_features'),
                                                                  (added_feature.name, 'categorical_features'),
                                                                  (feature_by_type_split.name, 'numerical_features')],
                                                                 pandas_concat_inputs)
                                    },
                                    cache_dirpath=config.env.cache_dirpath,
                                    **kwargs)

        groupby_aggregations_valid = Step(name='groupby_aggregations_valid',
                                          transformer=groupby_aggregations,
                                          input_steps=[feature_by_type_split_valid, added_feature_valid],
                                          adapter={'categorical_features': (
                                              [(feature_by_type_split_valid.name, 'categorical_features'),
                                               (added_feature_valid.name, 'categorical_features'),
                                               (feature_by_type_split_valid.name, 'numerical_features')],
                                              pandas_concat_inputs
                                          )
                                          },
                                          cache_dirpath=config.env.cache_dirpath,
                                          **kwargs)

        return groupby_aggregations, groupby_aggregations_valid

    else:
        feature_by_type_split = dispatchers
        added_feature = additional_features
        groupby_aggregations = Step(name='groupby_aggregations',
                                    transformer=fe.GroupbyAggregations(**config.groupby_aggregation),
                                    input_steps=[feature_by_type_split, added_feature],
                                    adapter={
                                        'categorical_features': ([(feature_by_type_split.name, 'categorical_features'),
                                                                  (added_feature.name, 'categorical_features'),
                                                                  (feature_by_type_split.name, 'numerical_features')],
                                                                 pandas_concat_inputs)
                                    },
                                    cache_dirpath=config.env.cache_dirpath,
                                    **kwargs)

        return groupby_aggregations


def _join_features(numerical_features, numerical_features_valid,
                   categorical_features, categorical_features_valid,
                   config, train_mode, **kwargs):
    if train_mode:
        feature_joiner = Step(name='feature_joiner',
                              transformer=fe.FeatureJoiner(),
                              input_steps=numerical_features + categorical_features,
                              adapter={
                                  'numerical_feature_list': (
                                      [(feature.name, 'numerical_features') for feature in numerical_features],
                                      identity_inputs),
                                  'categorical_feature_list': (
                                      [(feature.name, 'categorical_features') for feature in categorical_features],
                                      identity_inputs),
                              },
                              cache_dirpath=config.env.cache_dirpath, **kwargs)

        feature_joiner_valid = Step(name='feature_joiner_valid',
                                    transformer=feature_joiner,
                                    input_steps=numerical_features_valid + categorical_features_valid,
                                    adapter={'numerical_feature_list': (
                                        [(feature.name, 'numerical_features') for feature in numerical_features_valid],
                                        identity_inputs),
                                        'categorical_feature_list': (
                                            [(feature.name, 'categorical_features') for feature in
                                             categorical_features_valid],
                                            identity_inputs),
                                    },
                                    cache_dirpath=config.env.cache_dirpath, **kwargs)

        return feature_joiner, feature_joiner_valid

    else:
        feature_joiner = Step(name='feature_joiner',
                              transformer=fe.FeatureJoiner(),
                              input_steps=numerical_features + categorical_features,
                              adapter={
                                  'numerical_feature_list': (
                                      [(feature.name, 'numerical_features') for feature in numerical_features],
                                      identity_inputs),
                                  'categorical_feature_list': (
                                      [(feature.name, 'categorical_features') for feature in categorical_features],
                                      identity_inputs),
                              },
                              cache_dirpath=config.env.cache_dirpath, **kwargs)

        return feature_joiner


PIPELINES = {'main': {'train': partial(main, train_mode=True),
                      'inference': partial(main, train_mode=False)},
             }
