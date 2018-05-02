from functools import partial

import feature_extraction as fe
from hyperparameter_tuning import RandomSearchOptimizer, NeptuneMonitor, SaveResults
from steps.adapters import to_numpy_label_inputs, identity_inputs
from steps.base import Step, Dummy
from models import LightGBMLowMemory as LightGBM
from utils import root_mean_squared_error


def solution_1(config, train_mode):
    if train_mode:
        features, features_valid = feature_extraction_v1(config, train_mode,
                                                         save_output=True, cache_output=True, load_saved_output=True)
        light_gbm = classifier_lgbm((features, features_valid), config, train_mode)
    else:
        features = feature_extraction_v1(config, train_mode, cache_output=True)
        light_gbm = classifier_lgbm(features, config, train_mode)

    output = Step(name='output',
                  transformer=Dummy(),
                  input_steps=[light_gbm],
                  adapter={'y_pred': ([(light_gbm.name, 'prediction')]),
                           },
                  cache_dirpath=config.env.cache_dirpath)
    return output


def feature_extraction_v1(config, train_mode, **kwargs):
    if train_mode:
        feature_by_type_split, feature_by_type_split_valid = _feature_by_type_splits(config, train_mode)

        groupby_aggregation, groupby_aggregation_valid = _groupby_aggregations(
            (feature_by_type_split, feature_by_type_split_valid),
            config, train_mode, **kwargs)
        target_encoder, target_encoder_valid = _target_encoders((feature_by_type_split, feature_by_type_split_valid),
                                                                config, train_mode, **kwargs)

        feature_combiner, feature_combiner_valid = _join_features(numerical_features=[target_encoder,
                                                                                      groupby_aggregation],
                                                                  numerical_features_valid=[target_encoder_valid,
                                                                                            groupby_aggregation_valid],
                                                                  categorical_features=[target_encoder],
                                                                  categorical_features_valid=[target_encoder_valid],
                                                                  config=config, train_mode=train_mode)
        return feature_combiner, feature_combiner_valid
    else:
        feature_by_type_split = _feature_by_type_splits(config, train_mode)

        groupby_aggregation = _groupby_aggregations(feature_by_type_split, config, train_mode, **kwargs)
        target_encoder = _target_encoders(feature_by_type_split, config, train_mode, **kwargs)

        feature_combiner = _join_features(numerical_features=[target_encoder, groupby_aggregation],
                                          numerical_features_valid=[],
                                          categorical_features=[target_encoder],
                                          categorical_features_valid=[],
                                          config=config, train_mode=train_mode)
        return feature_combiner


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


def _groupby_aggregations(dispatchers, config, train_mode, **kwargs):
    if train_mode:
        feature_by_type_split, feature_by_type_split_valid = dispatchers
        groupby_aggregations = Step(name='groupby_aggregations',
                                    transformer=fe.GroupbyAggregations(**config.groupby_aggregation),
                                    input_data=['input'],
                                    input_steps=[feature_by_type_split],
                                    adapter={
                                        'categorical_features': ([(feature_by_type_split.name, 'categorical_features')])
                                    },
                                    cache_dirpath=config.env.cache_dirpath,
                                    **kwargs)

        groupby_aggregations_valid = Step(name='groupby_aggregations_valid',
                                          transformer=groupby_aggregations,
                                          input_data=['input'],
                                          input_steps=[feature_by_type_split_valid],
                                          adapter={'categorical_features': (
                                              [(feature_by_type_split_valid.name, 'categorical_features')])
                                          },
                                          cache_dirpath=config.env.cache_dirpath,
                                          **kwargs)

        return groupby_aggregations, groupby_aggregations_valid

    else:
        feature_by_type_split = dispatchers
        groupby_aggregations = Step(name='groupby_aggregations',
                                    transformer=fe.GroupbyAggregations(**config.groupby_aggregation),
                                    input_data=['input'],
                                    input_steps=[feature_by_type_split],
                                    adapter={
                                        'categorical_features': ([(feature_by_type_split.name, 'categorical_features')])
                                    },
                                    cache_dirpath=config.env.cache_dirpath,
                                    **kwargs)

        return groupby_aggregations


def _join_features(numerical_features, numerical_features_valid,
                   categorical_features, categorical_features_valid,
                   config, train_mode):
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
                              cache_dirpath=config.env.cache_dirpath)

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
                                    cache_dirpath=config.env.cache_dirpath)

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
                              cache_dirpath=config.env.cache_dirpath)

        return feature_joiner


PIPELINES = {'solution_1': {'train': partial(solution_1, train_mode=True),
                            'inference': partial(solution_1, train_mode=False)},
             }
