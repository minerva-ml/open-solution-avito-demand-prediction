from functools import partial

from steps.adapters import to_numpy_label_inputs, identity_inputs
from steps.base import Step, Dummy
import feature_extraction as fe
from hyperparameter_tuning import RandomSearchOptimizer, NeptuneMonitor, SaveResults
from models import HierarchicalInceptionResnet, LightGBMLowMemory as LightGBM
from loaders import MultiOutputImageLoader
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
        feature_by_type_split, feature_by_type_split_valid = _feature_by_type_splits(config, train_mode)

        dataframe_features_train, dataframe_features_valid = dataframe_features(
            (feature_by_type_split, feature_by_type_split_valid), config, train_mode, **kwargs)
        price_features, groupby_aggregation, target_encoder, label_encoder = dataframe_features_train
        price_features_valid, groupby_aggregation_valid, target_encoder_valid, label_encoder_valid = dataframe_features_valid

        feature_combiner, feature_combiner_valid = _join_features(numerical_features=[price_features,
                                                                                      target_encoder,
                                                                                      groupby_aggregation],
                                                                  numerical_features_valid=[price_features_valid,
                                                                                            target_encoder_valid,
                                                                                            groupby_aggregation_valid],
                                                                  categorical_features=[label_encoder, target_encoder],
                                                                  categorical_features_valid=[label_encoder_valid,
                                                                                              target_encoder_valid],
                                                                  config=config, train_mode=train_mode, **kwargs)
        return feature_combiner, feature_combiner_valid
    else:
        feature_by_type_split = _feature_by_type_splits(config, train_mode)

        price_features, groupby_aggregation, target_encoder, label_encoder = dataframe_features(feature_by_type_split,
                                                                                                config,
                                                                                                train_mode, **kwargs)

        feature_combiner = _join_features(numerical_features=[price_features, target_encoder, groupby_aggregation],
                                          numerical_features_valid=[],
                                          categorical_features=[label_encoder, target_encoder],
                                          categorical_features_valid=[],
                                          config=config, train_mode=train_mode, **kwargs)
        return feature_combiner


def image_model(config, train_mode):
    """ Pipeline predicting the 'parent_category_name, category_name' based on images
    """

    non_nan_subset = Step(name='non_nan_subset',
                          transformer=fe.SubsetNotNan(**config.subset_not_nan_image),
                          input_data=['input'],
                          cache_dirpath=config.env.cache_dirpath)

    image_column = Step(name='image_column',
                        transformer=fe.FetchColumns(**config.fetch_image_columns),
                        input_steps=[non_nan_subset],
                        cache_dirpath=config.env.cache_dirpath)

    label_encoder_image = Step(name='label_encoder_image',
                               transformer=fe.LabelEncoder(**config.label_encoder_image),
                               input_steps=[non_nan_subset],
                               adapter={'categorical_features': ([(non_nan_subset.name, 'y')]),
                                        },
                               cache_dirpath=config.env.cache_dirpath)

    if train_mode:

        non_nan_subset_valid = Step(name='non_nan_subset_valid',
                                    transformer=non_nan_subset,
                                    input_data=['input'],
                                    cache_dirpath=config.env.cache_dirpath)

        image_column_valid = Step(name='image_column_valid',
                                  transformer=image_column,
                                  input_steps=[non_nan_subset_valid],
                                  cache_dirpath=config.env.cache_dirpath)

        label_encoder_image_valid = Step(name='label_encoder_image_valid',
                                         transformer=label_encoder_image,
                                         input_steps=[non_nan_subset_valid],
                                         adapter={'categorical_features': ([(non_nan_subset_valid.name, 'y')]),
                                                  },
                                         cache_dirpath=config.env.cache_dirpath)

        loader = Step(name='loader',
                      transformer=MultiOutputImageLoader(**config.loader),
                      input_data=['specs'],
                      input_steps=[image_column, label_encoder_image,
                                   image_column_valid, label_encoder_image_valid],
                      adapter={'X': ([(image_column.name, 'X')]),
                               'y': ([(label_encoder_image.name, 'categorical_features')]),
                               'image_dir': ([('specs', 'image_dir')]),
                               'X_valid': ([(image_column_valid.name, 'X')]),
                               'y_valid': ([(label_encoder_image_valid.name, 'categorical_features')]),
                               },
                      cache_dirpath=config.env.cache_dirpath)

        inception_resnet = Step(name='inception_resnet',
                                transformer=HierarchicalInceptionResnet(**config.inception_resnet),
                                input_steps=[loader],
                                cache_dirpath=config.env.cache_dirpath)

        loader_valid = Step(name='loader_valid',
                            transformer=loader,
                            input_data=['specs'],
                            input_steps=[image_column_valid, label_encoder_image_valid],
                            adapter={'X': ([(image_column.name, 'X')]),
                                     'y': ([(label_encoder_image.name, 'categorical_features')]),
                                     'image_dir': ([('specs', 'image_dir')]),
                                     },
                            cache_dirpath=config.env.cache_dirpath)

        inception_resnet_valid = Step(name='inception_resnet_valid',
                                      transformer=inception_resnet,
                                      input_steps=[loader_valid],
                                      cache_dirpath=config.env.cache_dirpath)

        joined_image_predictions = Step(name='joined_image_predictions',
                                        transformer=fe.JoinWithNan(),
                                        input_steps=[inception_resnet, non_nan_subset],
                                        cache_dirpath=config.env.cache_dirpath)

        joined_image_predictions_valid = Step(name='joined_image_predictions_valid',
                                              transformer=joined_image_predictions,
                                              input_steps=[inception_resnet_valid, non_nan_subset_valid],
                                              cache_dirpath=config.env.cache_dirpath)

        return joined_image_predictions, joined_image_predictions_valid

    else:

        loader = Step(name='loader',
                      transformer=MultiOutputImageLoader(**config.loader),
                      input_data=['specs'],
                      input_steps=[image_column, label_encoder_image],
                      adapter={'X': ([(image_column.name, 'X')]),
                               'y': ([(label_encoder_image.name, 'categorical_features')]),
                               'image_dir': ([('specs', 'image_dir')]),
                               },
                      cache_dirpath=config.env.cache_dirpath)

        inception_resnet = Step(name='inception_resnet',
                                transformer=HierarchicalInceptionResnet(**config.inception_resnet),
                                input_steps=[loader],
                                cache_dirpath=config.env.cache_dirpath)

        joined_image_predictions = Step(name='joined_image_predictions',
                                        transformer=fe.JoinWithNan(),
                                        input_steps=[inception_resnet, non_nan_subset],
                                        cache_dirpath=config.env.cache_dirpath)

        return joined_image_predictions


def dataframe_features(dispatchers, config, train_mode, **kwargs):
    if train_mode:
        feature_by_type_split, feature_by_type_split_valid = dispatchers

        price_features, price_features_valid = _price_features(
            (feature_by_type_split, feature_by_type_split_valid),
            config, train_mode, **kwargs)

        groupby_aggregation, groupby_aggregation_valid = _groupby_aggregations(
            (feature_by_type_split, feature_by_type_split_valid),
            config, train_mode, **kwargs)
        target_encoder, target_encoder_valid = _target_encoders((feature_by_type_split, feature_by_type_split_valid),
                                                                config, train_mode, **kwargs)

        label_encoder, label_encoder_valid = _label_encoders((feature_by_type_split, feature_by_type_split_valid),
                                                             config, train_mode, **kwargs)

        return (price_features, groupby_aggregation, target_encoder, label_encoder), (
            price_features_valid, groupby_aggregation_valid, target_encoder_valid, label_encoder_valid)
    else:
        feature_by_type_split = dispatchers

        price_features = _price_features(feature_by_type_split, config, train_mode, **kwargs)
        groupby_aggregation = _groupby_aggregations(feature_by_type_split, config, train_mode, **kwargs)
        target_encoder = _target_encoders(feature_by_type_split, config, train_mode, **kwargs)
        label_encoder = _target_encoders(feature_by_type_split, config, train_mode, **kwargs)

        return price_features, groupby_aggregation, target_encoder, label_encoder


def image_features(config, train_mode=True):
    if train_mode:
        image_predictions, image_predictions_valid = image_model(config, train_mode=train_mode)
        image_features = Step(name='image_features',
                              transformer=fe.ImageFeatures(),
                              input_steps=[image_predictions],
                              cache_dirpath=config.env.cache_dirpath)

        image_features_valid = Step(name='image_features_valid',
                                    transformer=image_features,
                                    input_steps=[image_predictions],
                                    cache_dirpath=config.env.cache_dirpath)
        return image_features, image_features_valid
    else:
        image_predictions = image_model(config, train_mode=train_mode)
        image_features = Step(name='image_features',
                              transformer=fe.ImageFeatures(),
                              input_steps=[image_predictions],
                              cache_dirpath=config.env.cache_dirpath)
        return image_features


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


def _price_features(dispatchers, config, train_mode, **kwargs):
    if train_mode:
        feature_by_type_split, feature_by_type_split_valid = dispatchers
        price_features = Step(name='price_features',
                              transformer=Dummy(),
                              input_steps=[feature_by_type_split],
                              adapter={
                                  'numerical_features': ([(feature_by_type_split.name, 'numerical_features')])
                              },
                              cache_dirpath=config.env.cache_dirpath,
                              **kwargs)

        price_features_valid = Step(name='price_features_valid',
                                    transformer=price_features,
                                    input_steps=[feature_by_type_split_valid],
                                    adapter={'numerical_features': (
                                        [(feature_by_type_split_valid.name, 'numerical_features')])
                                    },
                                    cache_dirpath=config.env.cache_dirpath,
                                    **kwargs)

        return price_features, price_features_valid

    else:
        feature_by_type_split = dispatchers
        price_features = Step(name='price_features',
                              transformer=Dummy(),
                              input_steps=[feature_by_type_split],
                              adapter={
                                  'numerical_features': ([(feature_by_type_split.name, 'numerical_features')])
                              },
                              cache_dirpath=config.env.cache_dirpath,
                              **kwargs)

        return price_features


def _label_encoders(dispatchers, config, train_mode, **kwargs):
    if train_mode:
        feature_by_type_split, feature_by_type_split_valid = dispatchers
        label_encoder = Step(name='label_encoder',
                             transformer=fe.LabelEncoder(**config.label_encoder),
                             input_steps=[feature_by_type_split],
                             adapter={
                                 'categorical_features': ([(feature_by_type_split.name, 'categorical_features')]),
                             },
                             cache_dirpath=config.env.cache_dirpath,
                             **kwargs)

        label_encoder_valid = Step(name='label_encoder_valid',
                                   transformer=label_encoder,
                                   input_steps=[feature_by_type_split_valid],
                                   adapter={'categorical_features': (
                                       [(feature_by_type_split_valid.name, 'categorical_features')]),
                                   },
                                   cache_dirpath=config.env.cache_dirpath,
                                   **kwargs)

        return label_encoder, label_encoder_valid

    else:
        feature_by_type_split = dispatchers
        label_encoder = Step(name='label_encoder',
                             transformer=fe.LabelEncoder(**config.label_encoder),
                             input_steps=[feature_by_type_split],
                             adapter={
                                 'categorical_features': ([(feature_by_type_split.name, 'categorical_features')]),
                             },
                             cache_dirpath=config.env.cache_dirpath,
                             **kwargs)

        return label_encoder


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
                                    input_steps=[feature_by_type_split_valid],
                                    adapter={'categorical_features': (
                                        [(feature_by_type_split_valid.name, 'categorical_features')]),
                                    },
                                    cache_dirpath=config.env.cache_dirpath,
                                    **kwargs)

        return target_encoder, target_encoder_valid

    else:
        feature_by_type_split = dispatchers
        target_encoder = Step(name='target_encoder',
                              transformer=fe.TargetEncoderNSplits(**config.target_encoder),
                              input_steps=[feature_by_type_split],
                              adapter={
                                  'categorical_features': ([(feature_by_type_split.name, 'categorical_features')]),
                              },
                              cache_dirpath=config.env.cache_dirpath,
                              **kwargs)

        return target_encoder


def _groupby_aggregations(dispatchers, config, train_mode, **kwargs):
    if train_mode:
        feature_by_type_split, feature_by_type_split_valid = dispatchers
        groupby_aggregations = Step(name='groupby_aggregations',
                                    transformer=fe.GroupbyAggregations(**config.groupby_aggregation),
                                    input_steps=[feature_by_type_split],
                                    adapter={
                                        'categorical_features': ([(feature_by_type_split.name, 'categorical_features'),
                                                                  (feature_by_type_split.name, 'numerical_features')],
                                                                 pandas_concat_inputs)
                                    },
                                    cache_dirpath=config.env.cache_dirpath,
                                    **kwargs)

        groupby_aggregations_valid = Step(name='groupby_aggregations_valid',
                                          transformer=groupby_aggregations,
                                          input_steps=[feature_by_type_split_valid],
                                          adapter={'categorical_features': (
                                              [(feature_by_type_split_valid.name, 'categorical_features'),
                                               (feature_by_type_split_valid.name, 'numerical_features')],
                                              pandas_concat_inputs
                                          )
                                          },
                                          cache_dirpath=config.env.cache_dirpath,
                                          **kwargs)

        return groupby_aggregations, groupby_aggregations_valid

    else:
        feature_by_type_split = dispatchers
        groupby_aggregations = Step(name='groupby_aggregations',
                                    transformer=fe.GroupbyAggregations(**config.groupby_aggregation),
                                    input_steps=[feature_by_type_split],
                                    adapter={
                                        'categorical_features': ([(feature_by_type_split.name, 'categorical_features'),
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
             'image_features': {'train': partial(image_features, train_mode=True),
                                'inference': partial(image_features, train_mode=False)},
             }
