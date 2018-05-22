from functools import partial

from feature_cleaning import InputMissing
import feature_extraction as fe
from hyperparameter_tuning import RandomSearchOptimizer, NeptuneMonitor, SaveResults
from steps.adapters import to_numpy_label_inputs, identity_inputs
from steps.base import Step, Dummy
from models import LightGBMLowMemory as LightGBM
from postprocessing import Clipper
from utils import root_mean_squared_error, pandas_concat_inputs, pandas_subset_columns
import pipeline_config as cfg


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
                   adapter={'prediction': ([(light_gbm.name, 'prediction')]), },
                   cache_dirpath=config.env.cache_dirpath)

    output = Step(name='output',
                  transformer=Dummy(),
                  input_steps=[clipper],
                  adapter={'y_pred': ([(clipper.name, 'clipped_prediction')]), },
                  cache_dirpath=config.env.cache_dirpath)
    return output


def feature_extraction(config, train_mode, **kwargs):
    if train_mode:
        is_missing, is_missing_valid = _is_missing_features(config, train_mode, **kwargs)
        cleaned, cleaned_valid = _clean_features(config, train_mode)

        dataframe_features_train, dataframe_features_valid = dataframe_features(
            (cleaned, cleaned_valid), config, train_mode, **kwargs)
        categorical, timestamp, numerical, group_by, target_encoder = dataframe_features_train
        categorical_valid, timestamp_valid, numerical_valid, group_by_valid, target_encoder_valid = dataframe_features_valid

        text, text_valid = text_features((cleaned, cleaned_valid), config, train_mode, **kwargs)
        hand_crafted_text, word_overlap, tfidf = text
        hand_crafted_text_valid, word_overlap_valid, tfidf_valid = text_valid

        image_stats, image_stats_valid = image_features((cleaned, cleaned_valid), config, train_mode, **kwargs)

        feature_combiner, feature_combiner_valid = _join_features(numerical_features=[numerical,
                                                                                      target_encoder,
                                                                                      group_by,
                                                                                      hand_crafted_text,
                                                                                      word_overlap,
                                                                                      image_stats],
                                                                  numerical_features_valid=[numerical_valid,
                                                                                            target_encoder_valid,
                                                                                            group_by_valid,
                                                                                            hand_crafted_text_valid,
                                                                                            word_overlap_valid,
                                                                                            image_stats_valid],
                                                                  categorical_features=[timestamp,
                                                                                        is_missing,
                                                                                        categorical,
                                                                                        target_encoder],
                                                                  categorical_features_valid=[timestamp_valid,
                                                                                              is_missing_valid,
                                                                                              categorical_valid,
                                                                                              target_encoder_valid],
                                                                  sparse_features=[tfidf],
                                                                  sparse_features_valid=[tfidf_valid],
                                                                  config=config, train_mode=train_mode, **kwargs)
        return feature_combiner, feature_combiner_valid
    else:
        is_missing = _is_missing_features(config, train_mode, **kwargs)
        cleaned = _clean_features(config, train_mode)

        categorical, timestamp, prices, group_by, target_encoder = dataframe_features(
            cleaned, config, train_mode, **kwargs)

        hand_crafted_text, word_overlap, tfidf = text_features(cleaned, config, train_mode, **kwargs)

        image_stats = image_features(cleaned, config, train_mode, **kwargs)

        feature_combiner = _join_features(
            numerical_features=[prices, target_encoder, group_by, hand_crafted_text, word_overlap, image_stats],
            numerical_features_valid=[],
            categorical_features=[timestamp, is_missing, categorical, target_encoder],
            categorical_features_valid=[],
            sparse_features=[tfidf],
            sparse_features_valid=[],
            config=config, train_mode=train_mode, **kwargs)
        return feature_combiner


def dataframe_features(clean_features, config, train_mode, **kwargs):
    if train_mode:
        clean, clean_valid = clean_features

        encoded_categorical, encoded_categorical_valid = _encode_categorical(
            (clean, clean_valid),
            config, train_mode, **kwargs)

        timestamp_features, timestamp_features_valid = _timestamp_features(
            (clean, clean_valid),
            config, train_mode, **kwargs)

        numerical_features, numerical_features_valid = _numerical_features(
            (clean, clean_valid),
            config, train_mode, **kwargs)

        groupby_aggregation, groupby_aggregation_valid = _groupby_aggregations(
            (clean, clean_valid), (timestamp_features, timestamp_features_valid),
            config, train_mode, **kwargs)
        target_encoder, target_encoder_valid = _target_encoders((clean, clean_valid),
                                                                config, train_mode, **kwargs)
        train_features = (encoded_categorical,
                          timestamp_features,
                          numerical_features,
                          groupby_aggregation,
                          target_encoder)
        valid_features = (encoded_categorical_valid,
                          timestamp_features_valid,
                          numerical_features_valid,
                          groupby_aggregation_valid,
                          target_encoder_valid)
        return train_features, valid_features
    else:
        clean = clean_features

        encoded_categorical = _encode_categorical(clean, config, train_mode, **kwargs)
        timestamp_features = _timestamp_features(clean, config, train_mode, **kwargs)
        numerical_features = _numerical_features(clean, config, train_mode, **kwargs)
        groupby_aggregation = _groupby_aggregations(clean, timestamp_features, config, train_mode, **kwargs)
        target_encoder = _target_encoders(clean, config, train_mode, **kwargs)

        train_features = (encoded_categorical,
                          timestamp_features,
                          numerical_features,
                          groupby_aggregation,
                          target_encoder)
        return train_features


def text_features(clean_features, config, train_mode, **kwargs):
    if train_mode:
        clean, clean_valid = clean_features

        hand_crafted_text = Step(name='hand_crafted_text',
                                 transformer=fe.TextFeatures(**config.text_features),
                                 input_steps=[clean],
                                 adapter={'X': ([(clean.name, 'clean_features')])},
                                 cache_dirpath=config.env.cache_dirpath, **kwargs)

        hand_crafted_text_valid = Step(name='hand_crafted_text_valid',
                                       transformer=hand_crafted_text,
                                       input_steps=[clean_valid],
                                       adapter={'X': ([(clean_valid.name, 'clean_features')])},
                                       cache_dirpath=config.env.cache_dirpath, **kwargs)

        word_overlap = Step(name='word_overlap',
                            transformer=fe.WordOverlap(**config.word_overlap),
                            input_steps=[clean],
                            adapter={'X': ([(clean.name, 'clean_features')])},
                            cache_dirpath=config.env.cache_dirpath, **kwargs)

        word_overlap_valid = Step(name='word_overlap_valid',
                                  transformer=word_overlap,
                                  input_steps=[clean_valid],
                                  adapter={'X': ([(clean_valid.name, 'clean_features')])},
                                  cache_dirpath=config.env.cache_dirpath, **kwargs)

        tfidf = Step(name='tfidf',
                     transformer=fe.MultiColumnTfidfVectorizer(**config.tfidf),
                     input_steps=[clean],
                     adapter={'X': ([(clean.name, 'clean_features')])},
                     cache_dirpath=config.env.cache_dirpath, **kwargs)

        tfidf_valid = Step(name='tfidf_valid',
                           transformer=tfidf,
                           input_steps=[clean_valid],
                           adapter={'X': ([(clean_valid.name, 'clean_features')])},
                           cache_dirpath=config.env.cache_dirpath, **kwargs)

        return (hand_crafted_text, word_overlap, tfidf), (hand_crafted_text_valid, word_overlap_valid, tfidf_valid)

    else:
        clean = clean_features

        hand_crafted_text = Step(name='hand_crafted_text',
                                 transformer=fe.TextFeatures(**config.text_features),
                                 input_steps=[clean],
                                 adapter={'X': ([(clean.name, 'clean_features')])},
                                 cache_dirpath=config.env.cache_dirpath, **kwargs)

        word_overlap = Step(name='word_overlap',
                            transformer=fe.WordOverlap(**config.word_overlap),
                            input_steps=[clean],
                            adapter={'X': ([(clean.name, 'clean_features')])},
                            cache_dirpath=config.env.cache_dirpath, **kwargs)

        tfidf = Step(name='tfidf',
                     transformer=fe.MultiColumnTfidfVectorizer(**config.tfidf),
                     input_steps=[clean],
                     adapter={'X': ([(clean.name, 'clean_features')])},
                     cache_dirpath=config.env.cache_dirpath, **kwargs)

        return hand_crafted_text, word_overlap, tfidf


def image_features(clean_features, config, train_mode, **kwargs):
    if train_mode:
        clean, clean_valid = clean_features

        image_stats = Step(name='image_stats',
                           transformer=fe.ImageStatistics(**config.image_stats),
                           input_data=['specs'],
                           input_steps=[clean],
                           adapter={'X': ([(clean.name, 'clean_features')]),
                                    'is_train': ([('specs', 'is_train')])},
                           cache_dirpath=config.env.cache_dirpath, **kwargs)

        image_stats_valid = Step(name='image_stats_valid',
                                 transformer=image_stats,
                                 input_data=['specs'],
                                 input_steps=[clean_valid],
                                 adapter={'X': ([(clean_valid.name, 'clean_features')]),
                                          'is_train': ([('specs', 'is_train')])},
                                 cache_dirpath=config.env.cache_dirpath, **kwargs)

        return image_stats, image_stats_valid

    else:
        clean = clean_features

        image_stats = Step(name='image_stats',
                           transformer=fe.ImageStatistics(**config.image_stats),
                           input_data=['specs'],
                           input_steps=[clean],
                           adapter={'X': ([(clean.name, 'clean_features')]),
                                    'is_train': ([('specs', 'is_train')])},
                           cache_dirpath=config.env.cache_dirpath, **kwargs)

        return image_stats


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
                         cache_dirpath=config.env.cache_dirpath, **kwargs)
    else:
        light_gbm = Step(name='light_gbm',
                         transformer=LightGBM(**config.light_gbm),
                         input_steps=[features],
                         adapter={'X': ([(features.name, 'features')]), },
                         cache_dirpath=config.env.cache_dirpath, **kwargs)
    return light_gbm


def _clean_features(config, train_mode):
    if train_mode:

        input_missing = Step(name='input_missing',
                             transformer=InputMissing(**config.input_missing),
                             input_data=['input'],
                             adapter={'X': ([('input', 'X')]), },
                             cache_dirpath=config.env.cache_dirpath)

        input_missing_valid = Step(name='input_missing_valid',
                                   transformer=input_missing,
                                   input_data=['input'],
                                   adapter={'X': ([('input', 'X_valid')]), },
                                   cache_dirpath=config.env.cache_dirpath)

        return input_missing, input_missing_valid
    else:
        input_missing = Step(name='input_missing',
                             transformer=InputMissing(**config.input_missing),
                             input_data=['input'],
                             adapter={'X': ([('input', 'X')]), },
                             cache_dirpath=config.env.cache_dirpath)

    return input_missing


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


def _encode_categorical(clean_features, config, train_mode, **kwargs):
    if train_mode:
        clean, clean_valid = clean_features
        categorical_encoder = Step(name='categorical_encoder',
                                   transformer=fe.OrdinalEncoder(**config.categorical_encoder),
                                   input_steps=[clean],
                                   adapter={
                                       'categorical_features': (
                                           [(clean.name, 'clean_features')], partial(pandas_subset_columns,
                                                                                     cols=cfg.CATEGORICAL_COLUMNS))
                                   },
                                   cache_dirpath=config.env.cache_dirpath, **kwargs)

        categorical_encoder_valid = Step(name='categorical_encoder_valid',
                                         transformer=categorical_encoder,
                                         input_steps=[clean_valid],
                                         adapter={'categorical_features': (
                                             [(clean_valid.name, 'clean_features')], partial(pandas_subset_columns,
                                                                                             cols=cfg.CATEGORICAL_COLUMNS))
                                         },
                                         cache_dirpath=config.env.cache_dirpath, **kwargs)

        return categorical_encoder, categorical_encoder_valid

    else:
        clean = clean_features
        categorical_encoder = Step(name='categorical_encoder',
                                   transformer=fe.OrdinalEncoder(**config.categorical_encoder),
                                   input_steps=[clean],
                                   adapter={
                                       'categorical_features': (
                                           [(clean.name, 'clean_features')], partial(pandas_subset_columns,
                                                                                     cols=cfg.CATEGORICAL_COLUMNS))
                                   },
                                   cache_dirpath=config.env.cache_dirpath,
                                   **kwargs)

        return categorical_encoder


def _timestamp_features(clean_features, config, train_mode, **kwargs):
    if train_mode:
        clean, clean_valid = clean_features
        timestamp_features = Step(name='timestamp_features',
                                  transformer=fe.DateFeatures(**config.date_features),
                                  input_steps=[clean],
                                  adapter={
                                      'timestamp_features': (
                                          [(clean.name, 'clean_features')], partial(pandas_subset_columns,
                                                                                    cols=cfg.TIMESTAMP_COLUMNS))
                                  },
                                  cache_dirpath=config.env.cache_dirpath, **kwargs)

        timestamp_features_valid = Step(name='timestamp_features_valid',
                                        transformer=timestamp_features,
                                        input_steps=[clean_valid],
                                        adapter={'timestamp_features': (
                                            [(clean_valid.name, 'clean_features')], partial(pandas_subset_columns,
                                                                                            cols=cfg.TIMESTAMP_COLUMNS))
                                        },
                                        cache_dirpath=config.env.cache_dirpath, **kwargs)

        return timestamp_features, timestamp_features_valid

    else:
        clean = clean_features
        timestamp_features = Step(name='timestamp_features',
                                  transformer=fe.DateFeatures(**config.date_features),
                                  input_steps=[clean],
                                  adapter={
                                      'timestamp_features': (
                                          [(clean.name, 'clean_features')], partial(pandas_subset_columns,
                                                                                    cols=cfg.TIMESTAMP_COLUMNS))
                                  },
                                  cache_dirpath=config.env.cache_dirpath, **kwargs)

        return timestamp_features


def _numerical_features(clean_features, config, train_mode, **kwargs):
    if train_mode:
        clean, clean_valid = clean_features
        numerical_features = Step(name='numerical_features',
                                  transformer=fe.ProcessNumerical(),
                                  input_steps=[clean],
                                  adapter={
                                      'numerical_features': (
                                          [(clean.name, 'clean_features')], partial(pandas_subset_columns,
                                                                                    cols=cfg.NUMERICAL_COLUMNS))
                                  },
                                  cache_dirpath=config.env.cache_dirpath,
                                  **kwargs)

        numerical_features_valid = Step(name='numerical_features_valid',
                                        transformer=numerical_features,
                                        input_steps=[clean_valid],
                                        adapter={'numerical_features': (
                                            [(clean_valid.name, 'clean_features')], partial(pandas_subset_columns,
                                                                                            cols=cfg.NUMERICAL_COLUMNS))
                                        },
                                        cache_dirpath=config.env.cache_dirpath, **kwargs)

        return numerical_features, numerical_features_valid

    else:
        clean = clean_features
        numerical_features = Step(name='numerical_features',
                                  transformer=fe.ProcessNumerical(),
                                  input_steps=[clean],
                                  adapter={
                                      'numerical_features': (
                                          [(clean.name, 'clean_features')], partial(pandas_subset_columns,
                                                                                    cols=cfg.NUMERICAL_COLUMNS))
                                  },
                                  cache_dirpath=config.env.cache_dirpath, **kwargs)

        return numerical_features


def _target_encoders(clean_features, config, train_mode, **kwargs):
    if train_mode:
        clean, clean_valid = clean_features
        target_encoder = Step(name='target_encoder',
                              transformer=fe.TargetEncoderNSplits(**config.target_encoder),
                              input_data=['input'],
                              input_steps=[clean],
                              adapter={
                                  'categorical_features': (
                                      [(clean.name, 'clean_features')], partial(pandas_subset_columns,
                                                                                cols=cfg.CATEGORICAL_COLUMNS)),
                                  'target': ([('input', 'y')])
                              },
                              cache_dirpath=config.env.cache_dirpath, **kwargs)

        target_encoder_valid = Step(name='target_encoder_valid',
                                    transformer=target_encoder,
                                    input_data=['input'],
                                    input_steps=[clean_valid],
                                    adapter={'categorical_features': (
                                        [(clean_valid.name, 'clean_features')], partial(pandas_subset_columns,
                                                                                        cols=cfg.CATEGORICAL_COLUMNS)),
                                    },
                                    cache_dirpath=config.env.cache_dirpath, **kwargs)

        return target_encoder, target_encoder_valid

    else:
        clean = clean_features
        target_encoder = Step(name='target_encoder',
                              transformer=fe.TargetEncoderNSplits(**config.target_encoder),
                              input_data=['input'],
                              input_steps=[clean],
                              adapter={
                                  'categorical_features': (
                                      [(clean.name, 'clean_features')], partial(pandas_subset_columns,
                                                                                cols=cfg.CATEGORICAL_COLUMNS)),
                              },
                              cache_dirpath=config.env.cache_dirpath, **kwargs)

        return target_encoder


def _groupby_aggregations(clean_features, additional_features, config, train_mode, **kwargs):
    if train_mode:
        clean, clean_valid = clean_features
        added_feature, added_feature_valid = additional_features
        groupby_aggregations = Step(name='groupby_aggregations',
                                    transformer=fe.GroupbyAggregations(**config.groupby_aggregation),
                                    input_steps=[clean, added_feature],
                                    adapter={
                                        'X': ([(clean.name, 'clean_features'),
                                               (added_feature.name, 'categorical_features')],
                                              pandas_concat_inputs)
                                    },
                                    cache_dirpath=config.env.cache_dirpath, **kwargs)

        groupby_aggregations_valid = Step(name='groupby_aggregations_valid',
                                          transformer=groupby_aggregations,
                                          input_steps=[clean_valid, added_feature_valid],
                                          adapter={'X': ([(clean_valid.name, 'clean_features'),
                                                          (added_feature_valid.name, 'categorical_features')],
                                                         pandas_concat_inputs
                                                         )
                                                   },
                                          cache_dirpath=config.env.cache_dirpath, **kwargs)

        return groupby_aggregations, groupby_aggregations_valid

    else:
        clean = clean_features
        added_feature = additional_features
        groupby_aggregations = Step(name='groupby_aggregations',
                                    transformer=fe.GroupbyAggregations(**config.groupby_aggregation),
                                    input_steps=[clean, added_feature],
                                    adapter={
                                        'X': ([(clean.name, 'clean_features'),
                                               (added_feature.name, 'categorical_features')],
                                              pandas_concat_inputs)
                                    },
                                    cache_dirpath=config.env.cache_dirpath, **kwargs)

        return groupby_aggregations


def _join_features(numerical_features, numerical_features_valid,
                   categorical_features, categorical_features_valid,
                   sparse_features, sparse_features_valid,
                   config, train_mode, **kwargs):
    if train_mode:
        feature_joiner = Step(name='feature_joiner',
                              transformer=fe.FeatureJoiner(),
                              input_steps=numerical_features + categorical_features + sparse_features,
                              adapter={
                                  'numerical_feature_list': (
                                      [(feature.name, 'numerical_features') for feature in numerical_features],
                                      identity_inputs),
                                  'categorical_feature_list': (
                                      [(feature.name, 'categorical_features') for feature in categorical_features],
                                      identity_inputs),
                                  'sparse_feature_list': (
                                      [(feature.name, 'sparse_features') for feature in sparse_features],
                                      identity_inputs),
                              },
                              cache_dirpath=config.env.cache_dirpath, **kwargs)

        feature_joiner_valid = Step(name='feature_joiner_valid',
                                    transformer=feature_joiner,
                                    input_steps=numerical_features_valid + categorical_features_valid + sparse_features_valid,
                                    adapter={'numerical_feature_list': (
                                        [(feature.name, 'numerical_features') for feature in numerical_features_valid],
                                        identity_inputs),
                                        'categorical_feature_list': (
                                            [(feature.name, 'categorical_features') for feature in
                                             categorical_features_valid],
                                            identity_inputs),
                                        'sparse_feature_list': (
                                            [(feature.name, 'sparse_features') for feature in sparse_features_valid],
                                            identity_inputs),
                                    },
                                    cache_dirpath=config.env.cache_dirpath, **kwargs)

        return feature_joiner, feature_joiner_valid

    else:
        feature_joiner = Step(name='feature_joiner',
                              transformer=fe.FeatureJoiner(),
                              input_steps=numerical_features + categorical_features + sparse_features,
                              adapter={
                                  'numerical_feature_list': (
                                      [(feature.name, 'numerical_features') for feature in numerical_features],
                                      identity_inputs),
                                  'categorical_feature_list': (
                                      [(feature.name, 'categorical_features') for feature in categorical_features],
                                      identity_inputs),
                                  'sparse_feature_list': (
                                      [(feature.name, 'sparse_features') for feature in sparse_features],
                                      identity_inputs),
                              },
                              cache_dirpath=config.env.cache_dirpath, **kwargs)

        return feature_joiner


PIPELINES = {'main': {'train': partial(main, train_mode=True),
                      'inference': partial(main, train_mode=False)},
             }
