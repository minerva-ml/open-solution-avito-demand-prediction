import os
import re
import string
from multiprocessing import Pool

import category_encoders as ce
import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageStat
from sklearn.model_selection import KFold
from sklearn.externals import joblib
from sklearn import preprocessing as prep
from sklearn.feature_extraction import text
from scipy.sparse import hstack, csr_matrix

from steps.base import BaseTransformer
from steps.utils import get_logger

logger = get_logger()


class DataFrameByTypeSplitter(BaseTransformer):
    def __init__(self, numerical_columns, categorical_columns, timestamp_columns):
        self.numerical_columns = numerical_columns
        self.categorical_columns = categorical_columns
        self.timestamp_columns = timestamp_columns

    def transform(self, X, y=None, **kwargs):
        outputs = {}

        if self.numerical_columns is not None:
            outputs['numerical_features'] = X[self.numerical_columns]

        if self.categorical_columns is not None:
            outputs['categorical_features'] = X[self.categorical_columns]

        if self.timestamp_columns is not None:
            outputs['timestamp_features'] = X[self.timestamp_columns]

        return outputs


class FeatureJoiner(BaseTransformer):
    def transform(self, numerical_feature_list, categorical_feature_list, sparse_feature_list, **kwargs):
        features = numerical_feature_list + categorical_feature_list
        for feature in features:
            feature.reset_index(drop=True, inplace=True)
        dense_features = pd.concat(features, axis=1).astype(np.float32)

        outputs = {}
        if len(sparse_feature_list) != 0:
            sparse_features = hstack(sparse_feature_list)
            all_features = hstack([csr_matrix(dense_features.values), sparse_features])
            outputs['features'] = all_features
            outputs['feature_names'] = list(dense_features.columns) + self._get_sparse_names(sparse_features.shape)
            outputs['categorical_features'] = self._get_feature_names(categorical_feature_list)
        else:
            outputs['features'] = dense_features
            outputs['feature_names'] = self._get_feature_names(features)
            outputs['categorical_features'] = self._get_feature_names(categorical_feature_list)
        return outputs

    def _get_feature_names(self, dataframes):
        feature_names = []
        for dataframe in dataframes:
            try:
                feature_names.extend(list(dataframe.columns))
            except Exception as e:
                print(e)
                feature_names.append(dataframe.name)

        return feature_names

    def _get_sparse_names(self, shape):
        return ['sparse_feature_{}'.format(i) for i in range(shape[1])]


class CategoricalFilter(BaseTransformer):
    def __init__(self, categorical_columns, min_frequencies, impute_value=np.nan):
        self.categorical_columns = categorical_columns
        self.min_frequencies = min_frequencies
        self.impute_value = impute_value
        self.category_levels_to_remove = {}

    def fit(self, categorical_features):
        for column, threshold in zip(self.categorical_columns, self.min_frequencies):
            value_counts = categorical_features[column].value_counts()
            self.category_levels_to_remove[column] = value_counts[value_counts <= threshold].index.tolist()
        return self

    def transform(self, categorical_features):
        for column, levels_to_remove in self.category_levels_to_remove.items():
            if levels_to_remove:
                categorical_features[column].replace(levels_to_remove, self.impute_value, inplace=True)
            categorical_features['{}_infrequent'.format(column)] = categorical_features[column] == self.impute_value
            categorical_features['{}_infrequent'.format(column)] = categorical_features[
                '{}_infrequent'.format(column)].astype(int)
        return {'categorical_features': categorical_features}

    def load(self, filepath):
        params = joblib.load(filepath)
        self.categorical_columns = params['categorical_columns']
        self.min_frequencies = params['min_frequencies']
        self.impute_value = params['impute_value']
        self.category_levels_to_remove = params['category_levels_to_remove']
        return self

    def save(self, filepath):
        params = {}
        params['categorical_columns'] = self.categorical_columns
        params['min_frequencies'] = self.min_frequencies
        params['impute_value'] = self.impute_value
        params['category_levels_to_remove'] = self.category_levels_to_remove
        joblib.dump(params, filepath)


class TargetEncoderNSplits(BaseTransformer):
    def __init__(self, n_splits, **kwargs):
        self.k_folds = KFold(n_splits=n_splits)
        self.target_means_map = {}

    def _target_means_names(self, columns):
        confidence_rate_names = ['target_mean_{}'.format(column) for column in columns]
        return confidence_rate_names

    def _is_null_names(self, columns):
        is_null_names = ['target_mean_is_missing_{}'.format(column) for column in columns]
        return is_null_names

    def fit(self, categorical_features, target, **kwargs):
        feature_columns, target_column = categorical_features.columns, target.columns[0]

        X_target_means = []
        self.k_folds.get_n_splits(target)
        for train_index, test_index in self.k_folds.split(target):
            X_train, y_train = categorical_features.iloc[train_index], target.iloc[train_index]
            X_test, y_test = categorical_features.iloc[test_index], target.iloc[test_index]

            train = pd.concat([X_train, y_train], axis=1)
            for column, target_mean_name in zip(feature_columns, self._target_means_names(feature_columns)):
                group_object = train.groupby(column)
                train_target_means = group_object[target_column].mean(). \
                    reset_index().rename(index=str, columns={target_column: target_mean_name})

                X_test = X_test.merge(train_target_means, on=column, how='left')
            X_target_means.append(X_test)
        X_target_means = pd.concat(X_target_means, axis=0)

        for column, target_mean_name in zip(feature_columns, self._target_means_names(feature_columns)):
            group_object = X_target_means.groupby(column)
            self.target_means_map[column] = group_object[target_mean_name].mean().reset_index()

        return self

    def transform(self, categorical_features, **kwargs):
        columns = categorical_features.columns

        for column, target_mean_name, is_null_name in zip(columns,
                                                          self._target_means_names(columns),
                                                          self._is_null_names(columns)):
            categorical_features = categorical_features.merge(self.target_means_map[column],
                                                              on=column,
                                                              how='left')
            categorical_features[is_null_name] = pd.isnull(categorical_features[target_mean_name]).astype(int)
            categorical_features[target_mean_name].fillna(0, inplace=True)

        return {'numerical_features': categorical_features[self._target_means_names(columns)].astype(np.float32),
                'categorical_features': categorical_features[self._is_null_names(columns)]}

    def load(self, filepath):
        self.target_means_map = joblib.load(filepath)
        return self

    def save(self, filepath):
        joblib.dump(self.target_means_map, filepath)


class TargetEncoder(BaseTransformer):
    def __init__(self, **kwargs):
        self.params = kwargs
        self.encoder_class = ce.TargetEncoder

    def fit(self, X, y, **kwargs):
        categorical_columns = list(X.columns)
        self.target_encoder = self.encoder_class(cols=categorical_columns, **self.params)
        self.target_encoder.fit(X, y)
        return self

    def transform(self, X, y=None, **kwargs):
        X_ = self.target_encoder.transform(X)
        return {'numerical_features': X_}

    def load(self, filepath):
        self.target_encoder = joblib.load(filepath)
        return self

    def save(self, filepath):
        joblib.dump(self.target_encoder, filepath)


class OrdinalEncoder(BaseTransformer):
    def __init__(self, **kwargs):
        self.params = kwargs
        self.encoder_class = ce.ordinal.OrdinalEncoder

    def fit(self, categorical_features, **kwargs):
        categorical_columns = list(categorical_features.columns)
        self.encoder = self.encoder_class(cols=categorical_columns, **self.params)
        self.encoder.fit(categorical_features)
        return self

    def transform(self, categorical_features, **kwargs):
        X_ = self.encoder.transform(categorical_features)
        return {'categorical_features': X_}

    def load(self, filepath):
        self.encoder = joblib.load(filepath)
        return self

    def save(self, filepath):
        joblib.dump(self.encoder, filepath)


class BinaryEncoder(BaseTransformer):
    def __init__(self, **kwargs):
        self.params = kwargs
        self.encoder_class = ce.binary.BinaryEncoder

    def fit(self, X, **kwargs):
        categorical_columns = list(X.columns)
        self.binary_encoder = self.encoder_class(cols=categorical_columns, **self.params)
        self.binary_encoder.fit(X)
        return self

    def transform(self, X, **kwargs):
        X_ = self.binary_encoder.transform(X)
        return {'numerical_features': X_}

    def load(self, filepath):
        self.target_encoder = joblib.load(filepath)
        return self

    def save(self, filepath):
        joblib.dump(self.target_encoder, filepath)


class TextFeatures(BaseTransformer):
    def __init__(self, cols):
        self.cols = cols

    def transform(self, X):
        X_text_features = self._extract_text_features(X)
        X_all_features = pd.concat(X_text_features, axis=1)
        return {'numerical_features': X_all_features}

    def _extract_text_features(self, X):
        X_text_features = []
        for text_column in self.cols:
            logger.info('processing {}'.format(text_column))
            X_text = X[[text_column]].astype(str)
            X_text = X_text[text_column].apply(self._extract_first_level)
            X_text = self._extract_second_level(X_text)
            X_text.columns = ['{}_{}'.format(text_column, col) for col in X_text.columns]
            X_text.fillna(0.0, inplace=True)
            X_text_features.append(X_text)
        return X_text_features

    def _extract_first_level(self, x):
        features = {}
        features['char_count'] = len(x)
        features['word_count'] = len(x.split())
        features['punctuation_count'] = sum([1 for i in x if i in string.punctuation])
        features['upper_case_count'] = sum(c.isupper() for c in x)
        features['lower_case_count'] = sum(c.islower() for c in x)
        features['digit_count'] = sum(c.isdigit() for c in x)
        features['space_count'] = sum(c.isspace() for c in x)
        features['newline_count'] = x.count('\n')
        features['num_symbols'] = sum(x.count(w) for w in '*&$%')
        features['num_words'] = len(x.split())
        features['num_unique_words'] = len(set(w for w in x.split()))
        features['mean_word_len'] = np.mean([len(w) for w in x.split()])
        return pd.Series(features)

    def _extract_second_level(self, X):
        X['caps_vs_length'] = X['upper_case_count'].astype(float) / X['char_count'].astype(float)
        X['words_vs_unique'] = X['num_unique_words'].astype(float) / X['num_words'].astype(float)
        return X


class WordOverlap(BaseTransformer):
    def __init__(self, overlap_cols):
        self.overlap_cols = overlap_cols

    def transform(self, X):
        X_overlap_features = self._extract_overlap_features(X)
        X_overlap_features = pd.concat(X_overlap_features, axis=1)
        return {'numerical_features': X_overlap_features}

    def _extract_overlap_features(self, X):
        X_overlap_features = []
        for text_col1, text_col2 in self.overlap_cols:
            logger.info('processing {} with {}'.format(text_col1, text_col2))
            X_overlap = self._word_overlap(X[[text_col1, text_col2]].astype(str))
            X_overlap_features.append(X_overlap)
        return X_overlap_features

    def _word_overlap(self, X):
        col1, col2 = list(X.columns)

        def overlap(x):
            words1, words2 = x[col1].lower().split(), x[col2].lower().split()
            return len(set(words1) & set(words2))

        X['overlap_{}_{}'.format(col1, col2)] = X.apply(overlap, axis=1)
        X.drop([col1, col2], axis=1, inplace=True)
        return X


class TextCleaner(BaseTransformer):
    def __init__(self, text_features, drop_punctuation, all_lower_case):
        self.text_features = text_features
        self.drop_punctuation = drop_punctuation
        self.all_lower_case = all_lower_case

    @property
    def text_cleaner_names(self):
        text_cleaner_names = ['{}_clean'.format(feature) for feature in self.text_features]
        return text_cleaner_names

    def transform(self, X):
        X = pd.DataFrame(X, columns=self.text_features).astype(str)
        for feature, text_cleaner_name in zip(self.text_features, self.text_cleaner_names):
            X[text_cleaner_name] = X[feature].apply(self._transform)
        return {'categorical_features': X[self.text_cleaner_names]}

    def _transform(self, x):
        if self.all_lower_case:
            x = x.lower()
        if self.drop_punctuation:
            x = re.sub(r'[^\w\s]', ' ', x)
        return x


class TimeDelta(BaseTransformer):
    def __init__(self, groupby_specs, timestamp_column):
        self.groupby_specs = groupby_specs
        self.timestamp_column = timestamp_column

    @property
    def time_delta_names(self):
        time_delta_names = ['time_delta_{}'.format('_'.join(groupby_spec))
                            for groupby_spec in self.groupby_specs]
        return time_delta_names

    @property
    def is_null_names(self):
        is_null_names = ['time_delta_is_missing_{}'.format('_'.join(groupby_spec))
                         for groupby_spec in self.groupby_specs]
        return is_null_names

    def transform(self, categorical_features, timestamp_features):
        X = pd.concat([categorical_features, timestamp_features], axis=1)
        for groupby_spec, time_delta_name, is_null_name in zip(self.groupby_specs,
                                                               self.time_delta_names,
                                                               self.is_null_names):
            X[time_delta_name] = X.groupby(groupby_spec)[self.timestamp_column].apply(self._time_delta).reset_index(
                level=list(range(len(groupby_spec))), drop=True)
            X[is_null_name] = pd.isnull(X[time_delta_name]).astype(int)
            X[time_delta_name].fillna(0, inplace=True)
        return {'numerical_features': X[self.time_delta_names].astype(np.float32),
                'categorical_features': X[self.is_null_names]}

    def _time_delta(self, groupby_object):
        if len(groupby_object) == 1:
            return pd.Series(np.nan, index=groupby_object.index)
        else:
            groupby_object = groupby_object.sort_values().diff().dt.seconds
            return groupby_object


class GroupbyAggregations(BaseTransformer):
    def __init__(self, groupby_aggregations):
        self.groupby_aggregations = groupby_aggregations

    @property
    def groupby_aggregations_names(self):
        groupby_aggregations_names = ['{}_{}_{}'.format('_'.join(spec['groupby']), spec['agg'], spec['select'])
                                      for spec in self.groupby_aggregations]
        return groupby_aggregations_names

    def transform(self, X, **kwargs):
        for spec, groupby_aggregations_name in zip(self.groupby_aggregations, self.groupby_aggregations_names):
            logger.info('processing {}'.format(groupby_aggregations_name))
            group_object = X.groupby(spec['groupby'])

            X = X.merge(
                group_object[spec['select']].agg(spec['agg']).reset_index().rename(index=str, columns={
                    spec['select']: groupby_aggregations_name})[spec['groupby'] + [groupby_aggregations_name]],
                on=spec['groupby'], how='left')

        return {'numerical_features': X[self.groupby_aggregations_names].astype(np.float32)}


class IsMissing(BaseTransformer):
    def __init__(self, columns):
        self.columns = columns

    @property
    def missing_names(self):
        return ['{}_is_missing'.format(col) for col in self.columns]

    def transform(self, X, **kwargs):
        for name, missing_name in zip(self.columns, self.missing_names):
            X[missing_name] = pd.isnull(X[name]).astype(int)
        return {'categorical_features': X[self.missing_names]}


class HashingCategoricalEncoder(BaseTransformer):
    def __init__(self, **kwargs):
        self.hashing_encoder = ce.HashingEncoder(**kwargs)

    def fit(self, categorical_features, **kwargs):
        self.hashing_encoder.fit(categorical_features)
        return self

    def transform(self, categorical_features, **kwargs):
        categorical_features = self.hashing_encoder.transform(categorical_features)
        return {'categorical_features': categorical_features}

    def load(self, filepath):
        self.hashing_encoder = joblib.load(filepath)
        return self

    def save(self, filepath):
        joblib.dump(self.hashing_encoder, filepath)


class CategoricalEncoder(BaseTransformer):
    def __init__(self, columns_to_encode):
        self.columns_to_encode = columns_to_encode
        self.columns_with_encoders = [(col_name, prep.LabelEncoder()) for col_name in columns_to_encode]

    def fit(self, categorical_features, **kwargs):
        for column_name, encoder in self.columns_with_encoders:
            logger.info('fitting {}'.format(column_name))
            encoder.fit(categorical_features[column_name].astype(str).values)
        return self

    def transform(self, categorical_features, **kwargs):
        for column_name, encoder in self.columns_with_encoders:
            logger.info('transforming {}'.format(column_name))
            categorical_features[column_name], encoder = self._input_unknown(categorical_features[column_name], encoder)
            categorical_features[column_name] = encoder.transform(categorical_features[column_name].astype(str).values)
        return {'categorical_features': categorical_features}

    def _input_unknown(self, column, encoder):
        def func(x):
            return '<unknown>' if x not in encoder.classes_ else x

        column = column.apply(func)
        encoder.classes_ = np.append(encoder.classes_, '<unknown>')
        return column, encoder

    def load(self, filepath):
        self.columns_with_encoders = joblib.load(filepath)
        return self

    def save(self, filepath):
        joblib.dump(self.columns_with_encoders, filepath)


class DateFeatures(BaseTransformer):
    def __init__(self, date_column):
        self.date_column = date_column

    @property
    def date_features_names(self):
        date_features_names = ['{}_month'.format(self.date_column),
                               '{}_day'.format(self.date_column),
                               '{}_weekday'.format(self.date_column),
                               '{}_week'.format(self.date_column),
                               ]
        return date_features_names

    def transform(self, timestamp_features, **kwargs):
        date_index = pd.DatetimeIndex(timestamp_features[self.date_column])
        timestamp_features['{}_month'.format(self.date_column)] = date_index.month
        timestamp_features['{}_day'.format(self.date_column)] = date_index.day
        timestamp_features['{}_weekday'.format(self.date_column)] = date_index.weekday
        timestamp_features['{}_week'.format(self.date_column)] = date_index.week
        return {'categorical_features': timestamp_features[self.date_features_names].astype(int)}


class ProcessNumerical(BaseTransformer):
    def transform(self, numerical_features, **kwargs):
        numerical_features['price'] = np.log1p(numerical_features['price'].values)
        return {'numerical_features': numerical_features}


class Blacklist(BaseTransformer):
    def __init__(self, blacklist):
        self.blacklist = blacklist

    @property
    def blacklist_names(self):
        blacklist_names = ['{}_on_blacklist'.format(category) for category in self.blacklist]
        return blacklist_names

    def transform(self, categorical_features):
        for category, blacklist_name in zip(self.blacklist, self.blacklist_names):
            categorical_features[blacklist_name] = (
                categorical_features[category].isin(self.blacklist[category])).astype(int)

        return {'categorical_features': categorical_features[self.blacklist_names]}


class ConfidenceRate(BaseTransformer):
    def __init__(self, confidence_level=100, categories=[]):
        self.confidence_level = confidence_level
        self.categories = categories
        self.confidence_rates_map = {}

    @property
    def confidence_rate_names(self):
        confidence_rate_names = ['confidence_rate_{}'.format('_'.join(category))
                                 for category in self.categories]
        return confidence_rate_names

    @property
    def is_null_names(self):
        is_null_names = ['confidence_rate_is_nan_{}'.format('_'.join(category))
                         for category in self.categories]
        return is_null_names

    def fit(self, categorical_features, target):
        concatenated_dataframe = pd.concat([categorical_features, target], axis=1)

        for category, confidence_rate_name in zip(self.categories, self.confidence_rate_names):
            group_object = concatenated_dataframe.groupby(category)

            self.confidence_rates_map['_'.join(category)] = \
                group_object['is_attributed'].apply(self._rate_calculation).reset_index().rename(
                    index=str,
                    columns={'is_attributed': confidence_rate_name})[category + [confidence_rate_name]]

        return self

    def transform(self, categorical_features, **kwargs):

        for category, confidence_rate_name, is_null_name in zip(self.categories,
                                                                self.confidence_rate_names,
                                                                self.is_null_names):
            categorical_features = categorical_features.merge(self.confidence_rates_map['_'.join(category)],
                                                              on=category,
                                                              how='left')
            categorical_features[is_null_name] = pd.isnull(categorical_features[confidence_rate_name]).astype(int)
            categorical_features[confidence_rate_name].fillna(0, inplace=True)

        return {'numerical_features': categorical_features[self.confidence_rate_names].astype(np.float32),
                'categorical_features': categorical_features[self.is_null_names]}

    def load(self, filepath):
        self.confidence_rates_map = joblib.load(filepath)
        return self

    def save(self, filepath):
        joblib.dump(self.confidence_rates_map, filepath)

    def _rate_calculation(self, x):
        rate = x.sum() / float(x.count())
        confidence = np.min([1, np.log(x.count()) / np.log(self.confidence_level)])

        return rate * confidence * 100


class MultiColumnTfidfVectorizer(BaseTransformer):
    def __init__(self, cols_params):
        self.cols_vectorizers = self._get_vectorizers(cols_params)

    def fit(self, X, **kwargs):
        for col, vectorizer in self.cols_vectorizers:
            vectorizer.fit(X[col].values)
        return self

    def transform(self, X, **kwargs):
        sparse_features = []
        for col, vectorizer in self.cols_vectorizers:
            sparse_feature = vectorizer.transform(X[col].values)
            sparse_features.append(sparse_feature)
        sparse_features = hstack(sparse_features)
        return {'sparse_features': sparse_features}

    def load(self, filepath):
        self.cols_vectorizers = joblib.load(filepath)
        return self

    def save(self, filepath):
        joblib.dump(self.cols_vectorizers, filepath)

    def _get_vectorizers(self, cols_params):
        return [(col, text.TfidfVectorizer(**params)) for col, params in cols_params]


class ImageStatistics(BaseTransformer):
    PIL_FEATURES_NR = 12
    CV2_FEATURES_NR = 262

    def __init__(self, cols, img_dir_train, img_dir_test, n_jobs, log_features):
        self.cols = cols
        self.img_dir_train = img_dir_train
        self.img_dir_test = img_dir_test
        self.n_jobs = n_jobs
        self.log_features = log_features

    def transform(self, X, is_train, **kwargs):
        self.train_mode = is_train
        numerical_features = []
        for col in self.cols:
            numerical_features_col = self._get_column_image_stats(X[col], col)
            numerical_features.append(numerical_features_col)
        numerical_features = pd.concat(numerical_features, axis=1)
        if self.log_features:
            numerical_features = np.log1p(numerical_features)
        return {'numerical_features': numerical_features}

    def _pil_feature_names(self, colname):
        pil_feature_names = ['{}_pil_image_stat_{}'.format(colname, i)
                             for i in range(ImageStatistics.PIL_FEATURES_NR)]
        return pil_feature_names

    def _cv2_feature_names(self, colname):
        cv2_feature_names = ['{}_cv2_image_stat_{}'.format(colname, i)
                             for i in range(ImageStatistics.CV2_FEATURES_NR)]
        return cv2_feature_names

    def _get_column_image_stats(self, image_col, column_name):
        filepaths = [self._get_filepath(filename) for filename in image_col]

        with Pool(self.n_jobs) as executor:
            image_features = executor.map(extract_image_stats, filepaths)

        image_features = np.vstack(image_features)
        feature_names = self._pil_feature_names(column_name) + self._cv2_feature_names(column_name)
        return pd.DataFrame(image_features, columns=feature_names)

    def _get_filepath(self, filename):
        img_dir_path = self.img_dir_train if self.train_mode else self.img_dir_test
        filepath = os.path.join(img_dir_path, '{}.jpg'.format(filename))
        return filepath


def extract_image_stats(filepath):
    try:
        pil_img_stats = get_pil_image_stats(filepath)
        cv2_img_stats = get_cv2_image_stats(filepath)
    except Exception:
        pil_img_stats = [0] * ImageStatistics.PIL_FEATURES_NR
        cv2_img_stats = [0] * ImageStatistics.CV2_FEATURES_NR
    return np.hstack([pil_img_stats, cv2_img_stats])


def get_pil_image_stats(filepath):
    image = Image.open(filepath, 'r')
    img_stats_ = ImageStat.Stat(image)
    stats = []
    stats += img_stats_.mean
    stats += img_stats_.rms
    stats += img_stats_.var
    stats += img_stats_.stddev
    return stats


def get_cv2_image_stats(filepath):
    img = cv2.imread(filepath)
    bw = cv2.imread(filepath, 0)
    pixel_nr = float(bw.shape[0] * bw.shape[1])
    stats = []
    stats += list(cv2.calcHist([bw], [0], None, [256], [0, 256]).flatten() / pixel_nr)
    mean, std = cv2.meanStdDev(img)
    stats += list(mean)
    stats += list(std)
    stats += cv2.Laplacian(bw, cv2.CV_64F).var()
    stats += (bw < 10).mean()
    stats += (bw > 245).mean()
    return stats
