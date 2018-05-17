import os

from attrdict import AttrDict
from deepsense import neptune

from utils import read_params, safe_eval

ctx = neptune.Context()
params = read_params(ctx)

FEATURE_COLUMNS = ['user_id',
                   'region', 'city',
                   'parent_category_name', 'category_name',
                   'param_1', 'param_2', 'param_3',
                   'title', 'description',
                   'price',
                   'item_seq_number',
                   'activation_date',
                   'user_type',
                   'image',
                   'image_top_1']
CATEGORICAL_COLUMNS = ['user_id',
                       'region', 'city',
                       'parent_category_name', 'category_name',
                       'param_1', 'param_2', 'param_3',
                       'item_seq_number', 'user_type', 'image_top_1']
NUMERICAL_COLUMNS = ['price']
TEXT_COLUMNS = ['title', 'description']
IMAGE_COLUMNS = ['image']
TARGET_COLUMNS = ['deal_probability']
CV_COLUMN = ['user_id']
TIMESTAMP_COLUMNS = ['activation_date']
ITEM_ID_COLUMN = ['item_id']
USER_ID_COLUMN = ['user_id']

DEV_SAMPLE_SIZE = int(10e2)

COLUMN_TYPES = {'train': {'price': 'float64',
                          'item_seq_number': 'uint32',
                          'image_top_1': 'float64',
                          'deal_probability': 'float32',
                          },
                'inference': {'price': 'float64',
                              'item_seq_number': 'uint32',
                              'image_top_1': 'float64',
                              }
                }

SOLUTION_CONFIG = AttrDict({
    'env': {'cache_dirpath': params.experiment_dir
            },
    'random_search': {'light_gbm': {'n_runs': safe_eval(params.lgbm_random_search_runs),
                                    'callbacks': {'neptune_monitor': {'name': 'light_gbm'
                                                                      },
                                                  'save_results': {'filepath': os.path.join(params.experiment_dir,
                                                                                            'random_search_light_gbm.pkl')
                                                                   }
                                                  }
                                    }
                      },
    'dataframe_by_type_splitter': {'numerical_columns': NUMERICAL_COLUMNS,
                                   'categorical_columns': CATEGORICAL_COLUMNS,
                                   'timestamp_columns': TIMESTAMP_COLUMNS,
                                   },

    'date_features': {'date_column': TIMESTAMP_COLUMNS[0]},
    'is_missing': {'columns': FEATURE_COLUMNS},
    'categorical_encoder': {'cols': CATEGORICAL_COLUMNS,
                            'n_components': params.categorical_encoder__n_components,
                            'hash_method': params.categorical_encoder__hash_method
                            },

    'groupby_aggregation': {'groupby_aggregations': [
        {'groupby': ['user_id', 'activation_date_weekday'], 'select': 'price', 'agg': 'mean'},
        {'groupby': ['user_id'], 'select': 'price', 'agg': 'mean'},
        {'groupby': ['user_id'], 'select': 'price', 'agg': 'var'},
        {'groupby': ['user_id'], 'select': 'parent_category_name', 'agg': 'nunique'},
        {'groupby': ['parent_category_name'], 'select': 'price', 'agg': 'mean'},
        {'groupby': ['parent_category_name'], 'select': 'price', 'agg': 'var'},
        {'groupby': ['parent_category_name', 'category_name'], 'select': 'price', 'agg': 'mean'},
        {'groupby': ['parent_category_name', 'category_name'], 'select': 'price', 'agg': 'var'},
        {'groupby': ['region'], 'select': 'parent_category_name', 'agg': 'count'},
        {'groupby': ['city'], 'select': 'parent_category_name', 'agg': 'count'},
    ]},

    'target_encoder': {'n_splits': safe_eval(params.target_encoder__n_splits),
                       },

    'light_gbm': {'boosting_type': safe_eval(params.lgbm__boosting_type),
                  'objective': safe_eval(params.lgbm__objective),
                  'metric': safe_eval(params.lgbm__metric),
                  'learning_rate': safe_eval(params.lgbm__learning_rate),
                  'max_depth': safe_eval(params.lgbm__max_depth),
                  'subsample': safe_eval(params.lgbm__subsample),
                  'colsample_bytree': safe_eval(params.lgbm__colsample_bytree),
                  'min_child_weight': safe_eval(params.lgbm__min_child_weight),
                  'reg_lambda': safe_eval(params.lgbm__reg_lambda),
                  'reg_alpha': safe_eval(params.lgbm__reg_alpha),
                  'subsample_freq': safe_eval(params.lgbm__subsample_freq),
                  'max_bin': safe_eval(params.lgbm__max_bin),
                  'min_child_samples': safe_eval(params.lgbm__min_child_samples),
                  'num_leaves': safe_eval(params.lgbm__num_leaves),
                  'nthread': safe_eval(params.num_workers),
                  'number_boosting_rounds': safe_eval(params.lgbm__number_boosting_rounds),
                  'early_stopping_rounds': safe_eval(params.lgbm__early_stopping_rounds),
                  'verbose': safe_eval(params.verbose)
                  },

    'clipper': {'min_val': 0,
                'max_val': 1}
})
