import os

from attrdict import AttrDict
from deepsense import neptune

from utils import read_params, safe_eval

ctx = neptune.Context()
params = read_params(ctx)

FEATURE_COLUMNS = ['region', 'city',
                   'parent_category_name', 'category_name',
                   'param_1', 'param_2', 'param_3',
                   'title', 'description',
                   'price',
                   'item_seq_number',
                   'activation_date',
                   'user_type',
                   'image',
                   'image_top_1']
CATEGORICAL_COLUMNS = ['region', 'city',
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

DEV_SAMPLE_SIZE = int(20e4)

COLUMN_TYPES = {'train': {'price': 'uint32',
                          'item_seq_number': 'uint32',
                          'image_top_1': 'uint32',
                          'deal_probability': 'float32',
                          },
                'inference': {'price': 'uint32',
                              'item_seq_number': 'uint32',
                              'image_top_1': 'uint32',
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

    'groupby_aggregation': {'groupby_aggregations': [
        {'groupby': ['region'], 'select': 'region', 'agg': 'count'},
        {'groupby': ['city'], 'select': 'city', 'agg': 'count'},
        {'groupby': ['parent_category_name'], 'select': 'parent_category_name', 'agg': 'count'},
        {'groupby': ['category_name'], 'select': 'category_name', 'agg': 'count'},
        {'groupby': ['image_top_1'], 'select': 'image_top_1', 'agg': 'count'},
        {'groupby': ['user_type'], 'select': 'user_type', 'agg': 'count'}
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
                  'scale_pos_weight': safe_eval(params.lgbm__scale_pos_weight),
                  'subsample_freq': safe_eval(params.lgbm__subsample_freq),
                  'max_bin': safe_eval(params.lgbm__max_bin),
                  'min_child_samples': safe_eval(params.lgbm__min_child_samples),
                  'num_leaves': safe_eval(params.lgbm__num_leaves),
                  'nthread': safe_eval(params.num_workers),
                  'number_boosting_rounds': safe_eval(params.lgbm__number_boosting_rounds),
                  'early_stopping_rounds': safe_eval(params.lgbm__early_stopping_rounds),
                  'verbose': safe_eval(params.verbose)
                  },
})
