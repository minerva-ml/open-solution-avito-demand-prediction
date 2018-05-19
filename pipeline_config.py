from itertools import product
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
                       'user_type', 'image_top_1']
NUMERICAL_COLUMNS = ['price', 'item_seq_number']
TEXT_COLUMNS = ['title', 'description']
IMAGE_COLUMNS = ['image']
IMAGE_TARGET_COLUMNS = ['parent_category_name', 'category_name']
TARGET_COLUMNS = ['deal_probability']
CV_COLUMN = ['user_id']
TIMESTAMP_COLUMNS = ['activation_date']
ITEM_ID_COLUMN = ['item_id']
USER_ID_COLUMN = ['user_id']
FEATURES_TO_TRANSLATE = ['category_name', 'city', 'description', 'param_1', 'param_2', 'param_3',
                         'parent_category_name', 'region', 'title']

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

AGGREGATION_RECIPIES = []
for time, agg in product([['activation_date_month'], ['activation_date_day', 'activation_date_month']],
                         ['mean', 'size', 'var', 'min', 'max']):
    for group in [['user_id'] + time,
                  ['category_name'] + time,
                  ['parent_category_name'] + time,
                  ['param_1'] + time,
                  ['param_2'] + time,
                  ['param_3'] + time,
                  ['user_id', 'category_name'] + time,
                  ['user_id', 'parent_category_name'] + time,
                  ['user_id', 'param_1'] + time,
                  ['user_id', 'param_2'] + time,
                  ['user_id', 'param_3'] + time,
                  ['user_type'] + time,
                  ['user_type', 'category_name'] + time,
                  ['user_type', 'parent_category_name'] + time,
                  ['user_type', 'param_1'] + time,
                  ['user_type', 'param_2'] + time,
                  ['user_type', 'param_3'] + time,
                  ['user_type', 'city'] + time,
                  ['user_type', 'city', 'category_name'] + time,
                  ['user_type', 'city', 'parent_category_name'] + time,
                  ['user_type', 'region'] + time,
                  ['user_type', 'region', 'category_name'] + time,
                  ['user_type', 'region', 'parent_category_name'] + time,
                  ['user_type', 'region', 'param_1'] + time,
                  ['user_type', 'region', 'param_2'] + time,
                  ['user_type', 'region', 'param_3'] + time,
                  ]:
        AGGREGATION_RECIPIES.append({'groupby': group, 'select': 'price', 'agg': agg})

IMAGE_PARAMS = AttrDict({'h': safe_eval(params.target_size)[0],
                         'w': safe_eval(params.target_size)[1]})

SOLUTION_CONFIG = AttrDict({
    'env': {'cache_dirpath': params.experiment_dir
            },
    'random_search': {'light_gbm': {'n_runs': safe_eval(params.lgbm_random_search_runs),
                                    'callbacks': {'neptune_monitor': {'name': 'light_gbm'
                                                                      },
                                                  'save_results': {
                                                      'filepath': os.path.join(params.experiment_dir,
                                                                               'random_search_light_gbm.pkl')
                                                  }
                                                  }
                                    }
                      },
    'dataframe_by_type_splitter': {'numerical_columns': NUMERICAL_COLUMNS,
                                   'categorical_columns': CATEGORICAL_COLUMNS,
                                   'timestamp_columns': TIMESTAMP_COLUMNS,
                                   },

    'fetch_image_columns': {'columns': IMAGE_COLUMNS},
    'subset_not_nan_image': {'nan_column': IMAGE_COLUMNS},
    'join_with_nan': {'index_column': ITEM_ID_COLUMN},
    'label_encoder_image': {'columns_to_encode': IMAGE_TARGET_COLUMNS},

    'date_features': {'date_column': TIMESTAMP_COLUMNS[0]},
    'is_missing': {'columns': FEATURE_COLUMNS},
    'categorical_encoder': {'cols': CATEGORICAL_COLUMNS,
                            'n_components': params.categorical_encoder__n_components,
                            'hash_method': params.categorical_encoder__hash_method
                            },

    'text_counter': {'text_column': 'description'},

    'text_cleaner': {'text_features': ['description', 'title'],
                     'drop_punctuation': True,
                     'all_lower_case': True
                     },

    'groupby_aggregation': {'groupby_aggregations': AGGREGATION_RECIPIES
                            },

    'target_encoder': {'n_splits': safe_eval(params.target_encoder__n_splits),
                       },

    'loader': {'loader_params': {'training': {'batch_size': params.batch_size_train,
                                              'shuffle': True,
                                              'num_classes': safe_eval(params.num_classes),
                                              'target_size': safe_eval(params.target_size),
                                              },
                                 'inference': {'batch_size': params.batch_size_inference,
                                               'shuffle': False,
                                               'num_classes': safe_eval(params.num_classes),
                                               'target_size': safe_eval(params.target_size),
                                               },
                                 }
               },

    'inception_resnet': {'architecture_config': {'num_classes': safe_eval(params.num_classes),
                                                 'target_size': safe_eval(params.target_size),
                                                 'loss_weights': [0.5, 0.5],
                                                 'trainable_threshold': -1,
                                                 'lr': params.lr,
                                                 },
                         'training_config': {'epochs': params.epochs,
                                             'workers': params.num_workers
                                             },
                         'callbacks_config': {'lr_scheduler': {'gamma': params.lr_gamma},
                                              'model_checkpoint': {'filepath': os.path.join(params.experiment_dir,
                                                                                            'checkpoints',
                                                                                            'inception_resnet',
                                                                                            'best_model.h5'),
                                                                   'save_best_only': True,
                                                                   'save_weights_only': False},
                                              'early_stopping': {'patience': params.patience},
                                              'neptune_monitor': {'model_name': 'inception_resnet'}
                                              }
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
                  'scale_pos_weight': safe_eval(params.lgbm__scale_pos_weight),
                  'verbose': safe_eval(params.verbose)
                  },

    'clipper': {'min_val': 0,
                'max_val': 1}
})
