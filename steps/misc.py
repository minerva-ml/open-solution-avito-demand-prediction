import lightgbm as lgb
from attrdict import AttrDict
from sklearn.externals import joblib

from steps.base import BaseTransformer
from steps.utils import get_logger

logger = get_logger()


class LightGBM(BaseTransformer):
    def __init__(self, **params):
        self.params = params
        self.training_params = ['number_boosting_rounds', 'early_stopping_rounds']
        self.evaluation_results = {}
        self.evaluation_function = None

    @property
    def model_config(self):
        return AttrDict({param: value for param, value in self.params.items()
                         if param not in self.training_params})

    @property
    def training_config(self):
        return AttrDict({param: value for param, value in self.params.items()
                         if param in self.training_params})

    def fit(self, X, y, X_valid, y_valid, feature_names=None, categorical_features=None, **kwargs):
        train = lgb.Dataset(X, label=y)
        valid = lgb.Dataset(X_valid, label=y_valid)

        self.estimator = lgb.train(self.model_config,
                                   train, valid_sets=[train, valid], valid_names=['train', 'valid'],
                                   feature_name=feature_names,
                                   categorical_feature=categorical_features,
                                   evals_result=self.evaluation_results,
                                   num_boost_round=self.training_config.number_boosting_rounds,
                                   early_stopping_rounds=self.training_config.early_stopping_rounds,
                                   verbose_eval=self.model_config.verbose,
                                   feval=self.evaluation_function)
        return self

    def transform(self, X, y=None, **kwargs):
        prediction = self.estimator.predict(X, num_iteration=self.estimator.best_iteration)
        return {'prediction': prediction}

    def load(self, filepath):
#         load_objects = joblib.load(filepath)
#         self.estimator = load_objects['estimator']
#         self.evals_result = load_objects['evals_result']
        self.estimator = lgb.Booster(filepath)
        return self

    def save(self, filepath):
#         save_objects = {'estimator': self.estimator,
#                         'evals_result': self.evaluation_results}
#         joblib.dump(save_objects, filepath)
        self.estimator.save_model(filepath, num_iteration=self.estimator.best_iteration)# =  load_objects['estimator']