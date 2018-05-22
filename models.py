import numpy as np
import lightgbm as lgb

from steps.misc import LightGBM


class LightGBMLowMemory(LightGBM):
    def fit(self, X, y, X_valid, y_valid, feature_names=None, categorical_features=None, **kwargs):
        X = X.astype(np.float32)
        y = y.astype(np.float32)

        X_valid = X_valid.astype(np.float32)
        y_valid = y_valid.astype(np.float32)

        train = lgb.Dataset(X, label=y,
                            feature_name=feature_names, categorical_feature=categorical_features)
        valid = lgb.Dataset(X_valid, label=y_valid,
                            feature_name=feature_names, categorical_feature=categorical_features)

        self.evaluation_results = {}
        self.estimator = lgb.train(self.model_config,
                                   train, valid_sets=[valid], valid_names=['valid'],
                                   evals_result=self.evaluation_results,
                                   feature_name=feature_names,
                                   categorical_feature=categorical_features,
                                   num_boost_round=self.training_config.number_boosting_rounds,
                                   early_stopping_rounds=self.training_config.early_stopping_rounds,
                                   verbose_eval=self.model_config.verbose,
                                   feval=self.evaluation_function)
        return self
