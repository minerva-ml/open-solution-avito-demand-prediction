import numpy as np

from steps.base import BaseTransformer


class InputMissing(BaseTransformer):
    def __init__(self, text_columns,
                 categorical_columns,
                 numerical_columns,
                 timestamp_columns):
        self.text_columns = text_columns
        self.categorical_columns = categorical_columns
        self.numerical_columns = numerical_columns
        self.timestamp_columns = timestamp_columns

    def transform(self, X, **kwargs):
        X_ = X.copy()
        for col, input_value in [self.text_columns,
                                 self.categorical_columns,
                                 self.numerical_columns,
                                 self.timestamp_columns]:
            X_[col] = X_[col].fillna(input_value)
        return {'clean_features': X_}
