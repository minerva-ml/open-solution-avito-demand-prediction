from steps.base import BaseTransformer


class InputMissing(BaseTransformer):
    def __init__(self, text_columns,
                 categorical_columns,
                 numerical_columns,
                 timestamp_columns):
        self.text_columns, self.text_input_value = text_columns
        self.categorical_columns, self.categorical_input_value = categorical_columns
        self.numerical_columns, self.numerical_input_value = numerical_columns
        self.timestamp_columns, self.timestamp_input_value = timestamp_columns

    def transform(self, X, **kwargs):
        X_ = X.copy()
        for column in self.text_columns:
            X_[column] = X_[column].apply(self._text_missing)
        for column in self.categorical_columns:
            X_[column] = X_[column].apply(self._categorical_missing)
        for column in self.numerical_columns:
            X_[column] = X_[column].apply(self._numerical_missing)
        for column in self.timestamp_columns:
            X_[column] = X_[column].apply(self._timestamp_missing)
        return {'clean_features': X}

    def _text_missing(self, x):
        if str(x) == 'NaN':
            return self.text_input_value
        else:
            return x

    def _categorical_missing(self, x):
        if str(x) == 'NaN':
            return self.categorical_input_value
        else:
            return x

    def _numerical_missing(self, x):
        if str(x) == 'NaN':
            return self.numerical_input_value
        else:
            return x

    def _timestamp_missing(self, x):
        if str(x) == 'NaN':
            return self.timestamp_input_value
        else:
            return x
