import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class FrequencyEncoder(BaseEstimator, TransformerMixin):
    """Encodes a categorical column as the frequency (proportion) of each
    category observed during fit. Unseen categories at inference time map
    to 0. Matches the encoder used to train the production XGBoost pipeline.
    """

    def __init__(self):
        self.freq_maps = {}
        self.columns_ = None

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        self.columns_ = X.columns.tolist()
        for col in self.columns_:
            self.freq_maps[col] = X[col].value_counts(normalize=True).to_dict()
        return self

    def transform(self, X):
        X = pd.DataFrame(X, columns=self.columns_).copy()
        for col in self.columns_:
            X[col] = X[col].map(self.freq_maps[col]).fillna(0)
        return X.values

    def get_feature_names_out(self, input_features=None):
        return [f"{col}_freq" for col in self.columns_]
