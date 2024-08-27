from typing import Optional, List
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    FunctionTransformer,
    LabelEncoder,
    OneHotEncoder,
    StandardScaler,
    MinMaxScaler,
)


class DropColumnTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Drop specified columns
        X_transformed = X.drop(columns=self.columns, axis=1)
        return X_transformed


class CustomImputer(BaseEstimator, TransformerMixin):
    def __init__(self, strategy="mean", columns: Optional[List[str]] = None):
        self.strategy = strategy
        self.columns = columns if columns is not None else []
        self.imputer = SimpleImputer(strategy=self.strategy)

    def fit(self, X, y=None):
        self.imputer.fit(X[self.columns])
        return self

    def transform(self, X):
        X_transformed = X.copy()
        X_transformed[self.columns] = self.imputer.transform(X[self.columns])
        return X_transformed


class CustomLabelEncoder(BaseEstimator, TransformerMixin):

    def __init__(self, columns: List[str]) -> None:
        self.columns = columns
        self.encoders = {}

    def fit(self, X, y=None):
        for column in self.columns:
            self.encoders[column] = LabelEncoder()
            self.encoders[column].fit(X[column])
        return self

    def transform(self, X):
        X_transformed = X.copy()
        for column in self.columns:
            X_transformed[column] = self.encoders[column].transform(X[column])
        return X_transformed


class CustomOneHotEncoder(BaseEstimator, TransformerMixin):

    def __init__(self, columns: List[str]) -> None:
        self.columns = columns
        self.encoders = {}

    def fit(self, X, y=None):
        for column in self.columns:
            self.encoders[column] = OneHotEncoder(sparse_output=False)
            self.encoders[column].fit(X[[column]])
        return self

    def transform(self, X):
        X_transformed = X.copy()
        for column in self.columns:
            encoded = pd.DataFrame(
                self.encoders[column].transform(X[[column]]),
                columns=self.encoders[column].get_feature_names_out([column]),
                index=X.index,
            )
            X_transformed = pd.concat(
                [X_transformed.drop(columns=column), encoded], axis=1
            )
        return X_transformed


class CustomStandardScaler(BaseEstimator, TransformerMixin):

    def __init__(self, columns: List[str]) -> None:
        self.columns = columns
        self.scaler = StandardScaler()

    def fit(self, X, y=None):
        self.scaler.fit(X[self.columns])
        return self

    def transform(self, X):
        X_transformed = X.copy()
        X_transformed[self.columns] = self.scaler.transform(X[self.columns])
        return X_transformed

class CustomOutlierDetector:
    def __init__(self, data):
        self.data = data
    
    def detect_outliers_iqr(self, column):
        Q1 = self.data[column].quantile(0.25)
        Q3 = self.data[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = self.data[(self.data[column] < lower_bound) | (self.data[column] > upper_bound)]
        return outliers

class CustomOutlierRemover(BaseEstimator, TransformerMixin):
    def __init__(self, threshold = 3):
        self.threshold = threshold
        self.numeric_cols = None
        self._outliers = None
    
    # This function identifies the numerical columns
    def fit(self, X, y=None):
        self.numeric_cols = X.select_dtypes(include=np.number).columns
        return self
    
    def transform(self, X):
        if self.numeric_cols is None:
            raise ValueError("Call 'fit' before 'transform'.")
        
        # Make a copy of numerical columns
        X_transformed = X.copy()

        z_scores = stats.zscore(X_transformed[self.numeric_cols])

        # Concat with non-numerical columns
        self._outliers = (abs(z_scores) > self.threshold).any(axis=1)
        return X_transformed[~self._outliers] 
    
    @property
    def outliers(self):
        return self._outliers
    

class CustomMinMaxScaler(BaseEstimator, TransformerMixin):

    def __init__(self, columns: List[str]) -> None:
        self.columns = columns
        self.scaler = MinMaxScaler()

    def fit(self, X, y=None):
        self.scaler.fit(X[self.columns])
        return self

    def transform(self, X):
        X_transformed = X.copy()
        X_transformed[self.columns] = self.scaler.transform(X[self.columns])
        return X_transformed