import numpy as np
from mne.decoding import Vectorizer
from sklearn.base import TransformerMixin
from sklearn.preprocessing import StandardScaler


class NDScaler(TransformerMixin):
    """
    Scaler that scales ndarrays while preserving their dimensions.
    """

    def __init__(self, scaler: TransformerMixin = None):
        if scaler is None:
            scaler = StandardScaler()
        self._scaler = scaler
        self._vectorizer = Vectorizer()

    def fit(self, x: np.ndarray, **kwargs) -> 'NDScaler':
        x = self._vectorizer.fit_transform(x)
        self._scaler.fit(x, **kwargs)
        return self

    def transform(self, x, **kwargs) -> np.ndarray:
        x = self._vectorizer.transform(x)
        x = self._scaler.transform(x, **kwargs)
        return self._vectorizer.inverse_transform(x)

    def inverse_transform(self, x, **kwargs) -> np.ndarray:
        x = self._vectorizer.transform(x)
        x = self._scaler.inverse_transform(x)
        return self._vectorizer.inverse_transform(x)
