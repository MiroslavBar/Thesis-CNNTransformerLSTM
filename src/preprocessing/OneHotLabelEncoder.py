from typing import Any

from sklearn.preprocessing import LabelBinarizer
import numpy as np


class OneHotLabelEncoder(LabelBinarizer):

    def transform(self, y: np.ndarray):
        y = super().transform(y)
        if self.y_type_ == 'binary':
            return np.hstack((y, 1 - y))
        else:
            return y

    def inverse_transform(self, y: np.ndarray, threshold: Any = None):
        if self.y_type_ == 'binary':
            return super().inverse_transform(y[:, 0], threshold)
        else:
            return super().inverse_transform(y, threshold)
