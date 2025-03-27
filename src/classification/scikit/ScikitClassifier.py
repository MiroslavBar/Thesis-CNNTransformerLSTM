from abc import ABC, abstractmethod
import logging as log

import numpy as np
from mne.decoding import Vectorizer
from sklearn import svm
from sklearn.base import ClassifierMixin
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import StandardScaler

from src.augmentation.AugmentationMethod import AugmentationMethod
from src.classification.Classifier import Classifier
from src.classification.ClassificationMetrics import ClassificationMetrics
from src.utils import file_manager


class ScikitClassifier(Classifier, ABC):

    def train_and_evaluate(self, augmentation_method: AugmentationMethod, data: tuple) -> ClassificationMetrics:
        _, (x_test, y_test) = data
        metrics = ClassificationMetrics()

        object_type = 'classification_pipeline'
        best_model = file_manager.load_pickle_object(self.get_name(), object_type, augmentation_method.get_name())

        if best_model is None:
            best_model = self._train(data, metrics)
            file_manager.save_pickle_object(self.get_name(), object_type, best_model, augmentation_method.get_name())
        else:
            self._evaluate(best_model, x_test, y_test, metrics)

        metrics.classifier_name = self.get_name()
        metrics.augmentation_name = augmentation_method.get_name()

        return metrics

    def _train(self, data: tuple, metrics: ClassificationMetrics) -> Pipeline:
        best_model = None

        (x, y), (x_test, y_test) = data

        cross_validator = self.get_cross_validation()
        folds = cross_validator.get_n_splits(x, y)
        fold = 1
        for train_idx, val_idx in cross_validator.split(x, y):
            log.info(f"Cross validation fold: {fold}/{folds}.")

            x_train, y_train = x[train_idx], y[train_idx]

            model = make_pipeline(Vectorizer(), StandardScaler(), self._create_model())

            model.fit(x_train, y_train)
            self._evaluate(model, x_test, y_test, metrics)

            if metrics.is_best_model():
                best_model = model

            fold += 1

        return best_model

    def _evaluate(self, model: Pipeline, x_test: np.ndarray, y_test: np.ndarray,
                  metrics: ClassificationMetrics) -> None:
        y_prob = model.predict_proba(x_test)
        metrics.calculate_metrics(y_test, y_prob,
                                  one_hot_to_class_label=lambda y_true, y_pred:
                                  self._one_hot_to_class_label(model, y_true, y_pred)
                                  )

    @classmethod
    def _one_hot_to_class_label(cls, model: Pipeline, y_true: np.ndarray, y_pred: np.ndarray) -> tuple:
        return y_true, np.array([model.classes_[i] for i in np.argmax(y_pred, axis=-1)])

    @abstractmethod
    def _create_model(self) -> ClassifierMixin:
        pass


class SVM(ScikitClassifier):

    def _create_model(self) -> ClassifierMixin:
        return svm.SVC(probability=True)

    def get_name(self) -> str:
        return 'SVM'


class LDA(ScikitClassifier):

    def _create_model(self) -> ClassifierMixin:
        return LinearDiscriminantAnalysis()

    def get_name(self) -> str:
        return 'LDA'
