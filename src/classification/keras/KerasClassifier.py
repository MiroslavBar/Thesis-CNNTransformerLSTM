from abc import ABC, abstractmethod
import logging as log

import numpy as np
import tensorflow as tf
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger
from keras.models import Model

from src.augmentation.AugmentationMethod import AugmentationMethod
from src.classification.ClassificationMetrics import ClassificationMetrics
from src.classification.Classifier import Classifier
from src.preprocessing.OneHotLabelEncoder import OneHotLabelEncoder
from src.utils import process_utils
from src.utils import visualization, file_manager


class KerasClassifier(Classifier, ABC):

    def __init__(self):
        self._one_hot_encoder = OneHotLabelEncoder()
        self._epochs = 100

    def train_and_evaluate(self, augmentation_method: AugmentationMethod, data: tuple) -> ClassificationMetrics:
        (x_train, y_train), (x_test, y_test) = self._preprocess_dataset(data, augmentation_method.get_name())
        metrics = ClassificationMetrics()

        best_model = file_manager.load_model(self.get_name(), augmentation_name=augmentation_method.get_name())
        best_model_history = None

        if best_model is None:
            best_model, best_model_history = self._train(x_train, y_train, x_test, y_test, metrics)
            file_manager.save_model(self.get_name(), best_model, augmentation_method.get_name())
        else:
            self._evaluate(metrics, x_test, y_test, model=best_model)

        visualization.plot_training_history(best_model_history, self.get_name(), augmentation_method.get_name())
        visualization.save_model_graph(best_model, self.get_name())

        metrics.classifier_name = self.get_name()
        metrics.augmentation_name = augmentation_method.get_name()

        return metrics

    def _train_k_fold(self, x_train: np.ndarray, y_train: np.ndarray, x_validation: np.ndarray,
                      y_validation: np.ndarray, num_classes: int, fold: int) -> None:
        model = self._create_model(x_train.shape[1:], num_classes)

        checkpoint = ModelCheckpoint(filepath='model', monitor='val_loss', verbose=1,
                                                        save_best_only=True)
        early = EarlyStopping(monitor='val_loss', patience=20, verbose=1)
        redonplat = ReduceLROnPlateau(monitor='val_loss', patience=3, verbose=1)
        csv_logger = CSVLogger('log.csv', separator=',', append=True)

        callbacks_list = [
            checkpoint,
            early,
            redonplat,
            csv_logger,
        ]

        history = model.fit(x_train, y_train, validation_data=(x_validation, y_validation),
                            callbacks=callbacks_list, shuffle=True, epochs=self._epochs)

        file_manager.save_temporary_fold_model(model, fold)
        file_manager.save_temporary_fold_history(history, fold)

    def _train(self, data_train: np.ndarray, labels_train: np.ndarray, x_test: np.ndarray, y_test: np.ndarray,
               metrics: ClassificationMetrics) -> tuple[Model, dict]:
        best_model = None
        best_model_history = None
        num_classes = len(labels_train[0])

        cross_validator = self.get_cross_validation()
        folds = cross_validator.get_n_splits(data_train, labels_train)
        fold = 1
        for train_idx, validation_idx in cross_validator.split(data_train, labels_train):
            log.info(f"Cross validation fold: {fold}/{folds}.")

            x_train, y_train = data_train[train_idx], labels_train[train_idx]
            x_validation, y_validation = data_train[validation_idx], labels_train[validation_idx]

            # process_utils.run_in_separate_process(self._train_k_fold,
            #                                       (x_train, y_train, x_validation, y_validation, num_classes, fold))
            self._train_k_fold(x_train, y_train, x_validation, y_validation, num_classes, fold)
            model = file_manager.load_temporary_fold_model(fold)
            history = file_manager.load_temporary_fold_history(fold)

            self._evaluate(metrics, x_test, y_test, fold=fold)

            if metrics.is_best_model():
                best_model = model
                best_model_history = history

            fold += 1

        return best_model, best_model_history

    def _evaluate(self, metrics: ClassificationMetrics, x_test: np.ndarray, y_test: np.ndarray, fold: int = None,
                  model: Model = None) -> None:
        if model is None:
            y_prediction = process_utils.model_predict(x_test, fold)
        else:
            y_prediction = model.predict(x_test, verbose=False)

        metrics.calculate_metrics(y_test, y_prediction, self._one_hot_to_class_label)

    def _one_hot_to_class_label(self, y_true: np.ndarray, y_pred: np.ndarray) -> tuple:
        max_idx = np.argmax(y_pred, axis=-1)
        y_pred = np.zeros(y_pred.shape)
        y_pred[np.arange(y_pred.shape[0]), max_idx] = 1

        return self._one_hot_encoder.inverse_transform(y_true), self._one_hot_encoder.inverse_transform(y_pred)

    def _preprocess_dataset(self, data: tuple, augmentation_method_name: str) -> tuple:
        (x_train, y_train), (x_test, y_test) = data
        y_train = self._one_hot_encoder.fit_transform(y_train)
        y_test = self._one_hot_encoder.transform(y_test)

        return (x_train, y_train), (x_test, y_test)

    @abstractmethod
    def _create_model(self, input_shape: tuple, num_classes: int) -> Model:
        pass
