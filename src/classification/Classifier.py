import logging as log
import sys
from abc import ABC, abstractmethod

from sklearn.model_selection import KFold, BaseCrossValidator

from src.augmentation.AugmentationMethod import AugmentationMethod
from src.classification.ClassificationMetrics import ClassificationMetrics
from src.config.Config import config


class Classifier(ABC):

    @abstractmethod
    def train_and_evaluate(self, augmentation_method: AugmentationMethod, data: tuple) -> ClassificationMetrics:
        pass

    @abstractmethod
    def get_name(self) -> str:
        pass

    @staticmethod
    def get_cross_validation() -> BaseCrossValidator:
        return KFold(n_splits=config.k_folds(), shuffle=True)

    @classmethod
    def get_classifiers(cls, classifier_names: list) -> 'list[Classifier]':
        if all('' == classifier_name or classifier_name.isspace() for classifier_name in classifier_names):
            log.info('No classifiers specified, '
                     'only augmentation methods will be evaluated without any classification.')
            return []

        from src.classification.scikit.ScikitClassifier import SVM
        from src.classification.scikit.ScikitClassifier import LDA
        from src.classification.keras.MLP import MLP
        from src.classification.keras.CNN import CNN
        from src.classification.keras.CNNTransformerLSTM import CNNTransformerLSTM
        from src.classification.keras.LSTM import LSTMClassifier
        from src.classification.keras.GRU import GRUClassifier
        classifiers = [SVM(), LDA(), MLP(), CNN(), CNNTransformerLSTM(), LSTMClassifier(), GRUClassifier()]
        possible_classifier_names = [classifier.get_name() for classifier in classifiers]
        classifiers_to_use = []
        found = False
        for classifier_name in classifier_names:
            try:
                index = possible_classifier_names.index(classifier_name)
                classifiers_to_use.append(classifiers[index])
                found = True
            except ValueError:
                log.warning(f"{classifier_name} is not a valid classifier.")
        if not found:
            log.error(f"Neither of the requested classifiers '{', '.join(classifier_names)}' is available. "
                      f"Possible classifiers_to_use names are {', '.join([c.get_name() for c in classifiers])}.")
            sys.exit(-1)

        return classifiers_to_use
