import logging as log
import os
import time
from collections import defaultdict

import numpy as np
from sklearn.model_selection import train_test_split

from augmentation.AugmentationMethod import AugmentationMethod
from augmentation.AugmentationMetrics import AugmentationMetrics
from classification.ClassificationMetrics import ClassificationMetrics
from classification.Classifier import Classifier
from config.Config import config
from preprocessing import preprocessing
from utils import visualization, file_utils


def _format_execution_time(start: float, end: float) -> str:
    hours, rem = divmod(end - start, 3600)
    minutes, seconds = divmod(rem, 60)
    return f"{int(hours):0>2}:{int(minutes):0>2}:{seconds:05.3f}"


def _inter_subject_model(data: np.ndarray, labels: np.ndarray) -> None:
    augmentation_methods = AugmentationMethod.get_data_augmentation_methods(config.augmentation_methods)
    classifiers = Classifier.get_classifiers(config.model_names)

    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=config.test_size, shuffle=True)
    augmented_size = int(x_train.shape[0] * config.generated_data_multiplier)

    log.info(f"Train shape: {x_train.shape}.")
    log.info(f"Test shape: {x_test.shape}.")

    augmentation_metrics = AugmentationMetrics(config.augmentation_metrics)
    classification_metrics_per_classifier = defaultdict(list)
    for augmentation_method in augmentation_methods:
        do_augmentation = augmentation_method.get_name() != ''

        if do_augmentation:
            augmentation_start = time.perf_counter()
            x_generated, y_generated = augmentation_method.generate(x_train, y_train, augmented_size)
            log.info(f"Data augmentation took {_format_execution_time(augmentation_start, time.perf_counter())}.")

            augmentation_metrics.evaluate((x_train, y_train), (x_generated, y_generated), augmentation_method)

            x_train_augmented = np.vstack([x_train, x_generated])
            y_train_augmented = np.concatenate([y_train, y_generated])
        else:
            x_train_augmented = x_train
            y_train_augmented = y_train

        for classifier in classifiers:
            log.info(f"Running classification on data set of shape {np.vstack([x_test, x_train_augmented]).shape},"
                     f" using {classifier.get_name()} classifier.")
            classification_start = time.perf_counter()

            classification_data = ((x_train_augmented, y_train_augmented), (x_test, y_test))
            metrics = classifier.train_and_evaluate(augmentation_method, classification_data)

            log.info(f"{classifier.get_name()} model classification took "
                     f"{_format_execution_time(classification_start, time.perf_counter())}.")

            metrics.report(config.classification_metrics)
            classification_metrics_per_classifier[classifier.get_name()].append(metrics)

    augmentation_metrics.report()
    ClassificationMetrics.merge(classification_metrics_per_classifier).report(config.classification_metrics)


def main():
    log.info(config)
    start_time = time.perf_counter()
    log.info(f"Loading and preprocessing input data.")
    data, labels = preprocessing.load_data()
    log.info(f"Preprocessing took {_format_execution_time(start_time, time.perf_counter())}.")

    # _personal_models(data, labels)
    data, labels = np.concatenate(data), np.concatenate(labels)
    visualization.plot_input_data_tsne(data, labels)
    _inter_subject_model(data, labels)

    log.info(f"Total execution time {_format_execution_time(start_time, time.perf_counter())}.")

    if config.save_plots:
        log.info(f"All image output has been saved to {os.getcwd()}/{file_utils.IMAGES_OUTPUT_FOLDER}.")


if __name__ == '__main__':
    main()
