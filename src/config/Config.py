import argparse
import configparser
import logging as log
import multiprocessing
import os
import random
import sys
from pathlib import Path
from typing import Any

import matplotlib
import mne
import numpy as np
import pandas as pd
import tensorflow as tf

from src.classification.ClassificationType import ClassificationType
from src.preprocessing.DataRepresentation import DataRepresentation
from src.utils import file_utils
from sklearnex import patch_sklearn

DEFAULT_CONFIGURATION_FILE_PATH = 'config.ini'


def debugger_is_active() -> bool:
    """Return if the debugger is currently active."""
    return hasattr(sys, 'gettrace') and sys.gettrace() is not None


class SklearnIntelexLoggingFilter(log.Filter):
    def filter(self, record: log.LogRecord) -> bool:
        return 'running accelerated version on CPU' not in record.getMessage()


class Config:

    def __init__(self):
        # Logging
        self.log_level = log.INFO

        # Augmentation
        self.augmentation_methods = 'cVAE'
        self.generated_data_multiplier = 1.0

        # Classification
        self.model_names = 'CNNTransformerLSTM'
        self.use_pre_trained_models = False
        self._k_folds = 5

        # Metrics
        self.classification_metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'auc', 'confusion_matrix']

        # CPU/GPU
        self._use_gpu = True

        # Other
        self.classification_type = ClassificationType.BINARY
        self.data_representation = DataRepresentation.TIME_SERIES
        self.deterministic = False
        self.save_plots = True
        self.save_load_preprocessed_data = True
        self.num_samples = 700
        self.label_conversion = "two_class_ME"
        self.sfreq = 500

        # Internal configuration, not read from a config file
        self.parser = None
        self.config_file_path = None

        self.cpus = -1
        self.tmin = -3.5
        self.tmax = 0.5
        self.l_freq = 8
        self.h_freq = 30
        self.channels = ['Cz', 'C3', 'C4']
        self.test_size = 0.2
        self.tabulate_format = 'psql'

        if debugger_is_active():
            # tf.config.run_functions_eagerly(True)
            pass

        if not debugger_is_active():
            # Set plotting only to files without creating any interactive windows
            matplotlib.use('agg')

        self._load_from_file(self._parse_cmd_args().config_file)

    def data_type_suffix(self) -> str:
        return f"{self.data_representation}_{self.classification_type}"

    def k_folds(self) -> int:
        return self._k_folds

    @staticmethod
    def _parse_cmd_args() -> argparse.Namespace:
        parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

        optional_named = parser.add_argument_group('optional arguments')
        optional_named.add_argument('-f', '--config_file', help='path to a configuration .ini file that is to be used',
                                    required=False, default=DEFAULT_CONFIGURATION_FILE_PATH)
        args = parser.parse_args()

        return args

    def _load_from_file(self, file_path: str = DEFAULT_CONFIGURATION_FILE_PATH) -> None:
        self._setup_logging()
        self._parse_config(file_path)
        self._setup_augmentation()
        self._setup_classification()
        self._setup_metrics()
        self._setup_other()
        self._setup_cpu_gpu()

    def _parse_config(self, file_path: str) -> None:
        self.config_file_path = Path(file_path)
        log.info(f"Reading configuration from {self.config_file_path.absolute()}.")
        self.parser = configparser.ConfigParser()
        success = self.parser.read(self.config_file_path)
        if not success:
            log.error(f"Unable to read configuration from file {self.config_file_path.absolute()}!")
            sys.exit(-1)

    def _setup_augmentation(self) -> None:
        section = 'Augmentation'
        self.augmentation_methods = [method.strip() for method in
                                     pd.unique(self._get(section, 'method_names').split(','))]
        self.generated_data_multiplier = self._get_float(section, 'generated_data_multiplier')

    def _setup_classification(self) -> None:
        section = 'Classification'
        self.model_names = [model.strip() for model in pd.unique(self._get(section, 'model_names').split(','))]
        self.use_pre_trained_models = self._get_bool(section, 'use_pre_trained_models')
        self._k_folds = self._get_int(section, 'k_folds')

    def _setup_metrics(self) -> None:
        section = 'Metrics'
        self.classification_metrics = [metric.strip() for metric in
                                       self._get(section, 'classification_metrics').split(',')]
        self.augmentation_metrics = [metric.strip() for metric in self._get(section, 'augmentation_metrics').split(',')]

    def _setup_logging(self) -> None:
        logging_format = '%(asctime)s [%(levelname)s] %(message)s'
        file_utils.create_folders(file_utils.LOGS_FOLDER)
        log.basicConfig(level=self.log_level, format=logging_format,
                        handlers=[log.StreamHandler(sys.stdout),
                                  log.FileHandler(file_utils.MAIN_LOG_FILE)])
        log.getLogger().addFilter(SklearnIntelexLoggingFilter())
        mne.set_log_level('warning')

    def _setup_cpu_gpu(self) -> None:
        section = 'GPU'
        self._use_gpu = self._get_bool(section, 'use_gpu')
        keras_cpus = self.cpus if self.cpus > 0 else multiprocessing.cpu_count() + 1 + self.cpus
        tf.config.threading.set_intra_op_parallelism_threads(keras_cpus)
        tf.config.threading.set_inter_op_parallelism_threads(keras_cpus)

        if not self.data_representation == DataRepresentation.TIME_FREQUENCY:
            patch_sklearn()

        if self._use_gpu:
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                try:
                    # Set tensorflow GPU memory allocation as needed and not allocate whole GPU memory at initialization
                    for gpu in gpus:
                        tf.config.experimental.set_memory_growth(gpu, True)

                except RuntimeError as e:
                    log.error(e)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    def _setup_other(self) -> None:
        section = 'Other'
        self.classification_type = ClassificationType.get_type(self._get(section, 'classification_type'))
        self.data_representation = DataRepresentation.get_representation(
            self._get(section, 'data_representation')
        )
        self.save_plots = self._get_bool(section, 'save_plots')
        self.deterministic = self._get_bool(section, 'deterministic')
        self._set_deterministic()
        self.save_load_preprocessed_data = self._get_bool(section, 'save_load_preprocessed_data')
        self.num_samples = self._get_int(section, 'num_samples')
        self.label_conversion = self._get_str(section, 'label_conversion')
        self.sfreq = self._get_float(section, 'sfreq')
        self.data_dir = self._get_str(section, 'data_dir')

    def _set_deterministic(self) -> None:
        if not self.deterministic:
            return

        seed = 1
        os.environ['PYTHONHASHSEED'] = str(seed)
        random.seed(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)

    def _get(self, section: str, option: str) -> Any:
        self._assert_exists_option(section, option)
        return self.parser.get(section, option)

    def _get_int(self, section: str, option: str) -> int:
        self._assert_exists_option(section, option)
        return self.parser.getint(section, option)

    def _get_str(self, section: str, option: str) -> str:
        self._assert_exists_option(section, option)
        return self.parser.get(section, option)


    def _get_float(self, section: str, option: str) -> float:
        self._assert_exists_option(section, option)
        return self.parser.getfloat(section, option)

    def _get_bool(self, section: str, option: str) -> bool:
        self._assert_exists_option(section, option)
        return self.parser.getboolean(section, option)

    def _assert_exists_option(self, section: str, option: str) -> None:
        if not self.parser.has_section(section) or not self.parser.has_option(section, option):
            log.error(f"Required option {option} is missing for section {section}"
                      f" in configuration file {self.config_file_path.absolute()}!")
            sys.exit(-2)

    def __str__(self):
        classification_type_hint = 'movement vs resting' if config.classification_type == ClassificationType.BINARY \
            else 'left movement vs right movement vs resting'
        settings_log = f"Classification type is set to {config.classification_type} ({classification_type_hint})."
        return settings_log


config = Config()
