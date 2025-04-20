import logging as log
import os
import random
import time
from collections import defaultdict
from typing import Optional, Callable, Tuple, Any

import mne
import numpy as np
from sklearn.model_selection import train_test_split

from src.augmentation.AugmentationMethod import AugmentationMethod
from src.augmentation.AugmentationMetrics import AugmentationMetrics
from src.classification.ClassificationMetrics import ClassificationMetrics
from src.classification.Classifier import Classifier
from src.config.Config import config
from src.utils import file_utils

def load_csv_data(file_path):
    return np.loadtxt(file_path, delimiter=',')

def label_conversion_map(conversion_type: str) -> Optional[Callable[[int], Optional[int]]]:
    """
    Returns the appropriate label conversion function based on configuration.

    Args:
        conversion_type: The type of label conversion to apply.

    Returns:
        Optional function to convert labels or None if not found.
    """
    conversion_functions = {
        'two_class_ME': two_class_ME_labels,
        'two_class_MI': two_class_MI_labels,
        'four_class_ME': four_class_ME_labels,
        'four_class_MI': four_class_MI_labels,
        'two_class_labels': two_class_labels,
        'four_class_labels': four_class_labels
    }
    return conversion_functions.get(conversion_type)


def two_class_labels(label: int) -> Optional[int]:
    if label in [2, 3, 5, 6, 8, 9, 11, 12]: return 0  # Movement
    if label in [1, 4, 7, 10]: return 1  # Relax
    return None


def two_class_MI_labels(label: int) -> Optional[int]:
    if label in [5, 6, 11, 12]: return 0  # MI Movement
    if label in [4, 10]: return 1  # MI Relax
    return None


def two_class_ME_labels(label: int) -> Optional[int]:
    if label in [2, 3, 8, 9]: return 0  # ME Movement
    if label in [1, 7]: return 1  # ME Relax
    return None


def four_class_MI_labels(label: int) -> Optional[int]:
    if label in [5]: return 0  # MI left fist movement
    if label in [6]: return 1  # MI right first movement
    if label in [12]: return 2  # MI both feet movement
    if label in [4, 10]:
        if random.randint(1, 4) == 4:  # Reducing the amount of rest trials to balance the data
            return 3  # MI relax
    return None


def four_class_ME_labels(label: int) -> Optional[int]:
    if label in [2]: return 0  # ME left fist movement
    if label in [3]: return 1  # ME right first movement
    if label in [9]: return 2  # ME both feet movement
    if label in [1, 7]:
        if random.randint(1, 4) == 4:  # Reducing the amount of rest trials to balance the data
            return 3  # ME relax
    return None


def four_class_labels(label: int) -> Optional[int]:
    if label in [2, 5]: return 0  # Left first movement
    if label in [3, 6]: return 1  # Right first movement
    if label in [9, 12]: return 2  # Both feet movement
    if label in [1, 4, 7, 10]:
        if random.randint(1, 4) == 4:  # Reducing the amount of rest trials to balance the data
            return 3  # Relax
    return None

def get_subjects(csv_dir):
    subject_ids = set()
    for file in os.listdir(csv_dir):
        if 'SIG' in file:
            subject_id = file.split('_')[1]
            subject_ids.add(subject_id)
    return sorted(subject_ids)


def compute_tfr(eeg_data: np.ndarray, sfreq: float, freqs: np.ndarray, n_cycles: np.ndarray) -> np.ndarray:
    """
     Compute Time-Frequency Representation using Morlet wavelets.

    Args:
        eeg_data: EEG signal (n_channels, n_samples)
        sfreq: Sampling frequency
        freqs: List of frequencies to analyze
        n_cycles: Number of cycles per frequency

    Returns:
        Time-Frequency representation (n_channels, n_frequencies, n_times))
    """
    eeg_data = eeg_data[np.newaxis, :, :]  # Reshape to (1, n_channels, n_samples)

    # Apply Morlet wavelet transform
    tfr = mne.time_frequency.tfr_array_morlet(eeg_data, sfreq=sfreq, freqs=freqs, n_cycles=n_cycles, output="power")

    return tfr[0]  # Remove extra dimension


def preprocess_subject(
        csv_dir: str,
        subject_id: str,
        config: Any,
        freqs: np.ndarray = np.linspace(1, 40, 20)
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Preprocess a single subject's data and convert to time-frequency format.

    Args:
        csv_dir: Directory containing CSV files
        subject_id: ID of the subject to process
        config: Configuration object containing processing parameters
        freqs: Frequencies to use for time-frequency analysis

    Returns:
        Tuple of (processed_data, labels) or (None, None) if processing fails
    """
    signal_files = sorted([f for f in os.listdir(csv_dir) if f'SUB_{subject_id}_SIG' in f])
    annotation_files = sorted([f for f in os.listdir(csv_dir) if f'SUB_{subject_id}_ANN' in f])
    num_samples = config.num_samples
    conversion_type = config.label_conversion
    sfreq = config.sfreq
    convert_label = label_conversion_map(conversion_type)

    if len(signal_files) != len(annotation_files):
        print(f"Mismatch for subject {subject_id}")
        return None, None

    all_data, all_labels = [], []
    n_cycles = freqs / 2  # Define cycles per frequency

    for signal_file, annotation_file in zip(signal_files, annotation_files):
        signal_data = load_csv_data(os.path.join(csv_dir, signal_file))  # (n_samples, n_channels)
        annotation_data = load_csv_data(os.path.join(csv_dir, annotation_file))

        for trial_idx in range(annotation_data.shape[0]):
            trial_label = int(annotation_data[trial_idx, 0])
            converted_label = convert_label(trial_label)
            if converted_label is None:
                continue

            start_idx = int(annotation_data[trial_idx, 3]) - 1
            end_idx = int(annotation_data[trial_idx, 4]) - 1
            trial_signal = signal_data[start_idx:end_idx, :].T  # (n_channels, n_samples)

            # Ensure shape consistency
            if trial_signal.shape[1] < num_samples:
                trial_signal = np.pad(trial_signal, ((0, 0), (0, num_samples - trial_signal.shape[1])), mode='constant')
            elif trial_signal.shape[1] > num_samples:
                trial_signal = trial_signal[:, :num_samples]

            # Normalize
            mean = np.mean(trial_signal, axis=1, keepdims=True)
            std = np.std(trial_signal, axis=1, keepdims=True)
            std[std == 0] = 1
            trial_signal = ((trial_signal - mean) / std).astype(np.float32)

            # Convert to time-frequency representation
            trial_tfr = compute_tfr(trial_signal, sfreq, freqs, n_cycles)
            # trial_tfr = trial_signal
            all_data.append(trial_tfr)  # Shape: (n_channels, n_frequencies, n_times)
            all_labels.append(converted_label)

    if not all_data:
        return None, None

    all_data = np.array(all_data, dtype=np.float32)  # (n_samples, n_channels, n_frequencies, n_times)
    all_labels = np.array(all_labels, dtype=np.int64)

    return all_data, all_labels

def load_dataset(
    csv_dir: str,
    config: Any,
    freqs: np.ndarray = np.linspace(1, 40, 20)
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Load the complete dataset for all subjects.

    Args:
        csv_dir: Directory containing CSV files
        config: Configuration object containing processing parameters
        freqs: Frequencies to use for time-frequency analysis

    Returns:
        Tuple of (processed_data, labels) where data is in time-frequency format,
        or (None, None) if processing fails
    """
    all_subjects_data = []
    all_subjects_labels = []

    subjects = get_subjects(csv_dir)
    for subject_id in subjects:
        data, labels = preprocess_subject(csv_dir, subject_id, config, freqs)
        if data is not None and labels is not None:
            all_subjects_data.append(data)  # Shape: (n_samples, n_channels, n_frequencies, n_times)
            all_subjects_labels.append(labels)

    if not all_subjects_data:
        print("No valid data found.")
        return None, None

    all_subjects_data = np.array(all_subjects_data, dtype=np.float32)  # (n_people, n_samples, n_channels, n_frequencies, n_times)
    all_subjects_labels = np.array(all_subjects_labels, dtype=np.int64)

    return all_subjects_data, all_subjects_labels

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


def main() -> None:
    log.info(config)
    start_time = time.perf_counter()
    log.info(f"Loading and preprocessing input data.")
    data, labels = load_dataset(config.data_dir, config)
    log.info(f"Preprocessing took {_format_execution_time(start_time, time.perf_counter())}.")
    data, labels = np.concatenate(data), np.concatenate(labels)
    _inter_subject_model(data, labels)

    log.info(f"Total execution time {_format_execution_time(start_time, time.perf_counter())}.")

    if config.save_plots:
        log.info(f"All image output has been saved to {os.getcwd()}/{file_utils.IMAGES_OUTPUT_FOLDER}.")

if __name__ == '__main__':
    main()
