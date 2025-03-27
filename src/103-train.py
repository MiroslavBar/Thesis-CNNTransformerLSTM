import logging as log
import os
import time
from collections import defaultdict

import mne
import numpy as np
from sklearn.model_selection import train_test_split

from augmentation.AugmentationMethod import AugmentationMethod
from augmentation.AugmentationMetrics import AugmentationMetrics
from classification.ClassificationMetrics import ClassificationMetrics
from classification.Classifier import Classifier
from config.Config import config
from preprocessing import preprocessing
from utils import visualization, file_utils


# Load CSV Data
def load_csv_data(file_path):
    return np.loadtxt(file_path, delimiter=',')

# Convert Labels (Modify this for your specific dataset)
def convert_label(label):
    return two_class_labels(label)

def two_class_labels(label):
    if label in [2, 3, 5, 6, 8, 9, 11, 12]: return 0  # Movement
    if label in [1, 4, 7, 10]: return 1  # Relax
    return None  # Exclude other labels

# Get unique subjects from filenames
def get_subjects(csv_dir):
    subject_ids = set()
    for file in os.listdir(csv_dir):
        if 'SIG' in file:
            subject_id = file.split('_')[1]
            subject_ids.add(subject_id)
    return sorted(subject_ids)


# Compute Time-Frequency Representation using Morlet wavelets
def compute_tfr(eeg_data, sfreq, freqs, n_cycles):
    """
    Compute Time-Frequency Representation using Morlet wavelets.

    :param eeg_data: EEG signal (n_samples, n_channels)
    :param sfreq: Sampling frequency
    :param freqs: List of frequencies to analyze
    :param n_cycles: Number of cycles per frequency
    :return: Time-Frequency representation (n_channels, n_frequencies, n_times)
    """
    eeg_data = eeg_data[np.newaxis, :, :]  # Reshape to (1, n_channels, n_samples)

    # Apply Morlet wavelet transform
    tfr = mne.time_frequency.tfr_array_morlet(eeg_data, sfreq=sfreq, freqs=freqs, n_cycles=n_cycles, output="power")

    return tfr[0]  # Remove extra dimension

# Process a single subject
def preprocess_subject(csv_dir, subject_id, num_samples=700, sfreq=500, freqs=np.linspace(1, 40, 20)):
    """Preprocess and return subject data in time-frequency format."""
    signal_files = sorted([f for f in os.listdir(csv_dir) if f'SUB_{subject_id}_SIG' in f])
    annotation_files = sorted([f for f in os.listdir(csv_dir) if f'SUB_{subject_id}_ANN' in f])

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

# Process the full dataset
def load_dataset(csv_dir, num_samples=700, sfreq=500, freqs=np.linspace(1, 40, 20)):
    """Loads all subjects and returns a dataset in time-frequency format."""
    all_subjects_data = []
    all_subjects_labels = []

    subjects = get_subjects(csv_dir)
    for subject_id in range(1,2):
        data, labels = preprocess_subject(csv_dir, "001", num_samples, sfreq, freqs)
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







def main():

    CSV_DIR = "C:\FAV\FAV\\3.rocnik\\bakalarka\Thienuv_navrh\\103_data\eegmmidb"
    log.info(config)
    start_time = time.perf_counter()
    log.info(f"Loading and preprocessing input data.")
    data, labels = load_dataset(CSV_DIR)
    log.info(f"Preprocessing took {_format_execution_time(start_time, time.perf_counter())}.")

    # _personal_models(data, labels)
    data, labels = np.concatenate(data), np.concatenate(labels)
    # visualization.plot_input_data_tsne(data, labels)
    _inter_subject_model(data, labels)

    log.info(f"Total execution time {_format_execution_time(start_time, time.perf_counter())}.")

    if config.save_plots:
        log.info(f"All image output has been saved to {os.getcwd()}/{file_utils.IMAGES_OUTPUT_FOLDER}.")



# import multiprocessing  #TODO tohle odmazat na fav gpu
# multiprocessing.set_start_method("spawn") #TODO tohle odmazat na fav gpu
if __name__ == '__main__':
    main()
