import numpy as np
import os
import mne
import logging as log
import time
from collections import defaultdict

from sklearn.model_selection import train_test_split

from augmentation.AugmentationMethod import AugmentationMethod
from augmentation.AugmentationMetrics import AugmentationMetrics
from classification.ClassificationMetrics import ClassificationMetrics
from classification.Classifier import Classifier
from config.Config import config
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


def load_BCI42_data(dataset_path, data_file):
    """ Load EEG data and labels from .npy files """
    data_path = os.path.join(dataset_path, data_file + '_data.npy')
    label_path = os.path.join(dataset_path, data_file + '_label.npy')

    data = np.load(data_path)  # Shape: (n_samples, n_channels, n_times)
    label = np.load(label_path).squeeze() - 1  # Adjust labels

    print(f"{data_file} loaded successfully.")

    # Shuffle the data
    data, label = shuffle_data(data, label)

    print("Data shape:", data.shape)  # Expected: (n_samples, n_channels, n_times)
    print("Label shape:", label.shape)  # Expected: (n_samples,)

    return data, label

def shuffle_data(data, label):
    """ Shuffle the dataset randomly """
    index = np.arange(len(data))
    np.random.shuffle(index)
    return data[index], label[index]

def compute_tfr(eeg_data, sfreq, freqs, n_cycles):
    """
    Compute Time-Frequency Representation using Morlet wavelets.

    :param eeg_data: EEG signal (n_samples, n_channels, n_times)
    :param sfreq: Sampling frequency
    :param freqs: List of frequencies to analyze
    :param n_cycles: Number of cycles per frequency
    :return: Time-Frequency representation (n_samples, n_channels, n_frequencies, n_times)
    """
    # eeg_data = eeg_data[:, np.newaxis, :, :]  # Reshape to (n_samples, 1, n_channels, n_times)
    tfr = mne.time_frequency.tfr_array_morlet(eeg_data, sfreq=sfreq, freqs=freqs, n_cycles=n_cycles, output='power')
    return np.array(tfr.squeeze(), dtype=np.float32) #TODO tohle tam bylo puvodne

def load_all_data(data_path, frequencies, sampling_rate, n_cycles):
    """
    Loads and transforms EEG data from multiple subjects using Morlet wavelet transformation.

    :param data_path: Path to the dataset folder.
    :param frequencies: List of frequencies for wavelet decomposition.
    :param sampling_rate: Sampling rate of the EEG data.
    :param n_cycles: Number of cycles per frequency.
    :return: Tuple (all_data, all_labels)
    """
    all_data = []
    all_labels = []

    for subId in range(1, 10):
        train_datafile = f'A0{subId}T'
        test_datafile = f'A0{subId}E'

        train_data, train_labels = load_BCI42_data(data_path, train_datafile)
        test_data, test_labels = load_BCI42_data(data_path, test_datafile)

        # Apply wavelet transformation
        train_data_transformed = compute_tfr(train_data, sampling_rate, frequencies, n_cycles)
        test_data_transformed = compute_tfr(test_data, sampling_rate, frequencies, n_cycles)

        all_data.append(train_data_transformed)
        all_labels.append(train_labels)
        all_data.append(test_data_transformed)
        all_labels.append(test_labels)

    # Convert lists to numpy arrays
    all_data = np.array(all_data)
    all_labels = np.array(all_labels)

    return all_data, all_labels

def main():
    log.info(config)
    start_time = time.perf_counter()
    log.info(f"Loading and preprocessing input data.")

    # data_path = "../../competition_dataset/bci_iv_2a"
    data_path = "/auto/plzen1/home/mbartik/Thienuv_model/competition_preprocessed"
    sampling_rate = 250  # Adjust based on dataset
    frequencies = np.linspace(1, 40, 20)  # Define 20 frequency bands from 1Hz to 40Hz
    n_cycles = frequencies / 2

    data, labels = load_all_data(data_path, frequencies, sampling_rate, n_cycles)
    log.info(f"Preprocessing took {_format_execution_time(start_time, time.perf_counter())}.")

    data, labels = np.concatenate(data), np.concatenate(labels)
    # visualization.plot_input_data_tsne(data, labels)
    _inter_subject_model(data, labels)

    log.info(f"Total execution time {_format_execution_time(start_time, time.perf_counter())}.")

    if config.save_plots:
        log.info(f"All image output has been saved to {os.getcwd()}/{file_utils.IMAGES_OUTPUT_FOLDER}.")


    print(f"Final Data Shape: {data.shape}")  # (n_people, n_samples, n_channels, n_frequencies, n_times)
    print(f"Final Labels Shape: {labels.shape}")  # (n_people, n_samples)

# TODO daty pouzit kod z Thienova navrhu

if __name__ == '__main__':
    main()
