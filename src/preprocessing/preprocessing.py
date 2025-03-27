import logging as log
import random
import sys
import warnings

import mne
import numpy as np
from mne import Epochs
from mne.time_frequency import EpochsTFR

from src.classification.ClassificationType import ClassificationType
from src.config.Config import config, DataRepresentation
from src.preprocessing.EpochEvent import EpochEvent
from src.preprocessing.MovementType import MovementType
from src.preprocessing.file_formats.FileFormat import FileFormat
from src.utils import file_manager, visualization

warnings.filterwarnings('ignore', message='Concatenation of Annotations within Epochs is not supported yet.'
                                          ' All annotations will be dropped.')

_EPOCH_DROP_EQUALIZE = 'equalize epoch events'
_EPOCH_DROP_HALF_RESTING = 'half resting'
_REJECTION_THRESHOLD = 100e-6


def load_data() -> tuple[np.ndarray, np.ndarray]:
    """
    Reads the input data, either the already saved and preprocessed data if the file exits
    and config.save_load_preprocessed data has been set to true, otherwise reads the raw data signals from data folder.

    Each raw data are grouped together by person that the data belongs to. The raw signals are then preprocessed
    to the desired data representation set in config.data_representation.

    The shape of the returned array data array is:
        time_series: n_people, n_samples, n_channels, n_times
        time_frequency: n_people, n_samples, n_channels, n_frequencies, n_times
        frequency: n_people, n_samples, n_channels, n_frequencies

    The shape of the returned labels array is:
        n_people, n_samples

    :return: a tuple of 2 elements, where the first element is the preprocessed data
     and the second is labels for each data sample
    """
    data = []
    labels = []

    preprocessed_data = file_manager.load_preprocessed_data()
    if preprocessed_data[0] is not None:
        return preprocessed_data

    files_per_person = file_manager.group_input_files_per_person()

    sampling_frequency = _find_min_sampling_frequency(files_per_person)
    personal_epochs = []
    for i, person_files in enumerate(files_per_person):
        log.info(f"Gathering data for person {i + 1}.")
        if config.classification_type == ClassificationType.MULTICLASS:
            left = [left for left in person_files if left.movement_type is MovementType.LEFT]
            right = [right for right in person_files if right.movement_type is MovementType.RIGHT]

            left_epochs, left_labels = _get_epochs(left, MovementType.LEFT.get_epoch_event(), sampling_frequency)
            right_epochs, right_labels = _get_epochs(right, MovementType.RIGHT.get_epoch_event(), sampling_frequency)

            if left_epochs is None or right_epochs is None:
                continue

            # Dropping half of the epochs representing the resting state of the patient from each set, in order to
            # try to maintain a balanced overall dataset where 1/3 is resting 1/3 is left movement and 1/3 is right
            # movement, otherwise the resting state would be much larger than the movements
            _drop_half_resting(left_epochs)
            left_labels = left_epochs.events[:, 2]
            _drop_half_resting(right_epochs)
            right_labels = right_epochs.events[:, 2]

            personal_epochs.append(mne.concatenate_epochs([left_epochs, right_epochs]))

            left_data = _transform_data_representation(left_epochs)
            right_data = _transform_data_representation(right_epochs)

            data.append(np.concatenate((left_data, right_data)))
            labels.append(np.concatenate((left_labels, right_labels)))

        elif config.classification_type == ClassificationType.BINARY:
            epochs, epochs_labels = _get_epochs(person_files, EpochEvent.MOVEMENT_START, sampling_frequency)

            if epochs is None:
                continue

            personal_epochs.append(epochs)

            epochs_data = _transform_data_representation(epochs)
            data.append(epochs_data)
            labels.append(epochs_labels)

    data = np.array(data, dtype=object)
    labels = np.array(labels, dtype=object)

    _log_label_statistics(labels)

    for person_id, person_epochs in enumerate(personal_epochs):
        visualization.plot_erd_ers(person_epochs, f"Person_{person_id + 1}")

    global_epochs = mne.concatenate_epochs(personal_epochs)
    visualization.plot_compare_input_data(global_epochs)
    visualization.plot_erd_ers(global_epochs, 'Global')

    ignored = [reason[0] for reason in tuple(set(global_epochs.drop_log))
               if reason and reason[0] not in config.channels]
    log.info(f"Rejected {global_epochs.drop_log_stats(ignore=ignored) :.2f}% "
             f"based on peak to peak amplitude threshold {_REJECTION_THRESHOLD * 1e6}uV.")

    file_manager.save_preprocessed_data(data, labels)

    return data, labels


def _log_label_statistics(labels: np.ndarray) -> None:
    resting_total = 0
    left_total = 0
    right_total = 0
    log.info("Personal label statistics:")
    for i, label in enumerate(labels):
        resting = len([resting for resting in label if resting == MovementType.RESTING.get_epoch_event()])
        left = len([left for left in label if left == MovementType.LEFT.get_epoch_event()])
        right = len([right for right in label if right == MovementType.RIGHT.get_epoch_event()])

        resting_total += resting
        left_total += left
        right_total += right
        if config.classification_type == ClassificationType.BINARY:
            avg = np.average([resting, left])
            std = np.std([resting, left])
            right_part = ''
            left_part = f"Movement: {left}, "
        else:
            avg = np.average([resting, left, right])
            std = np.std([resting, left, right])
            right_part = f'Right: {right}, '
            left_part = f"Left: {left}, "

        log.info(f"Person: {i + 1}, "
                 f"Resting: {resting}, "
                 f"{left_part}"
                 f"{right_part}"
                 f"Avg: {avg:.2f}, "
                 f"Std: {std:.2f}, "
                 f"Sum: {len(label)}")

    if config.classification_type == ClassificationType.BINARY:
        avg = np.average([resting_total, left_total])
        std = np.std([resting_total, left_total])
        right_part = ''
        left_part = f"Movement: {left_total}, "
    else:
        avg = np.average([resting_total, left_total, right_total])
        std = np.std([resting_total, left_total, right_total])
        right_part = f'Right: {right_total}, '
        left_part = f"Left: {left_total}, "

    log.info("Global label statistics:")
    log.info(f"Resting: {resting_total}, "
             f"{left_part}"
             f"{right_part}"
             f"Avg: {avg:.2f}, "
             f"Std: {std:.2f}, "
             f"Sum: {len(np.concatenate(labels))}")


def _transform_data_representation(epochs: Epochs) -> np.ndarray:
    if config.data_representation == DataRepresentation.TIME_SERIES:
        return epochs.get_data()

    elif config.data_representation == DataRepresentation.TIME_FREQUENCY:
        return epochs_to_time_frequency(epochs).data

    elif config.data_representation == DataRepresentation.FREQUENCY:
        return epochs.compute_psd(fmin=config.l_freq, fmax=config.h_freq).get_data()


def epochs_to_time_frequency(epochs: Epochs) -> EpochsTFR:
    """
    Converts given epochs object to time frequency domain using Morlet wavelet method.
    The frequency range is set by config.l_freq and config.h_freq, the time domain is decimated by a factor of 4.

    :param epochs: epochs that are to be converted to time frequency domain
    :return: the converted EpochsTFR object instance
    """
    freqs = np.arange(config.l_freq, config.h_freq + 1)
    n_cycles = freqs / 2
    epochs_tfr = mne.time_frequency.tfr_morlet(epochs, freqs, n_cycles,
                                               use_fft=True, decim=4, average=False, return_itc=False)

    return epochs_tfr


def _drop_half_resting(epochs: Epochs) -> None:
    epoch_events = epochs.events[:, 2]
    resting_indices = [i for i, event in enumerate(epoch_events) if event == MovementType.RESTING.get_epoch_event()]

    amount_to_delete = len(resting_indices) // 2
    # Randomly pick half of the resting indices to drop
    resting_indices_to_remove = random.sample(resting_indices, amount_to_delete)
    epochs.drop(resting_indices_to_remove, reason=_EPOCH_DROP_HALF_RESTING)


def _find_min_sampling_frequency(files_per_person: list[list[FileFormat]]) -> int:
    min_sampling_frequency = sys.float_info.max
    for person_files in files_per_person:
        for file in person_files:
            if file.raw is not None:
                min_sampling_frequency = min(file.raw.info['sfreq'], min_sampling_frequency)

    return min_sampling_frequency


def _get_epochs(files: list[FileFormat], movement_event: int, sample_frequency: int) \
        -> tuple[None, None] or tuple[Epochs, np.ndarray]:
    raws = [file.raw for file in files if file.raw is not None]

    if not raws:
        return None, None

    raw = mne.concatenate_raws(raws)

    events, _ = mne.events_from_annotations(raw, verbose=False)
    epochs = mne.Epochs(raw,
                        tmin=config.tmin, tmax=config.tmax,
                        events=events,
                        picks=config.channels,
                        preload=True,
                        verbose=False,
                        baseline=(None, config.tmin + 0.5))
    # Cropping the time because when the sampling frequency is e.g. 500 and trying to get 1 sec epoch,
    # the constructor returns 501 samples, by calling crop with include_tmax set to False we get the expected 500
    # samples
    epochs.crop(include_tmax=False)
    epochs.resample(sample_frequency)
    epochs.filter(config.l_freq, config.h_freq, verbose=False)
    epochs.drop_bad(reject={'eeg': _REJECTION_THRESHOLD}, verbose=False)

    _equalize_epoch_events(epochs, movement_event)

    return epochs, epochs.events[:, 2]


def _equalize_epoch_events(epochs: Epochs, movement_start_side_marker: int) -> None:
    events = epochs.events
    events_to_keep = []
    indices_to_drop = []
    for i in range(len(events)):
        keep = False
        marker = events[i][2]
        last_marker = None if len(events_to_keep) == 0 else events_to_keep[len(events_to_keep) - 1][2]

        # Only keep the resting epoch if there was a movement epoch between this epoch and the last resting epoch
        if marker == EpochEvent.RESTING_MIDDLE and marker != last_marker:
            keep = True
        # Only keep the first movement epoch between two resting epochs
        elif (marker == EpochEvent.MOVEMENT_START or marker == EpochEvent.MOVEMENT_ADDITIONAL) and \
                (last_marker == EpochEvent.RESTING_MIDDLE):
            epochs.events[i][2] = movement_start_side_marker
            keep = True

        if keep:
            events_to_keep.append(events[i])
        else:
            indices_to_drop.append(i)

    epochs.event_id = {f"{MovementType.RESTING.get_epoch_event()}": int(MovementType.RESTING.get_epoch_event()),
                       f"{movement_start_side_marker}": movement_start_side_marker}
    epochs.drop(indices_to_drop, reason=_EPOCH_DROP_EQUALIZE)
