import logging as log

import numpy as np
import pandas as pd
import seaborn as sns
from keras import Model
from keras.callbacks import History
from keras.utils import plot_model
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize
from mne import Epochs
from mne.decoding import Vectorizer
from scipy.signal.windows import gaussian
from sklearn.manifold import TSNE
from sklearn.metrics import ConfusionMatrixDisplay

from src.classification.ClassificationType import ClassificationType
from src.config.Config import config
from src.preprocessing import preprocessing
from src.preprocessing.DataRepresentation import DataRepresentation
from src.preprocessing.MovementType import MovementType
from src.utils import file_manager, file_utils


def _gaussian_smoothing(data: np.ndarray, win_len: int) -> np.ndarray:
    gauss_win = gaussian(win_len, std=win_len / 6.0)
    return np.convolve(data, gauss_win, 'same') / np.sum(gauss_win)


def plot_erd_ers(epochs: Epochs, file_prefix: str) -> None:
    if not config.save_plots:
        return

    def square(data):
        return data ** 2

    log.info(f"Plotting input data ERD/ERS for {file_prefix}.")

    power = epochs.copy().apply_function(square)
    average = power.average(by_event_type=True)
    samples_per_sec = int(epochs.info['sfreq'])

    erd_per_event = {}
    for average_event in average:
        R = np.sum(average_event.data[:, :samples_per_sec], axis=-1) / samples_per_sec
        R = np.expand_dims(R, axis=-1)
        ERD = ((average_event.data - R) / R) * 100
        erd_per_event[int(average_event.comment)] = ERD

    times = np.linspace(config.tmin, config.tmax, num=epochs.get_data().shape[-1])
    smooth_time_interval = 0.20
    win_len = int(samples_per_sec * smooth_time_interval)
    fig, axs = plt.subplots(len(config.channels))
    for row, channel in enumerate(config.channels):
        axs[row].set_title(channel)
        for (event, erd) in erd_per_event.items():
            axs[row].plot(times, _gaussian_smoothing(erd[row], win_len),
                          label=MovementType.label_to_readable_str(event))
            axs[row].set_xlabel('Time (s)')
            axs[row].set_ylabel('ERDS (%)')

    fig.suptitle(f"{file_prefix} ERD/ERS\n "
                 f"(smoothed gaussian window of {int(smooth_time_interval * 1000)}ms)")

    handles, labels = plt.gca().get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=len(erd_per_event.keys()), bbox_to_anchor=(0.5, 0))

    file_manager.save_plot(plt, f"{file_prefix}", file_utils.ERD_ERS_IMAGES_FOLDER)


def plot_compare_input_data(epochs: Epochs) -> None:
    if not config.save_plots:
        return

    log.info("Plotting input data comparison between different labels.")

    unique_labels = np.unique(epochs.events[:, 2])
    num_unique_labels = len(unique_labels)
    num_channels = len(config.channels)

    if config.data_representation == DataRepresentation.TIME_SERIES:
        times = np.linspace(config.tmin, config.tmax, num=epochs.get_data().shape[-1])
        sfreq = epochs.info['sfreq']
        smooth_time_interval = 0.20
        win_len = int(sfreq * smooth_time_interval)

        fig, axs = plt.subplots(num_channels, 1)
        for row, channel in enumerate(config.channels):
            axs[row].set_title(channel)
            for label in unique_labels:
                display_label = MovementType.label_to_readable_str(label)
                labeled_epochs = epochs[str(label)].pick(channel)
                labeled_epochs_data = labeled_epochs.get_data()
                average_labeled_epochs_data = np.average(labeled_epochs_data, axis=0)[0] * 1e6  # Convert from V to uV
                smoothed_average_data = _gaussian_smoothing(average_labeled_epochs_data, win_len)
                axs[row].plot(times, smoothed_average_data, label=display_label)
                axs[row].set_ylabel('Amplitude (μV)')
                axs[row].set_xlabel('Time (s)')

        handles, labels = plt.gca().get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center', ncol=num_unique_labels, bbox_to_anchor=(0.5, 0))
        title = f"Average epochs time series\n" \
                f"(bandpass filtered={config.l_freq}-{config.h_freq}Hz, " \
                f"gaussian smoothing={int(smooth_time_interval * 1000)}ms)"
        fig.suptitle(title)

    if config.data_representation == DataRepresentation.TIME_FREQUENCY:
        fig, axs = plt.subplots(len(config.channels), num_unique_labels)

        for row, channel in enumerate(config.channels):
            for col, label in enumerate(unique_labels):
                display_label = MovementType.label_to_readable_str(label)
                labeled_epochs = epochs[str(label)].pick(channel)
                averaged_labeled_epochs_tfr = preprocessing.epochs_to_time_frequency(labeled_epochs).average()
                averaged_labeled_epochs_tfr.plot(axes=axs[row, col], show=False, vmin=0)
                axs[row, col].set_title(f"{display_label}({channel})")

        fig.suptitle(f"Average epochs time frequency domain")

    if config.data_representation == DataRepresentation.FREQUENCY:
        fig, axs = plt.subplots(num_channels, 1)

        for row, channel in enumerate(config.channels):
            colors = ['blue', 'orange']
            if config.classification_type == ClassificationType.MULTICLASS:
                colors.append('green')
            for color, label in zip(colors, unique_labels):
                labeled_epochs = epochs[str(label)].pick(channel)
                labeled_epochs.compute_psd(fmin=config.l_freq, fmax=config.h_freq).average() \
                    .plot(axes=axs[row], show=False, spatial_colors=False, color=color, alpha=1.0)
                axs[row].set_xlabel('Frequency (Hz)')
                axs[row].lines[-1].set_label(MovementType.label_to_readable_str(label))

            axs[row].set_title(channel)

        handles, labels = plt.gca().get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center', ncol=num_unique_labels, bbox_to_anchor=(0.5, 0))
        fig.suptitle(f"Average epochs frequency domain")

    file_manager.save_plot(plt, f"average_per_label_{config.data_type_suffix()}", file_utils.INPUT_DATA_IMAGES_FOLDER)


def plot_input_data_tsne(data: np.ndarray, labels: np.ndarray) -> None:
    if not config.save_plots:
        return

    log.info("Plotting input data as 2D t-SNE representation.")

    data = Vectorizer().fit_transform(data)
    tsne = TSNE(n_components=2, init='pca', learning_rate='auto')
    tsne_data = tsne.fit_transform(data)

    data_frame = pd.DataFrame()

    data_frame['t-SNE-2d-one'] = tsne_data[:, 0]
    data_frame['t-SNE-2d-two'] = tsne_data[:, 1]
    data_frame['Label'] = [MovementType.label_to_readable_str(label) for label in labels]

    sns.scatterplot(
        x="t-SNE-2d-one", y="t-SNE-2d-two",
        hue="Label",
        palette=sns.color_palette("hls", len(np.unique(labels))),
        data=data_frame,
        alpha=0.5)

    file_manager.save_plot(plt, f"t-SNE_{config.data_type_suffix()}", file_utils.INPUT_DATA_IMAGES_FOLDER)


def plot_real_vs_generated_tsne(data_real: tuple[np.ndarray, np.ndarray], data_generated: tuple[np.ndarray, np.ndarray],
                                augmentation_method: str) -> None:
    if not config.save_plots:
        return

    log.info(f"Plotting t-SNE 2D representation of real input data and "
             f"generated data by augmentation method {augmentation_method}")

    x_real, y_real = data_real
    x_generated, y_generated = data_generated
    # Need to somehow distinguish between which label is a part of the generated dataset and which is real. (the
    # solution is prone to issues in the case that the labels contain the value after the multiplication with the
    # generated_type, e.g. real labels would contain 5 and 50, all generated 5s are converted to 50s, but then it is
    # indistinguishable if the 50 is a generated 5 or a real 50)
    real_type = 1
    generated_type = 10
    real_type_label = np.ones(x_real.shape[0]) * real_type
    generated_type_label = np.ones(x_generated.shape[0]) * generated_type

    data_type = np.concatenate([generated_type_label, real_type_label])

    data = np.vstack([x_generated, x_real])
    labels = np.concatenate([y_generated * generated_type, y_real])

    data = Vectorizer().fit_transform(data)
    tsne = TSNE(n_components=2, init='pca', learning_rate='auto', n_jobs=config.cpus)
    tsne_data = tsne.fit_transform(data)

    data_frame = pd.DataFrame()

    data_frame['t-SNE-2d-one'] = tsne_data[:, 0]
    data_frame['t-SNE-2d-two'] = tsne_data[:, 1]

    def label_to_string(label_index, label):
        prefix = 'Real'
        label_value = label
        if data_type[label_index] == generated_type:
            prefix = 'Generated'
            label_value = label / generated_type

        return f"{prefix} {MovementType.label_to_readable_str(label_value)}"

    data_frame['Label'] = [label_to_string(label_index, label) for label_index, label in enumerate(labels)]

    sns.scatterplot(
        x="t-SNE-2d-one", y="t-SNE-2d-two",
        hue="Label",
        palette=sns.color_palette("hls", len(np.unique(labels) * 2)),
        data=data_frame,
        alpha=0.3)

    file_manager.save_plot(plt,
                           f"t-SNE_{augmentation_method}_{config.data_type_suffix()}",
                           file_utils.AUGMENTATION_IMAGES_FOLDER)


def plot_training_history(history: History or dict, model_name: str, augmentation_name: str = '') -> None:
    if not config.save_plots or history is None:
        return

    augmentation_prefix = f"{augmentation_name}_" if augmentation_name else ''
    full_model_name = f"{augmentation_prefix}{model_name}"
    log.info(f"Plotting training history for model {model_name}.")

    if isinstance(history, History):
        history_dict = history.history
    else:
        history_dict = history

    losses = dict(filter(lambda item: 'loss' in item[0], history_dict.items()))
    accuracies = dict(filter(lambda item: 'accuracy' in item[0], history_dict.items()))

    if len(losses.items()) > 0:
        for loss_label, loss_value in losses.items():
            plt.plot(loss_value, label=loss_label)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f"Model {model_name} training loss")
        plt.legend(loc='upper right')
        file_manager.save_training_history(plt, full_model_name)

    if len(accuracies.items()) > 0:
        for accuracy_label, accuracy_value in accuracies.items():
            plt.plot(accuracy_value, label=accuracy_label)
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title(f"Model {model_name} training accuracy")
        plt.legend(loc='lower right')

        file_manager.save_training_history(plt, full_model_name, 'accuracy')


def save_model_graph(model: Model, folder_name: str) -> None:
    if not config.save_plots:
        return

    output_folder = f"{file_utils.MODELS_IMAGE_GRAPH_FOLDER}/{folder_name}/"
    file_utils.create_folders(output_folder)
    output_file = f"{output_folder}{model.name}{file_utils.PLOT_FILE_EXTENSION}"
    try:
        log.info(f"Saving model '{model.name}' graph to {output_file}.")
        plot_model(model, to_file=output_file, show_shapes=True)
    except ImportError as e:
        log.warning(f"Unable to save graph model '{model.name}'. {e}")


def plot_real_vs_generated(data_real: tuple, data_generated: tuple, augmentation_name: str, file_name_suffix: str = '',
                           average: bool = True) -> None:
    if not config.save_plots:
        return

    log.info(f"Plotting comparison between real data and generated data by {augmentation_name} method.")

    data_real, labels_real = data_real
    data_generated, labels_generated = data_generated

    unique_labels = np.unique(labels_real)

    image_data = config.data_representation == DataRepresentation.TIME_FREQUENCY or \
                 config.data_representation == DataRepresentation.OTHER
    cols = 1
    if image_data:
        cols += 1

    title_prefix = 'Average' if average else 'Random'
    times = np.linspace(config.tmin, config.tmax, num=data_real.shape[-1])
    freqs = np.arange(config.l_freq, config.h_freq + 1)

    for channel_index, channel in enumerate(config.channels):
        fig, axs = plt.subplots(len(unique_labels), cols, squeeze=False)
        for row, label in enumerate(unique_labels):
            display_label = MovementType.label_to_readable_str(label)
            real_labeled = data_real[labels_real == label]
            generated_labeled = data_generated[labels_generated == label]
            real_sample = _get_sample(real_labeled, channel_index, average)
            generated_sample = _get_sample(generated_labeled, channel_index, average)

            samples = [real_sample, generated_sample]
            titles = [f"Real {display_label}", f"Generated {display_label}"]

            if image_data:
                for col, (sample, title) in enumerate(zip(samples, titles)):
                    if config.data_representation == DataRepresentation.TIME_FREQUENCY:
                        axs[row, col].set_title(title)
                        cmap = 'RdBu_r'
                        vmax = np.abs(sample).max()
                        vmin = 0

                        dt = np.median(np.diff(times)) / 2. if len(times) > 1 else 0.1
                        dy = np.median(np.diff(freqs)) / 2. if len(freqs) > 1 else 0.5
                        extent = [times[0] - dt, times[-1] + dt,
                                  freqs[0] - dy, freqs[-1] + dy]
                        im_args = dict(interpolation='nearest', origin='lower', extent=extent,
                                       aspect='auto')

                        time_lims = [extent[0], extent[1]]
                        ylim = [extent[2], extent[3]]

                        img = axs[row, col].imshow(sample, cmap=cmap, norm=Normalize(vmin=vmin, vmax=vmax), **im_args)
                        axs[row, col].set_xlim(time_lims[0], time_lims[-1])
                        axs[row, col].set_ylim(ylim)
                        axs[row, col].set_ylabel('Frequency (Hz)')
                        axs[row, col].set_xlabel('Time (s)')
                        plt.colorbar(mappable=img, ax=axs[row, col])

                    if config.data_representation == DataRepresentation.OTHER:
                        axs[0, 0].set_title('Real')
                        axs[0, 1].set_title('Generated')
                        axs[row, col].imshow(sample, cmap='Greys_r')
                        axs[row, col].axis('off')
            else:
                col = 0
                axs[row, col].set_title(f"{display_label}")
                if config.data_representation == DataRepresentation.TIME_SERIES:
                    axs[row, col].plot(times, generated_sample * 1e6, label='Generated')
                    axs[row, col].plot(times, real_sample * 1e6, label='Real')
                    axs[row, col].set_ylabel('Amplitude (μV)')
                    axs[row, col].set_xlabel('Time (s)')
                elif config.data_representation == DataRepresentation.FREQUENCY:

                    def psd_to_log_scale_uv(data):
                        # scale to uV^2
                        data *= 1e6 * 1e6
                        return 10 * np.log10(data + 1e-12)

                    freqs = np.linspace(config.l_freq, config.h_freq, num=data_real.shape[-1])
                    axs[row, col].plot(freqs, psd_to_log_scale_uv(generated_sample), label='Generated')
                    axs[row, col].plot(freqs, psd_to_log_scale_uv(real_sample), label='Real')
                    axs[row, col].set_ylabel('μV\u00b2/Hz (dB)')
                    axs[row, col].set_xlabel('Frequency (Hz)')

        if not image_data:
            handles, labels = plt.gca().get_legend_handles_labels()
            fig.legend(handles, labels, loc='upper center', ncol=len(unique_labels), bbox_to_anchor=(0.5, 0))

        fig.suptitle(f"{title_prefix} samples comparison (channel={channel})")
        file_name = f"{augmentation_name}_real_vs_generated_{config.data_type_suffix()}" \
                    f"_{channel}_{file_name_suffix}"
        file_manager.save_plot(plt, file_name, file_utils.AUGMENTATION_IMAGES_FOLDER)


def _get_sample(samples: np.ndarray, channel_index: int, average: bool) -> np.ndarray:
    if average:
        sample = np.average(samples, axis=0)
    else:
        random_index = np.random.choice(samples.shape[0], 1)
        sample = samples[random_index, :][0]

    # Assuming n_epochs, n_channels, n_times -> return sample for the plot chanel
    if config.data_representation == DataRepresentation.OTHER:
        return sample

    return sample[channel_index]


def plot_confusion_matrix(confusion_matrix: np.ndarray, classifier_name: str) -> None:
    cmd = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=MovementType.get_display_labels())
    cmd.plot(cmap=plt.cm.Reds, values_format='.3g')
    file_manager.save_plot(plt, f"{classifier_name}_{config.data_type_suffix()}", file_utils.CONFUSION_MATRIX_FOLDER)


def plot_roc_auc(tprs: list, aucs: list, augmentations: list, classifiers: list) -> None:
    plt.title('Average ROC curves')
    plt.plot([0, 1], [0, 1], "k--", label="Random chance (AUC = 0.5)")
    for tpr, auc, augmentation, classifier in zip(tprs, aucs, augmentations, classifiers):
        augmentation_header = f"{augmentation}_" if augmentation else ''
        classification_header = f"{augmentation_header}{classifier}"

        avg_tpr = np.mean(tpr, axis=0)
        fpr_grid = np.linspace(0, 1, avg_tpr.shape[0])

        plt.plot(fpr_grid, avg_tpr, label=f"{classification_header} (AUC={np.mean(auc, axis=-1):.3f})")

    plt.axis("square")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    handles, labels = plt.gca().get_legend_handles_labels()
    fig = plt.gcf()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0))
    file_name_prefix = 'comparison_between_classifiers' if len(classifiers) > 1 else classification_header
    file_manager.save_plot(plt, f"{file_name_prefix}_{config.data_type_suffix()}", file_utils.ROC_AUC_FOLDER)
