import logging as log
import os
import pickle
from typing import Any

import keras.models
import numpy as np
from keras import Model
from keras.callbacks import History
from matplotlib import pyplot

from src.config.Config import config
from src.preprocessing.file_formats.FileFormat import FileFormat
from src.preprocessing.file_formats.SalehFileFormat import SalehFileFormat
from src.preprocessing.file_formats.MochuraFileFormat import MochuraFileFormat
from src.utils import file_utils


def group_input_files_per_person() -> list[list[FileFormat]]:
    """
    Reads files from the ./data folder and groups the files that are measurements of the same person in a list.
    Returns a matrix, where a row represents a single person and the column is a single FileFormat
    """
    formats = [MochuraFileFormat, SalehFileFormat]

    # Parse the format of each input file
    processed_files = []
    for path, _, files in os.walk(file_utils.DATA_FOLDER):
        for file in files:
            file_path = path + '/' + file
            for file_format in formats:
                processed_file = file_format.process(file_path)
                if processed_file is not None:
                    processed_file.read_raw()
                    processed_files.append(processed_file)
                    break

    # Group files belonging to the same person in a list
    files_per_person = []
    for i in range(len(processed_files)):
        file = processed_files[i]
        # This file already belongs to a person which has been processed
        if file is None:
            continue

        person_files = [file]
        for j in range(i + 1, len(processed_files)):
            other_file = processed_files[j]
            if file.same_person(other_file):
                person_files.append(other_file)
                processed_files[j] = None  # Mark the file as processed

        processed_files[i] = None  # Mark the file as processed
        files_per_person.append(person_files)

    return files_per_person


def save_training_history(plt: pyplot, model_name: str, metric: str = 'loss') -> None:
    file_utils.create_folders(file_utils.TRAINING_HISTORY_FOLDER_PATH)
    output_file = f"{file_utils.TRAINING_HISTORY_FOLDER_PATH}{model_name}_{metric}{file_utils.PLOT_FILE_EXTENSION}"
    log.info(f"Saving training history {metric} plot of model {model_name} to file {output_file}.")
    _save_fig(plt, output_file)


def save_plot(plt: pyplot, file_name: str, folder: str = file_utils.IMAGES_OUTPUT_FOLDER) -> None:
    file_utils.create_folders(folder)
    output_file = f"{folder}{file_name}{file_utils.PLOT_FILE_EXTENSION}"
    log.debug(f"Saving plot to {output_file}.")
    _save_fig(plt, output_file)


def _save_fig(plt: pyplot, file: str) -> None:
    plt.tight_layout()
    plt.savefig(file, bbox_inches='tight', dpi=300)
    plt.close()


def load_model(model_architecture: str, sub_model_name: str = None, custom_objects: dict = None,
               augmentation_name: str = '') -> Model or None:
    if sub_model_name is None:
        sub_model_name = model_architecture

    augmentation_prefix = f"{augmentation_name}_" if augmentation_name else ''
    full_model_name = f"{augmentation_prefix}{sub_model_name}"

    model_file_folder = f"{file_utils.TRAINED_MODELS_FOLDER}{model_architecture}/"
    model_file_path = f"{model_file_folder}{full_model_name}_{config.data_type_suffix()}"

    if file_utils.exists(model_file_path) and config.use_pre_trained_models:
        log.info(f"Loading already pretrained model from {model_file_path}.")
        return keras.models.load_model(model_file_path, custom_objects=custom_objects, compile=False)
    elif config.use_pre_trained_models:
        log.warning(
            f"Unable to load pretrained {model_architecture}/{full_model_name} model"
            f" for data representation {config.data_representation}."
            f" The model file {model_file_path} does not exist."
        )

    return None


def save_model(model_architecture: str, model: Model, augmentation_name: str = '') -> None:
    model_file_folder = f"{file_utils.TRAINED_MODELS_FOLDER}{model_architecture}/"
    file_utils.create_folders(model_file_folder)

    augmentation_prefix = f"{augmentation_name}_" if augmentation_name else ''
    full_model_name = f"{augmentation_prefix}{model.name}"

    model_file_path = f"{model_file_folder}{full_model_name}_{config.data_type_suffix()}"
    log.info(f"Saving trained model {model.name} configuration to file {model_file_path}.")
    model.save(model_file_path)


def save_temporary_fold_model(model: Model, fold: int) -> None:
    file_path = f"{file_utils.FOLD_MODEL_FILE_NAME}_{fold}"
    log.info(f"Saving temporary train model {model.name} of fold {fold} to file {file_path}.")
    model.save(file_path)


def load_temporary_fold_model(fold: int, remove=False, custom_objects: dict = None) -> Model:
    model = None
    try:
        log.info(f"Loading temporary train model for fold {fold}.")
        model = keras.models.load_model(f"{file_utils.FOLD_MODEL_FILE_NAME}_{fold}", compile=False,
                                        custom_objects=custom_objects)
        if remove:
            remove_temporary_fold_model(fold)
    except IOError or ImportError as e:
        log.error('Unable to read temporarily saved model ' + str(e))

    return model


def save_temporary_fold_history(history: History, fold: int) -> None:
    with open(f"{file_utils.FOLD_HISTORY_FILE_NAME}_{fold}.pkl", 'wb') as history_file:
        pickle.dump(history.history, history_file)


def load_temporary_fold_history(fold: int, remove=True) -> dict:
    history = None
    try:
        with open(f"{file_utils.FOLD_HISTORY_FILE_NAME}_{fold}.pkl", 'rb') as history_file:
            history = pickle.load(history_file)

        if remove:
            remove_temporary_fold_history(fold)
    except IOError as e:
        log.error('Unable to read temporarily saved training history ' + str(e))

    return history


def save_model_prediction(prediction: np.ndarray, fold: int) -> None:
    np.save(f"{file_utils.FOLD_MODEL_PREDICTION_FILE_NAME}_{fold}.npy", prediction, allow_pickle=True)


def load_model_prediction(fold: int) -> np.ndarray:
    prediction = np.load(f"{file_utils.FOLD_MODEL_PREDICTION_FILE_NAME}_{fold}.npy", allow_pickle=True)
    remove_temporary_model_prediction(fold)

    return prediction


def remove_temporary_fold_model(fold: int) -> None:
    log.info(f"Removing temporary trained model for fold {fold}.")
    file_utils.remove_folder(f"{file_utils.FOLD_MODEL_FILE_NAME}_{fold}")


def remove_temporary_fold_history(fold: int) -> None:
    file_utils.remove_file(f"{file_utils.FOLD_HISTORY_FILE_NAME}_{fold}.pkl")


def remove_temporary_model_prediction(fold: int) -> None:
    file_utils.remove_file(f"{file_utils.FOLD_MODEL_PREDICTION_FILE_NAME}_{fold}.npy")


def load_pickle_object(model_name: str, object_type: str, augmentation_name: str = '') -> Any:
    model_file_folder = f"{file_utils.TRAINED_MODELS_FOLDER}{model_name}/"
    augmentation_prefix = f"{augmentation_name}_" if augmentation_name else ''
    full_model_name = f"{augmentation_prefix}{model_name}"

    object_file_path = f"{model_file_folder}{full_model_name}_{object_type}_{config.data_type_suffix()}.pkl"

    if file_utils.exists(object_file_path) and config.use_pre_trained_models:
        log.info(f"Loading serialized object for {full_model_name} from {object_file_path}.")
        with open(object_file_path, 'rb') as scaler_file:
            return pickle.load(scaler_file)
    elif config.use_pre_trained_models:
        log.warning(f"Unable to load object {object_file_path}. The file doesn't exist.")

    return None


def save_pickle_object(model_name: str, object_type: str, obj: Any, augmentation_name: str = '') -> None:
    model_file_folder = f"{file_utils.TRAINED_MODELS_FOLDER}{model_name}/"
    file_utils.create_folders(model_file_folder)

    augmentation_prefix = f"{augmentation_name}_" if augmentation_name else ''
    full_model_name = f"{augmentation_prefix}{model_name}"
    object_file_path = f"{model_file_folder}{full_model_name}_{object_type}_{config.data_type_suffix()}.pkl"

    log.info(f"Serializing object for {full_model_name} to {object_file_path}.")
    with open(object_file_path, 'wb') as scaler_file:
        pickle.dump(obj, scaler_file)


def save_preprocessed_data(data: np.ndarray, labels: np.ndarray) -> None:
    if not config.save_load_preprocessed_data:
        return

    file_utils.create_folders(file_utils.PREPROCESSED_DATA_FOLDER)

    data_file_path = f"{file_utils.PREPROCESSED_DATA_FOLDER}data_{config.data_type_suffix()}.npy"
    labels_file_path = f"{file_utils.PREPROCESSED_DATA_FOLDER}labels_{config.data_type_suffix()}.npy"

    log.info(f"Saving preprocessed data to file {data_file_path} as a numpy ndarray.")
    log.info(f"Saving preprocessed labels to file {labels_file_path} as a numpy ndarray.")

    np.save(data_file_path, data, allow_pickle=True)
    np.save(labels_file_path, labels, allow_pickle=True)


def load_preprocessed_data() -> tuple:
    if not config.save_load_preprocessed_data:
        return None, None

    data_file_path = f"{file_utils.PREPROCESSED_DATA_FOLDER}data_{config.data_type_suffix()}.npy"
    labels_file_path = f"{file_utils.PREPROCESSED_DATA_FOLDER}labels_{config.data_type_suffix()}.npy"

    if not file_utils.exists(data_file_path):
        log.warning(f"Unable to load preprocessed data from path {data_file_path}."
                    f" Maybe the data has not been preprocessed and saved yet?")
        return None, None

    log.info(f"Loading preprocessed data from {data_file_path}.")
    log.info(f"Loading preprocessed labels from {labels_file_path}.")

    data = np.load(data_file_path, allow_pickle=True)
    labels = np.load(labels_file_path, allow_pickle=True)

    return data, labels
