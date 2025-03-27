import os
import shutil

IMAGES_OUTPUT_FOLDER = 'images/'
TRAINED_MODELS_FOLDER = 'trained_models/'
MODELS_IMAGE_GRAPH_FOLDER = f"{IMAGES_OUTPUT_FOLDER}models/"
TRAINING_HISTORY_FOLDER_PATH = f"{IMAGES_OUTPUT_FOLDER}training_history/"
DATA_FOLDER = '../../../ZCU_dataset/data'
LOGS_FOLDER = 'logs/'
PREPROCESSED_DATA_FOLDER = 'preprocessed_data/'
ERD_ERS_IMAGES_FOLDER = f"{IMAGES_OUTPUT_FOLDER}erd_ers/"
INPUT_DATA_IMAGES_FOLDER = f"{IMAGES_OUTPUT_FOLDER}input_data/"
AUGMENTATION_IMAGES_FOLDER = f"{IMAGES_OUTPUT_FOLDER}augmentation/"
CONFUSION_MATRIX_FOLDER = f"{IMAGES_OUTPUT_FOLDER}confusion_matrix/"
ROC_AUC_FOLDER = f"{IMAGES_OUTPUT_FOLDER}roc_auc/"
MAIN_LOG_FILE = f"{LOGS_FOLDER}eeg-motion-detection.log"

PLOT_FILE_EXTENSION = '.pdf'

FOLD_MODEL_FILE_NAME = 'fold_model_temp'
FOLD_HISTORY_FILE_NAME = 'fold_history_temp'
FOLD_MODEL_PREDICTION_FILE_NAME = 'fold_model_prediction_temp'


def remove_file(file_path: str) -> None:
    os.remove(file_path)


def remove_folder(path: str) -> None:
    shutil.rmtree(path)


def exists(path: str) -> bool:
    return os.path.exists(path)


def create_folders(folder_path: str) -> None:
    os.makedirs(folder_path, exist_ok=True)
