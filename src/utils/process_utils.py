import multiprocessing
from multiprocessing import Process
from typing import Any

import numpy as np

from src.utils import file_manager


def run_in_separate_process(target: callable, args: tuple) -> None:
    multiprocessing.set_start_method("spawn", force=True)  # TODO odmazat
    p = Process(target=target, args=args)
    p.start()
    p.join()


def _model_predict(data: Any, fold: int, remove: bool, custom_objects: dict = None) -> None:
    model = file_manager.load_temporary_fold_model(fold, remove, custom_objects)
    prediction = model.predict(data, verbose=False)
    file_manager.save_model_prediction(prediction, fold)


def model_predict(data: Any, fold: int, remove: bool = True, custom_objects: dict = None) -> np.ndarray:
    run_in_separate_process(_model_predict, (data, fold, remove, custom_objects))
    return file_manager.load_model_prediction(fold)
