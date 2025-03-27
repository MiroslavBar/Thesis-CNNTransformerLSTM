import logging as log
from abc import ABC, abstractmethod

import numpy as np

import src.augmentation


class AugmentationMethod(ABC):

    @abstractmethod
    def get_name(self) -> str:
        pass

    @abstractmethod
    def generate(self, data: np.ndarray, labels: np.ndarray, amount_to_generate: int) \
            -> tuple[np.ndarray, np.ndarray]:
        pass

    @classmethod
    def get_data_augmentation_methods(cls, method_names: list) -> 'list[AugmentationMethod]':
        methods = [NoAugmentation(),
                   src.augmentation.CVAE.CVAE(),
                   src.augmentation.NoiseInjection.NoiseInjection(),
                   src.augmentation.CWGANGP.CWGANGP()]
        possible_method_names = [method.get_name() for method in methods]
        methods_to_use = []
        found = False
        for method_name in method_names:
            try:
                index = possible_method_names.index(method_name)
                methods_to_use.append(methods[index])
                found = True
            except ValueError:
                log.warning(f"{method_name} is not a valid classifier.")
        if not found:
            log.warning(f"None of the requested augmentation methods {', '.join(method_names)} are available. "
                        f"Continuing with no augmentation.\n Possible augmentation method names are "
                        f"{','.join([method.get_name() for method in methods[1:]])}.")
            methods_to_use.append(methods[0])

        return methods_to_use


class NoAugmentation(AugmentationMethod):

    def get_name(self) -> str:
        return ''

    def generate(self, data: np.ndarray, labels: np.ndarray, amount_to_generate: int) \
            -> tuple[np.ndarray, np.ndarray]:
        log.info("No augmentation is being done, using original data.")
        return np.empty(shape=data.shape), np.empty(shape=labels.shape)
