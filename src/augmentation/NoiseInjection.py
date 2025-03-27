import logging as log

import numpy as np

from src.augmentation.AugmentationMethod import AugmentationMethod
from src.utils import visualization


class NoiseInjection(AugmentationMethod):

    def generate(self, data: np.ndarray, labels: np.ndarray, amount_to_generate: int) \
            -> tuple[np.ndarray, np.ndarray]:
        log.info(f"Using {self.get_name()} method to generate {amount_to_generate} new data points.")
        generated_data, generated_data_labels = self._generate_new_data(data, labels, amount_to_generate)

        visualization.plot_real_vs_generated((data, labels), (generated_data, generated_data_labels), self.get_name())

        return generated_data, generated_data_labels

    @staticmethod
    def _generate_new_data(data: np.ndarray, labels: np.ndarray, amount_to_generate: int) -> tuple:
        unique_labels = np.unique(labels)
        amount_to_generate_per_label = int(amount_to_generate / len(np.unique(labels)))
        generated_data = []
        generated_data_labels = []

        for label in unique_labels:
            shape = (amount_to_generate_per_label,) + data.shape[1:]
            noise = np.random.randn(*shape)
            random_data_index_sample = np.random.choice(np.where(labels == label)[0], amount_to_generate_per_label)
            source_data = data[random_data_index_sample]
            generated_data_for_label = source_data + (source_data * noise)
            generated_labels = np.array([label for _ in range(amount_to_generate_per_label)])

            generated_data.append(generated_data_for_label)
            generated_data_labels.append(generated_labels)

        return np.concatenate(generated_data), np.concatenate(generated_data_labels)

    def get_name(self) -> str:
        return 'NI'
