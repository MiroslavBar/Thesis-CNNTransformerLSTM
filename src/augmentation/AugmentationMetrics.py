import logging as log

import numpy as np
import scipy
from mne.decoding import Vectorizer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tabulate import tabulate

from src.augmentation.AugmentationMethod import AugmentationMethod
from src.config.Config import config
from src.preprocessing.DataRepresentation import DataRepresentation
from src.preprocessing.NDScaler import NDScaler
from src.utils import visualization


class AugmentationMetrics:

    def __init__(self, report_metrics: list):
        self._report_metrics = report_metrics
        self._report_header = ['Method']
        self._calculated_metrics_rows = []

    @classmethod
    def _calculate_fid_score(cls, data_real: np.ndarray, data_generated: np.ndarray) -> float:
        log.info("Calculating Frechet Inception Distance score.")
        data_real = make_pipeline(Vectorizer(), StandardScaler()).fit_transform(data_real)
        data_generated = make_pipeline(Vectorizer(), StandardScaler()).fit_transform(data_generated)

        mu_real = np.mean(data_real, axis=0)
        sigma_real = np.cov(data_real, rowvar=False)

        mu_generated = np.mean(data_generated, axis=0)
        sigma_generated = np.cov(data_generated, rowvar=False)

        ssdiff = np.sum((mu_real - mu_generated) ** 2.0)
        covmean = scipy.linalg.sqrtm(sigma_real.dot(sigma_generated))
        if np.iscomplexobj(covmean):
            covmean = covmean.real

        return ssdiff + np.trace(sigma_real + sigma_generated - 2.0 * covmean)

    @classmethod
    def _calculate_snr(cls, data_generated: np.ndarray) -> float:
        log.info("Calculating Signal to noise ratio.")
        data_generated = NDScaler().fit_transform(data_generated)

        signal_power = np.mean(np.square(data_generated))

        noise = data_generated - np.mean(data_generated, axis=1, keepdims=True)
        noise_power = np.mean(np.square(noise))

        snr = 10 * np.log10(signal_power / noise_power)

        return snr

    @classmethod
    def _calculate_rmse(cls, data_real: np.ndarray, data_generated: np.ndarray) -> float:
        log.info("Calculating root mean square error.")
        data_real = NDScaler().fit_transform(data_real)
        data_generated = NDScaler().fit_transform(data_generated)

        num_samples = min(data_real.shape[0], data_generated.shape[0])
        data_real = data_real[np.random.choice(data_real.shape[0], num_samples, replace=False)]
        data_generated = data_generated[np.random.choice(data_generated.shape[0], num_samples, replace=False)]

        mse = np.mean(np.square(data_real - data_generated))

        rmse = np.sqrt(mse)

        return rmse

    @classmethod
    def _calculate_cross_correlation(cls, data_real: tuple, data_generated: tuple) -> float:
        log.info("Calculating cross correlation.")
        x_real, y_real = data_real
        x_generated, y_generated = data_generated

        x_real = NDScaler().fit_transform(x_real)
        x_generated = NDScaler().fit_transform(x_generated)

        mean_cross_correlation_per_label = []
        for label in np.unique(y_real):
            label_real_data = x_real[y_real == label]
            label_generated_data = x_generated[y_generated == label]
            label_cross_correlation = []
            for real_sample in label_real_data:
                for generated_sample in label_generated_data:
                    label_cross_correlation.append(scipy.signal.correlate(real_sample, generated_sample, mode='valid'))

            label_cross_correlation = np.squeeze(label_cross_correlation)
            mn, mx = np.min(label_cross_correlation), np.max(label_cross_correlation)
            label_cross_correlation = (label_cross_correlation - mn) / (mx - mn)

            mean_cross_correlation_per_label.append(np.mean(label_cross_correlation))

        return np.mean(mean_cross_correlation_per_label)

    def evaluate(self, data_real: tuple[np.ndarray, np.ndarray], data_generated: tuple[np.ndarray, np.ndarray],
                 method: AugmentationMethod) -> None:
        if not method.get_name():
            return

        log.info(f"Calculating evaluation metrics for augmentation method {method.get_name()}.")

        x_real, y_real = data_real
        x_generated, y_generated = data_generated

        visualization.plot_real_vs_generated_tsne(data_real, data_generated, method.get_name())
        method_metrics = [method.get_name()]

        # Since time frequency sample has dimensions of 3x23x500 the score has to calculate a square root of matrix
        # of dimensions 34500x34500, which can be of type complex128. That would mean it needs 34500*34500*16 bytes,
        # which is 18GB of memory.
        if 'fid' in self._report_metrics and config.data_representation != DataRepresentation.TIME_FREQUENCY:
            self._report_header.append('FID')
            fid = self._calculate_fid_score(x_real, x_generated)
            method_metrics.append(f"{fid:.3f}")

        if 'snr' in self._report_metrics:
            self._report_header.append('SNR')
            snr = self._calculate_snr(x_generated)
            method_metrics.append(f"{snr:.3f}")

        if 'rmse' in self._report_metrics:
            self._report_header.append('RMSE')
            rmse = self._calculate_rmse(x_real, x_generated)
            method_metrics.append(f"{rmse}")

        if 'cc' in self._report_metrics:
            self._report_header.append('CC')
            cross_correlation = self._calculate_cross_correlation(data_real, data_generated)
            method_metrics.append(f"{cross_correlation:.3f}")

        self._calculated_metrics_rows.append(method_metrics)
        self.report([method_metrics])

    def report(self, metrics: list = None) -> None:
        if not metrics:
            metrics = self._calculated_metrics_rows

        if not metrics:
            return

        log.info(f"\n{tabulate(metrics, headers=self._report_header, tablefmt=config.tabulate_format)}")
