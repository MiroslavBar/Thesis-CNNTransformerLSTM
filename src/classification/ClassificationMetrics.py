import logging as log

import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import LabelBinarizer
from tabulate import tabulate

from src.config.Config import config
from src.utils import visualization
import seaborn as sns
import matplotlib.pyplot as plt


class ClassificationMetrics:

    def __init__(self):
        self.classifier_name = None
        self.augmentation_name = None

        self._classifiers = []
        self._augmentations = []

        self._accuracies = []
        self._precisions = []
        self._recalls = []
        self._f1_scores = []
        self._fpr = []
        self._tpr = []
        self._auc = []
        self._confusion_matrices = []

    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, one_hot_to_class_label: callable) -> None:
        log.info("Calculating classification metrics.")
        y_true_labels, y_pred_labels = one_hot_to_class_label(y_true, y_pred)

        self._accuracies.append(accuracy_score(y_true_labels, y_pred_labels))
        self._precisions.append((precision_score(y_true_labels, y_pred_labels, average='macro', zero_division=0)))
        self._recalls.append((recall_score(y_true_labels, y_pred_labels, average='macro', zero_division=0)))
        self._calculate_auc(y_true_labels, y_pred_labels)
        self._f1_scores.append((f1_score(y_true_labels, y_pred_labels, average='macro', zero_division=0)))
        self._confusion_matrices.append(confusion_matrix(y_true_labels, y_pred_labels, normalize="true"))

    def _calculate_auc(self, y_true_labels: np.ndarray, y_pred_labels: np.ndarray) -> None:
        label_binarizer = LabelBinarizer()
        y_true = label_binarizer.fit_transform(y_true_labels)
        y_pred = label_binarizer.transform(y_pred_labels)

        classes = len(y_true[0])
        current_fpr = [0] * classes
        current_tpr = [0] * classes
        current_auc_score = [0] * classes

        fpr_grid = np.linspace(0, 1, 1000)
        mean_tpr = np.zeros_like(fpr_grid)

        for idx in range(classes):
            current_fpr[idx], current_tpr[idx], _ = roc_curve(y_true[:, idx], y_pred[:, idx])
            mean_tpr += np.interp(fpr_grid, current_fpr[idx], current_tpr[idx])
            current_auc_score[idx] = auc(current_fpr[idx], current_tpr[idx])

        self._tpr.append(list(mean_tpr / classes))
        self._auc.append(np.mean(current_auc_score))

    def report(self, display_metrics: list, transpose: bool = True) -> None:
        if not self._accuracies:
            return

        header = ['Metric']

        merged = len(self._augmentations) > 0

        if not merged:
            self._augmentations.append(self.augmentation_name)
            self._classifiers.append(self.classifier_name)

        for augmentation_name, classifier_name in zip(self._augmentations, self._classifiers):
            augmentation_header = f"{augmentation_name} " if augmentation_name else ''
            classification_header = f"{augmentation_header}{classifier_name}"
            header.append(classification_header)

        tabular_data = [header]

        if 'accuracy' in display_metrics:
            accuracy_row = ['Accuracy']
            accuracy_means = np.atleast_1d(np.mean(self._accuracies, axis=-1))
            accuracy_stds = np.atleast_1d(np.std(self._accuracies, axis=-1))
            for accuracy_mean, accuracy_std in zip(accuracy_means, accuracy_stds):
                accuracy_row.append(f"{accuracy_mean * 100:.2f}±{accuracy_std * 100:.2f}")
            tabular_data.append(accuracy_row)

        if 'precision' in display_metrics:
            precision_row = ['Precision']
            precision_means = np.atleast_1d(np.mean(self._precisions, axis=-1))
            precision_stds = np.atleast_1d(np.std(self._precisions, axis=-1))
            for precision_mean, precision_std in zip(precision_means, precision_stds):
                precision_row.append(f"{precision_mean * 100:.2f}±{precision_std * 100:.2f}")
            tabular_data.append(precision_row)

        if 'recall' in display_metrics:
            recall_row = ['Recall']
            recall_means = np.atleast_1d(np.mean(self._recalls, axis=-1))
            recall_stds = np.atleast_1d(np.std(self._recalls, axis=-1))
            for recall_mean, recall_std in zip(recall_means, recall_stds):
                recall_row.append(f"{recall_mean * 100:.2f}±{recall_std * 100:.2f}")
            tabular_data.append(recall_row)

        if 'f1_score' in display_metrics:
            f1_score_row = ['F1 Score']
            f1_score_means = np.atleast_1d(np.mean(self._f1_scores, axis=-1))
            f1_score_stds = np.atleast_1d(np.std(self._f1_scores, axis=-1))
            for f1_score_mean, f1_score_std in zip(f1_score_means, f1_score_stds):
                f1_score_row.append(f"{f1_score_mean * 100:.2f}±{f1_score_std * 100:.2f}")
            tabular_data.append(f1_score_row)

        if 'auc' in display_metrics:
            auc_row = ['AUC']
            auc_means = np.atleast_1d(np.mean(self._auc, axis=-1))
            auc_stds = np.atleast_1d(np.std(self._auc, axis=-1))
            for auc_mean, auc_std in zip(auc_means, auc_stds):
                auc_row.append(f"{auc_mean:.3f}±{auc_std:.2f}")

            aucs = self._auc
            tprs = self._tpr
            if not merged:
                tprs = np.expand_dims(self._tpr, axis=0)
                aucs = np.expand_dims(self._auc, axis=0)

            visualization.plot_roc_auc(tprs, aucs, self._augmentations, self._classifiers)

            tabular_data.append(auc_row)

        if 'confusion_matrix' in display_metrics:
            if merged:
                conf_matrix_avg = np.mean(np.atleast_3d(self._confusion_matrices), axis=1)
                for matrix, augmentation_name, classifier_name \
                        in zip(np.atleast_3d(conf_matrix_avg), self._augmentations, self._classifiers):
                    augmentation_header = f"{augmentation_name}_" if augmentation_name else ''
                    classification_header = f"{augmentation_header}{classifier_name}"

                    visualization.plot_confusion_matrix(matrix, classification_header)

                    # Plot confusion matrix using seaborn heatmap
                    #plt.figure(figsize=(8, 6))
                    #sns.set(font_scale=1.2)
                    #sns.heatmap(matrix, cmap='Reds', annot=True, fmt='.2f')
                    #plt.title(f'Confusion Matrix - {classification_header}')
                    #plt.xlabel('Predicted Labels')
                    #plt.ylabel('True Labels')
                    #plt.savefig(f'Confusion_Matrix_{classification_header}.png', bbox_inches='tight', dpi=300)
                    #plt.show()

            else:
                conf_matrix_avg = np.mean(self._confusion_matrices, axis=0)

                augmentation_header = f"{self.augmentation_name}_" if self.augmentation_name else ''
                classification_header = f"{augmentation_header}{self.classifier_name}"
                visualization.plot_confusion_matrix(conf_matrix_avg, classification_header)

                # Plot confusion matrix using seaborn heatmap
                #plt.figure(figsize=(8, 6))
                #sns.set(font_scale=1.2)
                #sns.heatmap(conf_matrix_avg, cmap='Reds', annot=True, fmt='.2f')
                #plt.title(f'Confusion Matrix - {classification_header}')
                #plt.xlabel('Predicted Labels')
                #plt.ylabel('True Labels')
                #plt.savefig(f'Confusion_Matrix_{classification_header}.png', bbox_inches='tight', dpi=300)
                #plt.show()

        if transpose:
            # Remove the metric header
            tabular_data[0].pop(0)
            new_header = ['Method']
            for row in tabular_data[1:]:
                new_header.append(row.pop(0))

            tabular_data = np.array(tabular_data).T.tolist()
            tabular_data.insert(0, new_header)

        log.info(f"\n{tabulate(tabular_data, headers='firstrow', tablefmt=config.tabulate_format)}")

    def is_best_model(self) -> bool:
        return self._accuracies[-1] == np.max(self._accuracies)

    def __lt__(self, other):
        return self.augmentation_name < other.augmentation_name

    @classmethod
    def merge(cls, all_metrics: dict) -> 'ClassificationMetrics':
        combined_metrics = ClassificationMetrics()

        for classifier, metrics in all_metrics.items():
            for metric in sorted(metrics):
                combined_metrics._classifiers.append(metric.classifier_name)
                combined_metrics._augmentations.append(metric.augmentation_name)

                combined_metrics._accuracies.append(metric._accuracies)
                combined_metrics._precisions.append(metric._precisions)
                combined_metrics._recalls.append(metric._recalls)
                combined_metrics._f1_scores.append(metric._f1_scores)
                combined_metrics._tpr.append(metric._tpr)
                combined_metrics._auc.append(metric._auc)
                combined_metrics._confusion_matrices.append(metric._confusion_matrices)

        return combined_metrics
