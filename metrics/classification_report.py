"""Classification report + confusion matrix reporting for a completed set
of predictions (e.g. the composed labels run_onnx.OnnxPipeline.classify()
produces over a test set).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, f1_score


class ClassificationReporter:
    """Builds and saves a classification report + confusion matrix for
    one set of (true, predicted) labels.

    Usage:
        reporter = ClassificationReporter(y_true, y_pred)
        print(reporter.report_text())
        reporter.save("run_onnx_test_set")
    """

    def __init__(self, y_true: np.ndarray | list[str], y_pred: np.ndarray | list[str],
                 labels: list[str] | None = None,
                 output_dir: str = "metrics/classification_logs") -> None:
        """Store the predictions and resolve the label set to report over.

        Input:
            y_true: ground-truth class name for every row.
            y_pred: predicted class name for every row (same length and
                order as y_true).
            labels: class names to include in the report, in report
                order. Defaults to the sorted union of classes seen in
                y_true and y_pred.
            output_dir: directory save() writes its files into (created
                if it doesn't exist).
        Output:
            None.
        """
        self.y_true: np.ndarray = np.asarray(y_true)
        self.y_pred: np.ndarray = np.asarray(y_pred)
        self.labels: list[str] = labels if labels is not None else sorted(
            set(self.y_true) | set(self.y_pred))
        self.output_dir: Path = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def report_text(self, digits: int = 4) -> str:
        """Build the sklearn per-class precision/recall/F1 text report.

        Input:
            digits: number of decimal places in the printed report.
        Output:
            The formatted classification report string.
        """
        return classification_report(self.y_true, self.y_pred, labels=self.labels,
                                     zero_division=0, digits=digits)

    def report_dict(self) -> dict:
        """Build the same report as report_text(), as a nested dict.

        Input:
            None.
        Output:
            Dict mapping each label (plus "accuracy"/"macro avg"/
            "weighted avg") to its metrics.
        """
        return classification_report(self.y_true, self.y_pred, labels=self.labels,
                                     zero_division=0, output_dict=True)

    def macro_f1(self) -> float:
        """Compute the macro-averaged F1 score over self.labels.

        Input:
            None.
        Output:
            The macro-F1 score, as a float in [0, 1].
        """
        return f1_score(self.y_true, self.y_pred, labels=self.labels,
                        average="macro", zero_division=0)

    def confusion_matrix(self) -> pd.DataFrame:
        """Build the confusion matrix, rows=true label, columns=predicted
        label, ordered per self.labels.

        Input:
            None.
        Output:
            A (n_labels, n_labels) DataFrame of counts, indexed and
            columned by self.labels.
        """
        cm = confusion_matrix(self.y_true, self.y_pred, labels=self.labels)
        return pd.DataFrame(cm, index=self.labels, columns=self.labels)

    def plot_confusion_matrix(self, output_path: Path, normalize: bool = True) -> None:
        """Render the confusion matrix as a heatmap image.

        Input:
            output_path: destination image file path (e.g. .png).
            normalize: if True, each row is divided by its true-label
                support so cells read as within-class recall fractions;
                if False, cells are raw counts.
        Output:
            None (writes output_path).
        """
        import matplotlib.pyplot as plt

        cm = self.confusion_matrix().to_numpy().astype(np.float64)
        if normalize:
            row_sums = cm.sum(axis=1, keepdims=True)
            cm = np.divide(cm, row_sums, out=np.zeros_like(cm), where=row_sums != 0)

        n = len(self.labels)
        fig, ax = plt.subplots(figsize=(max(6, n * 0.4), max(5, n * 0.4)))
        im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1 if normalize else cm.max())
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(self.labels, rotation=90, fontsize=6)
        ax.set_yticklabels(self.labels, fontsize=6)
        ax.set_xlabel("predicted")
        ax.set_ylabel("true")
        ax.set_title("confusion matrix" + (" (row-normalized)" if normalize else ""))
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(output_path, dpi=150)
        plt.close(fig)

    def save(self, name: str) -> None:
        """Write the text report, dict report (JSON), raw confusion
        matrix (CSV), and a normalized confusion-matrix heatmap (PNG)
        under self.output_dir, all prefixed with `name`.

        Input:
            name: filename prefix for every output file.
        Output:
            None (writes <name>_report.txt, <name>_report.json,
            <name>_confusion_matrix.csv, <name>_confusion_matrix.png).
        """
        import json

        (self.output_dir / f"{name}_report.txt").write_text(self.report_text(), encoding="utf-8")
        (self.output_dir / f"{name}_report.json").write_text(
            json.dumps(self.report_dict(), indent=2), encoding="utf-8")
        self.confusion_matrix().to_csv(self.output_dir / f"{name}_confusion_matrix.csv")
        self.plot_confusion_matrix(self.output_dir / f"{name}_confusion_matrix.png")

