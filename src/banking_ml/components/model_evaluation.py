import json
from pathlib import Path
import numpy as np
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix, classification_report,
    brier_score_loss)

class ModelEvaluation:
    def evaluate(self, model, X, y, threshold=0.5, positive_label=1):
        classes = list(model.classes_)
        if positive_label not in classes:
            raise ValueError(f"Positive label {positive_label!r} not in model classes {classes!r}")
        pos_idx = classes.index(positive_label)
        probabilities = model.predict_proba(X)[:, pos_idx]
        predictions = np.where(probabilities >= threshold, positive_label,
                               next(c for c in classes if c != positive_label))
        metrics = {
            "accuracy": float(accuracy_score(y, predictions)),
            "precision": float(precision_score(y, predictions, pos_label=positive_label, zero_division=0)),
            "recall": float(recall_score(y, predictions, pos_label=positive_label, zero_division=0)),
            "f1": float(f1_score(y, predictions, pos_label=positive_label, zero_division=0)),
            "confusion_matrix": confusion_matrix(y, predictions, labels=classes).tolist(),
            "classification_report": classification_report(y, predictions, output_dict=True, zero_division=0),
            "brier_score": float(brier_score_loss((np.asarray(y) == positive_label).astype(int), probabilities)),
        }
        if len(set(y)) == 2:
            binary_y = (np.asarray(y) == positive_label).astype(int)
            metrics["roc_auc"] = float(roc_auc_score(binary_y, probabilities))
            metrics["pr_auc"] = float(average_precision_score(binary_y, probabilities))
        return metrics, probabilities, predictions

    @staticmethod
    def save(metrics, path):
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(metrics, indent=2, default=str), encoding="utf-8")
