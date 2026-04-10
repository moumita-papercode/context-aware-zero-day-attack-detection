from __future__ import annotations

from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score


def find_optimal_threshold(y_val: np.ndarray, proba_val: np.ndarray) -> float:
    candidates = np.linspace(0.4, 0.6, 41)
    best_thresh = 0.5
    best_f1 = -1.0
    for t in candidates:
        f1 = f1_score(y_val, (proba_val >= t).astype(int))
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = float(t)
    return best_thresh


def evaluate_model(
    y_true: np.ndarray,
    proba: np.ndarray,
    name: str,
    zero_day_mask: Optional[np.ndarray] = None,
    threshold: Optional[float] = None,
) -> Dict[str, float]:
    if threshold is None:
        threshold = 0.5
    y_pred = (proba >= threshold).astype(int)

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    auc = roc_auc_score(y_true, proba)

    zd_rec = 0.0
    if zero_day_mask is not None and zero_day_mask.any():
        zd_rec = recall_score(y_true[zero_day_mask], y_pred[zero_day_mask], zero_division=0)

    print(f"\n{name} (threshold={threshold:.4f}):")
    print(f"  Accuracy:    {acc * 100:6.2f}%")
    print(f"  Precision:   {prec * 100:6.2f}%")
    print(f"  Recall:      {rec * 100:6.2f}%")
    print(f"  F1-Score:    {f1 * 100:6.2f}%")
    print(f"  AUC-ROC:     {auc:6.4f}")
    print(f"  Zero-Day R:  {zd_rec * 100:6.2f}%")

    return {
        "Model": name,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1_Score": f1,
        "AUC_ROC": auc,
        "ZeroDay_Recall": zd_rec,
    }


def results_to_dataframe(results: Iterable[Dict[str, float]]) -> pd.DataFrame:
    return pd.DataFrame(results)
