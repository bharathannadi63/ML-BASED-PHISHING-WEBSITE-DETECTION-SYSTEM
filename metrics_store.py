from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any, Optional

import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

_ROOT = os.path.dirname(os.path.abspath(__file__))
METRICS_PATH = os.path.join(_ROOT, "model_metrics.json")

_DISPLAY_NAMES = {
    "random_forest": "Random Forest",
    "ffnn": "Feed Forward Neural Network (FFNN)",
    "wide_deep": "Wide & Deep Network",
    "tabnet": "TabNet",
}

_COMPLEXITY = {
    "random_forest": "Low-Medium",
    "ffnn": "Medium",
    "wide_deep": "High",
    "tabnet": "Medium-High",
}


def _row_for_model(y_true: np.ndarray, y_prob: np.ndarray, model_key: str, threshold: float = 0.5) -> dict[str, Any]:
    probs = np.asarray(y_prob, dtype=np.float64).flatten()
    y_pred = (probs > threshold).astype(int)
    acc = float(accuracy_score(y_true, y_pred))
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", pos_label=1, zero_division=0
    )
    prec_f, rec_f, f1_f = float(prec), float(rec), float(f1)
    anti = rec_f
    return {
        "display_name": _DISPLAY_NAMES[model_key],
        "accuracy_pct": round(acc * 100, 2),
        "precision_pct": round(prec_f * 100, 2),
        "recall_pct": round(rec_f * 100, 2),
        "f1_pct": round(f1_f * 100, 2),
        "anti_phishing_rate_pct": round(anti * 100, 2),
        "complexity": _COMPLEXITY[model_key],
    }


def write_performance_metrics(
    y_true: np.ndarray, 
    y_prob_ffnn: np.ndarray, 
    y_prob_rf: np.ndarray = None,
    y_prob_wd: np.ndarray = None,
    y_prob_tabnet: np.ndarray = None,
    threshold: float = 0.5
) -> dict[str, Any]:
    """
    Evaluate models and persist metrics for the performance page.
    """
    y_true = np.asarray(y_true, dtype=int).flatten()
    y_prob_ffnn = np.asarray(y_prob_ffnn, dtype=np.float64).flatten()
    payload: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "evaluation": {
            "threshold": threshold,
            "test_examples": int(len(y_true)),
        },
        "models": {},
    }
    for key in ("random_forest", "ffnn", "wide_deep", "tabnet"):
        if key == "random_forest":
            y_prob = y_prob_rf if y_prob_rf is not None else y_prob_ffnn
        elif key == "ffnn":
            y_prob = y_prob_ffnn
        elif key == "wide_deep":
            y_prob = y_prob_wd if y_prob_wd is not None else y_prob_ffnn
        elif key == "tabnet":
            y_prob = y_prob_tabnet if y_prob_tabnet is not None else y_prob_ffnn
        else:
            y_prob = y_prob_ffnn
            
        payload["models"][key] = _row_for_model(y_true, y_prob, key, threshold=threshold)
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"Performance metrics saved to {METRICS_PATH}")
    return payload


def load_metrics_raw() -> Optional[dict[str, Any]]:
    if not os.path.exists(METRICS_PATH):
        return None
    try:
        with open(METRICS_PATH, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def get_model_metrics_for_predict() -> dict[str, dict[str, str]]:
    """Shape expected by predict(): name, accuracy, precision as display strings."""
    raw = load_metrics_raw()
    out: dict[str, dict[str, str]] = {}
    for key in ("random_forest", "ffnn", "wide_deep", "tabnet"):
        row = None
        if raw and isinstance(raw.get("models"), dict):
            row = raw["models"].get(key)
        if row and all(k in row for k in ("accuracy_pct", "precision_pct")):
            out[key] = {
                "name": str(row.get("display_name", _DISPLAY_NAMES[key])),
                "accuracy": f"{float(row['accuracy_pct']):.1f}",
                "precision": f"{float(row['precision_pct']):.1f}",
            }
        else:
            out[key] = {
                "name": _DISPLAY_NAMES[key],
                "accuracy": "—",
                "precision": "—",
            }
    return out


def get_performance_page_context() -> dict[str, Any]:
    """Template context for performance.html (table + Chart.js)."""
    raw = load_metrics_raw()
    order = ("random_forest", "ffnn", "wide_deep", "tabnet")
    models: list[dict[str, Any]] = []
    has_data = False
    
    for key in order:
        if raw and isinstance(raw.get("models"), dict) and key in raw.get("models"):
            row = raw["models"][key]
            models.append({"key": key, "is_empty": False, **row})
            has_data = True
        else:
            models.append({
                "key": key,
                "display_name": _DISPLAY_NAMES.get(key, key),
                "accuracy_pct": 0.0,
                "precision_pct": 0.0,
                "recall_pct": 0.0,
                "f1_pct": 0.0,
                "anti_phishing_rate_pct": 0.0,
                "complexity": _COMPLEXITY.get(key, "Unknown"),
                "is_empty": True
            })
    chart_datasets: list[dict[str, Any]] = []
    colors = {
        "random_forest": ("rgba(155, 89, 182, 0.8)", "rgba(155, 89, 182, 1)", "rgba(155, 89, 182, 0.6)"),
        "ffnn": ("rgba(52, 152, 219, 0.8)", "rgba(52, 152, 219, 1)", "rgba(52, 152, 219, 0.6)"),
        "wide_deep": ("rgba(46, 204, 113, 0.8)", "rgba(46, 204, 113, 1)", "rgba(46, 204, 113, 0.6)"),
        "tabnet": ("rgba(241, 196, 15, 0.8)", "rgba(241, 196, 15, 1)", "rgba(241, 196, 15, 0.6)"),
    }
    for m in models:
        k = m.get("key", "ffnn")
        bg, border, hover = colors.get(k, colors["ffnn"])
        chart_datasets.append(
            {
                "label": m.get("display_name", _DISPLAY_NAMES.get(k, k)),
                "data": [
                    float(m["accuracy_pct"]),
                    float(m["precision_pct"]),
                    float(m["recall_pct"]),
                    float(m["f1_pct"]),
                    float(m["anti_phishing_rate_pct"]),
                ],
                "backgroundColor": bg,
                "borderColor": border,
                "borderWidth": 2,
                "borderRadius": 4,
                "hoverBackgroundColor": hover,
            }
        )
    return {
        "perf_has_data": has_data,
        "perf_generated_at": (raw or {}).get("generated_at"),
        "perf_test_examples": ((raw or {}).get("evaluation") or {}).get("test_examples"),
        "perf_models": models,
        "perf_chart": {
            "labels": ["Accuracy", "Precision", "Recall", "F1-Score", "Anti-phishing Rate"],
            "datasets": chart_datasets,
        },
    }
