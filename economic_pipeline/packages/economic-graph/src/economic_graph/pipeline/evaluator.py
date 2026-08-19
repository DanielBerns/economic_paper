from dataclasses import dataclass

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


@dataclass
class EvaluationMetrics:
    rmse: float
    mae: float
    spearman_rho: float
    auroc: float
    auprc: float
    top10_precision: float
    top10_recall: float
    brier_score: float


def compute_metrics(
    pred_growth: np.ndarray,
    pred_prob: np.ndarray,
    target_growth: np.ndarray,
    target_tail: np.ndarray,
) -> EvaluationMetrics:
    """Compute continuous, ranking, tail classification, and calibration metrics."""
    # Continuous metrics
    rmse = float(np.sqrt(np.mean((pred_growth - target_growth) ** 2)))
    mae = float(np.mean(np.abs(pred_growth - target_growth)))

    # Spearman rank correlation
    rho, _ = spearmanr(pred_growth, target_growth)
    spearman_rho = float(rho) if not np.isnan(rho) else 0.0

    # Tail risk classification metrics
    try:
        auroc = float(roc_auc_score(target_tail, pred_prob))
    except Exception:
        auroc = 0.5

    try:
        auprc = float(average_precision_score(target_tail, pred_prob))
    except Exception:
        auprc = 0.1

    # Top 10% predicted risk metrics
    k = max(1, int(len(pred_prob) * 0.10))
    top_k_indices = np.argsort(pred_prob)[-k:]

    actual_positives = np.where(target_tail == 1.0)[0]
    num_actual_pos = max(1, len(actual_positives))

    tp = np.sum(target_tail[top_k_indices] == 1.0)
    top10_prec = float(tp / k)
    top10_rec = float(tp / num_actual_pos)

    # Calibration Brier score
    brier = float(brier_score_loss(target_tail, np.clip(pred_prob, 0.0, 1.0)))

    return EvaluationMetrics(
        rmse=rmse,
        mae=mae,
        spearman_rho=spearman_rho,
        auroc=auroc,
        auprc=auprc,
        top10_precision=top10_prec,
        top10_recall=top10_rec,
        brier_score=brier,
    )


def diebold_mariano_test(e1: np.ndarray, e2: np.ndarray) -> float:
    """Diebold-Mariano test p-value for paired loss differential d = e1^2 - e2^2."""
    d = e1**2 - e2**2
    mean_d = np.mean(d)
    var_d = np.var(d, ddof=1) / max(1, len(d))
    if var_d <= 1e-12:
        return 1.0
    stat = mean_d / np.sqrt(var_d)
    # Two-sided p-value approximation
    from scipy.stats import norm

    p_val = 2.0 * (1.0 - norm.cdf(abs(stat)))
    return float(p_val)
