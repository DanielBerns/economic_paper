import numpy as np


def compute_economic_decision_utility(
    pred_prob: np.ndarray,
    target_tail: np.ndarray,
    thresholds: np.ndarray = np.linspace(0.05, 0.50, 10),
    cost_fn: float = 5.0,
    cost_fp: float = 1.0,
) -> dict[float, float]:
    """Compute net decision utility across policy intervention thresholds c."""
    utilities = {}

    # Maximum possible loss if no intervention (all FN penalized)
    max_loss = np.sum(target_tail == 1.0) * cost_fn + 1e-6

    for c in thresholds:
        intervene = (pred_prob > c).astype(float)
        fn = np.sum((intervene == 0.0) & (target_tail == 1.0))
        fp = np.sum((intervene == 1.0) & (target_tail == 0.0))

        total_cost = fn * cost_fn + fp * cost_fp
        net_utility = 1.0 - (total_cost / max_loss)
        utilities[float(round(c, 2))] = float(net_utility)

    return utilities
