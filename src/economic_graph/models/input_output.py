from typing import List, Tuple
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from economic_graph.data.loader import EconomicGraphSnapshot
from economic_graph.data.matrices import compute_leontief_inverse, compute_technical_coefficients
from economic_graph.models.base import BaseModel


class LeontiefIOModel(BaseModel):
    """Benchmark 4: Classical Leontief Input-Output Multiplier Model (I - A)^(-1) F."""

    def __init__(self):
        super().__init__("Leontief IO Model")
        self.calib_reg = LinearRegression()

    def fit(self, train_snapshots: List[EconomicGraphSnapshot]) -> None:
        raw_preds, y_growth_list = [], []

        for t in range(len(train_snapshots) - 1):
            snap = train_snapshots[t]
            next_snap = train_snapshots[t + 1]

            A = compute_technical_coefficients(snap.Z, snap.Y)
            L = compute_leontief_inverse(A)

            # Conditional demand shock prediction
            demand_shock = (1.0 + snap.common_shock[0]) * snap.F
            pred_X = L.dot(demand_shock)

            pred_g = np.log(np.maximum(1e-6, pred_X)) - np.log(np.maximum(1e-6, snap.Y))
            raw_preds.append(pred_g[:, np.newaxis])
            y_growth_list.append(next_snap.target_growth)

        X_all = np.vstack(raw_preds)
        y_all = np.concatenate(y_growth_list)
        self.calib_reg.fit(X_all, y_all)
        self.is_fitted = True

    def predict(self, snapshot: EconomicGraphSnapshot) -> Tuple[np.ndarray, np.ndarray]:
        A = compute_technical_coefficients(snapshot.Z, snapshot.Y)
        L = compute_leontief_inverse(A)

        demand_shock = (1.0 + snapshot.common_shock[0]) * snapshot.F
        pred_X = L.dot(demand_shock)
        pred_g_raw = np.log(np.maximum(1e-6, pred_X)) - np.log(np.maximum(1e-6, snapshot.Y))

        pred_growth = self.calib_reg.predict(pred_g_raw[:, np.newaxis])
        # Tail probability mapped via sigmoid of calibrated prediction
        pred_prob = 1.0 / (1.0 + np.exp(pred_growth * 10.0 + 1.5))
        return pred_growth, pred_prob


class CentralityRegressionModel(BaseModel):
    """Benchmark 5: Network Centrality Regression Baseline."""

    def __init__(self):
        super().__init__("Centrality Regression")
        self.reg_cont = LinearRegression()
        self.reg_tail = LogisticRegression(max_iter=1000)

    def fit(self, train_snapshots: List[EconomicGraphSnapshot]) -> None:
        X_list, y_growth_list, y_tail_list = [], [], []

        for t in range(len(train_snapshots) - 1):
            snap = train_snapshots[t]
            next_snap = train_snapshots[t + 1]

            cent_feats = snap.S[:, 5:]  # Degrees, strengths, centralities, Herfindahl
            feats = np.column_stack([snap.target_growth, cent_feats])

            X_list.append(feats)
            y_growth_list.append(next_snap.target_growth)
            y_tail_list.append(next_snap.target_tail)

        X_all = np.vstack(X_list)
        y_growth_all = np.concatenate(y_growth_list)
        y_tail_all = np.concatenate(y_tail_list)

        self.reg_cont.fit(X_all, y_growth_all)
        self.reg_tail.fit(X_all, y_tail_all)
        self.is_fitted = True

    def predict(self, snapshot: EconomicGraphSnapshot) -> Tuple[np.ndarray, np.ndarray]:
        cent_feats = snapshot.S[:, 5:]
        feats = np.column_stack([snapshot.target_growth, cent_feats])

        pred_growth = self.reg_cont.predict(feats)
        pred_prob = self.reg_tail.predict_proba(feats)[:, 1]
        return pred_growth, pred_prob


class SpatialARModel(BaseModel):
    """Benchmark 6: Spatial / Network Autoregression (Spatial AR)."""

    def __init__(self):
        super().__init__("Spatial Autoregression")
        self.reg_cont = LinearRegression()
        self.reg_tail = LogisticRegression(max_iter=1000)

    def fit(self, train_snapshots: List[EconomicGraphSnapshot]) -> None:
        X_list, y_growth_list, y_tail_list = [], [], []

        for t in range(len(train_snapshots) - 1):
            snap = train_snapshots[t]
            next_snap = train_snapshots[t + 1]

            spatial_lag = snap.W_share.dot(snap.target_growth)
            feats = np.column_stack([snap.target_growth, spatial_lag, snap.S[:, :5]])

            X_list.append(feats)
            y_growth_list.append(next_snap.target_growth)
            y_tail_list.append(next_snap.target_tail)

        X_all = np.vstack(X_list)
        y_growth_all = np.concatenate(y_growth_list)
        y_tail_all = np.concatenate(y_tail_list)

        self.reg_cont.fit(X_all, y_growth_all)
        self.reg_tail.fit(X_all, y_tail_all)
        self.is_fitted = True

    def predict(self, snapshot: EconomicGraphSnapshot) -> Tuple[np.ndarray, np.ndarray]:
        spatial_lag = snapshot.W_share.dot(snapshot.target_growth)
        feats = np.column_stack([snapshot.target_growth, spatial_lag, snapshot.S[:, :5]])

        pred_growth = self.reg_cont.predict(feats)
        pred_prob = self.reg_tail.predict_proba(feats)[:, 1]
        return pred_growth, pred_prob
