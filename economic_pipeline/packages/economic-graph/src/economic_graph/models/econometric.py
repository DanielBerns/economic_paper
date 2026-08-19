from typing import List, Tuple

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, LogisticRegression

from economic_graph.data.loader import EconomicGraphSnapshot
from economic_graph.models.base import BaseModel


class PanelARModel(BaseModel):
    """Benchmark 1: Panel Autoregression with fixed effects & macro controls."""

    def __init__(self):
        super().__init__("Panel AR (Fixed Effects)")
        self.reg_cont = LinearRegression()
        self.reg_tail = LogisticRegression(max_iter=1000)

    def fit(self, train_snapshots: List[EconomicGraphSnapshot]) -> None:
        X_list, y_growth_list, y_tail_list = [], [], []

        for t in range(len(train_snapshots) - 1):
            snap = train_snapshots[t]
            next_snap = train_snapshots[t + 1]
            lag_growth = snap.target_growth
            macro_feats = snap.S[:, :5]  # Output, VA, Exports, Imports, Demand
            shock_feat = np.repeat(snap.common_shock[0], snap.S.shape[0])[:, np.newaxis]

            feats = np.column_stack([lag_growth, macro_feats, shock_feat])
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
        lag_growth = snapshot.target_growth
        macro_feats = snapshot.S[:, :5]
        shock_feat = np.repeat(snapshot.common_shock[0], snapshot.S.shape[0])[:, np.newaxis]

        feats = np.column_stack([lag_growth, macro_feats, shock_feat])
        pred_growth = self.reg_cont.predict(feats)
        pred_prob = self.reg_tail.predict_proba(feats)[:, 1]
        return pred_growth, pred_prob


class DynamicFactorModel(BaseModel):
    """Benchmark 2: Dynamic Factor Model estimating aggregate factors via PCA."""

    def __init__(self, num_factors: int = 3):
        super().__init__("Dynamic Factor Model")
        self.num_factors = num_factors
        self.pca = PCA(n_components=num_factors)
        self.reg_cont = LinearRegression()
        self.reg_tail = LogisticRegression(max_iter=1000)

    def fit(self, train_snapshots: List[EconomicGraphSnapshot]) -> None:
        growth_matrix = np.array([snap.target_growth for snap in train_snapshots])  # T x N
        factors = self.pca.fit_transform(growth_matrix)  # T x K

        X_list, y_growth_list, y_tail_list = [], [], []
        for t in range(len(train_snapshots) - 1):
            snap = train_snapshots[t]
            next_snap = train_snapshots[t + 1]

            f_t = np.tile(factors[t], (snap.S.shape[0], 1))
            feats = np.column_stack([snap.target_growth, f_t])

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
        # Estimate test factor from node growth
        f_test = self.pca.transform(snapshot.target_growth.reshape(1, -1))[0]
        f_t = np.tile(f_test, (snapshot.S.shape[0], 1))
        feats = np.column_stack([snapshot.target_growth, f_t])

        pred_growth = self.reg_cont.predict(feats)
        pred_prob = self.reg_tail.predict_proba(feats)[:, 1]
        return pred_growth, pred_prob


class PanelVARModel(BaseModel):
    """Benchmark 3: Panel Vector Autoregression."""

    def __init__(self):
        super().__init__("Panel VAR")
        self.reg_cont = LinearRegression()
        self.reg_tail = LogisticRegression(max_iter=1000)

    def fit(self, train_snapshots: List[EconomicGraphSnapshot]) -> None:
        X_list, y_growth_list, y_tail_list = [], [], []

        for t in range(len(train_snapshots) - 1):
            snap = train_snapshots[t]
            next_snap = train_snapshots[t + 1]

            N = len(snap.target_growth)
            num_ind = 50 if N >= 50 and N % 50 == 0 else max(1, N // 4)
            num_c = max(1, N // num_ind)
            c_means = snap.target_growth[: num_c * num_ind].reshape(num_c, num_ind).mean(axis=1)
            country_means = np.repeat(c_means, num_ind)[:N, np.newaxis]
            feats = np.column_stack([snap.target_growth, country_means])

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
        N = len(snapshot.target_growth)
        num_ind = 50 if N >= 50 and N % 50 == 0 else max(1, N // 4)
        num_c = max(1, N // num_ind)
        c_means = snapshot.target_growth[: num_c * num_ind].reshape(num_c, num_ind).mean(axis=1)
        country_means = np.repeat(c_means, num_ind)[:N, np.newaxis]
        feats = np.column_stack([snapshot.target_growth, country_means])

        pred_growth = self.reg_cont.predict(feats)
        pred_prob = self.reg_tail.predict_proba(feats)[:, 1]
        return pred_growth, pred_prob
