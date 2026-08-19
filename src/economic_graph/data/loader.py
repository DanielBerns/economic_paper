from dataclasses import dataclass
from typing import Dict, List
import numpy as np
from economic_graph.config import Config
from economic_graph.data.features import (
    compute_topological_centralities,
    construct_node_state_matrix,
)
from economic_graph.data.matrices import compute_input_share_matrix


@dataclass
class EconomicGraphSnapshot:
    year: int
    Z: np.ndarray  # N x N intermediate flow matrix
    W_share: np.ndarray  # N x N input share matrix
    Y: np.ndarray  # Real gross output N
    VA: np.ndarray  # Value added N
    F: np.ndarray  # Final demand N
    S: np.ndarray  # Node state matrix N x d
    target_growth: np.ndarray  # 1-year ahead growth target g_{i, t+1}
    target_tail: np.ndarray  # 1-year ahead tail contraction indicator D_{i, t+1}
    common_shock: np.ndarray  # Global macro shock vector z_t


class ICIODatasetLoader:
    """Dataset loader generating harmonized OECD ICIO multi-year production graphs."""

    def __init__(self, config: Config):
        self.config = config
        self.num_countries = config.data.num_countries
        self.num_industries = config.data.num_industries
        self.N = self.num_countries * self.num_industries
        self.start_year = config.data.start_year
        self.end_year = config.data.end_year
        self.rng = np.random.RandomState(config.model.seed)

    def generate_synthetic_icio(self) -> List[EconomicGraphSnapshot]:
        """Generate a realistic synthetic ICIO dataset reproducing structural gravity & crisis shocks."""
        years = list(range(self.start_year, self.end_year + 1))

        # Base structural interaction matrix with gravity effect
        base_Z = self.rng.exponential(scale=10.0, size=(self.N, self.N))
        np.fill_diagonal(base_Z, base_Z.diagonal() * 5.0)

        snapshots = []
        raw_outputs = []

        # Generate base trajectories
        for year in years:
            # Macro shock signal
            if year == 2020:
                shock_val = -0.08  # Pandemic crisis shock
            elif year in (2008, 2009):
                shock_val = -0.05  # GFC shock
            else:
                shock_val = 0.025 + self.rng.normal(0.0, 0.01)

            z_t = np.array([shock_val, self.rng.normal(0, 1), self.rng.normal(0, 1)])

            # Perturb Z matrix over time
            drift = 1.0 + 0.02 * (year - self.start_year) + z_t[0]
            Z_t = np.maximum(0.0, base_Z * drift + self.rng.normal(0, 0.5, size=(self.N, self.N)))

            F_t = Z_t.sum(axis=1) * 0.4 + self.rng.gamma(2.0, 5.0, size=self.N)
            Y_t = Z_t.sum(axis=1) + F_t
            VA_t = Y_t - Z_t.sum(axis=0)
            X_exp = Z_t.sum(axis=1) * 0.3
            M_imp = Z_t.sum(axis=0) * 0.3

            W_share_t = compute_input_share_matrix(Z_t)
            centralities_t = compute_topological_centralities(Z_t, self.config.data.edge_threshold)
            S_t = construct_node_state_matrix(Y_t, VA_t, X_exp, M_imp, F_t, centralities_t)

            raw_outputs.append(Y_t)
            snapshots.append(
                {
                    "year": year,
                    "Z": Z_t,
                    "W_share": W_share_t,
                    "Y": Y_t,
                    "VA": VA_t,
                    "F": F_t,
                    "S": S_t,
                    "z": z_t,
                }
            )

        # Compute continuous target growth g_{t+1} = ln(Y_{t+1}) - ln(Y_t)
        train_growths = []
        processed_snapshots: List[EconomicGraphSnapshot] = []

        for idx in range(len(snapshots) - 1):
            curr_snap = snapshots[idx]
            next_snap = snapshots[idx + 1]

            growth = np.log(np.maximum(1e-6, next_snap["Y"])) - np.log(
                np.maximum(1e-6, curr_snap["Y"])
            )
            curr_snap["target_growth"] = growth

            if curr_snap["year"] <= self.config.data.train_end_year:
                train_growths.extend(growth.tolist())

        # Determine 10th percentile tail threshold strictly on training sample
        train_q10 = (
            np.percentile(train_growths, self.config.model.tail_quantile * 100)
            if train_growths
            else -0.05
        )

        for idx in range(len(snapshots) - 1):
            curr_snap = snapshots[idx]
            growth = curr_snap["target_growth"]
            tail_indicator = (growth < train_q10).astype(float)

            processed_snapshots.append(
                EconomicGraphSnapshot(
                    year=curr_snap["year"],
                    Z=curr_snap["Z"],
                    W_share=curr_snap["W_share"],
                    Y=curr_snap["Y"],
                    VA=curr_snap["VA"],
                    F=curr_snap["F"],
                    S=curr_snap["S"],
                    target_growth=growth,
                    target_tail=tail_indicator,
                    common_shock=curr_snap["z"],
                )
            )

        return processed_snapshots
