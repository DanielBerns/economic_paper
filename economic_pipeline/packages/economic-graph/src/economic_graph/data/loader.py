from dataclasses import dataclass
import time
from typing import List

import numpy as np

from economic_graph.config import Config
from economic_graph.data.features import (
    compute_topological_centralities,
    construct_node_state_matrix,
)
from economic_graph.data.matrices import compute_input_share_matrix
from economic_graph.logging import setup_logger


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
        self.logger = setup_logger("economic_graph.data.loader", getattr(config, "logging", None))

        self.logger.info(
            f"Initialized ICIODatasetLoader: num_countries={self.num_countries}, "
            f"num_industries={self.num_industries}, total_nodes (N)={self.N}"
        )
        self.logger.info(
            f"Dataset year span: {self.start_year} to {self.end_year} "
            f"(train_end_year={config.data.train_end_year}, val_end_year={config.data.val_end_year})"
        )
        self.logger.info(
            f"Centrality strategy: '{config.data.centrality_strategy}', "
            f"edge_threshold={config.data.edge_threshold}, quicktest={config.app.quicktest}"
        )

    def generate_synthetic_icio(self) -> List[EconomicGraphSnapshot]:
        """Generate a realistic synthetic ICIO dataset reproducing structural gravity & crisis shocks."""
        start_time = time.time()
        years = list(range(self.start_year, self.end_year + 1))
        self.logger.info(
            f"Starting synthetic ICIO dataset generation for N={self.N} nodes over {len(years)} years ({self.start_year}-{self.end_year})..."
        )

        try:
            # Base structural interaction matrix with gravity effect
            matrix_mb = (self.N * self.N * 8) / (1024 * 1024)
            self.logger.info(
                f"Generating base structural interaction matrix (shape: {self.N}x{self.N}, ~{matrix_mb:.2f} MB)..."
            )
            base_Z = self.rng.exponential(scale=10.0, size=(self.N, self.N))
            np.fill_diagonal(base_Z, base_Z.diagonal() * 5.0)
            self.logger.debug(
                f"Base Z matrix generated: min={base_Z.min():.4f}, max={base_Z.max():.4f}, "
                f"mean={base_Z.mean():.4f}, sum={base_Z.sum():.2f}"
            )

            snapshots = []
            raw_outputs = []

            # Generate base trajectories
            for idx, year in enumerate(years):
                year_start_time = time.time()
                self.logger.info(f"[{idx+1}/{len(years)}] Processing trajectory snapshot for year {year}...")

                # Macro shock signal
                if year == 2020:
                    shock_val = -0.08  # Pandemic crisis shock
                elif year in (2008, 2009):
                    shock_val = -0.05  # GFC shock
                else:
                    shock_val = 0.025 + self.rng.normal(0.0, 0.01)

                z_t = np.array([shock_val, self.rng.normal(0, 1), self.rng.normal(0, 1)])
                self.logger.debug(f"Year {year}: macro shock vector z_t = [{z_t[0]:.4f}, {z_t[1]:.4f}, {z_t[2]:.4f}]")

                # Perturb Z matrix over time
                drift = 1.0 + 0.02 * (year - self.start_year) + z_t[0]
                Z_t = np.maximum(0.0, base_Z * drift + self.rng.normal(0, 0.5, size=(self.N, self.N)))
                self.logger.debug(
                    f"Year {year}: Z_t matrix perturbed (drift={drift:.4f}): min={Z_t.min():.4f}, "
                    f"max={Z_t.max():.4f}, sum={Z_t.sum():.2f}"
                )

                F_t = Z_t.sum(axis=1) * 0.4 + self.rng.gamma(2.0, 5.0, size=self.N)
                Y_t = Z_t.sum(axis=1) + F_t
                VA_t = Y_t - Z_t.sum(axis=0)
                X_exp = Z_t.sum(axis=1) * 0.3
                M_imp = Z_t.sum(axis=0) * 0.3

                # Sanity checks on macroeconomic values
                if np.any(np.isnan(Y_t)) or np.any(np.isinf(Y_t)):
                    self.logger.error(f"Year {year}: Invalid values detected in output vector Y_t (NaN or Inf)")
                if np.any(VA_t < 0):
                    self.logger.warning(f"Year {year}: Negative value-added detected in {np.sum(VA_t < 0)} nodes")

                self.logger.debug(
                    f"Year {year}: aggregates calculated - Total Output Y={Y_t.sum():.2f}, "
                    f"Value Added VA={VA_t.sum():.2f}, Final Demand F={F_t.sum():.2f}"
                )

                self.logger.debug(f"Year {year}: computing input share matrix W_share_t...")
                W_share_t = compute_input_share_matrix(Z_t)

                self.logger.info(
                    f"Year {year}: computing topological centralities (strategy='{self.config.data.centrality_strategy}', "
                    f"edge_threshold={self.config.data.edge_threshold})..."
                )
                cent_start = time.time()
                try:
                    centralities_t = compute_topological_centralities(
                        Z_t,
                        self.config.data.edge_threshold,
                        strategy=self.config.data.centrality_strategy,
                    )
                except Exception as ce:
                    self.logger.exception(f"Year {year}: Failed to compute topological centralities: {ce}")
                    raise

                cent_dur = time.time() - cent_start
                self.logger.info(
                    f"Year {year}: centralities computed in {cent_dur:.2f}s "
                    f"(betweenness min/max: {centralities_t['betweenness'].min():.4f}/{centralities_t['betweenness'].max():.4f}, "
                    f"eigenvector min/max: {centralities_t['eigenvector'].min():.4f}/{centralities_t['eigenvector'].max():.4f})"
                )

                S_t = construct_node_state_matrix(Y_t, VA_t, X_exp, M_imp, F_t, centralities_t)
                self.logger.debug(f"Year {year}: node state matrix S_t constructed (shape: {S_t.shape})")

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

                self.logger.info(f"Year {year} snapshot processing completed in {time.time() - year_start_time:.2f}s.")

            # Compute continuous target growth g_{t+1} = ln(Y_{t+1}) - ln(Y_t)
            self.logger.info("Computing target growth trajectories g_{i, t+1} across consecutive snapshot years...")
            train_growths = []
            processed_snapshots: List[EconomicGraphSnapshot] = []

            for idx in range(len(snapshots) - 1):
                curr_snap = snapshots[idx]
                next_snap = snapshots[idx + 1]

                growth = np.log(np.maximum(1e-6, next_snap["Y"])) - np.log(
                    np.maximum(1e-6, curr_snap["Y"])
                )
                curr_snap["target_growth"] = growth

                self.logger.debug(
                    f"Growth transition {curr_snap['year']} -> {next_snap['year']}: "
                    f"min={growth.min():.4f}, max={growth.max():.4f}, mean={growth.mean():.4f}, std={growth.std():.4f}"
                )

                if curr_snap["year"] <= self.config.data.train_end_year:
                    train_growths.extend(growth.tolist())

            # Determine 10th percentile tail threshold strictly on training sample
            self.logger.info(
                f"Collected {len(train_growths)} target growth values from training period "
                f"(years <= {self.config.data.train_end_year})."
            )
            if train_growths:
                quantile_pct = self.config.model.tail_quantile * 100
                train_q10 = float(np.percentile(train_growths, quantile_pct))
                self.logger.info(
                    f"Calculated {quantile_pct:.1f}th percentile tail threshold on training set: train_q10={train_q10:.6f}"
                )
            else:
                train_q10 = -0.05
                self.logger.warning("No training growth data available; fallback train_q10 default used (-0.05).")

            self.logger.info("Computing binary tail contraction indicators and creating EconomicGraphSnapshot objects...")
            for idx in range(len(snapshots) - 1):
                curr_snap = snapshots[idx]
                growth = curr_snap["target_growth"]
                tail_indicator = (growth < train_q10).astype(float)
                tail_count = int(np.sum(tail_indicator))
                self.logger.debug(
                    f"Year {curr_snap['year']}: tail contraction active for {tail_count}/{self.N} nodes "
                    f"({(tail_count / self.N) * 100:.2f}%)"
                )

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

            total_dur = time.time() - start_time
            self.logger.info(
                f"Successfully generated {len(processed_snapshots)} EconomicGraphSnapshots in {total_dur:.2f}s."
            )
            return processed_snapshots

        except Exception as e:
            self.logger.exception(f"Fatal error during generate_synthetic_icio execution: {e}")
            raise

