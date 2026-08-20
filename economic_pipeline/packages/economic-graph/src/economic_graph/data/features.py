from abc import ABC, abstractmethod
from typing import Union

import networkx as nx
import numpy as np


class CentralityStrategy(ABC):
    """Abstract Strategy interface for computing topological network centralities."""

    @abstractmethod
    def compute(
        self,
        Z: np.ndarray,
        W_share: np.ndarray,
        in_strength: np.ndarray,
        out_strength: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute and return (betweenness, eigenvector) centralities."""
        pass


class FastCentralityStrategy(CentralityStrategy):
    """Optimized Strategy: Vectorized power iteration eigenvector centrality + sampled sparse betweenness."""

    def compute(
        self,
        Z: np.ndarray,
        W_share: np.ndarray,
        in_strength: np.ndarray,
        out_strength: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        N = Z.shape[0]

        # Vectorized power iteration for eigenvector centrality (fast BLAS/NumPy)
        try:
            eigenvector = np.ones(N) / N
            for _ in range(50):
                eigen_next = W_share @ eigenvector
                norm = np.linalg.norm(eigen_next)
                if norm < 1e-12:
                    break
                eigen_next /= norm
                if np.allclose(eigenvector, eigen_next, atol=1e-6):
                    eigenvector = eigen_next
                    break
                eigenvector = eigen_next
        except Exception:
            eigenvector = in_strength / (in_strength.sum() + 1e-6)

        # Fast sampled betweenness centrality on thresholded graph
        try:
            if N > 500:
                thresh = max(1e-4, float(np.percentile(W_share, 98)))
                W_sparse = np.where(W_share >= thresh, W_share, 0.0)
                G_sparse = nx.from_numpy_array(W_sparse, create_using=nx.DiGraph)
                betweenness_dict = nx.betweenness_centrality(
                    G_sparse, k=min(N, 50), weight="weight", seed=42
                )
            else:
                G = nx.from_numpy_array(W_share, create_using=nx.DiGraph)
                betweenness_dict = nx.betweenness_centrality(
                    G, k=min(N, 50), weight="weight", seed=42
                )
            betweenness = np.array([betweenness_dict.get(i, 0.0) for i in range(N)])
        except Exception:
            betweenness = (in_strength * out_strength) / (N * (in_strength.sum() + 1e-6))

        return betweenness, eigenvector


class ExactNetworkXCentralityStrategy(CentralityStrategy):
    """Original Strategy: Recovered full NetworkX implementation for exact betweenness & eigenvector centralities."""

    def compute(
        self,
        Z: np.ndarray,
        W_share: np.ndarray,
        in_strength: np.ndarray,
        out_strength: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        N = Z.shape[0]
        G = nx.from_numpy_array(W_share, create_using=nx.DiGraph)

        try:
            betweenness_dict = nx.betweenness_centrality(G, weight="weight")
            betweenness = np.array([betweenness_dict.get(i, 0.0) for i in range(N)])
        except Exception:
            betweenness = np.zeros(N)

        try:
            eigen_dict = nx.eigenvector_centrality(G, max_iter=200, weight="weight")
            eigenvector = np.array([eigen_dict.get(i, 0.0) for i in range(N)])
        except Exception:
            eigenvector = in_strength / (in_strength.sum() + 1e-6)

        return betweenness, eigenvector


def get_centrality_strategy(
    strategy: Union[str, CentralityStrategy] = "fast"
) -> CentralityStrategy:
    """Strategy Factory resolving strategy instances by string name or direct Strategy object."""
    if isinstance(strategy, CentralityStrategy):
        return strategy
    if isinstance(strategy, str):
        strat_key = strategy.lower().strip()
        if strat_key in ("fast", "vectorized", "sampled"):
            return FastCentralityStrategy()
        elif strat_key in ("exact_networkx", "networkx", "exact", "original"):
            return ExactNetworkXCentralityStrategy()
    return FastCentralityStrategy()


def compute_topological_centralities(
    Z: np.ndarray,
    threshold: float = 1.0,
    strategy: Union[str, CentralityStrategy] = "fast",
) -> dict[str, np.ndarray]:
    """Compute in/out degrees, strengths, betweenness, eigenvector centralities, and Herfindahl index using Strategy pattern."""
    N = Z.shape[0]

    in_strength = Z.sum(axis=0)
    out_strength = Z.sum(axis=1)

    # Thresholded adjacency for degrees
    adj_binary = (Z >= threshold).astype(float)
    in_degree = adj_binary.sum(axis=0)
    out_degree = adj_binary.sum(axis=1)

    # Supplier Herfindahl concentration index
    col_sums = np.where(in_strength > 1e-6, in_strength, 1.0)
    W_share = Z / col_sums[np.newaxis, :]
    herfindahl = (W_share**2).sum(axis=0)

    # Compute betweenness and eigenvector via Strategy Pattern
    strat_obj = get_centrality_strategy(strategy)
    betweenness, eigenvector = strat_obj.compute(Z, W_share, in_strength, out_strength)

    return {
        "in_degree": in_degree,
        "out_degree": out_degree,
        "in_strength": in_strength,
        "out_strength": out_strength,
        "betweenness": betweenness,
        "eigenvector": eigenvector,
        "herfindahl": herfindahl,
    }


def construct_node_state_matrix(
    Y: np.ndarray,
    VA: np.ndarray,
    X_exp: np.ndarray,
    M_imp: np.ndarray,
    F_demand: np.ndarray,
    centralities: dict[str, np.ndarray],
) -> np.ndarray:
    """Construct N x d state matrix S_t containing economic and topological indicators."""
    S = np.column_stack(
        [
            Y,
            VA,
            X_exp,
            M_imp,
            F_demand,
            centralities["in_degree"],
            centralities["out_degree"],
            centralities["in_strength"],
            centralities["out_strength"],
            centralities["betweenness"],
            centralities["eigenvector"],
            centralities["herfindahl"],
        ]
    )
    return S
