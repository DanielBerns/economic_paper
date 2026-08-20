import networkx as nx
import numpy as np


def compute_topological_centralities(
    Z: np.ndarray, threshold: float = 1.0
) -> dict[str, np.ndarray]:
    """Compute in/out degrees, in/out strengths, betweenness, eigenvector centralities, and supplier Herfindahl index."""
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
