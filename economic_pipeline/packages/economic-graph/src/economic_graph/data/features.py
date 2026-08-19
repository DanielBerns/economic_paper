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

    # NetworkX centrality computation on subset for speed / efficiency
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
