import numpy as np

from economic_graph.config import AppConfig, Config
from economic_graph.data.features import compute_topological_centralities
from economic_graph.data.loader import ICIODatasetLoader
from economic_graph.data.matrices import (
    compute_input_share_matrix,
    compute_leontief_inverse,
    compute_technical_coefficients,
)


def test_input_output_matrices():
    N = 10
    Z = np.random.uniform(1.0, 10.0, size=(N, N))
    Y = Z.sum(axis=1) + 5.0

    A = compute_technical_coefficients(Z, Y)
    assert A.shape == (N, N)
    assert np.all(A >= 0.0)

    L = compute_leontief_inverse(A)
    assert L.shape == (N, N)

    W_share = compute_input_share_matrix(Z)
    assert W_share.shape == (N, N)
    assert np.allclose(W_share.sum(axis=0), 1.0)


def test_topological_centralities():
    N = 10
    Z = np.random.uniform(0.5, 5.0, size=(N, N))
    cents = compute_topological_centralities(Z, threshold=1.0)

    assert "in_degree" in cents
    assert "out_degree" in cents
    assert "betweenness" in cents
    assert "eigenvector" in cents
    assert "herfindahl" in cents
    assert len(cents["in_degree"]) == N


def test_icio_loader_small():
    app_config = AppConfig()
    app_config.data.num_countries = 4
    app_config.data.num_industries = 5
    app_config.data.start_year = 2015
    app_config.data.end_year = 2020
    cfg = Config(app_config)

    loader = ICIODatasetLoader(cfg)
    snapshots = loader.generate_synthetic_icio()

    assert len(snapshots) == 5
    assert snapshots[0].Z.shape == (20, 20)
    assert snapshots[0].S.shape[0] == 20
    assert len(snapshots[0].target_growth) == 20
