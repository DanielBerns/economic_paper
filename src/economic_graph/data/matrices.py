import numpy as np


def compute_technical_coefficients(Z: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Compute direct technical coefficient matrix A = Z * diag(X)^(-1)."""
    X_safe = np.where(X > 1e-6, X, 1.0)
    A = Z / X_safe[np.newaxis, :]
    return A


def compute_leontief_inverse(A: np.ndarray) -> np.ndarray:
    """Compute Leontief Inverse L = (I - A)^(-1)."""
    N = A.shape[0]
    I = np.eye(N)
    # Ensure spectral radius < 1 for numerical stability
    A_scaled = np.clip(A, 0.0, 0.95)
    try:
        L = np.linalg.inv(I - A_scaled)
    except np.linalg.LinAlgError:
        L = np.linalg.pinv(I - A_scaled)
    return L


def compute_input_share_matrix(Z: np.ndarray) -> np.ndarray:
    """Normalize input matrix Z by purchasing node's total intermediate expenditure: W_share[i,j] = Z[i,j] / sum_k Z[k,j]."""
    col_sums = Z.sum(axis=0)
    col_sums_safe = np.where(col_sums > 1e-6, col_sums, 1.0)
    W_share = Z / col_sums_safe[np.newaxis, :]
    return W_share
