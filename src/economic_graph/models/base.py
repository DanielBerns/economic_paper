from abc import ABC, abstractmethod
from typing import Dict, List, Tuple
import numpy as np
from economic_graph.data.loader import EconomicGraphSnapshot


class BaseModel(ABC):
    """Abstract Base Model for economic production network forecasting."""

    def __init__(self, name: str):
        self.name = name
        self.is_fitted = False

    @abstractmethod
    def fit(self, train_snapshots: List[EconomicGraphSnapshot]) -> None:
        """Fit model parameters strictly on historical training snapshots."""
        pass

    @abstractmethod
    def predict(
        self, snapshot: EconomicGraphSnapshot
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Predict continuous growth g_{i, t+1} and severe contraction probability P(D_{i, t+1} = 1)."""
        pass
