from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Tuple

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

    def get_safe_filename(self) -> str:
        """Return safe string for checkpoint filename."""
        return (
            self.name.lower()
            .replace(" ", "_")
            .replace("(", "")
            .replace(")", "")
            .replace("/", "_")
        )

    def save_checkpoint(self, checkpoint_dir: Path) -> Path:
        """Save model checkpoint to directory."""
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        path = checkpoint_dir / f"{self.get_safe_filename()}.pt"
        # Default fallback if subclass does not override
        return path

    def load_checkpoint(self, checkpoint_dir: Path) -> bool:
        """Load model checkpoint if it exists. Return True if successful."""
        path = checkpoint_dir / f"{self.get_safe_filename()}.pt"
        return path.exists()

