from pathlib import Path
from typing import List, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from economic_graph.config import Config
from economic_graph.data.loader import EconomicGraphSnapshot
from economic_graph.models.base import BaseModel


class GraphOnlyRGCN(nn.Module):
    """PyTorch PyG/GNN Module for Model A: Relational Graph Convolutional Network."""

    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        self.fc_self = nn.Linear(in_dim, hidden_dim)
        self.fc_neigh = nn.Linear(in_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc_out = nn.Linear(hidden_dim, 1)
        self.fc_tail = nn.Linear(hidden_dim, 1)

    def forward(
        self, x: torch.Tensor, W_share: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        h_self = self.fc_self(x)
        h_neigh = self.fc_neigh(torch.matmul(W_share, x))
        h = self.relu(h_self + h_neigh)

        pred_g = self.fc_out(h).squeeze(-1)
        pred_tail = torch.sigmoid(self.fc_tail(h)).squeeze(-1)
        return pred_g, pred_tail


class ModelAGraphOnly(BaseModel):
    """Model A: Graph-Only Relational GNN Model."""

    def __init__(self, config: Config):
        super().__init__("Graph-Only Model (Model A)")
        self.config = config
        self.net: GraphOnlyRGCN = None

    def fit(self, train_snapshots: List[EconomicGraphSnapshot]) -> None:
        in_dim = train_snapshots[0].S.shape[1] + 1
        self.net = GraphOnlyRGCN(in_dim, self.config.model.hidden_dim)
        optimizer = optim.Adam(self.net.parameters(), lr=self.config.model.learning_rate)
        criterion_cont = nn.MSELoss()
        criterion_tail = nn.BCELoss()

        self.net.train()
        for epoch in range(self.config.model.epochs):
            for t in range(len(train_snapshots) - 1):
                snap = train_snapshots[t]
                next_snap = train_snapshots[t + 1]

                x_np = np.column_stack([snap.target_growth, snap.S])
                x_t = torch.tensor(x_np, dtype=torch.float32)
                W_t = torch.tensor(snap.W_share, dtype=torch.float32)
                y_g = torch.tensor(next_snap.target_growth, dtype=torch.float32)
                y_d = torch.tensor(next_snap.target_tail, dtype=torch.float32)

                optimizer.zero_grad()
                pred_g, pred_d = self.net(x_t, W_t)
                pred_d_clamped = torch.clamp(pred_d, 1e-6, 1.0 - 1e-6)

                loss = criterion_cont(pred_g, y_g) + criterion_tail(pred_d_clamped, y_d)
                loss.backward()
                optimizer.step()

        self.is_fitted = True

    def predict(self, snapshot: EconomicGraphSnapshot) -> Tuple[np.ndarray, np.ndarray]:
        self.net.eval()
        with torch.no_grad():
            x_np = np.column_stack([snapshot.target_growth, snapshot.S])
            x_t = torch.tensor(x_np, dtype=torch.float32)
            W_t = torch.tensor(snapshot.W_share, dtype=torch.float32)

            pred_g, pred_d = self.net(x_t, W_t)
            return pred_g.cpu().numpy(), pred_d.cpu().numpy()

    def save_checkpoint(self, checkpoint_dir: Path) -> Path:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        path = checkpoint_dir / f"{self.get_safe_filename()}.pt"
        if self.net is not None:
            torch.save(self.net.state_dict(), path)
        return path

    def load_checkpoint(self, checkpoint_dir: Path, in_dim: int = 13) -> bool:
        path = checkpoint_dir / f"{self.get_safe_filename()}.pt"
        if not path.exists():
            return False
        try:
            state_dict = torch.load(path)
            # Infer in_dim from state_dict fc_self.weight shape [hidden_dim, in_dim]
            saved_in_dim = state_dict["fc_self.weight"].shape[1]
            self.net = GraphOnlyRGCN(saved_in_dim, self.config.model.hidden_dim)
            self.net.load_state_dict(state_dict)
            self.is_fitted = True
            return True
        except Exception:
            return False



class DynamicGraphAgentNet(nn.Module):
    """PyTorch Architecture for Model B: Dynamic Graph/Agent Model with VAE belief states & GRU cell."""

    def __init__(self, in_dim: int, hidden_dim: int, shock_dim: int = 3):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Relational Message Passing Attention
        self.W_q = nn.Linear(in_dim, hidden_dim)
        self.W_k = nn.Linear(in_dim, hidden_dim)
        self.W_v = nn.Linear(in_dim, hidden_dim)

        # Belief State VAE encoder
        self.fc_belief_mu = nn.Linear(in_dim, 8)
        self.fc_belief_logvar = nn.Linear(in_dim, 8)

        # GRU Temporal Cell
        gru_in_dim = hidden_dim + 8 + shock_dim
        self.gru = nn.GRUCell(gru_in_dim, hidden_dim)

        # Agent Readout Heads
        self.head_g = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.head_d = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        x: torch.Tensor,
        W_share: torch.Tensor,
        z_t: torch.Tensor,
        h_prev: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        N = x.shape[0]

        # VAE Latent Belief sampling
        b_mu = self.fc_belief_mu(x)
        b_logvar = self.fc_belief_logvar(x)
        eps = torch.randn_like(b_mu)
        b_t = b_mu + torch.exp(0.5 * b_logvar) * eps

        # Relational Message Aggregation
        m_t = torch.matmul(W_share, self.W_v(x))

        # Replicate common shock z_t across nodes
        z_node = z_t.unsqueeze(0).repeat(N, 1)

        # GRU State Update
        gru_input = torch.cat([m_t, b_t, z_node], dim=-1)
        h_next = self.gru(gru_input, h_prev)

        pred_g = self.head_g(h_next).squeeze(-1)
        pred_d = self.head_d(h_next).squeeze(-1)

        return pred_g, pred_d, h_next


class ModelBDynamicGraphAgent(BaseModel):
    """Model B: Dynamic Graph/Agent Model."""

    def __init__(self, config: Config):
        super().__init__("Graph/Agent Model (Model B)")
        self.config = config
        self.net: DynamicGraphAgentNet = None

    def fit(self, train_snapshots: List[EconomicGraphSnapshot]) -> None:
        in_dim = train_snapshots[0].S.shape[1] + 1
        self.net = DynamicGraphAgentNet(in_dim, self.config.model.hidden_dim)
        optimizer = optim.Adam(self.net.parameters(), lr=self.config.model.learning_rate)
        criterion_cont = nn.MSELoss()
        criterion_tail = nn.BCELoss()

        self.net.train()
        for epoch in range(self.config.model.epochs):
            N = train_snapshots[0].S.shape[0]
            h_state = torch.zeros(N, self.config.model.hidden_dim, dtype=torch.float32)

            for t in range(len(train_snapshots) - 1):
                snap = train_snapshots[t]
                next_snap = train_snapshots[t + 1]

                x_np = np.column_stack([snap.target_growth, snap.S])
                x_norm = np.clip((x_np - np.mean(x_np, axis=0)) / (np.std(x_np, axis=0) + 1e-5), -5.0, 5.0)
                x_t = torch.tensor(x_norm, dtype=torch.float32)
                W_t = torch.tensor(snap.W_share, dtype=torch.float32)
                z_t = torch.tensor(snap.common_shock, dtype=torch.float32)
                y_g = torch.tensor(next_snap.target_growth, dtype=torch.float32)
                y_d = torch.tensor(next_snap.target_tail, dtype=torch.float32)

                optimizer.zero_grad()
                pred_g, pred_d, next_h = self.net(x_t, W_t, z_t, h_state)
                pred_d_clamped = torch.clamp(pred_d, 1e-5, 1.0 - 1e-5)

                loss = criterion_cont(pred_g, y_g) + criterion_tail(pred_d_clamped, y_d)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                optimizer.step()
                h_state = torch.nan_to_num(next_h.detach(), nan=0.0)
        self.is_fitted = True

    def predict(self, snapshot: EconomicGraphSnapshot) -> Tuple[np.ndarray, np.ndarray]:
        self.net.eval()
        with torch.no_grad():
            N = snapshot.S.shape[0]
            h_state = torch.zeros(N, self.config.model.hidden_dim, dtype=torch.float32)

            x_np = np.column_stack([snapshot.target_growth, snapshot.S])
            x_norm = np.clip((x_np - np.mean(x_np, axis=0)) / (np.std(x_np, axis=0) + 1e-5), -5.0, 5.0)
            x_t = torch.tensor(x_norm, dtype=torch.float32)
            W_t = torch.tensor(snapshot.W_share, dtype=torch.float32)
            z_t = torch.tensor(snapshot.common_shock, dtype=torch.float32)

            pred_g, pred_d, _ = self.net(x_t, W_t, z_t, h_state)
            return pred_g.cpu().numpy(), pred_d.cpu().numpy()

    def save_checkpoint(self, checkpoint_dir: Path) -> Path:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        path = checkpoint_dir / f"{self.get_safe_filename()}.pt"
        if self.net is not None:
            torch.save(self.net.state_dict(), path)
        return path

    def load_checkpoint(self, checkpoint_dir: Path, in_dim: int = 13) -> bool:
        path = checkpoint_dir / f"{self.get_safe_filename()}.pt"
        if not path.exists():
            return False
        try:
            state_dict = torch.load(path)
            saved_in_dim = state_dict["W_q.weight"].shape[1]
            self.net = DynamicGraphAgentNet(saved_in_dim, self.config.model.hidden_dim)
            self.net.load_state_dict(state_dict)
            self.is_fitted = True
            return True
        except Exception:
            return False

