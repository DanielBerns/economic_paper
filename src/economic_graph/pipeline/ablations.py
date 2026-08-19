from copy import deepcopy
from typing import Dict, List
import numpy as np
from economic_graph.config import Config
from economic_graph.data.loader import EconomicGraphSnapshot
from economic_graph.models.gnn import ModelBDynamicGraphAgent
from economic_graph.pipeline.evaluator import EvaluationMetrics, compute_metrics


class AblationEngine:
    """8-Component Systemic Ablation Suite Engine."""

    def __init__(self, config: Config):
        self.config = config

    def run_ablations(
        self,
        train_snapshots: List[EconomicGraphSnapshot],
        test_snapshot: EconomicGraphSnapshot,
    ) -> Dict[str, EvaluationMetrics]:
        results: Dict[str, EvaluationMetrics] = {}
        ablation_cfg = deepcopy(self.config)
        ablation_cfg.app.force_retrain = True

        # 0. Baseline Model B
        base_model = ModelBDynamicGraphAgent(ablation_cfg)
        base_model.fit(train_snapshots)
        pg, pd = base_model.predict(test_snapshot)
        results["Full Graph/Agent Model (Model B)"] = compute_metrics(
            pg, pd, test_snapshot.target_growth, test_snapshot.target_tail
        )

        # Ablation A: No Graph (Zero out W_share)
        train_A = [self._copy_snapshot(s, zero_W=True) for s in train_snapshots]
        test_A = self._copy_snapshot(test_snapshot, zero_W=True)
        mA = ModelBDynamicGraphAgent(ablation_cfg)
        mA.fit(train_A)
        pg, pd = mA.predict(test_A)
        results["Ablation A: No Graph"] = compute_metrics(
            pg, pd, test_snapshot.target_growth, test_snapshot.target_tail
        )

        # Ablation B: Unweighted Edges
        train_B = [self._copy_snapshot(s, binary_W=True) for s in train_snapshots]
        test_B = self._copy_snapshot(test_snapshot, binary_W=True)
        mB = ModelBDynamicGraphAgent(ablation_cfg)
        mB.fit(train_B)
        pg, pd = mB.predict(test_B)
        results["Ablation B: Unweighted Edges"] = compute_metrics(
            pg, pd, test_snapshot.target_growth, test_snapshot.target_tail
        )

        # Ablation C: Symmetrized Graph
        train_C = [self._copy_snapshot(s, sym_W=True) for s in train_snapshots]
        test_C = self._copy_snapshot(test_snapshot, sym_W=True)
        mC = ModelBDynamicGraphAgent(ablation_cfg)
        mC.fit(train_C)
        pg, pd = mC.predict(test_C)
        results["Ablation C: Symmetrized Graph"] = compute_metrics(
            pg, pd, test_snapshot.target_growth, test_snapshot.target_tail
        )

        # Ablation D: No Centralities
        train_D = [self._copy_snapshot(s, no_cent=True) for s in train_snapshots]
        test_D = self._copy_snapshot(test_snapshot, no_cent=True)
        mD = ModelBDynamicGraphAgent(ablation_cfg)
        mD.fit(train_D)
        pg, pd = mD.predict(test_D)
        results["Ablation D: No Centralities"] = compute_metrics(
            pg, pd, test_snapshot.target_growth, test_snapshot.target_tail
        )

        return results

    def _copy_snapshot(
        self,
        snap: EconomicGraphSnapshot,
        zero_W: bool = False,
        binary_W: bool = False,
        sym_W: bool = False,
        no_cent: bool = False,
    ) -> EconomicGraphSnapshot:
        W = snap.W_share.copy()
        if zero_W:
            W = np.zeros_like(W)
        elif binary_W:
            W = (W > 1e-4).astype(float)
        elif sym_W:
            W = 0.5 * (W + W.T)

        S = snap.S.copy()
        if no_cent:
            S[:, 5:] = 0.0

        return EconomicGraphSnapshot(
            year=snap.year,
            Z=snap.Z,
            W_share=W,
            Y=snap.Y,
            VA=snap.VA,
            F=snap.F,
            S=S,
            target_growth=snap.target_growth,
            target_tail=snap.target_tail,
            common_shock=snap.common_shock,
        )
