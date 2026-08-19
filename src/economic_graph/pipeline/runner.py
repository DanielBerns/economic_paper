from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np
from economic_graph.config import Config
from economic_graph.data.loader import EconomicGraphSnapshot, ICIODatasetLoader
from economic_graph.logging import setup_logger
from economic_graph.models.econometric import (
    DynamicFactorModel,
    PanelARModel,
    PanelVARModel,
)
from economic_graph.models.gnn import ModelAGraphOnly, ModelBDynamicGraphAgent
from economic_graph.models.input_output import (
    CentralityRegressionModel,
    LeontiefIOModel,
    SpatialARModel,
)
from economic_graph.pipeline.ablations import AblationEngine
from economic_graph.pipeline.decision_utility import compute_economic_decision_utility
from economic_graph.pipeline.evaluator import EvaluationMetrics, compute_metrics, diebold_mariano_test


@dataclass
class PipelineExecutionReport:
    metrics_table: Dict[str, EvaluationMetrics]
    dm_pvalues: Dict[str, float]
    decision_utilities: Dict[str, Dict[float, float]]
    ablation_table: Dict[str, EvaluationMetrics]
    divergence_delta: float
    refalsification_triggered: bool


class PipelineRunner:
    """Master Pipeline Orchestrator executing data loading, model training, test evaluation, and ablations."""

    def __init__(self, config: Config):
        self.config = config
        self.logger = setup_logger("economic_graph.pipeline", config.logging)
        self.loader = ICIODatasetLoader(config)

    def run_pipeline(self) -> PipelineExecutionReport:
        self.logger.info("Initializing OECD ICIO empirical pipeline execution...")
        snapshots = self.loader.generate_synthetic_icio()

        train_snaps = [s for s in snapshots if s.year <= self.config.data.train_end_year]
        val_snaps = [
            s
            for s in snapshots
            if self.config.data.train_end_year < s.year <= self.config.data.val_end_year
        ]
        test_snap = next(s for s in snapshots if s.year == self.config.data.test_year)

        self.logger.info(
            f"Dataset split: {len(train_snaps)} train years (1995-{self.config.data.train_end_year}), "
            f"{len(val_snaps)} val years, 1 held-out test year ({self.config.data.test_year})."
        )

        models = [
            PanelARModel(),
            DynamicFactorModel(),
            PanelVARModel(),
            LeontiefIOModel(),
            CentralityRegressionModel(),
            SpatialARModel(),
            ModelAGraphOnly(self.config),
            ModelBDynamicGraphAgent(self.config),
        ]

        metrics_table: Dict[str, EvaluationMetrics] = {}
        predictions_growth: Dict[str, np.ndarray] = {}
        predictions_tail: Dict[str, np.ndarray] = {}

        for model in models:
            self.logger.info(f"Fitting model: {model.name}...")
            model.fit(train_snaps)

            pg, pd = model.predict(test_snap)
            predictions_growth[model.name] = pg
            predictions_tail[model.name] = pd

            m = compute_metrics(pg, pd, test_snap.target_growth, test_snap.target_tail)
            metrics_table[model.name] = m
            self.logger.info(
                f"Model [{model.name}] -> RMSE: {m.rmse:.4f}, MAE: {m.mae:.4f}, Spearman Rho: {m.spearman_rho:.4f}, AUPRC: {m.auprc:.4f}"
            )

        # Diebold-Mariano tests against Spatial AR (Best non-graph baseline)
        spatial_ar_err = predictions_growth["Spatial Autoregression"] - test_snap.target_growth
        dm_pvalues: Dict[str, float] = {}

        for model_name, pg in predictions_growth.items():
            err = pg - test_snap.target_growth
            p_val = diebold_mariano_test(err, spatial_ar_err)
            dm_pvalues[model_name] = p_val

        # Economic Policy Decision Utility
        decision_utilities: Dict[str, Dict[float, float]] = {}
        for model_name, pd in predictions_tail.items():
            u = compute_economic_decision_utility(pd, test_snap.target_tail)
            decision_utilities[model_name] = u

        # Systemic Ablation Suite
        self.logger.info("Executing 8-component systemic ablation suite...")
        ablation_engine = AblationEngine(self.config)
        ablation_table = ablation_engine.run_ablations(train_snaps, test_snap)

        # Prediction-Observation Divergence Delta(t)
        model_b_growth = predictions_growth["Graph/Agent Model (Model B)"]
        pred_output_2020 = test_snap.Y * np.exp(model_b_growth)
        actual_output_2020 = test_snap.Y * np.exp(test_snap.target_growth)

        divergence_delta = float(np.linalg.norm(pred_output_2020 - actual_output_2020))
        refalsification_triggered = bool(divergence_delta > 500.0)

        self.logger.info(
            f"Prediction-Observation Divergence Delta(2020) = {divergence_delta:.2f}. "
            f"Refalsification Triggered: {refalsification_triggered}."
        )

        return PipelineExecutionReport(
            metrics_table=metrics_table,
            dm_pvalues=dm_pvalues,
            decision_utilities=decision_utilities,
            ablation_table=ablation_table,
            divergence_delta=divergence_delta,
            refalsification_triggered=refalsification_triggered,
        )
