import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from economic_graph.config import Config
from economic_graph.data.loader import ICIODatasetLoader
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
from economic_graph.pipeline.evaluator import (
    EvaluationMetrics,
    compute_metrics,
    diebold_mariano_test,
)
from economic_graph.pipeline.exporters import PipelineExporter


@dataclass
class PipelineExecutionReport:
    metrics_table: Dict[str, EvaluationMetrics]
    dm_pvalues: Dict[str, float]
    decision_utilities: Dict[str, Dict[float, float]]
    ablation_table: Dict[str, EvaluationMetrics]
    divergence_delta: float
    refalsification_triggered: bool
    output_paths: Dict[str, str]


class PipelineRunner:
    """Master Pipeline Orchestrator executing dataset loading, model training, test evaluation, checkpointing, and exports."""

    def __init__(self, config: Config):
        self.config = config
        self.logger = setup_logger("economic_graph.pipeline", config.logging)
        self.loader = ICIODatasetLoader(config)
        self.checkpoint_dir = Path(config.app.checkpoint_dir)
        self.output_dir = Path(config.app.output_dir)
        self.exporter = PipelineExporter(self.output_dir)

    def run_pipeline(self) -> PipelineExecutionReport:
        start_time = time.time()
        self.logger.info("=" * 80)
        self.logger.info("Initializing OECD ICIO empirical pipeline execution...")
        self.logger.info(f"Output Directory: {self.output_dir.resolve()}")
        self.logger.info(f"Checkpoint Directory: {self.checkpoint_dir.resolve()}")
        self.logger.info("=" * 80)

        snapshots = self.loader.generate_synthetic_icio()
        self.logger.info(f"Snapshots: {len(snapshots)}")

        train_snaps = [s for s in snapshots if s.year <= self.config.data.train_end_year]
        self.logger.info(f"train_snaps: {len(train_snaps)}")

        val_snaps = [
            s
            for s in snapshots
            if self.config.data.train_end_year < s.year <= self.config.data.val_end_year
        ]
        self.logger.info(f"val_snaps: {len(val_snaps)}")

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
        total_models = len(models)

        for idx, model in enumerate(models, 1):
            model_start = time.time()
            elapsed_total = time.time() - start_time
            self.logger.info(
                f"[Progress {idx}/{total_models} ({idx/total_models*100:.1f}%)] "
                f"Processing model: '{model.name}' | Total elapsed: {elapsed_total:.1f}s"
            )

            # Check if checkpoint exists and force_retrain is False
            loaded = False
            if not self.config.app.force_retrain:
                try:
                    loaded = model.load_checkpoint(self.checkpoint_dir)
                except Exception:
                    loaded = False

            if loaded:
                self.logger.info(f"-> Loaded pre-trained checkpoint for '{model.name}'. Skipping training.")
            else:
                self.logger.info(f"-> Fitting '{model.name}' on training sample...")
                model.fit(train_snaps)
                ckpt_path = model.save_checkpoint(self.checkpoint_dir)
                self.logger.info(f"-> Saved checkpoint to {ckpt_path}")

            pg, p_tail = model.predict(test_snap)
            predictions_growth[model.name] = pg
            predictions_tail[model.name] = p_tail

            m = compute_metrics(pg, p_tail, test_snap.target_growth, test_snap.target_tail)
            metrics_table[model.name] = m

            dur = time.time() - model_start
            self.logger.info(
                f"-> Completed [{model.name}] in {dur:.2f}s | RMSE: {m.rmse:.4f}, MAE: {m.mae:.4f}, "
                f"Spearman Rho: {m.spearman_rho:.4f}, AUPRC: {m.auprc:.4f}"
            )

        # Diebold-Mariano tests against Spatial AR (Best non-graph baseline)
        self.logger.info("Computing pairwise Diebold-Mariano statistical forecast comparison tests...")
        spatial_ar_err = predictions_growth["Spatial Autoregression"] - test_snap.target_growth
        dm_pvalues: Dict[str, float] = {}

        for model_name, pg in predictions_growth.items():
            err = pg - test_snap.target_growth
            p_val = diebold_mariano_test(err, spatial_ar_err)
            dm_pvalues[model_name] = p_val

        # Economic Policy Decision Utility
        self.logger.info("Computing Economic Policy Decision Utility curves across thresholds c...")
        decision_utilities: Dict[str, Dict[float, float]] = {}
        for model_name, p_tail in predictions_tail.items():
            u = compute_economic_decision_utility(p_tail, test_snap.target_tail)
            decision_utilities[model_name] = u

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

        # Export main publication artifacts (Table 1, Figures, Predictions CSV)
        self.logger.info("Exporting main publication tables, high-resolution figures, and predictions...")
        t1_csv, t1_tex = self.exporter.export_main_table(metrics_table, dm_pvalues)
        fig_png, fig_pdf = self.exporter.export_figures(decision_utilities)

        pred_df = pd.DataFrame(
            {
                "node_id": list(range(len(test_snap.target_growth))),
                "actual_growth": test_snap.target_growth,
                "actual_tail": test_snap.target_tail,
            }
        )
        for name, pg in predictions_growth.items():
            pred_df[f"pred_growth_{name}"] = pg
            pred_df[f"pred_prob_{name}"] = predictions_tail[name]

        pred_csv = self.exporter.predictions_dir / "predictions_2020.csv"
        pred_df.to_csv(pred_csv, index=False)

        # Systemic Ablation Suite
        ablation_table: Dict[str, EvaluationMetrics] = {}
        t2_csv, t2_tex = "", ""
        try:
            self.logger.info("Executing 8-component systemic ablation suite...")
            ablation_engine = AblationEngine(self.config)
            ablation_table = ablation_engine.run_ablations(train_snaps, test_snap)
            res_t2_csv, res_t2_tex = self.exporter.export_ablation_table(ablation_table)
            t2_csv, t2_tex = str(res_t2_csv), str(res_t2_tex)
        except Exception as ae:
            self.logger.error(f"Ablation suite encounter error: {ae}")

        # Export JSON report
        report_dict = {
            "metrics": {
                k: {
                    "rmse": v.rmse,
                    "mae": v.mae,
                    "spearman_rho": v.spearman_rho,
                    "auroc": v.auroc,
                    "auprc": v.auprc,
                }
                for k, v in metrics_table.items()
            },
            "divergence_delta": divergence_delta,
            "refalsification_triggered": refalsification_triggered,
        }
        json_report = self.exporter.export_report_json(report_dict)

        output_paths = {
            "table1_csv": str(t1_csv),
            "table1_tex": str(t1_tex),
            "table2_csv": t2_csv,
            "table2_tex": t2_tex,
            "figure_png": str(fig_png),
            "figure_pdf": str(fig_pdf),
            "predictions_csv": str(pred_csv),
            "report_json": str(json_report),
        }

        total_dur = time.time() - start_time
        self.logger.info("=" * 80)
        self.logger.info(f"Pipeline execution completed successfully in {total_dur:.1f}s.")
        self.logger.info(f"All outputs saved to: {self.output_dir.resolve()}")
        self.logger.info("=" * 80)

        return PipelineExecutionReport(
            metrics_table=metrics_table,
            dm_pvalues=dm_pvalues,
            decision_utilities=decision_utilities,
            ablation_table=ablation_table,
            divergence_delta=divergence_delta,
            refalsification_triggered=refalsification_triggered,
            output_paths=output_paths,
        )
