import numpy as np
from economic_graph.config import AppConfig, Config
from economic_graph.pipeline.decision_utility import compute_economic_decision_utility
from economic_graph.pipeline.evaluator import compute_metrics, diebold_mariano_test
from economic_graph.pipeline.runner import PipelineRunner


def test_compute_metrics():
    target_g = np.array([0.05, -0.10, 0.02, -0.15, 0.08])
    pred_g = np.array([0.04, -0.08, 0.01, -0.12, 0.07])
    target_d = np.array([0, 1, 0, 1, 0])
    pred_d = np.array([0.1, 0.8, 0.2, 0.9, 0.05])

    m = compute_metrics(pred_g, pred_d, target_g, target_d)
    assert m.rmse > 0.0
    assert m.mae > 0.0
    assert m.spearman_rho > 0.5
    assert m.auroc > 0.5
    assert m.auprc > 0.5


def test_diebold_mariano_test():
    e1 = np.random.normal(0, 0.05, size=100)
    e2 = np.random.normal(0, 0.05, size=100)
    p_val = diebold_mariano_test(e1, e2)
    assert 0.0 <= p_val <= 1.0


def test_economic_decision_utility():
    pred_d = np.array([0.1, 0.8, 0.2, 0.9, 0.05])
    target_d = np.array([0, 1, 0, 1, 0])

    util = compute_economic_decision_utility(pred_d, target_d)
    assert len(util) > 0
    assert all(isinstance(v, float) for v in util.values())


def test_pipeline_runner_small(tmp_path):
    app_config = AppConfig()
    app_config.app.checkpoint_dir = str(tmp_path / "checkpoints")
    app_config.app.output_dir = str(tmp_path / "output")
    app_config.data.num_countries = 3
    app_config.data.num_industries = 4
    app_config.data.start_year = 2015
    app_config.data.end_year = 2021
    app_config.data.train_end_year = 2018
    app_config.data.val_end_year = 2019
    app_config.data.test_year = 2020
    app_config.model.epochs = 1
    cfg = Config(app_config)

    runner = PipelineRunner(cfg)
    report = runner.run_pipeline()

    assert "Graph/Agent Model (Model B)" in report.metrics_table
    assert report.divergence_delta >= 0.0
    assert "table1_csv" in report.output_paths
    assert "figure_png" in report.output_paths
