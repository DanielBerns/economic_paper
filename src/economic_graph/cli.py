import argparse
import sys
from economic_graph.config import read_config
from economic_graph.logging import setup_logger
from economic_graph.pipeline.runner import PipelineRunner


def main(args_list=None):
    parser = argparse.ArgumentParser(
        description="Economic Graph Pipeline CLI for Dynamic Graph/Agent Forecasting"
    )
    parser.add_argument("--config", help="Path to YAML configuration file")
    parser.add_argument(
        "--create",
        action="store_true",
        help="Create a default configuration file if missing",
    )
    parser.add_argument(
        "--checkpoint-dir",
        help="Directory to save and load model checkpoints",
    )
    parser.add_argument(
        "--output-dir",
        help="Directory to export tables, figures, CSVs, and reports",
    )
    parser.add_argument(
        "--force-retrain",
        action="store_true",
        help="Force retrain models from scratch ignoring existing checkpoints",
    )
    parser.add_argument(
        "--init", action="store_true", help="Initialize workspace and configuration"
    )
    parser.add_argument(
        "--run", action="store_true", help="Execute the complete empirical pipeline"
    )
    parser.add_argument(
        "--report", action="store_true", help="Print summary evaluation tables"
    )

    args = parser.parse_args(args_list)

    # Read configuration using standards
    config = read_config(args_list=args_list)

    # Override CLI flags if provided
    if args.checkpoint_dir:
        config.app.checkpoint_dir = args.checkpoint_dir
    if args.output_dir:
        config.app.output_dir = args.output_dir
    if args.force_retrain:
        config.app.force_retrain = True

    logger = setup_logger("economic_graph.cli", config.logging)

    if args.init:
        logger.info("Initializing economic graph pipeline workspace...")
        print(f"Workspace initialized with configuration: {config.app.name}")
        return 0

    if args.run or not (args.init or args.report):
        logger.info("Starting empirical pipeline execution...")
        runner = PipelineRunner(config)
        report = runner.run_pipeline()

        print("\n" + "=" * 80)
        print("DECISIVE EMPIRICAL BASELINE COMPARISON TABLE (Held-Out 2020 Test Set)")
        print("=" * 80)
        print(
            f"{'Model Specification':<32} | {'RMSE':<7} | {'MAE':<7} | {'Spearman':<8} | {'AUPRC':<7} | {'DM p-val':<8}"
        )
        print("-" * 80)
        for model_name, m in report.metrics_table.items():
            p_val = report.dm_pvalues.get(model_name, 1.0)
            print(
                f"{model_name:<32} | {m.rmse:.4f}  | {m.mae:.4f}  | {m.spearman_rho:.4f}   | {m.auprc:.4f}  | {p_val:.4f}"
            )
        print("=" * 80)

        print("\nSYSTEMIC ABLATION STUDY RESULTS")
        print("-" * 80)
        for name, m in report.ablation_table.items():
            print(
                f"{name:<35} | RMSE: {m.rmse:.4f} | MAE: {m.mae:.4f} | Spearman: {m.spearman_rho:.4f} | AUPRC: {m.auprc:.4f}"
            )
        print("=" * 80)

        print(
            f"\nPrediction-Observation Divergence Delta(2020): {report.divergence_delta:.2f}"
        )
        print(f"Refalsification Triggered: {report.refalsification_triggered}")
        print("=" * 80)
        print("\nGENERATED PUBLICATION OUTPUTS:")
        for k, v in report.output_paths.items():
            print(f"  - {k:<15}: {v}")
        print("=" * 80 + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
