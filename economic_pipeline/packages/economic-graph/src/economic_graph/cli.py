import argparse
import sys
from pathlib import Path

import yaml

from economic_graph.config import AppConfig, read_config, resolve_config_path
from economic_graph.logging import setup_logger


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
        "--centrality-strategy",
        choices=["fast", "unweighted", "distance_inverted", "exact_networkx"],
        help="Select topological centrality strategy ('fast', 'unweighted', 'distance_inverted', 'exact_networkx')",
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
    if args.centrality_strategy:
        config.data.centrality_strategy = args.centrality_strategy

    logger = setup_logger("economic_graph.cli", config.logging)

    # Handle workspace initialization and configuration file creation
    if args.create or args.init:
        cfg_path, _ = resolve_config_path(args_list)
        cfg_path.parent.mkdir(parents=True, exist_ok=True)
        if not cfg_path.exists():
            with open(cfg_path, "w") as f:
                yaml.dump(AppConfig().model_dump(), f, default_flow_style=False)
            logger.info(f"Created default configuration file at {cfg_path}")

        Path(config.app.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        Path(config.app.output_dir).mkdir(parents=True, exist_ok=True)
        Path(config.logging.log_file).parent.mkdir(parents=True, exist_ok=True)

        if args.create:
            logger.info("Configuration processed via --create flag.")
            print(f"Configuration file processed for workspace: {config.app.name}")
        if args.init:
            logger.info("Initializing economic graph pipeline workspace...")
            print(f"Workspace initialized with configuration: {config.app.name}")

        if not args.run:
            return 0

    if args.run:
        from economic_graph.pipeline.runner import PipelineRunner

        logger.info("Starting empirical pipeline execution...")
        runner = PipelineRunner(config)
        report = runner.run_pipeline()

        print("\n" + "=" * 80)
        print("DECISIVE EMPIRICAL BASELINE COMPARISON TABLE (Held-Out Test Set)")
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
            f"\nPrediction-Observation Divergence Delta: {report.divergence_delta:.2f}"
        )
        print(f"Refalsification Triggered: {report.refalsification_triggered}")
        print("=" * 80)
        print("\nGENERATED PUBLICATION OUTPUTS:")
        for k, v in report.output_paths.items():
            print(f"  - {k:<15}: {v}")
        print("=" * 80 + "\n")
        return 0

    if args.report:
        tables_dir = Path(config.app.output_dir) / "tables"
        t1_csv = tables_dir / "table1_main_results.csv"
        t2_csv = tables_dir / "table2_ablations.csv"

        if t1_csv.exists():
            import pandas as pd
            df1 = pd.read_csv(t1_csv)
            print("\n" + "=" * 80)
            print("DECISIVE EMPIRICAL BASELINE COMPARISON TABLE (Held-Out Test Set)")
            print("=" * 80)
            print(df1.to_string(index=False))
            print("=" * 80)

            if t2_csv.exists():
                df2 = pd.read_csv(t2_csv)
                print("\nSYSTEMIC ABLATION STUDY RESULTS")
                print("-" * 80)
                print(df2.to_string(index=False))
                print("=" * 80 + "\n")
        else:
            print(f"\nNo empirical report table found at {t1_csv}.")
            print("Please run the pipeline first using: uv run economic-graph --run\n")
        return 0

    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())

