import json
from pathlib import Path
from typing import Any, Dict, Tuple

import matplotlib.pyplot as plt
import pandas as pd

from economic_graph.pipeline.evaluator import EvaluationMetrics


class PipelineExporter:
    """Exports empirical pipeline outputs into LaTeX tables, CSV files, high-res figures, and JSON reports."""

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.tables_dir = output_dir / "tables"
        self.figures_dir = output_dir / "figures"
        self.predictions_dir = output_dir / "predictions"
        self.reports_dir = output_dir / "reports"

        for d in [self.tables_dir, self.figures_dir, self.predictions_dir, self.reports_dir]:
            d.mkdir(parents=True, exist_ok=True)

    def export_main_table(
        self, metrics_table: Dict[str, EvaluationMetrics], dm_pvalues: Dict[str, float]
    ) -> Tuple[Path, Path]:
        """Export Table 1 (Main Empirical Benchmark) to CSV and LaTeX format."""
        rows = []
        for model_name, m in metrics_table.items():
            p_val = dm_pvalues.get(model_name, 1.0)
            rows.append(
                {
                    "Model Specification": model_name,
                    "RMSE": f"{m.rmse:.4f}",
                    "MAE": f"{m.mae:.4f}",
                    "Spearman Rho": f"{m.spearman_rho:.4f}",
                    "AUROC": f"{m.auroc:.4f}",
                    "AUPRC": f"{m.auprc:.4f}",
                    "Top-10% Precision": f"{m.top10_precision:.4f}",
                    "Top-10% Recall": f"{m.top10_recall:.4f}",
                    "Brier Score": f"{m.brier_score:.4f}",
                    "DM p-value": f"{p_val:.4f}",
                }
            )

        df = pd.DataFrame(rows)
        csv_path = self.tables_dir / "table1_main_results.csv"
        df.to_csv(csv_path, index=False)

        # LaTeX table export
        tex_path = self.tables_dir / "table1_main_results.tex"
        tex_lines = [
            "\\begin{table}[htbp]",
            "\\centering",
            "\\caption{\\textbf{Held-Out 2020 Out-of-Sample Prediction Results}}",
            "\\label{tab:main_results}",
            "\\small",
            "\\resizebox{\\textwidth}{!}{%",
            "\\begin{tabular}{l *{8}{c}}",
            "\\toprule",
            "\\textbf{Model Specification} & \\textbf{RMSE} $\\downarrow$ & \\textbf{MAE} $\\downarrow$ & \\textbf{Spearman }$\\rho$ $\\uparrow$ & \\textbf{AUROC} $\\uparrow$ & \\textbf{AUPRC} $\\uparrow$ & \\textbf{Top-10\\% Prec} & \\textbf{Top-10\\% Rec} & \\textbf{DM p-val} \\\\",
            "\\midrule",
        ]
        for row in rows:
            tex_lines.append(
                f"{row['Model Specification']} & {row['RMSE']} & {row['MAE']} & {row['Spearman Rho']} & {row['AUROC']} & {row['AUPRC']} & {row['Top-10% Precision']} & {row['Top-10% Recall']} & {row['DM p-value']} \\\\"
            )
        tex_lines.extend(
            [
                "\\bottomrule",
                "\\end{tabular}%",
                "}",
                "\\end{table}",
            ]
        )
        with open(tex_path, "w") as f:
            f.write("\n".join(tex_lines))

        return csv_path, tex_path

    def export_ablation_table(
        self, ablation_table: Dict[str, EvaluationMetrics]
    ) -> Tuple[Path, Path]:
        """Export Table 2 (Ablation Study) to CSV and LaTeX format."""
        rows = []
        base_rmse = ablation_table.get(
            "Full Graph/Agent Model (Model B)", list(ablation_table.values())[0]
        ).rmse

        for name, m in ablation_table.items():
            delta_rmse = m.rmse - base_rmse
            rows.append(
                {
                    "Ablation Specification": name,
                    "RMSE": f"{m.rmse:.4f}",
                    "MAE": f"{m.mae:.4f}",
                    "Spearman Rho": f"{m.spearman_rho:.4f}",
                    "AUPRC": f"{m.auprc:.4f}",
                    "Delta RMSE": f"{delta_rmse:+.4f}",
                }
            )

        df = pd.DataFrame(rows)
        csv_path = self.tables_dir / "table2_ablations.csv"
        df.to_csv(csv_path, index=False)

        tex_path = self.tables_dir / "table2_ablations.tex"
        tex_lines = [
            "\\begin{table}[htbp]",
            "\\centering",
            "\\caption{\\textbf{Ablation Study Results on Held-Out 2020 Test Set}}",
            "\\label{tab:ablations}",
            "\\small",
            "\\resizebox{\\textwidth}{!}{%",
            "\\begin{tabular}{l *{5}{c}}",
            "\\toprule",
            "\\textbf{Ablation Specification} & \\textbf{RMSE} $\\uparrow$ & \\textbf{MAE} $\\uparrow$ & \\textbf{Spearman }$\\rho$ $\\downarrow$ & \\textbf{AUPRC} $\\downarrow$ & \\textbf{$\\Delta$ RMSE} \\\\",
            "\\midrule",
        ]
        for row in rows:
            tex_lines.append(
                f"{row['Ablation Specification']} & {row['RMSE']} & {row['MAE']} & {row['Spearman Rho']} & {row['AUPRC']} & {row['Delta RMSE']} \\\\"
            )
        tex_lines.extend(
            [
                "\\bottomrule",
                "\\end{tabular}%",
                "}",
                "\\end{table}",
            ]
        )
        with open(tex_path, "w") as f:
            f.write("\n".join(tex_lines))

        return csv_path, tex_path

    def export_figures(self, decision_utilities: Dict[str, Dict[float, float]]) -> Tuple[Path, Path]:
        """Generate high-resolution PNG and PDF plots for Economic Policy Decision Utility."""
        fig, ax = plt.subplots(figsize=(8, 5), dpi=300)

        for model_name, utils in decision_utilities.items():
            thresholds = list(utils.keys())
            values = list(utils.values())

            style = "-" if "Graph/Agent" in model_name else "--"
            lw = 2.5 if "Graph/Agent" in model_name else 1.5
            ax.plot(thresholds, values, label=model_name, linestyle=style, linewidth=lw)

        ax.set_xlabel("Policy Risk Threshold c", fontsize=11, fontweight="bold")
        ax.set_ylabel("Economic Net Decision Utility", fontsize=11, fontweight="bold")
        ax.set_title(
            "Economic Decision Utility Across Risk Thresholds (Held-Out 2020 Test)",
            fontsize=12,
            fontweight="bold",
        )
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(fontsize=9, loc="lower right")

        png_path = self.figures_dir / "economic_decision_utility.png"
        pdf_path = self.figures_dir / "economic_decision_utility.pdf"

        fig.savefig(png_path, bbox_inches="tight")
        fig.savefig(pdf_path, bbox_inches="tight")
        plt.close(fig)

        return png_path, pdf_path

    def export_report_json(self, report_data: Dict[str, Any]) -> Path:
        """Export comprehensive JSON execution report."""
        path = self.reports_dir / "empirical_report.json"
        with open(path, "w") as f:
            json.dump(report_data, f, indent=2)
        return path
