# Economic Graph Agent Pipeline

The **Economic Graph Agent Pipeline** is an empirical forecasting and validation framework for dynamic graph-agent economic world models. It operationalizes high-dimensional international production networks—such as the OECD Inter-Country Input-Output (ICIO) database (1995–2022)—into an out-of-sample prediction benchmark on held-out systemic shocks (such as the 2020 COVID-19 pandemic episode). The framework compares econometric, input-output, and graph neural network models across continuous growth forecasting, rank preservation, tail-risk classification, economic policy decision utility, and systemic ablation suites.

## Installation

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/DanielBerns/economic_paper.git
cd economic_paper
uv sync --extra dev
```

### System Prerequisites
- Standard C/C++ compiler toolchain (for PyTorch and SciPy matrix acceleration).
- LaTeX environment (`pdflatex`, `latexmk`) if compiling documents under `latex/`.

## Configuration

The application dynamically reads configuration parameters using the following precedence:
1. Explicit CLI arguments: `--config`, `--checkpoint-dir`, `--output-dir`, `--force-retrain`
2. Environment variable: `CONFIG_FILE`
3. Default path: `config/settings.yaml`

To create a default configuration file automatically using the CLI:

```bash
uv run python -m economic_graph.cli --create
```

Or copy the standard template manually:

```bash
cp config/settings.example.yaml config/settings.yaml
```

### Example `config/settings.yaml`

```yaml
app:
  name: "Economic Graph Agent Pipeline"
  env: "development"
  output_dir: "output"
  checkpoint_dir: "checkpoints"
  force_retrain: false
  verbose: true
  progress_interval_sec: 10

data:
  num_countries: 80
  num_industries: 50
  start_year: 1995
  end_year: 2022
  train_end_year: 2017
  val_end_year: 2019
  test_year: 2020
  raw_monetary_scaling: 1000.0
  edge_threshold: 1.0

model:
  hidden_dim: 64
  num_layers: 2
  learning_rate: 0.005
  epochs: 50
  batch_size: 1
  tail_quantile: 0.10
  seed: 42

logging:
  level: "INFO"
  log_file: "logs/economic_graph.log"
  max_bytes: 10485760
  backup_count: 5
```

| Key                       | Required | Default                    | Description                                                   |
|---------------------------|----------|----------------------------|---------------------------------------------------------------|
| `app.name`                | No       | `Economic Graph Pipeline`  | Application title.                                            |
| `app.output_dir`          | No       | `output`                   | Directory for tables, figures, CSVs, and JSON reports.        |
| `app.checkpoint_dir`      | No       | `checkpoints`              | Directory for model weights and state checkpoints.            |
| `app.force_retrain`       | No       | `false`                    | Force retraining models from scratch, ignoring checkpoints.  |
| `data.num_countries`      | No       | `80`                       | Number of economies in production network graph.             |
| `data.num_industries`     | No       | `50`                       | Number of ISIC industries per country ($N \approx 4,000$).     |
| `data.start_year`         | No       | `1995`                     | Start year of ICIO historical series.                         |
| `data.end_year`           | No       | `2022`                     | End year of ICIO historical series.                           |
| `data.train_end_year`     | No       | `2017`                     | End of historical training split.                             |
| `data.val_end_year`       | No       | `2019`                     | End of hyperparameter validation split.                       |
| `data.test_year`          | No       | `2020`                     | Held-out out-of-sample crisis test year.                      |
| `model.hidden_dim`        | No       | `64`                       | Hidden dimension for neural message-passing layers.           |
| `model.learning_rate`     | No       | `0.005`                    | Learning rate for gradient optimization.                      |
| `model.epochs`            | No       | `50`                       | Maximum training epochs per model.                            |
| `model.tail_quantile`     | No       | `0.10`                     | Quantile threshold for severe contraction event $D_{i,t+1}$.  |
| `logging.level`           | No       | `INFO`                     | Verbosity level (`DEBUG`, `INFO`, `WARNING`, `ERROR`).        |
| `logging.log_file`        | No       | `logs/economic_graph.log`  | Log file output path with rotating file handler.              |

## Initialization

Initialize workspace directories and verify configuration settings:

```bash
uv run python -m economic_graph.cli --init
```

## Execution & Checkpointing

### Run Full Empirical Pipeline

Execute the full empirical benchmark (loads dataset, trains 8 candidate models, computes 2020 held-out metrics, calculates Diebold-Mariano test p-values, economic decision utility curves, and runs the 8-component systemic ablation suite):

```bash
uv run python -m economic_graph.cli --run
```

### Model Checkpoints & Fast Reruns

The pipeline automatically saves model weights and parameter state dicts to the `checkpoints/` directory.

- **Automatic Checkpoint Loading**: If you rerun the pipeline to update figures or report outputs, the system automatically detects pre-trained model checkpoints in `checkpoints/` and skips training.
- **Force Retraining**: To force retrain all models from scratch:
  ```bash
  uv run python -m economic_graph.cli --run --force-retrain
  ```

### Generated Output Artifacts

The pipeline automatically generates and exports all publication-ready tables and figures directly into the `output/` directory:

1. **LaTeX Tables & CSVs (`output/tables/`)**:
   - `table1_main_results.tex` & `table1_main_results.csv`: Table 1 main empirical benchmark results formatted in LaTeX.
   - `table2_ablations.tex` & `table2_ablations.csv`: Table 2 systemic ablation study results formatted in LaTeX.
2. **High-Resolution Figures (`output/figures/`)**:
   - `economic_decision_utility.png` & `economic_decision_utility.pdf`: 300 DPI vector PDF and PNG plots for policy decision utility curves across risk thresholds $c$.
3. **Node Predictions (`output/predictions/`)**:
   - `predictions_2020.csv`: Node-by-node ground truth output growth, tail indicators, and model predictions.
4. **JSON Execution Report (`output/reports/`)**:
   - `empirical_report.json`: Machine-readable summary of all evaluation metrics, DM p-values, and divergence $\Delta(2020)$.

## Testing

Run unit tests using `pytest` managed by `uv`:

```bash
uv run pytest
```

## Project Structure

```
economic_paper/
├── .python-version               # Specifies Python 3.12
├── pyproject.toml                # Managed by uv (project metadata & dependencies)
├── checkpoints/                  # Saved model state dicts & weights
├── output/                       # Publication outputs
│   ├── tables/                   # LaTeX table snippets (.tex) and CSV files
│   ├── figures/                  # Publication-ready plots (.png & .pdf)
│   ├── predictions/              # Node-level model prediction CSVs
│   └── reports/                  # JSON empirical reports
├── config/
│   ├── settings.example.yaml     # Standard settings template
│   └── settings.yaml             # Working settings
├── latex/                        # Separated LaTeX document & build artifacts
│   ├── main.tex
│   ├── references.bib
│   ├── pipeline.png
│   └── main.pdf
├── src/
│   └── economic_graph/
│       ├── __init__.py
│       ├── config.py             # Pydantic settings schema & YAML reader
│       ├── logging.py            # Console & RotatingFileHandler setup
│       ├── cli.py                # Command-line entry point
│       ├── data/
│       │   ├── loader.py         # ICIO data loader & snapshot generator
│       │   ├── matrices.py       # Input-output & share matrix calculations
│       │   └── features.py       # Node state vectors & topological centralities
│       ├── models/
│       │   ├── base.py           # Abstract Base Model interface
│       │   ├── econometric.py    # Panel AR, DFM, Panel VAR
│       │   ├── input_output.py   # Leontief IO, Centrality Reg, Spatial AR
│       │   └── gnn.py            # Model A (Graph-Only) & Model B (Graph/Agent)
│       └── pipeline/
│           ├── evaluator.py      # RMSE, MAE, Spearman, AUROC, AUPRC, DM test
│           ├── decision_utility.py # Economic decision cost curves
│           ├── ablations.py      # 8-component systemic ablation suite
│           ├── exporters.py      # LaTeX table, CSV, figure, & JSON export engine
│           └── runner.py         # Master pipeline orchestrator
└── tests/                        # Unit test suite
    ├── test_config.py
    ├── test_data.py
    ├── test_models.py
    └── test_pipeline.py
```
