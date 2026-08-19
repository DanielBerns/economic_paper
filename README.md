# Economic Graph Agent Pipeline

The **Economic Graph Agent Pipeline** is an empirical forecasting and validation framework for dynamic graph-agent economic world models. It operationalizes high-dimensional international production networks—such as the OECD Inter-Country Input-Output (ICIO) database (1995–2022)—into an out-of-sample prediction benchmark on held-out systemic shocks (such as the 2020 COVID-19 pandemic episode). The framework compares econometric, input-output, and graph neural network models across continuous growth forecasting, rank preservation, tail-risk classification, economic policy decision utility, and systemic ablation suites.

## Installation

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/economic-paper/economic-graph-pipeline.git
cd economic-paper
uv sync
```

### System Prerequisites
- Standard C/C++ compiler toolchain (for PyTorch and SciPy matrix acceleration).
- LaTeX environment (`pdflatex`, `latexmk`) if compiling documents under `latex/`.

## Configuration

The application dynamically reads configuration parameters using the following precedence:
1. Explicit CLI argument: `--config /path/to/settings.yaml`
2. Environment variable: `CONFIG_FILE`
3. Default path: `config/settings.yaml`

To create a working configuration file from the template:

```bash
cp config/settings.example.yaml config/settings.yaml
```

Or generate a default file automatically using the CLI:

```bash
uv run python -m economic_graph.cli --config config/settings.yaml --create
```

| Key                       | Required | Default                    | Description                                                   |
|---------------------------|----------|----------------------------|---------------------------------------------------------------|
| `app.name`                | No       | `Economic Graph Pipeline`  | Application title.                                            |
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

## Execution

### Run Full Empirical Pipeline

Execute the full empirical benchmark (loads dataset, trains 8 candidate models, computes 2020 held-out metrics, calculates Diebold-Mariano test p-values, economic decision utility curves, and runs the 8-component systemic ablation suite):

```bash
uv run python -m economic_graph.cli --run
```

### Options

```bash
# Run with custom configuration file
uv run python -m economic_graph.cli --config config/custom_settings.yaml --run

# Print evaluation report summary
uv run python -m economic_graph.cli --report
```

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
│           └── runner.py         # Master pipeline orchestrator
└── tests/                        # Unit test suite
    ├── test_config.py
    ├── test_data.py
    ├── test_models.py
    └── test_pipeline.py
```
