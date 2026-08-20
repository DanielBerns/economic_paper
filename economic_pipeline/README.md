# Economic Pipeline

Empirical Pipeline and Out-of-Sample Benchmark for Dynamic Graph-Agent Economic World Models.

`economic_pipeline` is a Python empirical validation framework designed to evaluate multi-country, multi-industry network models on high-dimensional economic datasets (such as the OECD Inter-Country Input-Output tables).

---

## Key Features

- **Multi-Country Input-Output Data Loader**: Generates and loads OECD ICIO snapshots ($80\text{ countries} \times 50\text{ industries} = 4,000$ network nodes) with empirical growth rates and tail-risk flags.
- **Model Suite**:
  - **Econometric Baselines**: Panel Autoregressions (Panel AR), Dynamic Factor Models (DFM), Panel Vector Autoregressions (Panel VAR).
  - **Input-Output & Spatial Models**: Leontief Input-Output Inverse, Network Centrality Regression, Spatial Autoregressions (Spatial AR).
  - **Graph Neural Network / Agent Models**: Model A (Graph-Only GNN), Model B (Dynamic Graph-Agent Network).
- **Strategy Pattern Centrality Engine**: Supports both ultra-fast vectorized feature calculation (`"fast"`) and exact original NetworkX graph centralities (`"exact_networkx"`).
- **Rigorous Statistical Evaluation**: Computes RMSE, MAE, Spearman $\rho$, AUPRC, pairwise Diebold-Mariano hypothesis tests, and economic policy decision utilities.
- **Systemic Ablation Suite**: Evaluates 8 component-level model variants to quantify the contribution of graph topology and agent state dynamics.
- **Publication Artifact Exporters**: Automatically generates high-resolution figures (`.png`, `.pdf`), formatted LaTeX tables (`.tex`), CSV outputs, and JSON evaluation reports.

---

## Prerequisites

- **Python**: $\ge 3.14$
- **Package Manager**: [`uv`](https://github.com/astral-sh/uv) (recommended)

---

## Installation

1. **Navigate into the `economic_pipeline` directory**:

   ```bash
   cd economic_pipeline
   ```

2. **Sync dependencies and build local workspace packages**:

   Sync the workspace virtual environment including development tools (`pytest`, `ruff`, `mypy`):

   ```bash
   uv sync --extra dev
   ```

   This automatically builds and links the workspace subpackage `economic-graph` and installs the `economic-graph` CLI executable into `.venv/bin/`.

---

## Configuration

The pipeline behavior is controlled via YAML configuration files located in `config/`.

### Configuration Structure (`config/settings.yaml`)

```yaml
app:
  name: "Economic Graph Agent Pipeline"
  env: "development"
  checkpoint_dir: "checkpoints"
  output_dir: "output"
  force_retrain: false

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
  centrality_strategy: "fast"   # Options: "fast" (default, ~9s/snap) or "exact_networkx"

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

### Centrality Computation Strategies

The framework uses the **Strategy Pattern** for topological feature extraction:

1. **`fast`** *(Default)*: Uses NumPy BLAS power iteration for eigenvector centrality and sampled sparse betweenness centrality ($k=50$). Dataset snapshot generation takes $\sim 4.2$ minutes for all 28 years.
2. **`exact_networkx`**: Uses full NetworkX shortest-path betweenness (`nx.betweenness_centrality`) and eigenvector centrality (`nx.eigenvector_centrality`).

### Initializing Configuration

To generate or verify a default configuration file:

```bash
uv run economic-graph --config config/settings.yaml --create
```

---

## Running the CLI (`economic-graph`)

You can run `economic-graph` directly through `uv run`.

### Command-Line Options

```
usage: economic-graph [-h] [--config CONFIG] [--create]
                      [--checkpoint-dir CHECKPOINT_DIR]
                      [--output-dir OUTPUT_DIR] [--force-retrain] [--init]
                      [--run] [--report]

options:
  -h, --help            Show help message and exit
  --config CONFIG       Path to YAML configuration file (default: config/settings.yaml)
  --create              Create a default configuration file if missing
  --checkpoint-dir DIR  Directory to save and load model checkpoints
  --output-dir DIR      Directory to export tables, figures, CSVs, and reports
  --force-retrain       Force retrain models from scratch, ignoring saved checkpoints
  --init                Initialize workspace structure and configuration
  --run                 Execute the complete empirical pipeline
  --report              Print summary evaluation tables
```

### Quick Start Examples

- **Display Help**:
  ```bash
  uv run economic-graph --help
  ```

- **Initialize Workspace**:
  ```bash
  uv run economic-graph --init
  ```

- **Run Empirical Pipeline**:
  ```bash
  uv run economic-graph --run
  ```

- **Run Pipeline with Custom Configuration & Paths**:
  ```bash
  uv run economic-graph --run --config config/settings.yaml --output-dir output/results --checkpoint-dir output/checkpoints --force-retrain
  ```

- **Alternative Python Module Invocation**:
  ```bash
  uv run python -m economic_graph.cli --run
  ```

---

## Testing

To run the full unit test suite:

```bash
uv run pytest
```

To run specific test modules:

```bash
uv run pytest tests/test_cli.py
uv run pytest tests/test_config.py
uv run pytest tests/test_data.py
uv run pytest tests/test_models.py
uv run pytest tests/test_pipeline.py
```

---

## Directory Structure

```
economic_pipeline/
├── config/
│   ├── settings.yaml            # Main configuration file
│   └── settings.example.yaml    # Template configuration file
├── packages/
│   └── economic-graph/          # Core Python subpackage
│       ├── pyproject.toml
│       └── src/
│           └── economic_graph/  # Module source (cli, data, models, pipeline, config)
├── tests/                       # Pytest test suite
├── pyproject.toml               # Workspace root configuration
├── README.md                    # Project documentation
└── uv.lock                      # Lockfile for reproducible builds
```
