# Contributing to Economic Paper

Thank you for your interest in contributing to `economic_paper`!

## Getting Started

1. **Fork and Clone**:
   ```bash
   git clone https://github.com/DanielBerns/economic_paper.git
   cd economic_paper
   ```

2. **Set up Environment**:
   We recommend using `uv` for quick environment setup:
   ```bash
   uv sync --extra dev
   ```

3. **Running CLI**:
   ```bash
   uv run python -m economic_graph.cli --help
   uv run python -m economic_graph.cli --create
   ```

4. **Running Tests & Linting**:
   ```bash
   uv run pytest
   uv run ruff check .
   ```

## Development Workflow

- Create a descriptive feature branch (`git checkout -b feature/my-feature`).
- Ensure all unit tests pass before submitting a pull request.
- Keep commits structured and clean.

## Code Style

We enforce code style using `ruff`. Run `ruff check .` before submitting PRs.
