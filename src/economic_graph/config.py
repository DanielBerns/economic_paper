import argparse
import os
from pathlib import Path
from typing import Optional, Tuple
import yaml
from pydantic import BaseModel, Field, ValidationError


class AppSettings(BaseModel):
    name: str = Field(default="Economic Graph Agent Pipeline")
    env: str = Field(default="development")


class DataSettings(BaseModel):
    num_countries: int = Field(default=80, ge=2)
    num_industries: int = Field(default=50, ge=1)
    start_year: int = Field(default=1995)
    end_year: int = Field(default=2022)
    train_end_year: int = Field(default=2017)
    val_end_year: int = Field(default=2019)
    test_year: int = Field(default=2020)
    raw_monetary_scaling: float = Field(default=1000.0)
    edge_threshold: float = Field(default=1.0)


class ModelSettings(BaseModel):
    hidden_dim: int = Field(default=64, ge=8)
    num_layers: int = Field(default=2, ge=1)
    learning_rate: float = Field(default=0.005, gt=0)
    epochs: int = Field(default=50, ge=1)
    batch_size: int = Field(default=1, ge=1)
    tail_quantile: float = Field(default=0.10, gt=0, lt=1)
    seed: int = Field(default=42)


class LoggingSettings(BaseModel):
    level: str = Field(default="INFO")
    log_file: str = Field(default="logs/economic_graph.log")
    max_bytes: int = Field(default=10485760)
    backup_count: int = Field(default=5)


class AppConfig(BaseModel):
    app: AppSettings = Field(default_factory=AppSettings)
    data: DataSettings = Field(default_factory=DataSettings)
    model: ModelSettings = Field(default_factory=ModelSettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)


class Config:
    """Central configuration object providing access to specific settings."""

    def __init__(self, settings: AppConfig):
        self._settings = settings

    @property
    def app(self) -> AppSettings:
        return self._settings.app

    @property
    def data(self) -> DataSettings:
        return self._settings.data

    @property
    def model(self) -> ModelSettings:
        return self._settings.model

    @property
    def logging(self) -> LoggingSettings:
        return self._settings.logging

    def get_num_nodes(self) -> int:
        return self._settings.data.num_countries * self._settings.data.num_industries


def resolve_config_path(args_list: Optional[list[str]] = None) -> Tuple[Path, bool]:
    """Resolve configuration file path via CLI, Environment Variable, or default path."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", help="Path to the YAML configuration file")
    parser.add_argument(
        "--create",
        action="store_true",
        help="Create a default configuration file if missing",
    )
    parsed_args, _ = parser.parse_known_args(args_list)

    config_path_str = "config/settings.yaml"
    if parsed_args.config:
        config_path_str = parsed_args.config
    elif os.environ.get("CONFIG_FILE"):
        config_path_str = os.environ.get("CONFIG_FILE")

    return Path(config_path_str), bool(parsed_args.create)


def read_config(
    config_path_override: Optional[Path] = None, args_list: Optional[list[str]] = None
) -> Config:
    """Read, parse, and validate YAML configuration file using Pydantic."""
    if config_path_override:
        config_path = config_path_override
        should_create = False
    else:
        config_path, should_create = resolve_config_path(args_list)

    if not config_path.exists():
        if should_create:
            print(f"Configuration file not found. Generating default at: {config_path}")
            default_settings = AppConfig()
            config_path.parent.mkdir(parents=True, exist_ok=True)
            with open(config_path, "w") as f:
                yaml.dump(default_settings.model_dump(), f, default_flow_style=False)
        else:
            # Fallback to default in-memory config if no file found to allow running seamlessly
            print(f"Notice: Config file not found at {config_path}. Using default configuration.")
            return Config(AppConfig())

    with open(config_path, "r") as f:
        raw_data = yaml.safe_load(f) or {}

    try:
        settings = AppConfig(**raw_data)
        return Config(settings)
    except ValidationError as e:
        print(f"Configuration validation error: {e}")
        raise
