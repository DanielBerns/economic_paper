from pathlib import Path
from economic_graph.config import AppConfig, Config, read_config, resolve_config_path


def test_app_config_defaults():
    config = AppConfig()
    assert config.app.name == "Economic Graph Agent Pipeline"
    assert config.data.num_countries == 80
    assert config.data.num_industries == 50
    assert config.model.hidden_dim == 64


def test_config_wrapper():
    app_config = AppConfig()
    cfg = Config(app_config)
    assert cfg.get_num_nodes() == 4000
    assert cfg.data.train_end_year == 2017


def test_resolve_config_path_default():
    path, should_create = resolve_config_path([])
    assert path == Path("config/settings.yaml")
    assert not should_create


def test_read_config_fallback(tmp_path):
    cfg_file = tmp_path / "test_settings.yaml"
    cfg = read_config(config_path_override=cfg_file)
    assert cfg.data.test_year == 2020
