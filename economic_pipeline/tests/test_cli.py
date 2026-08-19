from economic_graph.cli import main


def test_cli_no_args():
    # Calling CLI without flags should display help and return 0 without executing pipeline
    res = main([])
    assert res == 0


def test_cli_init():
    res = main(["--init"])
    assert res == 0


def test_cli_create(tmp_path):
    cfg_path = tmp_path / "settings.yaml"
    res = main(["--config", str(cfg_path), "--create"])
    assert res == 0
    assert cfg_path.exists()
