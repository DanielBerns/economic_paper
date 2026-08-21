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


def test_cli_quicktest_run(tmp_path):
    cfg_path = tmp_path / "settings.yaml"
    out_dir = tmp_path / "output"
    ckpt_dir = tmp_path / "checkpoints"
    with open(cfg_path, "w") as f:
        f.write(
            f"app:\n  quicktest: true\n  output_dir: {out_dir}\n  checkpoint_dir: {ckpt_dir}\n"
        )
    res = main(["--config", str(cfg_path), "--run"])
    assert res == 0
    assert (out_dir / "tables" / "table1_main_results.csv").exists()
    assert (out_dir / "figures" / "economic_decision_utility.png").exists()
    assert (out_dir / "reports" / "empirical_report.json").exists()

