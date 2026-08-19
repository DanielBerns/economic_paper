import numpy as np
from economic_graph.config import AppConfig, Config
from economic_graph.data.loader import ICIODatasetLoader
from economic_graph.models.econometric import DynamicFactorModel, PanelARModel, PanelVARModel
from economic_graph.models.gnn import ModelAGraphOnly, ModelBDynamicGraphAgent
from economic_graph.models.input_output import CentralityRegressionModel, LeontiefIOModel, SpatialARModel


def _get_test_dataset():
    app_config = AppConfig()
    app_config.data.num_countries = 3
    app_config.data.num_industries = 4
    app_config.data.start_year = 2015
    app_config.data.end_year = 2020
    app_config.model.epochs = 2
    cfg = Config(app_config)

    loader = ICIODatasetLoader(cfg)
    snaps = loader.generate_synthetic_icio()
    return snaps, cfg


def test_panel_ar_model():
    snaps, _ = _get_test_dataset()
    m = PanelARModel()
    m.fit(snaps[:3])
    pg, pd = m.predict(snaps[3])

    assert len(pg) == 12
    assert len(pd) == 12
    assert np.all(pd >= 0.0) and np.all(pd <= 1.0)


def test_leontief_io_model():
    snaps, _ = _get_test_dataset()
    m = LeontiefIOModel()
    m.fit(snaps[:3])
    pg, pd = m.predict(snaps[3])

    assert len(pg) == 12
    assert len(pd) == 12


def test_graph_models():
    snaps, cfg = _get_test_dataset()
    mA = ModelAGraphOnly(cfg)
    mA.fit(snaps[:3])
    pg_a, pd_a = mA.predict(snaps[3])

    assert len(pg_a) == 12

    mB = ModelBDynamicGraphAgent(cfg)
    mB.fit(snaps[:3])
    pg_b, pd_b = mB.predict(snaps[3])

    assert len(pg_b) == 12
