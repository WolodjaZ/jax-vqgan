import jax
import pytest
from omegaconf import OmegaConf

from modules.config import DiscConfig, TrainConfig, VQGANConfig


def test_VQGANConfig():
    "Test that the VQGANConfig class works"
    ch_mult_sample = (1, 1, 2, 2, 4, 4)
    config = VQGANConfig(
        ch_mult=ch_mult_sample,
    )
    assert config is not None
    assert config.num_resolutions == len(ch_mult_sample)


def test_hydra_VQGANConfig():
    "Test that the VQGANConfig class is instantiated correctly by Hydra"
    pass


def test_DiscConfig():
    "Test that the DiscConfig class works"
    config = DiscConfig()
    assert config is not None


def test_hydra_DiscConfig():
    "Test that the DiscConfig class is instantiated correctly by Hydra"
    pass


def test_TrainConfig():
    """Test that the TrainConfig class works"""
    config_files = "tests/config_test.yaml"
    cfg = OmegaConf.load(config_files)
    load_confg = OmegaConf.to_container(cfg)
    cfg = TrainConfig(**load_confg)
    assert cfg is not None

    # Test distributed
    dist_load_confg = load_confg.copy()
    dist_load_confg["distributed"] = True
    dist_load_confg["train_batch_size"] = 8
    dist_load_confg["test_batch_size"] = 8
    cfg = TrainConfig(**dist_load_confg)
    assert cfg is not None
    assert cfg.distributed
    assert cfg.train_batch_size == 8 // jax.device_count()
    assert cfg.test_batch_size == 8 // jax.device_count()

    # Test temp_scheduler
    temp_load_confg = load_confg.copy()
    temp_load_confg["temp_scheduler"] = None
    cfg = TrainConfig(**temp_load_confg)
    assert cfg is not None
    assert cfg.temp_scheduler is None
    temp_load_confg["temp_scheduler"] = {
        "_target_": "optax.linear_schedule",
        "init_value": 1.0,
        "end_value": 0.0,
        "transition_steps": 10,
    }
    cfg = TrainConfig(**temp_load_confg)
    assert cfg is not None

    # Test fail model_hparams
    faild_load_confg = load_confg.copy()
    faild_load_confg["model_hparams"] = {"_target_": "modules.config.DiscConfig"}
    with pytest.raises(AssertionError):
        cfg = TrainConfig(**faild_load_confg)
    faild_load_confg["model_hparams"] = DiscConfig()
    with pytest.raises(Exception):
        cfg = TrainConfig(**faild_load_confg)

    # Test fail disc_hparams
    faild_load_confg = load_confg.copy()
    faild_load_confg["disc_hparams"] = {"_target_": "modules.config.VQGANConfig"}
    with pytest.raises(AssertionError):
        cfg = TrainConfig(**faild_load_confg)
    faild_load_confg["disc_hparams"] = VQGANConfig()
    with pytest.raises(Exception):
        cfg = TrainConfig(**faild_load_confg)

    # Test fail input_shape
    faild_load_confg = load_confg.copy()
    faild_load_confg["input_shape"] = (1, 1)
    with pytest.raises(AssertionError):
        cfg = TrainConfig(**faild_load_confg)

    # Test fail dtype
    faild_load_confg = load_confg.copy()
    faild_load_confg["dtype"] = "int64"
    with pytest.raises(ValueError):
        cfg = TrainConfig(**faild_load_confg)
    faild_load_confg["dtype"] = 64
    with pytest.raises(ValueError):
        cfg = TrainConfig(**faild_load_confg)

    # Test fail optimizer
    faild_load_confg = load_confg.copy()
    faild_load_confg["optimizer"] = {"_target_": "modules.config.DiscConfig"}
    with pytest.raises(AssertionError):
        cfg = TrainConfig(**faild_load_confg)
    faild_load_confg["optimizer"] = DiscConfig()
    with pytest.raises(Exception):
        cfg = TrainConfig(**faild_load_confg)

    # Test fail optimizer disc
    faild_load_confg = load_confg.copy()
    faild_load_confg["optimizer_disc"] = {"_target_": "modules.config.DiscConfig"}
    with pytest.raises(AssertionError):
        cfg = TrainConfig(**faild_load_confg)
    faild_load_confg["optimizer_disc"] = DiscConfig()
    with pytest.raises(Exception):
        cfg = TrainConfig(**faild_load_confg)

    # Test fail optimizer temp_scheduler
    faild_load_confg = load_confg.copy()
    faild_load_confg["temp_scheduler"] = {"_target_": "modules.config.DiscConfig"}
    with pytest.raises(AssertionError):
        cfg = TrainConfig(**faild_load_confg)
    faild_load_confg["temp_scheduler"] = DiscConfig()
    with pytest.raises(Exception):
        cfg = TrainConfig(**faild_load_confg)

    del cfg, load_confg, dist_load_confg, temp_load_confg, faild_load_confg


def test_hydra_TrainConfig():
    "Test that the TrainConfig  class is instantiated correctly by Hydra"
    pass
