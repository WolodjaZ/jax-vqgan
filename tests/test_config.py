import jax
import pytest
from omegaconf import OmegaConf

from modules import config

TRAINCONFIG_PATH = "tests/config_test.yaml"
DATACONFIG_PATH = "tests/dataconfig_test.yaml"


def test_VQGANConfig():
    "Test that the VQGANConfig class works"
    ch_mult_sample = (1, 1, 2, 2, 4, 4)
    cfg = config.VQGANConfig(
        ch_mult=ch_mult_sample,
    )
    assert cfg is not None
    assert cfg.num_resolutions == len(ch_mult_sample)


def test_DiscConfig():
    "Test that the DiscConfig class works"
    cfg = config.DiscConfig()
    assert cfg is not None


def test_TrainConfig():
    """Test that the TrainConfig class works"""
    cfg_omgega = OmegaConf.load(TRAINCONFIG_PATH)
    load_confg = OmegaConf.to_container(cfg_omgega)
    cfg = config.TrainConfig(**load_confg)
    assert cfg is not None
    # Test VQGANConfig
    assert cfg.model_hparams is not None
    assert isinstance(cfg.model_hparams, config.VQGANConfig)
    # Test DiscConfig
    assert cfg.disc_hparams is not None
    assert isinstance(cfg.disc_hparams, config.DiscConfig)

    # Test distributed
    dist_load_confg = load_confg.copy()
    dist_load_confg["distributed"] = True
    dist_load_confg["train_batch_size"] = 8
    dist_load_confg["test_batch_size"] = 8
    cfg = config.TrainConfig(**dist_load_confg)
    assert cfg is not None
    assert cfg.distributed
    assert cfg.train_batch_size == 8 // jax.device_count()
    assert cfg.test_batch_size == 8 // jax.device_count()

    # Test temp_scheduler
    temp_load_confg = load_confg.copy()
    temp_load_confg["temp_scheduler"] = None
    cfg = config.TrainConfig(**temp_load_confg)
    assert cfg is not None
    assert cfg.temp_scheduler is None
    temp_load_confg["temp_scheduler"] = {
        "_target_": "optax.linear_schedule",
        "init_value": 1.0,
        "end_value": 0.0,
        "transition_steps": 10,
    }
    cfg = config.TrainConfig(**temp_load_confg)
    assert cfg is not None

    # Test fail model_hparams
    faild_load_confg = load_confg.copy()
    faild_load_confg["model_hparams"] = {"_target_": "modules.config.DiscConfig"}
    with pytest.raises(AssertionError):
        cfg = config.TrainConfig(**faild_load_confg)
    faild_load_confg["model_hparams"] = config.DiscConfig()
    with pytest.raises(Exception):
        cfg = config.TrainConfig(**faild_load_confg)

    # Test fail disc_hparams
    faild_load_confg = load_confg.copy()
    faild_load_confg["disc_hparams"] = {"_target_": "modules.config.VQGANConfig"}
    with pytest.raises(AssertionError):
        cfg = config.TrainConfig(**faild_load_confg)
    faild_load_confg["disc_hparams"] = config.VQGANConfig()
    with pytest.raises(Exception):
        cfg = config.TrainConfig(**faild_load_confg)

    # Test fail input_shape
    faild_load_confg = load_confg.copy()
    faild_load_confg["input_shape"] = (1, 1)
    with pytest.raises(AssertionError):
        cfg = config.TrainConfig(**faild_load_confg)

    # Test fail dtype
    faild_load_confg = load_confg.copy()
    faild_load_confg["dtype"] = "int64"
    with pytest.raises(ValueError):
        cfg = config.TrainConfig(**faild_load_confg)
    faild_load_confg["dtype"] = 64
    with pytest.raises(ValueError):
        cfg = config.TrainConfig(**faild_load_confg)

    # Test fail optimizer
    faild_load_confg = load_confg.copy()
    faild_load_confg["optimizer"] = {"_target_": "modules.config.DiscConfig"}
    with pytest.raises(AssertionError):
        cfg = config.TrainConfig(**faild_load_confg)
    faild_load_confg["optimizer"] = config.DiscConfig()
    with pytest.raises(Exception):
        cfg = config.TrainConfig(**faild_load_confg)

    # Test fail optimizer disc
    faild_load_confg = load_confg.copy()
    faild_load_confg["optimizer_disc"] = {"_target_": "modules.config.DiscConfig"}
    with pytest.raises(AssertionError):
        cfg = config.TrainConfig(**faild_load_confg)
    faild_load_confg["optimizer_disc"] = config.DiscConfig()
    with pytest.raises(Exception):
        cfg = config.TrainConfig(**faild_load_confg)

    # Test fail optimizer temp_scheduler
    faild_load_confg = load_confg.copy()
    faild_load_confg["temp_scheduler"] = {"_target_": "modules.config.DiscConfig"}
    with pytest.raises(AssertionError):
        cfg = config.TrainConfig(**faild_load_confg)
    faild_load_confg["temp_scheduler"] = config.DiscConfig()
    with pytest.raises(Exception):
        cfg = config.TrainConfig(**faild_load_confg)

    del cfg, load_confg, dist_load_confg, temp_load_confg, faild_load_confg


def test_DataConfig():
    """Test that the DataConfig class works"""
    cfg_omgega = OmegaConf.load(DATACONFIG_PATH)
    load_confg = OmegaConf.to_container(cfg_omgega)
    cfg = config.DataConfig(**load_confg)
    assert cfg is not None
    assert cfg.test_params is not None
    assert isinstance(cfg.test_params, config.DataParams)
    assert cfg.train_params is not None
    assert isinstance(cfg.train_params, config.DataParams)
    assert type(cfg.test_params.batch_size) == int
    assert type(cfg.train_params.batch_size) == int
    assert type(cfg.test_params.shuffle) == bool
    assert type(cfg.train_params.shuffle) == bool

    # Test fail test_params
    test_config = load_confg.copy()
    test_config["test_params"] = {"batch_size": 1}
    with pytest.raises(TypeError):
        cfg = config.DataConfig(**test_config)
    test_config["test_params"] = {"shuffle": True}
    with pytest.raises(TypeError):
        cfg = config.DataConfig(**test_config)
    test_config["test_params"] = {"batch_size": 1, "shuffle": True, "lorem": "ipsum"}
    with pytest.raises(TypeError):
        cfg = config.DataConfig(**test_config)

    # Test fail train_params
    test_config = load_confg.copy()
    test_config["train_params"] = {"batch_size": 1}
    with pytest.raises(TypeError):
        cfg = config.DataConfig(**test_config)
    test_config["train_params"] = {"shuffle": True}
    with pytest.raises(TypeError):
        cfg = config.DataConfig(**test_config)
    test_config["train_params"] = {"batch_size": 1, "shuffle": True, "lorem": "ipsum"}
    with pytest.raises(TypeError):
        cfg = config.DataConfig(**test_config)


def test_LoadConfig():
    """Test that the LoadConfig class works"""
    cfg_data_omgega = OmegaConf.load(DATACONFIG_PATH)
    load_data_confg = OmegaConf.to_container(cfg_data_omgega)

    cfg_train_omgega = OmegaConf.load(TRAINCONFIG_PATH)
    load_train_confg = OmegaConf.to_container(cfg_train_omgega)

    cfg = config.LoadConfig(data=load_data_confg, train=load_train_confg)
    assert cfg is not None
    assert cfg.data is not None
    assert cfg.train is not None
    assert isinstance(cfg.data, config.DataConfig)
    assert isinstance(cfg.train, config.TrainConfig)
    assert cfg.train.train_batch_size == cfg.data.train_params.batch_size
    assert cfg.train.test_batch_size == cfg.data.test_params.batch_size

    # Test batch_size assignment from data config
    load_data_confg["train_params"]["batch_size"] = 1
    load_data_confg["test_params"]["batch_size"] = 1
    load_train_confg["train_batch_size"] = 2
    load_train_confg["test_batch_size"] = 2

    cfg = config.LoadConfig(data=load_data_confg, train=load_train_confg)
    assert cfg.train.train_batch_size == cfg.data.train_params.batch_size
    assert cfg.train.test_batch_size == cfg.data.test_params.batch_size
