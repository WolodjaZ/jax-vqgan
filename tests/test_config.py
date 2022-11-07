from modules.config import DiscConfig, VQGANConfig


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


def test_hydra_TrainConfig():
    "Test that the TrainConfig  class is instantiated correctly by Hydra"
    pass
