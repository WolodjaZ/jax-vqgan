import pytest

from modules.config import VQGANConfig


def test_VQGANConfig():
    "Test that the VQGANConfig class works"
    ch_mult_sample = (1, 1, 2, 2, 4, 4)
    config = VQGANConfig(
        ch_mult=ch_mult_sample,
    )
    assert config is not None
    assert config.num_resolutions ==  len(ch_mult_sample)


def test_hydra_VQGANConfig():
    "Test that the VQGANConfig class is instantiated correctly by Hydra"
    pass