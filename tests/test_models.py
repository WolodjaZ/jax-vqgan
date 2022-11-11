from typing import Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
import pytest

from modules import config, models


@pytest.fixture()
def JAX_PRNG() -> Tuple[jax.random.PRNGKey, jnp.dtype]:
    """Seeding for random operations."""
    test_rng = jax.random.PRNGKey(42)
    dtype = jnp.float32
    return test_rng, dtype


@pytest.mark.parametrize(
    "hw_ch, use_conv",
    [
        (1, False),
        (4, False),
        (4, True),
    ],
)
def test_Upsample(JAX_PRNG, hw_ch, use_conv):
    "Test that the Upsample layer works"
    in_channels = 4
    rng, dtype = JAX_PRNG

    # Check instantiation
    upsample = models.Upsample(in_channels, use_conv=use_conv, dtype=dtype)
    assert upsample is not None
    assert upsample.dtype == dtype

    # Create a dummy input
    rng, init_rng = jax.random.split(rng)
    x = jnp.ones((1, hw_ch, hw_ch, in_channels), dtype=dtype)  # [BxHxWxC]

    # Check initialization
    params = upsample.init(init_rng, x)
    assert params is not None

    # Check forward pass
    y = upsample.apply(params, x)
    assert y is not None
    assert y.shape == (1, hw_ch * 2, hw_ch * 2, in_channels)
    assert y.dtype == dtype

    # Clean up
    del upsample, params, x, y


@pytest.mark.parametrize(
    "hw_ch, use_conv",
    [
        (2, False),
        (4, False),
        (4, True),
    ],
)
def test_Downsample(JAX_PRNG, hw_ch, use_conv):
    "Test that the Downsample layer works"
    in_channels = 4
    rng, dtype = JAX_PRNG

    # Check instantiation
    upsample = models.Downsample(in_channels, use_conv=use_conv, dtype=dtype)
    assert upsample is not None
    assert upsample.dtype == dtype

    # Create a dummy input
    rng, init_rng = jax.random.split(rng)
    x = jnp.ones((1, hw_ch, hw_ch, in_channels), dtype=dtype)  # [BxHxWxC]

    # Check initialization
    params = upsample.init(init_rng, x)
    assert params is not None

    # Check forward pass
    y = upsample.apply(params, x)
    assert y is not None
    assert y.shape == (1, hw_ch // 2, hw_ch // 2, in_channels)
    assert y.dtype == dtype

    # Clean up
    del upsample, params, x, y


@pytest.mark.parametrize(
    "in_channels, out_channels, use_conv_shortcut, temb_use",
    [
        (64, 64, False, False),
        (64, 0, False, False),
        (64, 128, False, False),
        (64, 32, True, False),
        (64, 32, True, False),
        (64, 32, True, True),
    ],
)
def test_ResNetBlock(JAX_PRNG, in_channels, out_channels, use_conv_shortcut, temb_use):
    "Test that the ResNetBlock works"
    hw_ch = 16
    dropout_prob = 0.0
    temb_channels = 128
    act_fn = nn.swish
    rng, dtype = JAX_PRNG

    if out_channels == 0:
        out_channels = None

    # Check instantiation
    resnetblock = models.ResNetBlock(
        in_channels,
        out_channels,
        act_fn=act_fn,
        use_conv_shortcut=use_conv_shortcut,
        temb_channels=temb_channels,
        dropout_prob=dropout_prob,
        dtype=dtype,
    )
    assert resnetblock is not None
    assert resnetblock.dtype == dtype

    # Create a dummy input
    rng, init_rng, dropout_init_rng = jax.random.split(rng, num=3)
    x = jnp.ones((1, hw_ch, hw_ch, in_channels), dtype=dtype)  # [BxHxWxC]
    temb = jnp.ones((1, temb_channels), dtype=dtype) if temb_use else None

    # Check initialization
    params = resnetblock.init(
        {"params": init_rng, "dropout": dropout_init_rng},
        x,
        temb=temb,
        deterministic=True,
    )["params"]
    assert params is not None

    # Check forward pass
    rng, dropout_apply_rng = jax.random.split(rng)
    y = resnetblock.apply(
        {"params": params},
        x,
        temb=temb,
        deterministic=True,
        rngs={"dropout": dropout_apply_rng},
    )
    assert y is not None
    if out_channels is None:
        assert y.shape == (1, hw_ch, hw_ch, in_channels)
    else:
        assert y.shape == (1, hw_ch, hw_ch, out_channels)
    assert y.dtype == dtype

    # Clean up
    del resnetblock, params, x, y


@pytest.mark.parametrize(
    "in_channels",
    [(32), (64)],
)
def test_AttentionBlock(JAX_PRNG, in_channels):
    "Test that the ResNetBlock works"
    hw_ch = 16
    rng, dtype = JAX_PRNG
    # Check instantiation
    attnblock = models.AttnBlock(in_channels, dtype=dtype)
    assert attnblock is not None
    assert attnblock.dtype == dtype

    # Create a dummy input
    rng, init_rng = jax.random.split(rng)
    x = jnp.ones((1, hw_ch, hw_ch, in_channels), dtype=dtype)  # [BxHxWxC]

    # Check initialization
    params = attnblock.init(init_rng, x)
    assert params is not None

    # Check forward pass
    y = attnblock.apply(params, x)
    assert y is not None
    assert y.shape == x.shape
    assert y.dtype == dtype

    # Clean up
    del attnblock, params, x, y


@pytest.mark.parametrize(
    "curr_res, ch_mult, temb_use",
    [
        (16, (1, 1, 2, 2, 4), False),
        (16, (1, 2, 4), False),
        (16, (1, 2), False),
        (0, (1, 2), False),
    ],
)
def test_UpsamplingBlock(JAX_PRNG, curr_res, ch_mult, temb_use):
    "Test that the UpsamplingBlock works"
    config_vqgan = config.VQGANConfig(
        ch_mult=ch_mult,
        attn_resolutions=(curr_res,),
        num_res_blocks=1,
        ch=32,
    )
    config_vqgan.act_fn = nn.swish
    temb_use = False
    if curr_res == 0:
        curr_res = 16
    hw_ch = 16
    temb_channels = 128
    rng, dtype = JAX_PRNG

    # Iterate to check every block
    for block_idx in range(len(ch_mult)):
        # Check instantiation
        upsampleblock = models.UpsamplingBlock(
            config_vqgan, curr_res, block_idx, dtype=dtype
        )
        assert upsampleblock is not None
        assert upsampleblock.dtype == dtype

        # Create a dummy input
        rng, init_rng, dropout_init_rng = jax.random.split(rng, num=3)
        if block_idx == len(ch_mult) - 1:
            x = jnp.ones(
                (1, hw_ch, hw_ch, config_vqgan.ch * ch_mult[-1]), dtype=dtype
            )  # [BxHxWxC]
        else:
            x = jnp.ones(
                (1, hw_ch, hw_ch, config_vqgan.ch * ch_mult[block_idx + 1]), dtype=dtype
            )  # [BxHxWxC]
        temb = jnp.ones((1, temb_channels), dtype=dtype) if temb_use else None

        # Check initialization
        params = upsampleblock.init(
            {"params": init_rng, "dropout": dropout_init_rng},
            x,
            temb=temb,
            deterministic=True,
        )["params"]
        assert params is not None

        # Check forward pass
        rng, dropout_apply_rng = jax.random.split(rng)
        y = upsampleblock.apply(
            {"params": params},
            x,
            temb=temb,
            deterministic=True,
            rngs={"dropout": dropout_apply_rng},
        )
        assert y is not None
        print(y.shape)
        if block_idx == 0:  # First block don't change the resolution
            assert y.shape == (1, hw_ch, hw_ch, config_vqgan.ch * ch_mult[block_idx])
        else:
            assert y.shape == (
                1,
                hw_ch * 2,
                hw_ch * 2,
                config_vqgan.ch * ch_mult[block_idx],
            )
        assert y.dtype == dtype

        # Clean up
        del upsampleblock, params, x, y


@pytest.mark.parametrize(
    "curr_res, ch_mult, temb_use, ch",
    [
        (16, (2, 2), True, 16),  # fail for temb embedding
        (16, (1, 2), False, 16),  # fail for too small resolution
    ],
)
def test_fail_UpsamplingBlock(JAX_PRNG, curr_res, ch_mult, temb_use, ch):
    "Test that the UpsamplingBlock fails on certain conditions"
    config_vqgan = config.VQGANConfig(
        ch_mult=ch_mult,
        attn_resolutions=(curr_res,),
        num_res_blocks=1,
        ch=ch,
    )
    config_vqgan.act_fn = nn.swish
    if curr_res == 0:
        curr_res = 16
    hw_ch = 16
    temb_channels = 128
    rng, dtype = JAX_PRNG

    # Iterate to check every block
    for block_idx in range(len(ch_mult)):
        # Check instantiation
        upsampleblock = models.UpsamplingBlock(
            config_vqgan, curr_res, block_idx, dtype=dtype
        )
        assert upsampleblock is not None
        assert upsampleblock.dtype == dtype

        # Create a dummy input
        rng, init_rng, dropout_init_rng = jax.random.split(rng, num=3)
        if block_idx == len(ch_mult) - 1:
            x = jnp.ones(
                (1, hw_ch, hw_ch, config_vqgan.ch * ch_mult[-1]), dtype=dtype
            )  # [BxHxWxC]
        else:
            x = jnp.ones(
                (1, hw_ch, hw_ch, config_vqgan.ch * ch_mult[block_idx + 1]), dtype=dtype
            )  # [BxHxWxC]
        temb = jnp.ones((1, temb_channels), dtype=dtype) if temb_use else None

        # Check initialization with fail
        if temb is None:
            with pytest.raises(AssertionError) as excinfo:
                _ = upsampleblock.init(
                    {"params": init_rng, "dropout": dropout_init_rng},
                    x,
                    temb=temb,
                    deterministic=True,
                )["params"]
            assert "block_in must be divisible by 32 for GroupNorm" in str(
                excinfo.value
            )
        else:
            with pytest.raises(AssertionError) as excinfo:
                _ = upsampleblock.init(
                    {"params": init_rng, "dropout": dropout_init_rng},
                    x,
                    temb=temb,
                    deterministic=True,
                )["params"]

            assert "UpsamplingBlock don't use temporal embedding" in str(excinfo.value)

        # Clean up
        del upsampleblock, x
        break


@pytest.mark.parametrize(
    "curr_res, ch_mult, temb_use",
    [
        (16, (1, 1, 2, 2, 4), False),
        (16, (1, 2, 4), False),
        (16, (1, 2), False),
        (0, (1, 2), False),
    ],
)
def test_DownsamplingBlock(JAX_PRNG, curr_res, ch_mult, temb_use):
    "Test that the DownsamplingBlock works"
    config_vqgan = config.VQGANConfig(
        ch_mult=ch_mult,
        attn_resolutions=(curr_res,),
        num_res_blocks=1,
        ch=32,
    )
    config_vqgan.act_fn = nn.swish
    temb_use = False
    if curr_res == 0:
        curr_res = 16
    hw_ch = 16
    temb_channels = 128
    in_ch_mult = (1,) + tuple(config_vqgan.ch_mult)
    rng, dtype = JAX_PRNG

    # Iterate to check every block
    for block_idx in range(len(ch_mult)):
        # Check instantiation
        downsampleblock = models.DownsamplingBlock(
            config_vqgan, curr_res, block_idx, dtype=dtype
        )
        assert downsampleblock is not None
        assert downsampleblock.dtype == dtype

        # Create a dummy input
        rng, init_rng, dropout_init_rng = jax.random.split(rng, num=3)
        x = jnp.ones(
            (1, hw_ch, hw_ch, config_vqgan.ch * in_ch_mult[block_idx]), dtype=dtype
        )  # [BxHxWxC]
        temb = jnp.ones((1, temb_channels), dtype=dtype) if temb_use else None

        # Check initialization
        params = downsampleblock.init(
            {"params": init_rng, "dropout": dropout_init_rng},
            x,
            temb=temb,
            deterministic=True,
        )["params"]
        assert params is not None

        # Check forward pass
        rng, dropout_apply_rng = jax.random.split(rng)
        y = downsampleblock.apply(
            {"params": params},
            x,
            temb=temb,
            deterministic=True,
            rngs={"dropout": dropout_apply_rng},
        )
        assert y is not None
        print(y.shape)
        if (
            block_idx == len(config_vqgan.ch_mult) - 1
        ):  # Last block don't change the resolution
            assert y.shape == (1, hw_ch, hw_ch, config_vqgan.ch * ch_mult[block_idx])
        else:
            assert y.shape == (
                1,
                hw_ch // 2,
                hw_ch // 2,
                config_vqgan.ch * ch_mult[block_idx],
            )
        assert y.dtype == dtype

        # Clean up
        del downsampleblock, params, x, y


@pytest.mark.parametrize(
    "curr_res, ch_mult, temb_use, ch",
    [
        (16, (2, 2), True, 32),  # fail for temb embedding
        (
            16,
            (2, 2),
            False,
            16,
        ),  # fail for too small resolution becouse first block is ch * 1
    ],
)
def test_fail_DownsamplingBlock(JAX_PRNG, curr_res, ch_mult, temb_use, ch):
    "Test that the DownsamplingBlock fails on certain conditions"
    config_vqgan = config.VQGANConfig(
        ch_mult=ch_mult,
        attn_resolutions=(curr_res,),
        num_res_blocks=1,
        ch=ch,
    )
    config_vqgan.act_fn = nn.swish
    if curr_res == 0:
        curr_res = 16
    hw_ch = 16
    temb_channels = 128
    rng, dtype = JAX_PRNG

    # Iterate to check every block
    for block_idx in range(len(ch_mult)):
        # Check instantiation
        upsampleblock = models.DownsamplingBlock(
            config_vqgan, curr_res, block_idx, dtype=dtype
        )
        assert upsampleblock is not None
        assert upsampleblock.dtype == dtype

        # Create a dummy input
        rng, init_rng, dropout_init_rng = jax.random.split(rng, num=3)
        if block_idx == len(ch_mult) - 1:
            x = jnp.ones(
                (1, hw_ch, hw_ch, config_vqgan.ch * ch_mult[-1]), dtype=dtype
            )  # [BxHxWxC]
        else:
            x = jnp.ones(
                (1, hw_ch, hw_ch, config_vqgan.ch * ch_mult[block_idx + 1]), dtype=dtype
            )  # [BxHxWxC]
        temb = jnp.ones((1, temb_channels), dtype=dtype) if temb_use else None

        # Check initialization with fail
        if temb is None:
            with pytest.raises(AssertionError) as excinfo:
                _ = upsampleblock.init(
                    {"params": init_rng, "dropout": dropout_init_rng},
                    x,
                    temb=temb,
                    deterministic=True,
                )["params"]
            assert "block_in must be divisible by 32 for GroupNorm" in str(
                excinfo.value
            )
        else:
            with pytest.raises(AssertionError) as excinfo:
                _ = upsampleblock.init(
                    {"params": init_rng, "dropout": dropout_init_rng},
                    x,
                    temb=temb,
                    deterministic=True,
                )["params"]

            assert "DownsamplingBlock don't use temporal embedding" in str(
                excinfo.value
            )

        # Clean up
        del upsampleblock, x
        break


@pytest.mark.parametrize(
    "in_channels, temb_use",
    [
        (32, False),
        (64, False),
    ],
)
def test_MidBlock(JAX_PRNG, in_channels, temb_use):
    "Test that the MidBlock works"
    hw_ch = 16
    act_fn = nn.swish
    dropout_prob = 0.0
    temb_channels = 64
    rng, dtype = JAX_PRNG
    # Check instantiation
    midblock = models.MidBlock(
        in_channels,
        act_fn=act_fn,
        temb_channels=temb_channels,
        dropout_prob=dropout_prob,
        dtype=dtype,
    )
    assert midblock is not None
    assert midblock.dtype == dtype

    # Create a dummy input
    rng, init_rng, dropout_init_rng = jax.random.split(rng, num=3)
    x = jnp.ones((1, hw_ch, hw_ch, in_channels), dtype=dtype)  # [BxHxWxC]
    temb = jnp.ones((1, temb_channels), dtype=dtype) if temb_use else None

    # Check initialization
    params = midblock.init(
        {"params": init_rng, "dropout": dropout_init_rng},
        x,
        temb=temb,
        deterministic=True,
    )["params"]
    assert params is not None

    # Check forward pass
    rng, dropout_apply_rng = jax.random.split(rng)
    y = midblock.apply(
        {"params": params},
        x,
        temb=temb,
        deterministic=True,
        rngs={"dropout": dropout_apply_rng},
    )
    assert y is not None
    assert y.shape == x.shape
    assert y.dtype == dtype

    # Clean up
    del midblock, params, x, y


@pytest.mark.parametrize(
    "ch, ch_mult, z_channels, double_z",
    [
        (32, (1, 2), 128, False),
        (64, (1, 2), 128, False),
        (32, (1, 1, 2, 2), 128, False),
        (32, (1, 2), 128, True),
        (32, (1, 2), 32, False),
    ],
)
def test_Encoder(JAX_PRNG, ch, ch_mult, z_channels, double_z):
    "Test that the Encoder works"
    config_vqgan = config.VQGANConfig(
        ch_mult=ch_mult,
        attn_resolutions=(0,),
        num_res_blocks=1,
        ch=ch,
        double_z=double_z,
        z_channels=z_channels,
    )
    config_vqgan.act_fn = nn.swish
    hw_ch = len(ch_mult) * 2
    out_channels = z_channels * 2 if double_z else z_channels
    rng, dtype = JAX_PRNG

    # Check instantiation
    encoder = models.Encoder(config_vqgan, dtype=dtype)
    assert encoder is not None
    assert encoder.dtype == dtype

    # Create a dummy input
    rng, init_rng, dropout_init_rng = jax.random.split(rng, num=3)
    x = jnp.ones((1, hw_ch, hw_ch, 3), dtype=dtype)  # [BxHxWxC]

    # Check initialization
    params = encoder.init(
        {"params": init_rng, "dropout": dropout_init_rng}, x, deterministic=True
    )["params"]
    assert params is not None

    # Check forward pass
    rng, dropout_apply_rng = jax.random.split(rng)
    y = encoder.apply(
        {"params": params}, x, deterministic=True, rngs={"dropout": dropout_apply_rng}
    )
    assert y is not None
    assert y.shape == (
        1,
        hw_ch // (2 ** (len(ch_mult) - 1)),
        hw_ch // (2 ** (len(ch_mult) - 1)),
        out_channels,
    )
    assert y.dtype == dtype

    # Clean up
    del encoder, params, x, y


def test_Encoder_with_configs(JAX_PRNG):
    "Test that the Encoder works with used configs"
    pass


#    config_vqgan = None # Load config
#    config_vqgan.act_fn = nn.swish
#    hw_ch = config_vqgan.num_resolutions*2
#    out_channels = config_vqgan.z_channels*2 if config_vqgan.double_z else config_vqgan.z_channels
#    rng, dtype = JAX_PRNG
#
#    # Check instantiation
#    encoder = models.Encoder(config_vqgan, dtype=dtype)
#    assert encoder is not None
#    assert encoder.dtype == dtype
#
#    # Create a dummy input
#    rng, init_rng, dropout_init_rng = jax.random.split(rng, num=3)
#    x = jnp.ones((1, hw_ch, hw_ch, 3), dtype=dtype) #[BxHxWxC]
#
#    # Check initialization
#    params = encoder.init({'params': init_rng, 'dropout': dropout_init_rng},
#                           x,
#                           deterministic=True)['params']
#    assert params is not None
#
#    # Check forward pass
#    rng, dropout_apply_rng = jax.random.split(rng)
#    y = encoder.apply({'params': params},
#                       x,
#                       deterministic=True,
#                       rngs={'dropout': dropout_apply_rng})
#    assert y is not None
#    assert y.shape == (1,
#                       hw_ch//(2**(config_vqgan.num_resolutions-1)),
#                       hw_ch//(2**(config_vqgan.num_resolutions-1)),
#                       out_channels)
#    assert y.dtype == dtype
#
#    # Clean up
#    del encoder, params, x, y


@pytest.mark.parametrize(
    "ch, ch_mult, z_channels, out_ch",
    [
        (32, (1, 2), 128, 3),
        (64, (1, 2), 128, 3),
        (32, (1, 1, 2, 2), 128, 3),
        (32, (1, 2), 128, 1),
        (32, (1, 2), 32, 3),
    ],
)
def test_Decoder(JAX_PRNG, ch, ch_mult, z_channels, out_ch):
    "Test that the Decoder works"
    config_vqgan = config.VQGANConfig(
        ch_mult=ch_mult,
        attn_resolutions=(0,),
        num_res_blocks=1,
        ch=ch,
        out_ch=out_ch,
        z_channels=z_channels,
    )
    config_vqgan.act_fn = nn.swish
    hw_ch = 1
    rng, dtype = JAX_PRNG

    # Check instantiation
    decoder = models.Decoder(config_vqgan, dtype=dtype)
    assert decoder is not None
    assert decoder.dtype == dtype

    # Create a dummy input
    rng, init_rng, dropout_init_rng = jax.random.split(rng, num=3)
    x = jnp.ones((1, hw_ch, hw_ch, 3), dtype=dtype)  # [BxHxWxC]

    # Check initialization
    params = decoder.init(
        {"params": init_rng, "dropout": dropout_init_rng}, x, deterministic=True
    )["params"]
    assert params is not None

    # Check forward pass
    rng, dropout_apply_rng = jax.random.split(rng)
    y = decoder.apply(
        {"params": params}, x, deterministic=True, rngs={"dropout": dropout_apply_rng}
    )
    assert y is not None
    assert y.shape == (
        1,
        int(hw_ch * (2 ** (len(ch_mult) - 1))),
        int(hw_ch * (2 ** (len(ch_mult) - 1))),
        out_ch,
    )
    assert y.dtype == dtype

    # Clean up
    del decoder, params, x, y


def test_Decoder_with_configs(JAX_PRNG):
    "Test that the Decoder works with used configs"
    pass


#    config_vqgan = None # Load config
#    config_vqgan.act_fn = nn.swish
#    hw_ch = 1
#    rng, dtype = JAX_PRNG
#
#    # Check instantiation
#    decoder = models.Decoder(config_vqgan, dtype=dtype)
#    assert decoder is not None
#    assert decoder.dtype == dtype
#
#    # Create a dummy input
#    rng, init_rng, dropout_init_rng = jax.random.split(rng, num=3)
#    x = jnp.ones((1, hw_ch, hw_ch, 3), dtype=dtype) #[BxHxWxC]
#
#    # Check initialization
#    params = decoder.init({'params': init_rng, 'dropout': dropout_init_rng},
#                           x,
#                           deterministic=True)['params']
#    assert params is not None
#
#    # Check forward pass
#    rng, dropout_apply_rng = jax.random.split(rng)
#    y = decoder.apply({'params': params},
#                       x, deterministic=True,
#                       rngs={'dropout': dropout_apply_rng})
#    assert y is not None
#    assert y.shape == (1,
#                       int(hw_ch*(2**(config_vqgan.num_resolutions-1))),
#                       int(hw_ch*(2**(config_vqgan.num_resolutions-1))),
#                       config_vqgan.out_ch)
#    assert y.dtype == dtype
#
#    # Clean up
#    del decoder, params, x, y
