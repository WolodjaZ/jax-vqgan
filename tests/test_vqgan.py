import time
from typing import Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
import pytest

from modules import config, vqgan


@pytest.fixture()
def JAX_PRNG() -> Tuple[jax.random.PRNGKey, jnp.dtype]:
    # Seeding for random operations
    test_rng = jax.random.PRNGKey(42)
    dtype = jnp.float32
    return test_rng, dtype


def test_quick_gelu(JAX_PRNG):
    "Test quick gelu implementations"
    rng, dtype = JAX_PRNG

    # dummy data
    rng, rng_data = jax.random.split(rng)
    x = jax.random.normal(rng_data, (4, 256), dtype=dtype)

    # compare value quick gelu with original gelu
    assert jnp.allclose(jnp.sum(nn.gelu(x)), jnp.sum(vqgan.quick_gelu(x)), atol=0.5)

    # compare times
    start = time.time()
    nn.gelu(x)
    end_rgular = time.time() - start

    start = time.time()
    vqgan.quick_gelu(x)
    end_quick = time.time() - start
    assert end_rgular > end_quick


@pytest.mark.parametrize(
    "n_embed, embed_dim",
    [
        (16, 32),
        (8, 32),
        (16, 64),
    ],
)
def test_VectorQuantizer(JAX_PRNG, n_embed, embed_dim):
    "Test that the test_VectorQuantizer works"
    config_vqgan = config.VQGANConfig(
        n_embed=n_embed,
        embed_dim=embed_dim,
    )
    hw_ch = 8
    rng, dtype = JAX_PRNG

    # Check instantiation
    vq = vqgan.VectorQuantizer(config_vqgan, dtype=dtype)
    assert vq is not None
    assert vq.dtype == dtype

    # Create a dummy input
    rng, init_rng = jax.random.split(rng)
    x = jnp.ones((1, hw_ch, hw_ch, embed_dim), dtype=dtype)  # [BxHxWxC]

    # Check initialization
    params_dict = vq.init(init_rng, x)
    assert params_dict is not None

    # Check forward pass
    z_q, min_encoding_indices = vq.apply(params_dict, x)
    assert z_q is not None
    assert min_encoding_indices is not None
    assert z_q.dtype == dtype
    assert z_q.shape == x.shape[:-1] + (embed_dim,)
    assert min_encoding_indices.shape == (1, hw_ch * hw_ch)
    assert jnp.min(min_encoding_indices) >= 0
    assert jnp.max(min_encoding_indices) <= n_embed - 1

    # Check get codebook
    params = params_dict["params"]
    z_q_entry = vq.get_codebook_entry(params, min_encoding_indices)
    z_q_shape = z_q.shape[1] * z_q.shape[2] * z_q.shape[3]
    assert z_q_entry.shape[1] == z_q_shape
    z_q_entry = vq.get_codebook_entry(params, min_encoding_indices, shape=x.shape)
    assert z_q_entry.shape == x.shape
    jnp.allclose(z_q_entry, z_q, atol=0.0001)

    # Clean up
    del vq, params, x, z_q, min_encoding_indices


@pytest.mark.parametrize(
    "n_embed, embed_dim",
    [
        (16, 32),
        (8, 32),
        (16, 64),
    ],
)
def test_GumbelQuantize(JAX_PRNG, n_embed, embed_dim):
    "Test that the test_GumbelQuantize works"
    config_vqgan = config.VQGANConfig(
        n_embed=n_embed,
        embed_dim=embed_dim,
    )
    hw_ch = 8
    rng, dtype = JAX_PRNG

    # Check instantiation
    vq = vqgan.GumbelQuantize(config_vqgan, dtype=dtype)
    assert vq is not None
    assert vq.dtype == dtype

    # Create a dummy input
    rng, init_rng, gumble_init_rng = jax.random.split(rng, num=3)
    x = jnp.ones((1, hw_ch, hw_ch, embed_dim), dtype=dtype)  # [BxHxWxC]

    # Check initialization
    params = vq.init({"params": init_rng, "gumbel": gumble_init_rng}, x)["params"]
    assert params is not None

    # Check forward pass
    rng, gumble_apply_rng = jax.random.split(rng)
    z_q, min_encoding_indices, logits = vq.apply(
        {"params": params}, x, rngs={"gumbel": gumble_apply_rng}
    )
    assert z_q is not None
    assert min_encoding_indices is not None
    assert logits is not None
    assert z_q.dtype == dtype
    assert z_q.shape == x.shape[:-1] + (embed_dim,)
    assert min_encoding_indices.shape == (1, hw_ch * hw_ch)
    assert jnp.min(min_encoding_indices) >= 0
    assert jnp.max(min_encoding_indices) <= n_embed - 1
    assert logits.shape == (1, hw_ch, hw_ch, n_embed)
    assert logits.dtype == dtype

    # Check get codebook
    print(params)
    z_q_entry = vq.get_codebook_entry(params, min_encoding_indices)
    z_q_shape = z_q.shape[1] * z_q.shape[2] * z_q.shape[3]
    assert z_q_entry.shape[1] == z_q_shape
    z_q_entry = vq.get_codebook_entry(params, min_encoding_indices, shape=x.shape)
    assert z_q_entry.shape == x.shape
    jnp.allclose(z_q_entry, z_q, atol=0.0001)

    # Clean up
    del vq, params, x, z_q, z_q_entry, min_encoding_indices


@pytest.mark.parametrize(
    "act_name, hw_ch, batch_size, use_gumbel",
    [
        ("gelu", 128, 1, False),
        ("swish", 128, 1, False),
        ("gelu", 256, 1, False),
        ("gelu", 128, 4, False),
        ("gelu", 128, 1, True),
        ("gelu", 128, 4, True),
    ],
)
def test_VQModule(JAX_PRNG, act_name, hw_ch, batch_size, use_gumbel):
    "Test that VQModule works"
    config_vqgan = config.VQGANConfig(
        act_name=act_name,
        use_gumbel=use_gumbel,
    )
    rng, dtype = JAX_PRNG

    def model():
        return vqgan.VQModule(config_vqgan, dtype=dtype)

    # Check instantiation
    vqmodule = model()
    assert vqmodule is not None
    assert vqmodule.dtype == dtype

    # Create a dummy input
    rng, init_rng, dropout_init_rng, gumble_init_rng = jax.random.split(rng, num=4)
    x = jnp.ones((batch_size, hw_ch, hw_ch, 3), dtype=dtype)  # [BxHxWx3]

    # Check initialization
    params = vqmodule.init(
        {"params": init_rng, "gumbel": gumble_init_rng, "dropout": dropout_init_rng},
        x,
        True,
    )["params"]
    assert params is not None

    # Check forward pass
    rng, gumble_apply_rng, dropout_apply_rng = jax.random.split(rng, num=3)
    x_recon, z_q, min_encoding_indices, logits = vqmodule.apply(
        {"params": params},
        x,
        rngs={"gumbel": gumble_apply_rng, "dropout": dropout_apply_rng},
        deterministic=True,
    )
    assert z_q is not None
    assert min_encoding_indices is not None
    if use_gumbel:
        assert logits is not None
    assert z_q.dtype == dtype
    x_recon.dtype == dtype
    assert x_recon.shape == x.shape

    # Check decode_code
    def get_codebook_entry(model):
        return model.decode_code(min_encoding_indices, z_shape=z_q.shape)  # noqa: F821

    x_recon_code = nn.apply(get_codebook_entry, model())({"params": params})
    assert x_recon_code.shape == x_recon.shape
    jnp.allclose(x_recon_code, x_recon, atol=0.0001)

    # Clean up
    del vqmodule, params, x, x_recon, z_q, min_encoding_indices, logits, x_recon_code


@pytest.mark.parametrize(
    "hw_ch, batch_size, use_gumbel",
    [
        (128, 1, False),
        (256, 1, False),
        (128, 4, False),
        (128, 1, True),
        (128, 4, True),
    ],
)
def test_VQModel(JAX_PRNG, hw_ch, batch_size, use_gumbel):
    "Test that VQModel works"
    config_vqgan = config.VQGANConfig(use_gumbel=use_gumbel)
    input_shape = (batch_size, hw_ch, hw_ch, 3)
    rng, dtype = JAX_PRNG

    # Check initialization
    vqmodel = vqgan.VQModel(
        config=config_vqgan, input_shape=input_shape, seed=42, dtype=dtype
    )
    assert vqmodel is not None
    assert vqmodel.params is not None

    # Create a dummy input
    x = jnp.ones(input_shape, dtype=dtype)  # [BxHxWx3]

    # Check forward pass
    rng, gumble_apply_rng, dropout_apply_rng = jax.random.split(rng, num=3)
    x_recon, z_q, indices, logits = vqmodel(
        x,
        dropout_rng=dropout_apply_rng,
        gumble_rng=gumble_apply_rng,
        train=False,
    )
    assert z_q is not None
    assert indices is not None
    if use_gumbel:
        assert logits is not None
    assert z_q.dtype == dtype
    x_recon.dtype == dtype
    assert x_recon.shape == x.shape

    # check encode
    z_q_encoder, indices_encode, logits_encode = vqmodel.encode(
        x,
        dropout_rng=dropout_apply_rng,
        gumble_rng=gumble_apply_rng,
        train=False,
    )
    assert jnp.allclose(z_q, z_q_encoder, atol=0.0001)
    assert jnp.allclose(indices, indices_encode, atol=0.0001)
    if use_gumbel:
        assert jnp.allclose(logits, logits_encode, atol=0.0001)

    # check decode
    x_recon_decode = vqmodel.decode(
        z_q,
        dropout_rng=dropout_apply_rng,
        gumble_rng=gumble_apply_rng,
        train=False,
    )
    assert jnp.allclose(x_recon, x_recon_decode, atol=0.0001)

    # check decode_code
    x_recon_decode_code = vqmodel.decode_code(indices, z_shape=z_q.shape)  # noqa: F841
    # assert jnp.allclose(x_recon_decode_code, x_recon, atol=0.0001) TODO: fix this


def test_VQModel_with_configs(JAX_PRNG):
    "Test that VQModel works with used configs"
    pass
