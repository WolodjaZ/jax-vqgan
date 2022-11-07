from typing import Tuple

import jax
import jax.numpy as jnp
import pytest

import modules.losses as losses


@pytest.fixture()
def JAX_PRNG() -> Tuple[jax.random.PRNGKey, jnp.dtype]:
    # Seeding for random operations
    test_rng = jax.random.PRNGKey(42)
    dtype = jnp.float32
    return test_rng, dtype


@pytest.mark.parametrize(
    "type_loss, test_numbs",
    [
        ("l2", 4),
        ("l2", 8),
        ("l1", 4),
        ("l1", 8),
    ],
)
def test_reconstruction_loss(JAX_PRNG, type_loss: str, test_numbs: int):
    "Test reconstruction loss"
    rng, dtype = JAX_PRNG

    # dummy data
    rng, rng_data = jax.random.split(rng)
    x = jax.random.normal(rng_data, (4, 256), dtype=dtype)
    y = x
    loss_prev = jnp.zeros_like(x)
    for _ in range(test_numbs):
        loss = losses.reconstruction_loss(y, x, type=type_loss)
        assert loss.shape == (4, 256)
        assert jnp.mean(loss) >= jnp.mean(loss_prev)
        rng, rng_data = jax.random.split(rng)
        y = x + jax.random.normal(rng_data, (4, 256), dtype=dtype)

    del loss_prev, loss, x, y


@pytest.mark.parametrize("type_loss", ["vanilla", "hinge"])
def test_dics_loss(JAX_PRNG, type_loss: str):
    "Test dics loss"
    _, dtype = JAX_PRNG

    # case perfect prediction
    real = jnp.ones((4, 70, 70, 1), dtype=dtype)
    fake = jnp.zeros((4, 70, 70, 1), dtype=dtype)
    loss_perfect = losses.dics_loss(real, fake, type=type_loss)
    assert jnp.allclose(float(loss_perfect), 0.5, atol=0.01)

    # case half good prediction
    real_bad = jnp.zeros((4, 70, 70, 1), dtype=dtype)
    loss_h_real = losses.dics_loss(real_bad, fake, type=type_loss)
    assert loss_h_real > loss_perfect

    fake_bad = jnp.ones((4, 70, 70, 1), dtype=dtype)
    loss_h_fake = losses.dics_loss(real, fake_bad, type=type_loss)
    assert loss_h_fake > loss_perfect
    if type_loss == "hinge":
        assert jnp.allclose(loss_h_fake, loss_h_real, atol=0.001)
    elif type_loss == "vanilla":
        assert loss_h_fake > loss_h_real

    # case bad prediction
    loss_bad = losses.dics_loss(real_bad, fake_bad, type=type_loss)
    assert loss_bad > loss_h_real
    assert loss_bad > loss_h_fake

    del loss_perfect, loss_h_real, loss_h_fake, loss_bad, real, fake, real_bad, fake_bad
