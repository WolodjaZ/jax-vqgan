from typing import Callable, Tuple

import jax
import jax.numpy as jnp
import pytest

import modules.losses as losses


@pytest.fixture()
def JAX_PRNG() -> Tuple[jax.random.PRNGKey, jnp.dtype]:
    """Seeding for random operations."""
    test_rng = jax.random.PRNGKey(42)
    dtype = jnp.float32
    return test_rng, dtype


@pytest.mark.parametrize(
    "loss_fn, test_numbs",
    [
        (losses.l2_loss, 4),
        (losses.l2_loss, 8),
        (losses.l1_loss, 4),
        (losses.l1_loss, 8),
        (losses.combo_loss, 4),
        (losses.combo_loss, 8),
        (losses.mape_loss, 4),
        (losses.mape_loss, 8),
    ],
)
def test_reconstruction_loss(JAX_PRNG, loss_fn: Callable, test_numbs: int):
    "Test reconstruction loss"
    rng, dtype = JAX_PRNG

    # dummy data
    rng, rng_data = jax.random.split(rng)
    x = jax.random.normal(rng_data, (4, 256), dtype=dtype)
    y = x
    loss_prev = jnp.zeros_like(x)
    for _ in range(test_numbs):
        loss = loss_fn(y, x)
        assert loss.shape == (4, 256)
        assert jnp.mean(loss) >= jnp.mean(loss_prev)
        rng, rng_data = jax.random.split(rng)
        y = x + jax.random.normal(rng_data, (4, 256), dtype=dtype)

    del loss_prev, loss, x, y


@pytest.mark.parametrize("loss_fn", [losses.disc_loss_vanilla, losses.disc_loss_hinge])
def test_disc_loss(JAX_PRNG, loss_fn: Callable):
    "Test dics loss"
    _, dtype = JAX_PRNG

    # case perfect prediction
    real = jnp.ones((4, 70, 70, 1), dtype=dtype)
    fake = jnp.zeros((4, 70, 70, 1), dtype=dtype)
    loss_perfect = loss_fn(real, fake)
    assert jnp.allclose(float(loss_perfect), 0.5, atol=0.01)

    # case half good prediction
    real_bad = jnp.zeros((4, 70, 70, 1), dtype=dtype)
    loss_h_real = loss_fn(real_bad, fake)
    assert loss_h_real > loss_perfect

    fake_bad = jnp.ones((4, 70, 70, 1), dtype=dtype)
    loss_h_fake = loss_fn(real, fake_bad)
    assert loss_h_fake > loss_perfect
    if loss_fn == losses.disc_loss_hinge:
        assert jnp.allclose(loss_h_fake, loss_h_real, atol=0.001)
    elif loss_fn == losses.disc_loss_vanilla:
        assert loss_h_fake > loss_h_real

    # case bad prediction
    loss_bad = loss_fn(real_bad, fake_bad)
    assert loss_bad > loss_h_real
    assert loss_bad > loss_h_fake

    del loss_perfect, loss_h_real, loss_h_fake, loss_bad, real, fake, real_bad, fake_bad
