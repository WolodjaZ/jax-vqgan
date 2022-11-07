import jax
import jax.numpy as jnp


def reconstruction_loss(
    predictions: jnp.ndarray, targets: jnp.ndarray, type: str = "l2"
) -> jnp.ndarray:
    """Compute reconstruction loss.
    Args:
        predictions: Predictions from the model.
        targets: Targets for the model.
        type: Type of loss to use. Currently only supports "l2" and "l1".
    Returns:
        Reconstruction loss.
    """
    if type == "l2":
        return (predictions - targets) ** 2
    elif type == "l1":
        return jnp.abs(predictions - targets)
    else:
        raise ValueError(f"Unknown loss type: {type}")


def dics_loss(real: jnp.ndarray, fake: jnp.ndarray, type: str = "hinge") -> float:
    """Compute discriminator loss.
    Args:
        predictions: Predictions from the model.
        targets: Targets for the model.
        type: Type of loss to use. Currently only supports "vanilla" and "hinge".
    Returns:
        Discriminator loss.
    """
    if type == "hinge":
        """Hinge loss. real and fake logits influence the loss the same."""
        real_loss = jnp.mean(jnp.maximum(1.0 - real, 0.0))
        loss_fake = jnp.mean(jnp.maximum(1.0 + fake, 0.0))
        return 0.5 * (real_loss + loss_fake)
    elif type == "vanilla":
        """Vanilla loss. Wrong fake logits impact more the loss than the bad real logits."""
        real_loss = jnp.mean(jax.nn.softplus(-real))
        generated_loss = jnp.mean(jax.nn.softplus(fake))
        return 0.5 * (real_loss + generated_loss)
    else:
        raise ValueError(f"Unknown loss type: {type}")
