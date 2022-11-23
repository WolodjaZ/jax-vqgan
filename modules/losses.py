import jax
import jax.numpy as jnp


def l2_loss(predictions: jnp.ndarray, targets: jnp.ndarray) -> jnp.ndarray:
    """Compute L2 loss.
    Args:
        predictions (jnp.ndarray): Predictions from the model.
        targets (jnp.ndarray): Targets for the model.
    Returns:
        Reconstruction loss.
    """
    return (predictions - targets) ** 2


def l1_loss(predictions: jnp.ndarray, targets: jnp.ndarray) -> jnp.ndarray:
    """Compute L1 loss.
    Args:
        predictions (jnp.ndarray): Predictions from the model.
        targets (jnp.ndarray): Targets for the model.
    Returns:
        Reconstruction loss.
    """
    return jnp.abs(predictions - targets)


def combo_loss(predictions: jnp.ndarray, targets: jnp.ndarray) -> jnp.ndarray:
    """Compute combined l1 and l2 loss : l1 if l2 < 0.5 else l2.
    Args:
        predictions (jnp.ndarray): Predictions from the model.
        targets (jnp.ndarray): Targets for the model.
    Returns:
        Reconstruction loss.
    """
    l1 = predictions - targets
    l2 = (predictions - targets) ** 2
    return jnp.where(l2 < 0.5, l1, l2)


def mape_loss(predictions: jnp.ndarray, targets: jnp.ndarray) -> jnp.ndarray:
    """Compute mean absolute percentage error loss.
    Args:
        predictions (jnp.ndarray): Predictions from the model.
        targets (jnp.ndarray): Targets for the model.
    Returns:
        Reconstruction loss.
    """
    return jnp.abs((targets - predictions) / targets)


def disc_loss_vanilla(real: jnp.ndarray, fake: jnp.ndarray) -> jnp.ndarray:
    """Compute discriminator loss for vanilla GAN.
    Wrong fake logits impact more the loss than the bad real logits.
    Args:
        real: Real images, received from dataset.
        fake: Fake images, produced by generator.
    Returns:
        Discriminator loss.
    """
    real_loss = jnp.mean(jax.nn.softplus(-real))
    generated_loss = jnp.mean(jax.nn.softplus(fake))
    return 0.5 * (real_loss + generated_loss)


def disc_loss_hinge(real: jnp.ndarray, fake: jnp.ndarray) -> jnp.ndarray:
    """Compute discriminator loss for hinge GAN.
    Real and fake logits influence the loss the same.
    Args:
        real: Real images, received from dataset.
        fake: Fake images, produced by generator.
    Returns:
        Discriminator loss.
    """
    real_loss = jnp.mean(jnp.maximum(1.0 - real, 0.0))
    loss_fake = jnp.mean(jnp.maximum(1.0 + fake, 0.0))
    return 0.5 * (real_loss + loss_fake)
