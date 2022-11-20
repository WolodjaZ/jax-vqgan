import logging
import os
from collections import defaultdict
from functools import partial
from typing import Any, Callable, Dict, Optional, Tuple

import jax
import jax.numpy as jnp
import optax
import tensorflow as tf
from flax.core.frozen_dict import FrozenDict
from flax.training import train_state
from tqdm.auto import tqdm
from transformers.modeling_flax_utils import FlaxPreTrainedModel

from modules import config, losses, utils, vqgan

logger = logging.getLogger(__name__)


class TrainerModule:
    """Helper functions for training."""

    def __init__(self, module_config: config.TrainConfig, model_class: FlaxPreTrainedModel):
        """Module for summarizing all common training functionalities.

        Args:
            module_config (TrainConfig): Configuration for training
                with all hyperparameters and train module parameters.
            model_class (FlaxPreTrainedModel): Model class to be trained.
        """
        super().__init__()
        self.module_config = module_config
        self.eval_key = module_config.monitor
        self.main_rng = jax.random.PRNGKey(self.module_config.seed)
        # Set model name
        self.model_name = self.module_config.model_name
        self.model_class = model_class
        self.model = model_class(
            self.module_config.model_hparams,
            input_shape=(1,) + self.module_config.input_shape,
            seed=self.module_config.seed,
            dtype=self.module_config.dtype,
        )

        # Set training parameters
        self.state = train_state.TrainState(
            step=0,
            apply_fn=self.model.__call__,
            params=self.model.params,
            tx=None,
            opt_state=None,
        )

        # Prepare logging
        self.log_dir: str = os.path.join(self.module_config.log_dir, f"{self.model_name}/")
        self.logger: tf.summary.SummaryWriter = tf.summary.create_file_writer(self.log_dir)
        self.save_dir: str = os.path.join(self.module_config.save_dir, f"{self.model_name}/")
        # Create jitted training and eval functions
        self.create_functions()

    def create_functions(self):
        """To be implemented in sub-classes."""
        raise NotImplementedError

    def init_optimizer(self):
        """Initialize optimizer and scheduler.
        By default, we decrease the learning rate with cosine annealing.
        """
        optimizer: optax.GradientTransformation = self.module_config.optimizer
        self.create_train_state(optimizer)

    def create_train_state(self, optimizer: optax.GradientTransformation):
        """Initialize training state."""
        self.state = train_state.TrainState.create(
            apply_fn=self.state.apply_fn, params=self.state.params, tx=optimizer
        )

    def train_model(self, train_loader: utils.DataLoader, val_loader: utils.DataLoader):
        """Train model for defined number of epochs.
        Args:
            train_loader (utils.DataLoader): Training data loader.
            val_loader (utils.DataLoader): Validation data loader.
        """
        # We first need to create optimizer and the scheduler for the given number of epochs
        self.init_optimizer()
        # Track best eval metric
        logger.info("Starting training 💃")
        best_eval = None
        with self.logger.as_default():
            for epoch_idx in range(1, self.module_config.num_epochs + 1):
                logger.info("Epoch: %d", epoch_idx)
                train_metrics = self.train_epoch(train_loader, epoch=epoch_idx)
                for key in train_metrics:
                    tf.summary.scalar(f"train/{key}", train_metrics[key], step=epoch_idx)

                if epoch_idx % self.module_config.check_val_every_n_epoch == 0:
                    eval_metrics = self.eval_model(val_loader)
                    for key in eval_metrics:
                        tf.summary.scalar(f"val/{key}", eval_metrics[key], step=epoch_idx)
                    if best_eval is None or eval_metrics[self.eval_key] > best_eval:
                        best_eval = eval_metrics[self.eval_key]
                        self.save_model()

                self.logger.flush()
        logger.info("Finished training ✅ with best eval metric: %f 😎", best_eval)

    def train_epoch(self, data_loader: utils.DataLoader, epoch: int) -> Dict[str, float]:
        """Train model for one epoch, and log avg metrics.
        Args:
            data_loader (utils.DataLoader): Data loader to train on.
            epoch (int): Current epoch.
        Returns:
            Dict[str, float]: Dictionary with all metrics.
        """
        metrics: Dict[str, float] = defaultdict(float)
        for batch in tqdm(data_loader(), desc="Training", leave=False):
            # ensure that model have actual parameters
            self.model.params = self.state.params
            batch_metrics: Dict[str, float]
            self.state, self.main_rng, batch_metrics = self.train_step(
                state=self.state,
                batch=batch,
                rng=self.main_rng,
                distributed=self.module_config.distributed,
            )
            for key in batch_metrics:
                metrics[key] += batch_metrics[key]

        count = len(data_loader)
        metrics = {key: metrics[key] / count for key in metrics}
        return metrics

    def train_step(
        state: train_state.TrainState,
        batch: Any,
        rng: jax.random.PRNGKey,
        distributed: bool = False,
        *args,
        **kwargs,
    ) -> Any:
        """Train model on a single batch.

        Args:
            state (TrainState): Current training state.
            batch (Any): Batch of data.
            rng (jax.random.PRNGKey): Random number generator.
            distributed (bool, optional): Whether to use distributed training. Defaults to False.
        Returns:
            Updated training states, rng and metrics.
        """
        raise NotImplementedError

    def eval_step(
        state: train_state.TrainState,
        batch: Any,
        rng: jax.random.PRNGKey,
        *args,
        **kwargs,
    ) -> Tuple[jax.random.PRNGKey, Dict[str, float]]:
        """Evaluate model on a single batch.
        Args:
            state (TrainState): Current training state.
            batch (Any): Batch of data.
            rng (jax.random.PRNGKey): Random number generator.
        Returns:
            Tuple[jax.random.PRNGKey, Dict[str, float]]: New rng and metrics.

        """
        raise NotImplementedError

    def eval_model(self, data_loader: utils.DataLoader) -> Dict[str, float]:
        """Test model on all images of a data loader and return avg metrics.
        Args:
            data_loader (utils.DataLoader): Data loader to evaluate on.
        Returns:
            Dict[str, float]: Dictionary with all metrics.
        """
        metrics: Dict[str, float] = defaultdict(float)
        count = 0
        for batch in data_loader():
            batch_metrics: Dict[str, float]
            self.main_rng, batch_metrics = self.eval_step(
                state=self.state, batch=batch, rng=self.main_rng
            )
            batch_size = (batch[0] if isinstance(batch, (tuple, list)) else batch).shape[0]
            count += batch_size
            for key in batch_metrics:
                metrics[key] += batch_metrics[key] * batch_size
        metrics = {key: metrics[key] / count for key in metrics}
        return metrics

    def save_model(self, hub=False):
        """Save current model."""
        if hub:
            self.model.save_pretrained(
                self.save_dir,
                push_to_hub=True,
                commit_message="Saving weights and logs",
            )
        else:
            self.model.save_pretrained(self.save_dir)

    def load_model(self):
        """Load model."""
        self.model = self.model_class.from_pretrained(self.save_dir)
        self.state = train_state.TrainState.create(
            apply_fn=self.state.apply_fn,
            params=self.model.params,
            tx=self.state.tx if self.state.tx else self.module_config.optimizer,
        )

    def checkpoint_exists(self) -> bool:
        """Check whether a pretrained model exist.
        Returns:
            bool: True if model exists, False otherwise.
        """
        return os.path.exists(self.save_dir) and len(os.listdir(self.log_dir)) > 0


class TrainStateDisc(train_state.TrainState):
    """Train state for discriminator.
    Arguments:
        apply_fn: The function that applies the model.
        step: The current step.
        params: The model parameters.
        batch_stats: The batch statistics. Defaults to None.
        tx: The optimizer. Defaults to None.
        opt_state: The optimizer state. Defaults to None.
    """

    batch_stats: Optional[FrozenDict[str, Any]] = None


class TrainerVQGan(TrainerModule):
    """Helper functions for training VQGAN.
    Arguments:
        recon_loss_fn (Callable): Reconstruction loss function. Defaults to l1 loss.
        disc_loss_fn (Callable): Discriminator loss function. Defaults to hinge.
    """

    recon_loss_fn: Callable = losses.l1_loss
    disc_loss_fn: Callable = losses.disc_loss_hinge

    def __init__(self, module_config: config.TrainConfig):
        # Initialize parent class
        self.module_config = module_config
        self.model_name = self.module_config.model_name
        if self.module_config.recon_loss == "l2":
            TrainerVQGan.recon_loss_fn = losses.l2_loss
        elif self.module_config.recon_loss == "l1":
            TrainerVQGan.recon_loss_fn = losses.l1_loss
        elif self.module_config.recon_loss == "combo":
            TrainerVQGan.recon_loss_fn = losses.combo_loss
        else:
            logger.warning(
                f"""Reconstruction loss function {self.module_config.recon_loss} not supported.
                Will be used default l2 loss instead."""
            )
        # Train state for discriminator
        self.model_disc: FlaxPreTrainedModel = vqgan.VQGanDiscriminator(
            self.module_config.disc_hparams
        )
        if self.module_config.disc_loss == "vanilla":
            TrainerVQGan.disc_loss_fn = losses.disc_loss_vanilla
        elif self.module_config.disc_loss == "hinge":
            TrainerVQGan.disc_loss_fn = losses.disc_loss_hinge
        else:
            logger.warning(
                f"""Discriminator loss function {self.module_config.disc_loss} not supported.
                Will be used default hinge loss instead."""
            )
        self.state_disc = TrainStateDisc(
            step=0,
            apply_fn=self.model_disc.__call__,
            params=self.model_disc.params["params"],
            batch_stats=self.model_disc.params["batch_stats"],
            tx=None,
            opt_state=None,
        )
        # Log dir discriminator loss
        self.save_dir_disc: str = os.path.join(
            self.module_config.save_dir, f"{self.model_name}_disc/"
        )
        self.temp_scheduler: Optional[Callable] = self.module_config.temp_scheduler

        super().__init__(module_config=module_config, model_class=vqgan.VQModel)

    def temperature_scheduling(self, epoch: int) -> float:
        """Temperature scheduling.
        Args:
            epoch (int): Current epoch.
        Returns:
            float: Temperature.
        """
        if self.temp_scheduler is not None:
            return self.model.update_temperature(self.temp_scheduler(epoch))
        else:
            return self.module_config.model_hparams.gumb_temp

    def init_optimizer(self):
        """Initialize optimizer and scheduler also for discriminator.
        By default, we decrease the learning rate with cosine annealing.
        """
        optimizer: optax.GradientTransformation = self.module_config.optimizer
        optimizer_disc: optax.GradientTransformation = self.module_config.optimizer_disc
        self.create_train_stat_full(optimizer, optimizer_disc)

    def create_train_stat_full(
        self,
        optimizer: optax.GradientTransformation,
        optimizer_disc: optax.GradientTransformation,
    ):
        """Initialize training state.
        Args:
            optimizer (optax.GradientTransformation): Optimizer for generator.
            optimizer_disc (optax.GradientTransformation): Optimizer for discriminator.
        """
        self.state = train_state.TrainState.create(
            apply_fn=self.state.apply_fn, params=self.state.params, tx=optimizer
        )
        self.state_disc = TrainStateDisc.create(
            apply_fn=self.state_disc.apply_fn,
            params=self.state_disc.params,
            batch_stats=self.state_disc.batch_stats,
            tx=optimizer_disc,
        )

    def save_model(self):
        """Save current model."""
        super().save_model()
        self.model_disc.save_pretrained(self.save_dir_disc)

    def load_model(self):
        """Load model."""
        super().load_model()
        self.model_disc = self.model_disc.from_pretrained(self.save_dir_disc)
        self.state_disc = TrainStateDisc.create(
            apply_fn=self.state_disc.apply_fn,
            params=self.model_disc.params["params"],
            batch_stats=self.model_disc.params["batch_stats"],
            tx=self.state_disc.tx if self.state_disc.tx else self.module_config.optimizer_disc,
        )

    def checkpoint_exists(self) -> bool:
        """Check whether a pretrained model exist.
        Returns:
            bool: True if model and discriminator exists, False otherwise.
        """
        main_model: bool = os.path.exists(self.save_dir) and len(os.listdir(self.log_dir)) > 0
        disc_model: bool = os.path.exists(self.save_dir_disc) and len(os.listdir(self.log_dir)) > 0
        return main_model and disc_model

    def train_epoch(self, data_loader: utils.DataLoader, epoch: int) -> Dict[str, float]:
        """Train model for one epoch, and log avg metrics.
        Args:
            data_loader (utils.DataLoader): Data loader to train on.
        Returns:
            Dict[str, float]: Dictionary with all metrics.
        """
        metrics: Dict[str, float] = defaultdict(float)
        metrics_disc = defaultdict(float)
        metrics_disc["step"] = 0.0
        new_temp: float = self.temperature_scheduling(epoch - 1)
        for batch in tqdm(data_loader(), desc="Training", leave=False):
            batch_metrics: Dict[str, float]
            self.state, self.state_disc, self.main_rng, batch_metrics = self.train_step(
                state=self.state,
                disc_state=self.state_disc,
                batch=batch,
                rng=self.main_rng,
                optimizer_idx=0,
                disc_use=self.module_config.disc_start > epoch,
                distributed=self.module_config.distributed,
            )
            # Update metrics
            for key, value in batch_metrics.items():
                metrics[key] += value

            if self.module_config.disc_start > epoch:
                batch_metrics_disc: Dict[str, float]
                (self.state, self.state_disc, self.main_rng, batch_metrics_disc,) = self.train_step(
                    state=self.state,
                    disc_state=self.state_disc,
                    batch=batch,
                    rng=self.main_rng,
                    optimizer_idx=1,
                    disc_use=True,
                    distributed=self.module_config.distributed,
                )
                # Update metrics discriminator
                for key, value in batch_metrics_disc.items():
                    metrics_disc[key] += value
                metrics_disc["step"] += 1.0

            # ensure that model have actual parameters
            self.model.params = self.state.params
            self.model_disc.params["params"] = self.state_disc.params
            self.model_disc.params["batch_stats"] = self.state_disc.batch_stats

        count = len(data_loader)
        metrics = {key: metrics[key] / count for key in metrics}
        metrics["temp"] = new_temp
        count_disc = metrics_disc["step"]
        del metrics_disc["step"]
        metrics_disc_resized: Dict[str, float] = {
            key: metrics_disc[key] / count_disc for key in metrics_disc
        }
        # merge metrics
        for key, value in metrics_disc_resized.items():
            metrics[key] = value

        return metrics

    def create_functions(self):
        """Create training and eval functions."""

        def calculate_loss_autoencoder(
            params: FrozenDict[str, Any],
            batch: jnp.ndarray,
            train: bool,
            rng: jax.random.PRNGKey,
            disc_use: bool,
            disc_variables: TrainStateDisc,
        ) -> Tuple[
            jnp.ndarray,
            Tuple[Dict[str, float], jax.random.PRNGKey, Optional[FrozenDict[str, Any]]],
        ]:
            """Function to calculate the loss autoencoder for a batch of images."""
            new_rng, gumble_apply_rng, dropout_apply_rng = jax.random.split(rng, num=3)
            outs = self.model(
                batch,
                params=params,
                dropout_rng=dropout_apply_rng,
                gumble_rng=gumble_apply_rng,
                train=train,
            )
            x_recon, z_q, codebook_loss, indices = outs
            # for now we will use l1 loss than it will be combined with perceptual loss
            rec_loss = TrainerVQGan.recon_loss_fn(x_recon, batch)
            nll_loss = jnp.mean(rec_loss)

            # Generator loss (autoencode)
            outs = self.model_disc(
                x_recon,
                params=disc_variables.params,
                batch_stats=disc_variables.batch_stats,
                train=train,
            )
            logits_fake, new_model_state = outs if train else (outs, None)
            # Original loss is
            # g_loss = -jnp.mean(logits_fake)
            # But we think that based on disc for generator should work we will use minimax loss
            # This loss is none negative and tries to maximize the probability of the fake_logits.
            g_loss = jnp.mean(jnp.maximum(1.0 - logits_fake, 0.0))
            disc_factor = 0.0
            if disc_use:
                disc_factor = self.module_config.disc_weight
            disc_factor = jax.lax.cond(
                disc_use, lambda _: self.module_config.disc_weight, lambda _: 0.0, None
            )
            loss = (
                nll_loss
                + disc_factor * g_loss
                + self.module_config.codebook_weight * codebook_loss.mean()
            )

            metrics = {
                "total_loss": loss,
                "quant_loss": jnp.mean(codebook_loss),
                "nll_loss": nll_loss,
                "rec_loss": jnp.mean(rec_loss),
                "g_loss": g_loss,
            }

            return loss, (metrics, new_rng, new_model_state)

        def calculate_loss_disc(
            params: FrozenDict[str, Any],
            batch: jnp.ndarray,
            train: bool,
            rng: jax.random.PRNGKey,
            disc_use: bool,
            batch_stats: FrozenDict[str, Any],
            model_params: Optional[FrozenDict[str, Any]],
        ) -> Tuple[jnp.ndarray, Tuple[Dict[str, float], jax.random.PRNGKey, FrozenDict[str, Any]]]:
            """Function to calculate the loss discriminator for a batch of images."""
            new_rng, gumble_apply_rng, dropout_apply_rng = jax.random.split(rng, num=3)
            outs = self.model(
                batch,
                params=model_params,
                dropout_rng=dropout_apply_rng,
                gumble_rng=gumble_apply_rng,
                train=train,
            )
            x_recon, z_q, codebook_loss, indices = outs

            # Discriminator loss
            outs = self.model_disc(batch, params=params, batch_stats=batch_stats, train=train)
            logits_real, new_model_state = outs if train else (outs, None)
            outs = self.model_disc(x_recon, params=params, batch_stats=batch_stats, train=train)
            logits_fake, new_model_state = outs if train else (outs, None)
            disc_factor = jax.lax.cond(
                disc_use, lambda _: self.module_config.disc_weight, lambda _: 0.0, None
            )
            loss = disc_factor * TrainerVQGan.disc_loss_fn(logits_real, logits_fake)
            metrics = {
                "disc_loss": loss,
                "logits_real": logits_real.mean(),
                "logits_fake": logits_fake.mean(),
            }

            return loss, (metrics, new_rng, new_model_state)

        def train_step_autoencoder(
            state: train_state.TrainState,
            disc_state: TrainStateDisc,
            batch: jnp.ndarray,
            rng: jax.random.PRNGKey,
            disc_use: bool,
            distributed: bool = False,
        ) -> Tuple[train_state.TrainState, TrainStateDisc, jax.random.PRNGKey, Dict[str, float]]:
            """Train step for autoencoder."""
            loss_fn = partial(
                calculate_loss_autoencoder,
                batch=batch,
                train=True,
                rng=rng,
                disc_use=disc_use,
                disc_variables=disc_state,
            )
            (_, (metrics, new_rng, new_model_state)), grads = jax.value_and_grad(
                loss_fn, has_aux=True
            )(state.params)
            # if distributed training, average grads
            if distributed:
                grads = jax.lax.pmean(grads, axis_name="batch")
            # Update parameters
            state = state.apply_gradients(grads=grads)
            return state, disc_state, new_rng, metrics

        def train_step_disc(
            state: train_state.TrainState,
            disc_state: TrainStateDisc,
            batch: jnp.ndarray,
            rng: jax.random.PRNGKey,
            disc_use: bool,
            distributed: bool = False,
        ) -> Tuple[train_state.TrainState, TrainStateDisc, jax.random.PRNGKey, Dict[str, float]]:
            """Train step for discriminator."""
            loss_fn = partial(
                calculate_loss_disc,
                batch=batch,
                train=True,
                rng=rng,
                disc_use=disc_use,
                batch_stats=disc_state.batch_stats,
                model_params=state.params,
            )
            (_, (metrics, new_rng, new_model_state)), grads = jax.value_and_grad(
                loss_fn, has_aux=True
            )(disc_state.params)
            # if distributed training, average grads
            if distributed:
                grads = jax.lax.pmean(grads, axis_name="batch")
            # Update parameters, batch statistics
            disc_state = disc_state.apply_gradients(
                grads=grads, batch_stats=new_model_state["batch_stats"]
            )
            return state, disc_state, new_rng, metrics

        def train_step(
            state: train_state.TrainState,
            disc_state: TrainStateDisc,
            batch: jnp.ndarray,
            rng: jax.random.PRNGKey,
            optimizer_idx: int,
            disc_use: bool,
            distributed: bool,
        ) -> Tuple[train_state.TrainState, TrainStateDisc, jax.random.PRNGKey, Dict[str, float]]:
            """Train model on a single batch."""
            # calculate loss
            if optimizer_idx == 0:
                outs = train_step_autoencoder(state, disc_state, batch, rng, disc_use, distributed)
            else:
                outs = train_step_disc(state, disc_state, batch, rng, disc_use, distributed)
            # outs = jax.lax.cond(
            #  optimizer_idx == 0,
            #    lambda _: train_step_autoencoder(state,
            #                                     disc_state,
            #                                     batch,
            #                                     rng,
            #                                     disc_use,
            #                                     distributed),
            #    lambda _: train_step_disc(state,
            #                              disc_state,
            #                              batch,
            #                              rng,
            #                              disc_use,
            #                              distributed),
            #    None)
            state, disc_state, new_rng, metrics = outs
            return state, disc_state, new_rng, metrics

        def eval_step(
            state: train_state.TrainState,
            batch: jnp.ndarray,
            disc_state: TrainStateDisc,
            rng: jax.random.PRNGKey,
        ) -> Tuple[jax.random.PRNGKey, Dict[str, float]]:
            """Evaluate model on a single batch."""
            _, (metrics, new_rng, _) = calculate_loss_autoencoder(
                state.params,
                batch=batch,
                train=False,
                rng=rng,
                disc_use=False,
                disc_variables=disc_state,
            )
            return new_rng, metrics

        # pmap or jit for efficiency
        if self.module_config.distributed:
            self.train_step = jax.pmap(
                train_step, axis_name="batch", static_broadcasted_argnums=(4, 5, 6)
            )
            self.eval_step = jax.jit(partial(eval_step, disc_state=self.state_disc))
        else:
            self.train_step = jax.jit(train_step, static_argnums=(4, 5, 6))
            self.eval_step = jax.jit(partial(eval_step, disc_state=self.state_disc))
