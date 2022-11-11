import os
import tempfile

import jax
import jax.numpy as jnp
import pytest
from jax.tree_util import tree_all, tree_map
from omegaconf import OmegaConf

from modules import config, training, utils, vqgan

CONFIG_FILE = "tests/config_test.yaml"


@pytest.fixture()
def LOAD_CONFIG():
    """Load the config file."""
    cfg = OmegaConf.load(CONFIG_FILE)
    return cfg


def test_TrainerVQGan_initalization(LOAD_CONFIG):
    """Test that the TrainerVQGan can be initialized."""
    # test loading the config file
    load_confg = OmegaConf.to_container(LOAD_CONFIG)
    with tempfile.TemporaryDirectory() as dp:
        load_confg["save_dir"] = os.path.join(dp, load_confg["save_dir"])
        load_confg["log_dir"] = os.path.join(dp, load_confg["log_dir"])
        cfg = config.TrainConfig(**load_confg)
        assert cfg is not None
        type(cfg) == config.TrainConfig
        # test the TrainerModule initialization
        model = training.TrainerVQGan(cfg)
        assert model is not None
        assert model.model is not None
        type(model.model) == vqgan.VQModel
        assert model.model_disc is not None
        type(model.model_disc) == vqgan.VQGanDiscriminator
        assert model.state is not None
        assert model.state.tx is None
        assert model.state_disc is not None
        assert model.state_disc.tx is None

    del model, cfg, load_confg


def test_init_create_optimizer(LOAD_CONFIG):
    """Test that the optimizer is created correctly."""
    load_confg = OmegaConf.to_container(LOAD_CONFIG)
    with tempfile.TemporaryDirectory() as dp:
        load_confg["save_dir"] = os.path.join(dp, load_confg["save_dir"])
        load_confg["log_dir"] = os.path.join(dp, load_confg["log_dir"])
        cfg = config.TrainConfig(**load_confg)
        model = training.TrainerVQGan(cfg)
        assert model is not None

        # test the optimizer creation
        model.init_optimizer()
        assert model.state is not None
        assert model.state.tx is not None
        assert model.state_disc is not None
        assert model.state_disc.tx is not None

    del model, cfg, load_confg


@pytest.mark.parametrize(
    "scheduler",
    [
        None,
        {
            "_target_": "optax.linear_schedule",
            "init_value": 1.0,
            "end_value": 0.0,
            "transition_steps": 10,
        },
    ],
)
def test_temperature_scheduling(LOAD_CONFIG, scheduler):
    """Test that the temperature is scheduled correctly."""
    load_confg = OmegaConf.to_container(LOAD_CONFIG)
    load_confg["temp_scheduler"] = scheduler
    with tempfile.TemporaryDirectory() as dp:
        load_confg["save_dir"] = os.path.join(dp, load_confg["save_dir"])
        load_confg["log_dir"] = os.path.join(dp, load_confg["log_dir"])
        cfg = config.TrainConfig(**load_confg)
        model = training.TrainerVQGan(cfg)
        assert model is not None

        # test the temperature scheduling
        if scheduler is not None:
            for i in range(scheduler["transition_steps"]):
                expected_value = 1.0 - i / scheduler["transition_steps"]
                assert jnp.allclose(
                    model.temperature_scheduling(i), expected_value, atol=1e-5
                )
        else:
            for i in range(10):
                assert model.temperature_scheduling(i) == cfg.model_hparams.gumb_temp

    del model, cfg, load_confg, scheduler


def test_save_load_model(LOAD_CONFIG):
    """Test that the model can be saved and loaded."""
    load_confg = OmegaConf.to_container(LOAD_CONFIG)
    with tempfile.TemporaryDirectory() as dp:
        load_confg["save_dir"] = os.path.join(dp, load_confg["save_dir"])
        load_confg["log_dir"] = os.path.join(dp, load_confg["log_dir"])
        cfg = config.TrainConfig(**load_confg)
        model = training.TrainerVQGan(cfg)
        assert model is not None

        assert not model.checkpoint_exists()

        # test the model saving
        model.init_optimizer()
        model.save_model()
        assert len(os.listdir(load_confg["save_dir"])) == 2
        assert (
            len(os.listdir(os.path.join(load_confg["save_dir"], f"{cfg.model_name}")))
            > 0
        )
        assert (
            len(
                os.listdir(
                    os.path.join(load_confg["save_dir"], f"{cfg.model_name}_disc")
                )
            )
            > 0
        )

        # test the model loading
        assert model.checkpoint_exists()
        model_new = training.TrainerVQGan(cfg)
        assert model_new.checkpoint_exists()
        model_new.load_model()
        assert model_new is not None
        assert tree_all(
            tree_map(
                lambda x, y: (x == y).all(), model.state.params, model_new.state.params
            )
        )
        assert model_new.state.tx is not None
        assert tree_all(
            tree_map(
                lambda x, y: (x == y).all(),
                model.state_disc.params,
                model_new.state_disc.params,
            )
        )
        assert tree_all(
            tree_map(
                lambda x, y: (x == y).all(),
                model.state_disc.batch_stats,
                model_new.state_disc.batch_stats,
            )
        )
        assert model_new.state_disc.tx is not None

    del model, cfg, load_confg, model_new


def test_train_step(LOAD_CONFIG):
    """Test that the model can be trained for one step."""
    load_confg = OmegaConf.to_container(LOAD_CONFIG)

    rng = jax.random.PRNGKey(load_confg["seed"])
    input_shape = [
        load_confg["train_batch_size"],
    ] + load_confg["input_shape"]
    batch = jax.random.normal(rng, input_shape)

    with tempfile.TemporaryDirectory() as dp:
        load_confg["save_dir"] = os.path.join(dp, load_confg["save_dir"])
        load_confg["log_dir"] = os.path.join(dp, load_confg["log_dir"])
        cfg = config.TrainConfig(**load_confg)
        model = training.TrainerVQGan(cfg)
        assert model is not None
        model.init_optimizer()

        # test the model training only autoencoder
        rng, train_rng = jax.random.split(rng)
        outs = model.train_step(
            model.state,
            model.state_disc,
            batch,
            rng=rng,
            optimizer_idx=0,
            disc_use=False,
            distributed=False,
        )

        state, state_disc, rng, batch_metrics = outs
        assert state is not None
        assert state != model.state
        assert state.step == 1
        assert state_disc is not None
        assert state_disc.step == model.state_disc.step
        assert tree_all(
            tree_map(
                lambda x, y: (x == y).all(), state_disc.params, model.state_disc.params
            )
        )
        assert tree_all(
            tree_map(
                lambda x, y: (x == y).all(),
                state_disc.batch_stats,
                model.state_disc.batch_stats,
            )
        )
        assert batch_metrics is not None
        assert len(batch_metrics) == 5

        # test the model training autoencoder and discriminator
        outs = model.train_step(
            model.state,
            model.state_disc,
            batch,
            rng=train_rng,
            optimizer_idx=0,
            disc_use=True,
            distributed=False,
        )

        state, state_disc, rng, batch_metrics = outs
        assert state is not None
        assert state != model.state
        assert state.step == 1
        assert state_disc is not None
        assert state_disc.step == model.state_disc.step
        assert tree_all(
            tree_map(
                lambda x, y: (x == y).all(), state_disc.params, model.state_disc.params
            )
        )
        assert tree_all(
            tree_map(
                lambda x, y: (x == y).all(),
                state_disc.batch_stats,
                model.state_disc.batch_stats,
            )
        )
        assert batch_metrics is not None
        assert len(batch_metrics) == 5

        # test the model training only discriminator
        outs = model.train_step(
            model.state,
            model.state_disc,
            batch,
            rng=train_rng,
            optimizer_idx=1,
            disc_use=True,
            distributed=False,
        )

        state, state_disc, rng, batch_metrics = outs
        assert state is not None
        assert state.step == model.state.step
        assert state_disc.step == 1
        assert tree_all(
            tree_map(lambda x, y: (x == y).all(), state.params, model.state.params)
        )
        assert batch_metrics is not None
        assert len(batch_metrics) == 3

    # clean up
    del model, cfg, load_confg, state, state_disc, batch_metrics, rng, train_rng


def test_eval_step(LOAD_CONFIG):
    """Test that the model can be evaluated for one step."""
    load_confg = OmegaConf.to_container(LOAD_CONFIG)

    rng = jax.random.PRNGKey(load_confg["seed"])
    input_shape = [
        load_confg["train_batch_size"],
    ] + load_confg["input_shape"]
    batch = jax.random.normal(rng, input_shape)

    with tempfile.TemporaryDirectory() as dp:
        load_confg["save_dir"] = os.path.join(dp, load_confg["save_dir"])
        load_confg["log_dir"] = os.path.join(dp, load_confg["log_dir"])
        cfg = config.TrainConfig(**load_confg)
        model = training.TrainerVQGan(cfg)
        assert model is not None
        model.init_optimizer()

        # test the model evaluation only autoencoder
        rng, eval_rng = jax.random.split(rng)
        outs = model.eval_step(model.state, batch, rng=eval_rng)

        rng, batch_metrics = outs
        assert rng[0] != eval_rng[0] or rng[1] != eval_rng[1]
        assert batch_metrics is not None
        assert len(batch_metrics) == 5

        # test train and than eval
        outs = model.train_step(
            model.state,
            model.state_disc,
            batch,
            rng=rng,
            optimizer_idx=0,
            disc_use=True,
            distributed=False,
        )

        state, state_disc, rng, batch_metrics_train = outs
        assert state is not None
        assert state_disc is not None
        assert rng is not None
        assert batch_metrics_train is not None

        model.state_disc = state_disc
        outs = model.eval_step(state, batch, rng=rng)

        rng, batch_metrics_new = outs
        for k, v in batch_metrics_new.items():
            assert v != batch_metrics[k], f"Metric {k} is unchanged after training."

    # clean up
    del model, cfg, load_confg, state, state_disc, batch_metrics, rng, eval_rng


@pytest.mark.training_long
@pytest.mark.parametrize(
    "use_scheduler, bath_size, batches, num_steps_disc_start",
    [
        (False, 2, 2, 2),
        (False, 2, 4, 1),
        (False, 4, 2, 2),
        (False, 2, 2, 2),
        (True, 2, 2, 2),
    ],
)
def test_train_epoch(
    LOAD_CONFIG, use_scheduler, bath_size, batches, num_steps_disc_start
):
    """Test that the model can be trained for one epoch."""
    load_confg = OmegaConf.to_container(LOAD_CONFIG)
    load_confg["train_batch_size"] = bath_size
    load_confg["disc_start"] = num_steps_disc_start
    if use_scheduler is not None:
        load_confg["temp_scheduler"] = {
            "_target_": "optax.linear_schedule",
            "init_value": 1.0,
            "end_value": 0.5,
            "transition_steps": num_steps_disc_start + 1,
        }
    else:
        load_confg["temp_scheduler"] = None
    with tempfile.TemporaryDirectory() as dp:
        load_confg["save_dir"] = os.path.join(dp, load_confg["save_dir"])
        load_confg["log_dir"] = os.path.join(dp, load_confg["log_dir"])
        cfg = config.TrainConfig(**load_confg)

        rng = jax.random.PRNGKey(cfg.seed)
        input_shape = (cfg.train_batch_size,) + cfg.input_shape
        loader = utils.DummyDataLoader(rng, input_shape, batches, cfg.dtype)
        model = training.TrainerVQGan(cfg)
        assert model is not None
        model.init_optimizer()
        # starting from 1 because in the pipeline we count from 1
        # I know it is not informatics way but we implement based on original code
        prev_metrics = None
        for i in range(1, num_steps_disc_start + 2):
            metrics = model.train_epoch(data_loader=loader, epoch=i)
            if prev_metrics is None:
                prev_metrics = metrics
            else:
                for k, v in metrics.items():
                    assert (
                        v != prev_metrics[k]
                    ), f"Metric {k} is unchanged after training."
                if "disc_loss" in metrics and "disc_loss" in prev_metrics:
                    assert (
                        metrics["disc_loss"] < prev_metrics["disc_loss"]
                    ), "Discriminator loss is not decreasing."
                if "total_loss" in metrics and "total_loss" in prev_metrics:
                    assert (
                        metrics["total_loss"] < prev_metrics["total_loss"]
                    ), "Total loss is not decreasing."
                if use_scheduler is None:
                    assert (
                        metrics["temp"] == prev_metrics["temp"]
                    ), "Temperature is not constant."
                else:
                    assert (
                        metrics["temp"] < prev_metrics["temp"]
                    ), "Temperature is not decreasing."
                prev_metrics = metrics
            if num_steps_disc_start > i:
                assert "disc_loss" in metrics

            assert tree_all(
                tree_map(
                    lambda x, y: (x == y).all(), model.model.params, model.state.params
                )
            ), "Model parameters are not updated."
            assert tree_all(
                tree_map(
                    lambda x, y: (x == y).all(),
                    model.model_disc.params["params"],
                    model.state_disc.params,
                )
            ), "Discriminator parameters are not updated."
            assert tree_all(
                tree_map(
                    lambda x, y: (x == y).all(),
                    model.model_disc.params["batch_stats"],
                    model.state_disc.batch_stats,
                )
            ), "Discriminator batch stats are not updated."

    # clean up
    del model, cfg, load_confg, prev_metrics, metrics, rng, loader


@pytest.mark.training_long
@pytest.mark.parametrize(
    "batch_size, batches",
    [(4, 2), (4, 1), (8, 2)],
)
def test_eval_model(LOAD_CONFIG, batch_size, batches):
    """Test that the model can be evaluated."""
    load_confg = OmegaConf.to_container(LOAD_CONFIG)
    load_confg["test_batch_size"] = batch_size

    with tempfile.TemporaryDirectory() as dp:
        load_confg["save_dir"] = os.path.join(dp, load_confg["save_dir"])
        load_confg["log_dir"] = os.path.join(dp, load_confg["log_dir"])
        cfg = config.TrainConfig(**load_confg)

        rng = jax.random.PRNGKey(cfg.seed)
        input_shape = (cfg.train_batch_size,) + cfg.input_shape
        loader = utils.DummyDataLoader(rng, input_shape, batches, cfg.dtype)
        model = training.TrainerVQGan(cfg)
        assert model is not None
        model.init_optimizer()
        # test the model evaluation
        prev_metrics = model.eval_model(loader)
        assert prev_metrics is not None
        assert len(prev_metrics) > 0
        metrics = model.eval_model(loader)
        print(metrics)
        assert metrics is not None
        assert len(metrics) > 0
        for k, v in metrics.items():
            assert jnp.allclose(
                v, prev_metrics[k], atol=1e-2
            ), f"Metric {k} is not the same after evaluation."

        # tain once and test the model evaluation
        _ = model.train_epoch(data_loader=loader, epoch=-1)
        new_metrics = model.eval_model(loader)
        assert new_metrics is not None
        assert len(new_metrics) > 0
        for k, v in metrics.items():
            assert v != new_metrics[k], f"Metric {k} is unchanged after training."


@pytest.mark.training_long
@pytest.mark.parametrize(
    "batch_size, batches, epochs",
    [(2, 2, 1), (2, 1, 1), (2, 2, 2), (2, 2, 4)],
)
def test_train_model(LOAD_CONFIG, batch_size, batches, epochs):
    """Test that the model can be trained for one epoch."""
    load_confg = OmegaConf.to_container(LOAD_CONFIG)
    load_confg["train_batch_size"] = batch_size
    load_confg["test_batch_size"] = batch_size
    load_confg["disc_start"] = 1
    load_confg["temp_scheduler"] = None
    load_confg["num_epochs"] = epochs

    with tempfile.TemporaryDirectory() as dp:
        load_confg["save_dir"] = os.path.join(dp, load_confg["save_dir"])
        load_confg["log_dir"] = os.path.join(dp, load_confg["log_dir"])
        cfg = config.TrainConfig(**load_confg)

        rng = jax.random.PRNGKey(cfg.seed)
        rng, eval_rng = jax.random.split(rng)
        input_shap_train = (cfg.train_batch_size,) + cfg.input_shape
        input_shap_eval = (cfg.test_batch_size,) + cfg.input_shape
        loader_train = utils.DummyDataLoader(rng, input_shap_train, batches, cfg.dtype)
        loader_val = utils.DummyDataLoader(
            eval_rng, input_shap_eval, batches, cfg.dtype
        )
        model = training.TrainerVQGan(cfg)
        assert model is not None
        assert model.checkpoint_exists() is False

        # test the model train
        model.train_model(loader_train, loader_val)
        assert model.checkpoint_exists()

    # clean up
    del model, cfg, load_confg, rng, loader_train, loader_val


@pytest.mark.training_long
def test_train_model_distriubuted(LOAD_CONFIG):
    """Test that the model can be trained for one epoch."""
    pass


def test_simple_train_model(LOAD_CONFIG):
    """Test that the model can be trained for one epoch. Simple version"""
    batch_size = 2
    batches = 2
    epochs = 1
    load_confg = OmegaConf.to_container(LOAD_CONFIG)
    load_confg["train_batch_size"] = batch_size
    load_confg["test_batch_size"] = batch_size
    load_confg["disc_start"] = 1
    load_confg["temp_scheduler"] = None
    load_confg["num_epochs"] = epochs

    with tempfile.TemporaryDirectory() as dp:
        load_confg["save_dir"] = os.path.join(dp, load_confg["save_dir"])
        load_confg["log_dir"] = os.path.join(dp, load_confg["log_dir"])
        cfg = config.TrainConfig(**load_confg)

        rng = jax.random.PRNGKey(cfg.seed)
        rng, eval_rng = jax.random.split(rng)
        input_shap_train = (cfg.train_batch_size,) + cfg.input_shape
        input_shap_eval = (cfg.test_batch_size,) + cfg.input_shape
        loader_train = utils.DummyDataLoader(rng, input_shap_train, batches, cfg.dtype)
        loader_val = utils.DummyDataLoader(
            eval_rng, input_shap_eval, batches, cfg.dtype
        )
        model = training.TrainerVQGan(cfg)
        assert model is not None
        assert model.checkpoint_exists() is False

        # test the model train
        model.train_model(loader_train, loader_val)
        assert model.checkpoint_exists()

    # clean up
    del model, cfg, load_confg, rng, loader_train, loader_val


@pytest.mark.training_long
def test_model_with_hydra_config(LOAD_CONFIG):
    """Test that the model can be trained for one epoch."""
    pass
