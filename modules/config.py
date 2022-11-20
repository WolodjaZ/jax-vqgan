from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple

import jax
import jax.numpy as jnp
import optax
from hydra.utils import instantiate
from transformers import PretrainedConfig


class VQGANConfig(PretrainedConfig):
    """Configuration class to store the configuration of a VQGAN model.

    Args:
        ch (int): number of channels.
        out_ch (int): number of output channels (RGB).
        in_channels (int): number of input channels (RGB).
        num_res_blocks (int): number of residual blocks.
        resolution (int): resolution of the input image (256x256).
        z_channels (int): number of channels in the latent space.
        ch_mult (Tuple[int]): channel multiplier for each layer.
        attn_resolutions (Tuple[int]): resolutions at which to apply attention.
        n_embed (int): number of embeddings, unique codes in the latent space.
        embed_dim (int): dimension of embedding from Encoder.
        dropout (float): dropout rate.
        double_z (bool): whether to double the latent space for.
        resamp_with_conv (bool): whether to use convolutions for upsampling.
        use_gumbel (bool): whether to use gumbel softmax for quantization.
        gumb_temp (float): temperature for gumbel softmax.
        act_name (str): activation function name to use.
        give_pre_end (bool): whether to give the pre-end layer for the decoder.
        kwargs: keyword arguments passed along to the super class.
    """

    def __init__(
        self,
        ch: int = 128,
        out_ch: int = 3,
        in_channels: int = 3,
        num_res_blocks: int = 2,
        resolution: int = 256,
        z_channels: int = 256,
        ch_mult: Tuple[int, ...] = tuple([1, 1, 2, 2, 4]),
        attn_resolutions: Tuple[int] = (16,),
        n_embed: int = 1024,
        embed_dim: int = 256,
        dropout: float = 0.0,
        double_z: bool = False,
        resamp_with_conv: bool = True,
        use_gumbel: bool = False,
        gumb_temp: float = 1.0,
        beta: float = 0.25,
        kl_weight: float = 5e-4,
        act_name: str = "swish",
        give_pre_end: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.ch = ch
        self.out_ch = out_ch
        self.in_channels = in_channels
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.z_channels = z_channels
        self.ch_mult = list(ch_mult)
        self.attn_resolutions = list(attn_resolutions)
        self.n_embed = n_embed
        self.embed_dim = embed_dim
        self.dropout = dropout
        self.double_z = double_z
        self.resamp_with_conv = resamp_with_conv
        self.use_gumbel = use_gumbel
        self.gumb_temp = gumb_temp
        self.beta = beta
        self.kl_weight = kl_weight
        self.act_name = act_name
        self.give_pre_end = give_pre_end
        self.num_resolutions = len(ch_mult)


class DiscConfig(PretrainedConfig):
    """Configuration class to store the configuration of a Discriminator model.

    Args:
        PretrainedConfig (_type_): _description_
    """

    def __init__(
        self,
        input_last_dim: int = 3,
        output_last_dim: int = 1,
        resolution: int = 256,
        ndf: int = 64,
        n_layers: int = 3,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.input_last_dim = input_last_dim
        self.output_last_dim = output_last_dim
        self.resolution = resolution
        self.ndf = ndf
        self.n_layers = n_layers


@dataclass
class TrainConfig:
    """Configuration class to store the configuration of a train model.

    Arguments:
        model_name (str): name of the model to train. Used for saving and logging.
        model_hparams (VQGANConfig): model hyperparameters. Check VQGANConfig for more details.
        disc_hparams (DiscConfig): discriminator hyperparameters.
            Check DiscConfig for more details.
        save_dir (str): directory to save the model.
        log_dir (str): directory to save the logs for tensorboard.
        check_val_every_n_epoch (int): number of epochs to run validation.
        input_shape (Tuple[int, int, int]): shape of the input image (H, W, C).
        train_batch_size (int): batch size for training.  Only informative.
        test_batch_size (optional, int): batch size for testing. Only informative.
        codebook_weight (float): weight for the codebook loss (Quantizer part).
        monitor (str): metric to monitor for saving best model.
        recon_loss (str): reconstruction loss to use.
        disc_loss (str): discriminator loss to use.
        disc_weight (float): weight for the discriminator loss.
        num_epochs (int): number of epochs to train.
        dtype (str): dtype to use for training.
            Supported ["float32", "float16", "float16", "bfloat16"].
        distributed (bool): whether to use distributed training.
        seed (int): seed for random number generation.
        optimizer (str): optimizer to use for training. Structure needs to be
            optax Optimizer (Check optax for more details) with '__target__' parameter,
            for specifing optax optimizer, and 'kwargs' parameter for passing to optimizer.
            check config_test.yaml for example.
        optimizer_disc (str): optimizer to use for discriminator training. Similar to optimizer.
        disc_start (int): number of epochs to past to start using the discriminator.
        temp_scheduler (optional): temperature scheduler to use for training. Similar to optimizer
            but uses optax scheduler with '__target__' parameter, for specifing optax scheduler.
            if None, then no scheduler is used. Check config_test.yaml for example.
    """

    model_name: str
    model_hparams: VQGANConfig
    disc_hparams: DiscConfig
    save_dir: str
    log_dir: str
    check_val_every_n_epoch: int
    input_shape: Tuple[int, int, int]
    train_batch_size: int
    test_batch_size: int
    codebook_weight: float
    monitor: str
    recon_loss: str
    disc_loss: str
    disc_weight: float
    num_epochs: int
    dtype: jnp.dtype
    distributed: bool
    seed: int
    optimizer: optax.GradientTransformation
    optimizer_disc: optax.GradientTransformation
    disc_start: int
    temp_scheduler: Optional[Callable]

    def __post_init__(self):
        # load model hparams
        self.model_hparams = instantiate(self.model_hparams)
        assert type(self.model_hparams) == VQGANConfig
        # load disc hparams
        self.disc_hparams = instantiate(self.disc_hparams)
        assert type(self.disc_hparams) == DiscConfig
        # conver shape list to tuple shape
        self.input_shape = tuple(self.input_shape)
        assert len(self.input_shape) == 3
        # set dtype
        if self.dtype == "float64":
            self.dtype = jnp.float64
        elif self.dtype == "float32":
            self.dtype = jnp.float32
        elif self.dtype == "float16":
            self.dtype = jnp.float16
        elif self.dtype == "bfloat16":
            self.dtype = jnp.bfloat16
        else:
            raise ValueError(
                f"""Invalid dtype {self.dtype}
                             expected one of float64, float32, float16, bfloat16"""
            )
        # if distributed: chunk the dataset
        if self.distributed:
            self.train_batch_size = self.train_batch_size // jax.device_count()
            self.test_batch_size = self.test_batch_size // jax.device_count()
        # instantiate the optimizer
        self.optimizer = instantiate(self.optimizer)
        assert type(self.optimizer) == optax.GradientTransformation
        # instantiate the optimizer for discriminator
        self.optimizer_disc = instantiate(self.optimizer_disc)
        assert type(self.optimizer_disc) == optax.GradientTransformation
        # if optimizer is a dict, instantiate it
        if self.temp_scheduler is not None:
            self.temp_scheduler: Callable = instantiate(self.temp_scheduler)
            assert hasattr(self.temp_scheduler, "__call__")


@dataclass
class DataParams:
    """Train and test data parameters.
    Arguments:
        batch_size (int): batch size for training.
        shuffle (bool): whether to shuffle the dataset.
    """

    batch_size: int
    shuffle: bool


@dataclass
class DataConfig:
    """Data configuration class.
    Arguments:
        train_params (DataParams): training data parameters. Check DataParams for more details.
        test_params (DataParams): testing data parameters. Check DataParams for more details.
        dataset_name (str): name of the dataset to use. Currently only supports "voc".
        dataset_root (str): root directory of the dataset.
        transform (optional, dict): transform to apply to the dataset. Default None for no transf.
            Transform dict comes from albumentations library. Check albumentations for more details.
            Loading transform should proceed with albumentations.from_dict method.
        size (int): size of image width and height.

    """

    train_params: DataParams
    test_params: DataParams
    dataset_name: str = ""
    dataset_root: str = ""
    transform: Optional[Dict[str, Any]] = None
    size: int = 224

    def __post_init__(self):
        # set train_params and test_params
        self.train_params = DataParams(**self.train_params)
        self.test_params = DataParams(**self.test_params)


@dataclass
class LoadConfig:
    """Load configuration class to store the configuration of a train model and data.
    Main configuration class to be used for training.

    Arguments:
        train (DataConfig): data configuration.
        data (TrainConfig): (DataConfig) training configuration.
    """

    train: TrainConfig
    data: DataConfig

    def __post_init__(self):
        self.train = TrainConfig(**self.train)
        self.data = DataConfig(**self.data)

        # set batch_size
        self.train.train_batch_size = self.data.train_params.batch_size
        self.train.test_batch_size = self.data.test_params.batch_size
