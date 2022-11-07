from dataclasses import dataclass
from typing import Any, Tuple

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
    disc_weight: float
    num_epochs: int
    dtype: str
    distributed: bool
    seed: int
    optimizer: Any
    optimizer_disc: Any
