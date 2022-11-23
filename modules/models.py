from typing import Callable, Optional, Tuple

import jax
import jax.numpy as jnp
from einops import rearrange
from flax import linen as nn

from modules.config import VQGANConfig


class Upsample(nn.Module):
    """Upsample the input by a factor of 2.

    Attributes:
        in_channels (int): Number of input channels.
        use_conv (bool): Whether to use a identity convolution.
        dtype (jnp.dtype): the dtype of the computation (default: float32).
    """

    in_channels: int
    use_conv: bool
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        batch, height, width, channels = x.shape
        x = jax.image.resize(
            x,
            shape=(batch, height * 2, width * 2, channels),
            method="nearest",
        )
        if self.use_conv:
            x = nn.Conv(
                features=self.in_channels,
                kernel_size=(3, 3),
                strides=(1, 1),
                padding=((1, 1), (1, 1)),
                dtype=self.dtype,
            )(x)
        return x


class Downsample(nn.Module):
    """Downsample the input by a factor of 2.

    Attributes:
        in_channels (int): Number of input channels.
        use_conv (bool): Whether to use a identity convolution.
        dtype (jnp.dtype): the dtype of the computation (default: float32).
    """

    in_channels: int
    use_conv: bool
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        if self.use_conv:
            pad = ((0, 0), (0, 1), (0, 1), (0, 0))  # pad height and width dim
            x = jnp.pad(x, pad, mode="constant", constant_values=0)
            x = nn.Conv(
                features=self.in_channels,
                kernel_size=(3, 3),
                strides=(2, 2),
                padding="VALID",
                dtype=self.dtype,
            )(x)
        else:
            x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2), padding="VALID")
        return x


class ResNetBlock(nn.Module):
    """ResNet block with optional bottleneck.

    Attributes:
        in_channels (int): number of input channels.
        out_channels (Optional[int]): number of output channels.
            If None, the output channels will be the same as the input channels.
        act_fn (Callable): activation function.
        use_conv_shortcut (bool): whether to use a convolutional shortcut.
        temb_channels (int): number of channels in the temporal embedding.
        dropout_prob (float): dropout probability.
        dtype (jnp.dtype): the dtype of the computation (default: float32).
    """

    in_channels: int
    out_channels: Optional[int] = None
    act_fn: Callable = nn.gelu
    use_conv_shortcut: bool = False
    temb_channels: int = 512
    dropout_prob: float = 0.0
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        # out_channels not specified, we will residual block will be identity
        self.out_channels_: int = (
            self.in_channels if self.out_channels is None else self.out_channels
        )

        #  First block
        self.block1 = nn.Sequential(
            [
                nn.GroupNorm(num_groups=32, epsilon=1e-6),
                self.act_fn,
                nn.Conv(
                    self.out_channels_,
                    kernel_size=(3, 3),
                    strides=(1, 1),
                    padding=((1, 1), (1, 1)),
                    dtype=self.dtype,
                ),
            ]
        )

        # Project temporary embedding to be add to hidden states
        if self.temb_channels:
            self.temb_proj = nn.Dense(self.out_channels_, dtype=self.dtype)

        # Second block
        self.block2_pre = nn.Sequential([nn.GroupNorm(num_groups=32, epsilon=1e-6), self.act_fn])
        self.block2_drop = nn.Dropout(self.dropout_prob)
        self.bloc2_conv = nn.Conv(
            self.out_channels_,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding=((1, 1), (1, 1)),
            dtype=self.dtype,
        )

        # if output x channels do not match residual channels, we need to project residual
        if self.in_channels != self.out_channels_:
            if self.use_conv_shortcut:
                # Reduce by conv(3,3) only channels with learned spatial features
                self.conv_shortcut = nn.Conv(
                    self.out_channels_,
                    kernel_size=(3, 3),
                    strides=(1, 1),
                    padding=((1, 1), (1, 1)),
                    dtype=self.dtype,
                )
            else:
                # Reduce by conv(1,1) only channels
                self.nin_shortcut = nn.Conv(
                    self.out_channels_,
                    kernel_size=(1, 1),
                    strides=(1, 1),
                    padding="VALID",
                    dtype=self.dtype,
                )

    def __call__(
        self,
        x: jnp.ndarray,
        temb: Optional[jnp.ndarray] = None,
        deterministic: bool = True,
    ) -> jnp.ndarray:
        """Forward pass of the block.
        Args:
            x (jnp.ndarray): input tensor.
            temb (Optional[int], optional): temporal embedding. Defaults to None.
            deterministic (bool, optional): deterministic flag. Defaults to True.
        Returns:
            output tensor with the out_channels dimension as the last dimension (C).
        """
        residual = x
        x = self.block1(x)
        if temb is not None:
            # transform temporal embedding to match hidden states [BxT] -> [BxC] -> [Bx1x1xC]
            x = x + self.temb_proj(self.act_fn(temb))[:, None, None, :]

        x = self.block2_pre(x)
        x = self.block2_drop(x, deterministic=deterministic)
        x = self.bloc2_conv(x)
        if self.in_channels != self.out_channels_:
            if self.use_conv_shortcut:
                residual = self.conv_shortcut(residual)
            else:
                residual = self.nin_shortcut(residual)

        return x + residual


class AttnBlock(nn.Module):
    """Attention block.

    Attributes:
        in_channels (int): number of input channels.
        dtype (jnp.dtype): the dtype of the computation (default: float32).
    """

    in_channels: int
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Forward pass of the block.
        Args:
            x (jnp.ndarray): input tensor.
        Returns:
            output tensor with the same shape as the input.
        """
        h_ = x
        h_ = nn.GroupNorm(num_groups=32, epsilon=1e-6)(h_)
        # get query, key, value
        q = nn.Conv(
            self.in_channels,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding="VALID",
            dtype=self.dtype,
        )(h_)
        k = nn.Conv(
            self.in_channels,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding="VALID",
            dtype=self.dtype,
        )(h_)
        v = nn.Conv(
            self.in_channels,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding="VALID",
            dtype=self.dtype,
        )(h_)

        # compute attention
        q = rearrange(q, "B H W C -> B (H W) C")
        k = rearrange(k, "B H W C -> B (H W) C")
        w_ = jnp.einsum("bqc,bkc->bqk", q, k)
        w_ *= int(x.shape[-1]) ** (-0.5)
        w_ = nn.softmax(w_, axis=2)

        # attend to values
        v = rearrange(v, "B H W C -> B (H W) C")
        h_ = jnp.einsum("bkc,bqk->bqc", v, w_)
        h_ = rearrange(h_, "B (H W) C -> B H W C", H=x.shape[1], W=x.shape[2])

        h_ = nn.Conv(
            self.in_channels,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding="VALID",
            dtype=self.dtype,
        )(h_)
        h_ += x
        return h_


class UpsamplingBlock(nn.Module):
    """Upsampling block for Decoder.

    Attributes:
        config (VQGANConfig): the config of the model.
        curr_res (int): current resolution.
        blck_idx (int): current block index.
        act_fn (Callable): activation function.
        dtype (jnp.dtype): the dtype of the computation (default: float32).
    """

    config: VQGANConfig
    curr_res: int
    block_idx: int
    act_fn: Callable = nn.gelu
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        # get input numb of channels based on config and current block index.
        # Looking in reverse on self.config.ch_mult variable.
        if self.block_idx == self.config.num_resolutions - 1:
            block_in: int = self.config.ch * self.config.ch_mult[-1]
        else:
            block_in: int = self.config.ch * self.config.ch_mult[self.block_idx + 1]

        # get output numb of channels based on config and current block index
        block_out: int = self.config.ch * self.config.ch_mult[self.block_idx]
        # temporary embedding channels UpsamplingBlock don't use temporal embedding
        self.temb_ch: int = 0

        # build blocks
        res_blocks = []
        attn_blocks = []
        for _ in range(self.config.num_res_blocks + 1):
            assert block_in % 32 == 0, "block_in must be divisible by 32 for GroupNorm"
            res_blocks.append(
                ResNetBlock(
                    block_in,
                    block_out,
                    act_fn=self.act_fn,
                    temb_channels=self.temb_ch,
                    dropout_prob=self.config.dropout,
                    dtype=self.dtype,
                )
            )
            # after channel resize rest are identity blocks.
            block_in = block_out
            # check if we need to add attention block based on configs
            if self.curr_res in self.config.attn_resolutions:
                assert block_in % 32 == 0, "block_in must be divisible by 32 for GroupNorm"
                attn_blocks.append(AttnBlock(block_in, dtype=self.dtype))

        self.blocks = res_blocks
        self.attns = attn_blocks

        # upsample if not first block
        self.upsample = None
        if self.block_idx != 0:
            self.upsample = Upsample(block_in, self.config.resamp_with_conv, dtype=self.dtype)

    def __call__(
        self, x: jnp.ndarray, temb: Optional[int] = None, deterministic: bool = True
    ) -> jnp.ndarray:
        """Forward pass of the block.
        Args:
            x (jnp.ndarray): input tensor.
            temb (Optional[int], optional): temporal embedding. Defaults to None.
            deterministic (bool, optional): deterministic flag. Defaults to True.
        """
        assert temb is None, "UpsamplingBlock don't use temporal embedding"
        for i, res_block in enumerate(self.blocks):
            x = res_block(x, temb, deterministic=deterministic)

            if self.attns:
                x = self.attns[i](x)

        if self.upsample is not None:
            x = self.upsample(x)

        return x


class DownsamplingBlock(nn.Module):
    """Downsampling block for Encoder.

    Attributes:
        config (VQGANConfig): the config of the model.
        curr_res (int): current resolution.
        blck_idx (int): current block index.
        act_fn (Callable): activation function.
        dtype (jnp.dtype): the dtype of the computation (default: float32).
    """

    config: VQGANConfig
    curr_res: int
    block_idx: int
    act_fn: Callable = nn.gelu
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        # get input and output numb of channels based on config and current block index.
        in_ch_mult: Tuple[int] = (1,) + tuple(self.config.ch_mult)
        block_in: int = self.config.ch * in_ch_mult[self.block_idx]
        block_out: int = self.config.ch * self.config.ch_mult[self.block_idx]

        # temporary embedding channels
        self.temb_ch: int = 0

        # build blocks
        res_blocks = []
        attn_blocks = []
        for _ in range(self.config.num_res_blocks):
            assert block_in % 32 == 0, "block_in must be divisible by 32 for GroupNorm"
            res_blocks.append(
                ResNetBlock(
                    block_in,
                    block_out,
                    act_fn=self.act_fn,
                    temb_channels=self.temb_ch,
                    dropout_prob=self.config.dropout,
                    dtype=self.dtype,
                )
            )
            # after channel downsample rest are identity blocks.
            block_in = block_out
            # check if we need to add attention block based on configs
            if self.curr_res in self.config.attn_resolutions:
                assert block_in % 32 == 0, "block_in must be divisible by 32 for GroupNorm"
                attn_blocks.append(AttnBlock(block_in, dtype=self.dtype))

        self.blocks = res_blocks
        self.attns = attn_blocks

        # downsample if not last block
        self.downsample = None
        if self.block_idx != self.config.num_resolutions - 1:
            self.downsample = Downsample(block_in, self.config.resamp_with_conv, dtype=self.dtype)

    def __call__(
        self, x: jnp.ndarray, temb: Optional[int] = None, deterministic: bool = True
    ) -> jnp.ndarray:
        """Forward pass of the block.
        Args:
            x (jnp.ndarray): input tensor.
            temb (Optional[int], optional): temporal embedding. Defaults to None.
            deterministic (bool, optional): deterministic flag. Defaults to True.
        """
        assert temb is None, "DownsamplingBlock don't use temporal embedding"
        for i, res_block in enumerate(self.blocks):
            x = res_block(x, temb, deterministic=deterministic)

            if self.attns:
                x = self.attns[i](x)

        if self.downsample is not None:
            x = self.downsample(x)

        return x


class MidBlock(nn.Module):
    """Mid block for Encoder and Decoder.

    Attributes:
        in_channels (int): number of input channels.
        act_fn (str): activation function.
        temb_channels (int): number of channels for temporal embedding.
        dropout_prob (float): dropout probability.
        dtype (jnp.dtype): the dtype of the computation (default: float32).
    """

    in_channels: int
    act_fn: Callable
    temb_channels: int
    dropout_prob: float
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(
        self, x: jnp.ndarray, temb: Optional[int] = None, deterministic: bool = True
    ) -> jnp.ndarray:
        """BxWxHxC --ResNet--> BxWxHxC --Attn--> BxWxHxC --ResNet--> BxWxHxC

        Args:
            x (jnp.ndarray): input tensor.
            temb (Optional[int], optional): temporal embedding. Defaults to None.
            deterministic (bool, optional): deterministic flag. Defaults to True.
        """
        assert self.in_channels % 32 == 0, "block_in must be divisible by 32 for GroupNorm"
        x = ResNetBlock(
            self.in_channels,
            self.in_channels,
            act_fn=self.act_fn,
            temb_channels=self.temb_channels,
            dropout_prob=self.dropout_prob,
            dtype=self.dtype,
        )(x, temb, deterministic=deterministic)
        x = AttnBlock(self.in_channels, dtype=self.dtype)(x)
        x = ResNetBlock(
            self.in_channels,
            self.in_channels,
            act_fn=self.act_fn,
            temb_channels=self.temb_channels,
            dropout_prob=self.dropout_prob,
            dtype=self.dtype,
        )(x, temb, deterministic=deterministic)

        return x


class Encoder(nn.Module):
    """
    Encoder of VQ-GAN to map input batch of images to latent space.
    Dimension Transformations originally:
    256x256x3 --Conv2d--> 256x256x32
    for loop:
        --DownsamplingBlock--> 128x128x64
        --DownsamplingBlock--> 64x64x128
        --DownsamplingBlock--> 32x32x256
        --DownsamplingBlock--> 32x32x512

    --MidBlock--> 32x32x512
    --GroupNorm-->
    --nonlinear-->
    --Conv2d-> 32x32x256

    Attributes:
        config (VQGANConfig): the config of the model.
        act_fn (str): activation function.
        dtype (jnp.dtype): the dtype of the computation (default: float32).
    """

    config: VQGANConfig
    act_fn: Callable = nn.gelu
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x: jnp.ndarray, deterministic: bool = True) -> jnp.ndarray:
        # time embedding
        temb_ch: int = 0
        temb: Optional[int] = None

        # downsampling
        x = nn.Conv(
            self.config.ch,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding=((1, 1), (1, 1)),
            dtype=self.dtype,
        )(x)
        curr_res = self.config.resolution
        for i in range(self.config.num_resolutions):
            x = DownsamplingBlock(self.config, curr_res, block_idx=i, dtype=self.dtype)(
                x, temb, deterministic=deterministic
            )

            # update resolution if not bottleneck
            curr_res = curr_res // 2 if i != self.config.num_resolutions - 1 else curr_res
        # middle
        mid_channels = self.config.ch * self.config.ch_mult[-1]
        x = MidBlock(
            mid_channels,
            act_fn=self.act_fn,
            temb_channels=temb_ch,
            dropout_prob=self.config.dropout,
            dtype=self.dtype,
        )(x, temb, deterministic=deterministic)

        # end CFN
        x = nn.GroupNorm(num_groups=32, dtype=self.dtype)(x)
        x = self.act_fn(x)
        x = nn.Conv(
            2 * self.config.z_channels if self.config.double_z else self.config.z_channels,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding=((1, 1), (1, 1)),
            dtype=self.dtype,
        )(x)

        return x


class Decoder(nn.Module):
    """
    Decoder of VQ-GAN to map input batch of latent space to images.
    Dimension Transformations originally:
    32x32x256 --Conv2d--> 32x32x512
    --MidBlock--> 32x32x512
    for loop:
        --UpsamplingBlock--> 64x64x256
        --UpsamplingBlock--> 128x128x128
        --UpsamplingBlock--> 256x256x64
        --UpsamplingBlock--> 256x256x32
    --GroupNorm-->
    --nonlinear-->
    --Conv2d-> 256x256x3

    Attributes:
        config (VQGANConfig): the config of the model.
        act_fn (str): activation function.
        dtype (jnp.dtype): the dtype of the computation (default: float32).
    """

    config: VQGANConfig
    act_fn: Callable = nn.gelu
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, z: jnp.ndarray, deterministic: bool = True) -> jnp.ndarray:
        # time embedding
        temb_ch: int = 0
        temb: Optional[int] = None

        # compute in_ch_mult and block_in at lowest res
        block_in = self.config.ch * self.config.ch_mult[self.config.num_resolutions - 1]

        # z_channel to block_in
        x = nn.Conv(
            block_in,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding=((1, 1), (1, 1)),
            dtype=self.dtype,
        )(z)

        # middle
        x = MidBlock(
            block_in,
            act_fn=self.act_fn,
            temb_channels=temb_ch,
            dropout_prob=self.config.dropout,
            dtype=self.dtype,
        )(x, temb, deterministic=deterministic)

        # upsampling
        # compute curr_res at lowest res
        curr_res = self.config.resolution // 2 ** (self.config.num_resolutions - 1)
        for i in reversed(range(self.config.num_resolutions)):
            x = UpsamplingBlock(self.config, curr_res, block_idx=i, dtype=self.dtype)(
                x, temb, deterministic=deterministic
            )

            # update resolution if not end
            curr_res = curr_res * 2 if i != self.config.num_resolutions - 1 else curr_res

        # end
        if self.config.give_pre_end:
            return x

        # CFN
        x = nn.GroupNorm(num_groups=32, dtype=self.dtype)(x)
        x = self.act_fn(x)
        x = nn.Conv(
            self.config.out_ch,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding=((1, 1), (1, 1)),
            dtype=self.dtype,
        )(x)

        return x
