from functools import partial
from typing import Callable, Optional, Set, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict, freeze, unfreeze
from flax.traverse_util import flatten_dict, unflatten_dict
from transformers import PretrainedConfig
from transformers.modeling_flax_utils import FlaxPreTrainedModel

from modules.config import DiscConfig, VQGANConfig
from modules.models import Decoder, Encoder


def quick_gelu(x):
    return x * jax.nn.sigmoid(1.702 * x)


ACTFUN = {
    "gelu": partial(nn.gelu, approximate=False),
    "relu": nn.relu,
    "silu": nn.swish,
    "swish": nn.swish,
    "gelu_new": partial(nn.gelu, approximate=True),
    "quick_gelu": quick_gelu,
}


class VectorQuantizer(nn.Module):
    """
    Discretization bottleneck part of the VQ-VAE.
    Module get z lattent vector (Encoder output)
    and maps it to a discrete one-hot vector
    that is the index of the closest embedding vector e_j
    z (continuous) -> z_q (discrete)
    z.shape = (batch, height, width, channel)
    quantization pipeline:
        1. get encoder input (B,H,W,C)
        2. flatten input to (B*H*W,C)

    See:
        https://github.com/MishaLaskin/vqvae/blob/d761a999e2267766400dc646d82d3ac3657771d4/models/quantizer.py

    Attributes:
        config (VQGANConfig): the config of the model.
        dtype (jnp.dtype): the dtype of the computation for embeddings (default: float32).

    Config Attributes:
        n_embed (int) : number of embeddings.
        emb_dim (int): dimension of embedding.
        beta (float): weight of commitment loss.
    """

    config: VQGANConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        init_embbeding = jax.nn.initializers.uniform(
            scale=-1.0 / self.config.n_embed, dtype=self.dtype
        )
        self.embedding = nn.Embed(
            self.config.n_embed,
            self.config.embed_dim,
            embedding_init=init_embbeding,
            dtype=self.dtype,
        )

    def __call__(self, z: jnp.ndarray) -> Tuple[jnp.ndarray, float, jnp.ndarray]:
        # flatten z
        z_flatten = z.reshape(-1, self.config.embed_dim)

        # dummy op to init the weights, so we can access them below
        self.embedding(jnp.ones((1, 1), dtype="i4"))

        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z
        emb_weights = self.variables["params"]["embedding"]["embedding"]
        distance = (
            jnp.sum(z_flatten**2, axis=1, keepdims=True)
            + jnp.sum(emb_weights**2, axis=1)
            - 2 * jnp.dot(z_flatten, emb_weights.T)
        )

        # get quantized latent vectors
        min_encoding_indices = jnp.argmin(distance, axis=1)
        z_q = self.embedding(min_encoding_indices).reshape(z.shape)

        # reshape to (batch, num_tokens)
        min_encoding_indices = min_encoding_indices.reshape(z.shape[0], -1)

        # compute the codebook_loss (q_loss)
        q_loss = self.config.beta * jnp.mean(
            (jax.lax.stop_gradient(z_q) - z) ** 2
        ) + jnp.mean((z_q - jax.lax.stop_gradient(z)) ** 2)

        # here we return the embeddings and indices
        return z_q, q_loss, min_encoding_indices

    @staticmethod
    def get_codebook_entry(
        params: FrozenDict, indices: jnp.ndarray, shape: Optional[Tuple[int]] = None
    ) -> jnp.ndarray:
        """Get the codebook entry for a given index.
        Input is expected to be of shape (batch, num_tokens)"""
        # indices are expected to be of shape (batch, num_tokens)
        # get quantized latent vectors
        B = indices.shape[0]
        indices = indices.reshape(
            -1,
        )
        emb_weights = params["embedding"]["embedding"]
        z_q = jnp.take(emb_weights, indices, axis=0).reshape(B, -1)
        # z_q = self.embedding(indices) can't be used because of not accessibility of self.embedding

        if shape is not None:
            z_q = z_q.reshape(shape)

        return z_q


class GumbelQuantize(nn.Module):
    """Gumbel Softmax trick quantizer
    Categorical Reparameterization with Gumbel-Softmax, Jang et al. 2016.
    z (continuous) -> z_q (discrete)
    z.shape = (batch, height, width, channel)
    quantization pipeline:
        1. get encoder input (B,H,W,C)
        2. get logits(prob) of input (B,H,W,n_embed)

    See:
        https://arxiv.org/abs/1611.01144

    Attributes:
        config (VQGANConfig): the config of the model.
        dtype (jnp.dtype): the dtype of the computation (default: float32).

    Config Attributes:
        n_embed (int) : number of embeddings.
        emb_dim (int): dimension of embedding.
        kl_weight (float): weight of kl loss.
    """

    config: VQGANConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        # Project the input to the embedding space
        self.proj = nn.Conv(
            self.config.n_embed,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding="VALID",
            dtype=self.dtype,
        )
        # Embeddings (codebook)
        init_embbeding = jax.nn.initializers.uniform(
            scale=-1.0 / self.config.n_embed, dtype=self.dtype
        )
        self.embedding = nn.Embed(
            self.config.n_embed,
            self.config.embed_dim,
            embedding_init=init_embbeding,
            dtype=self.dtype,
        )

    def __call__(self, z: jnp.ndarray) -> Tuple[jnp.ndarray, float, jnp.ndarray]:
        # project z to get logits
        logits = self.proj(z)

        # given logits, sample from the Gumbel-Softmax distribution
        gumbel_rng = self.make_rng("gumbel")
        gumbels = jax.random.gumbel(gumbel_rng, logits.shape, dtype=self.dtype)
        indicies_prob = nn.softmax((logits + gumbels) / self.config.gumb_temp, axis=-1)

        # dummy op to init the weights, so we can access them below
        self.embedding(jnp.ones((1, 1), dtype="i4"))

        # get quantized latent vectors
        emb_weights = self.variables["params"]["embedding"]["embedding"]
        z_q = jnp.einsum("bhwp,pd->bhwd", indicies_prob, emb_weights).reshape(z.shape)

        # get indices [BxHxWxP] -> [BxH*W]
        indices = jnp.argmax(indicies_prob, axis=-1).reshape(z.shape[0], -1)

        # compute the codebook_loss (q_loss)
        qy = nn.softmax(logits)
        q_loss = self.config.kl_weight * jnp.mean(
            jnp.sum(qy * jnp.log(qy * self.config.n_embed + 1e-10), axis=-1)
        )

        # here we return the embeddings, indices and logits (for loss)
        return z_q, q_loss, indices

    @staticmethod
    def get_codebook_entry(
        params: FrozenDict, indices: jnp.ndarray, shape: Optional[Tuple[int]] = None
    ) -> jnp.ndarray:
        """Get the codebook entry for a given index.
        Input is expected to be of shape (batch, num_tokens)"""
        # indices are expected to be of shape (batch, num_tokens)
        # get quantized latent vectors
        B = indices.shape[0]
        indices = indices.reshape(
            -1,
        )
        emb_weights = params["embedding"]["embedding"]
        z_q = jnp.take(emb_weights, indices, axis=0).reshape(B, -1)
        # z_q = self.embedding(indices) can't be used because of not accessibility of self.embedding

        if shape is not None:
            z_q = z_q.reshape(shape)

        return z_q


class VQModule(nn.Module):
    """VQ-VAE module.
    See:
        https://arxiv.org/abs/1711.00937v2

    Attributes:
        config (VQGANConfig): the config of the model.
        dtype (jnp.dtype): the dtype of the computation (default: float32).
    """

    config: VQGANConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        """Setup the VQ-VAE module."""
        # Set activation function
        act_fn: Callable = ACTFUN[self.config.act_name]
        # Encoder
        self.encoder = Encoder(self.config, act_fn=act_fn, dtype=self.dtype)
        # Map last channel of encoder to embedding dim for VQ
        self.pre_quantizer = nn.Conv(
            self.config.embed_dim,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding="VALID",
            dtype=self.dtype,
        )
        # Which quantizer to use
        if self.config.use_gumbel:
            self.quantizer = GumbelQuantize(self.config, dtype=self.dtype)
        else:
            self.quantizer = VectorQuantizer(self.config, dtype=self.dtype)
        # Map last channel of VQ to z channels dim
        self.post_quantizer = nn.Conv(
            self.config.z_channels,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding="VALID",
            dtype=self.dtype,
        )
        # Decoder
        self.decoder = Decoder(self.config, act_fn=act_fn, dtype=self.dtype)

    def encode(
        self, x: jnp.ndarray, deterministic: bool = True
    ) -> Tuple[jnp.ndarray, float, jnp.ndarray]:
        """Encode the input.
        Args:
            x (jnp.ndarray): the input to the encoder.
        Returns:
            Tuple[jnp.ndarray, float, jnp.ndarray]: the encoded input, the loss and the indices.
        """
        # Encoder
        z = self.encoder(x, deterministic=deterministic)
        # Pre-quantizer
        z = self.pre_quantizer(z)
        # Quantizer
        z_q, q_loss, indices = self.quantizer(z)
        # Post-quantizer
        z_q = self.post_quantizer(z_q)
        return z_q, q_loss, indices

    def decode(self, z_q: jnp.ndarray, deterministic: bool = True) -> jnp.ndarray:
        """Decode the quantized latent vector.

        Args:
            z_q (jnp.ndarray): the quantized latent vector.
            deterministic (bool, optional): for Dropout. Defaults to True.

        Returns:
            jnp.ndarray: the reconstructed image.
        """
        # Post-quantizer
        z_q = self.post_quantizer(z_q)
        # Decoder
        x_recon = self.decoder(z_q, deterministic=deterministic)
        return x_recon

    def decode_code(self, code: jnp.ndarray, z_shape: Tuple[int, ...]) -> jnp.ndarray:
        """Decode already created z_code"""
        params = self.variables["params"]["quantizer"]
        z_q = self.quantizer.get_codebook_entry(
            params=params, indices=code, shape=z_shape
        )
        x = self.decode(z_q, deterministic=True)
        return x

    def update_temperature(self, temperature: float) -> float:
        """Update the temperature of the Gumbel-Softmax distribution.

        Args:
            temperature (float): the new temperature of the Gumbel-Softmax distribution
        Returns:
            float: the new temperature of the Gumbel-Softmax distribution
        """
        self.quantizer.config.gumb_temp = temperature
        return self.quantizer.config.gumb_temp

    def __call__(
        self, x: jnp.ndarray, deterministic: bool = True
    ) -> Tuple[jnp.ndarray, jnp.ndarray, float, jnp.ndarray]:
        z_q, q_loss, indices = self.encode(x, deterministic=deterministic)
        x_recon = self.decode(z_q, deterministic=deterministic)
        return x_recon, z_q, q_loss, indices


class VQGANPreTrainedModel(FlaxPreTrainedModel):
    """An abstract class to handle weights initialization and a simple interface
    for downloading and loading pretrained models.

    Arguments:
        module_class (nn.Module): a class derived from nn.Module
            that defines the model's core computation.
        config_class (PretrainedConfig): a class derived from PretrainedConfig

    """

    module_class: nn.Module = None
    config_class: PretrainedConfig = None

    def __init__(
        self,
        config: PretrainedConfig = VQGANConfig(),
        input_shape: Tuple = (1, 256, 256, 3),
        seed: int = 0,
        dtype: jnp.dtype = jnp.float32,
        _do_init: bool = True,
        **kwargs,
    ):
        """Initialize the model.

        Args:
            config (PretrainedConfig, optional): the config of the model. Defaults to VQGANConfig.
            input_shape (Tuple, optional): the input shape of the model.
                Defaults to (1, 256, 256, 3).
            seed (int, optional): the seed of the model. Defaults to 0.
            dtype (jnp.dtype, optional): the dtype of the computation. Defaults to jnp.float32.
            _do_init (bool, optional): whether to initialize the model. Defaults to True.
        """
        self._missing_keys: Set[str] = set()
        assert isinstance(
            config, self.config_class
        ), f"""config: {config} has to be an instance
                                                            of {self.config_class}"""
        module = self.module_class(config=config, dtype=dtype, **kwargs)
        self.seed = seed
        self.dtype = dtype
        super().__init__(
            config,
            module,
            input_shape=input_shape,
            seed=seed,
            dtype=dtype,
            _do_init=_do_init,
        )

    def init_weights(
        self, rng: jax.random.PRNGKey, input_shape: Tuple, params: FrozenDict = None
    ) -> FrozenDict:
        """Initialize the weights of the model. Get the params

        Args:
            rng (jax.random.PRNGKey): the random number generator.
            input_shape (Tuple): the input shape of the model.
            params (FrozenDict, optional): the params of the model. Defaults to None.

        Returns:
            FrozenDict: initialized params of the model.
        """
        # initialize model
        input_x = jnp.zeros(input_shape, dtype=self.dtype)
        params_rng, dropout_rng, gumble_rng = jax.random.split(rng, num=3)
        rngs = {"params": params_rng, "dropout": dropout_rng, "gumbel": gumble_rng}

        random_params = self.module.init(rngs, input_x, True)["params"]

        # If params provided find unitialized params and replace with provided params
        if params is not None:
            random_params = flatten_dict(unfreeze(random_params))
            params = flatten_dict(unfreeze(params))
            for missing_key in self._missing_keys:
                params[missing_key] = random_params[missing_key]
            self._missing_keys = set()
            return freeze(unflatten_dict(params))
        else:
            return random_params

    def encode(
        self,
        pixel_values: jnp.ndarray,
        params: Optional[FrozenDict] = None,
        dropout_rng: Optional[jax.random.PRNGKey] = None,
        gumble_rng: Optional[jax.random.PRNGKey] = None,
        train: bool = False,
    ) -> Tuple[jnp.ndarray, float, jnp.ndarray]:
        """Encode the input.
        Args:
            pixel_values (jnp.ndarray): the input to the encoder.
            params (Optional[FrozenDict], optional): the params of the model. Defaults to None.
            dropout_rng (Optional[jax.random.PRNGKey], optional): the dropout rng. Defaults to None.
            gumble_rng (Optional[jax.random.PRNGKey], optional): the gumbel rng. Defaults to None.
            train (bool, optional): Training or inference mode. Defaults to False.
        """
        # Handle any PRNG if needed
        rngs = {"dropout": dropout_rng} if dropout_rng is not None else {}
        rngs["gumbel"] = gumble_rng if gumble_rng is not None else {}
        return self.module.apply(
            {"params": params or self.params},
            pixel_values,
            not train,
            rngs=rngs,
            method=self.module.encode,
        )

    def decode(
        self,
        z: jnp.ndarray,
        params: Optional[FrozenDict] = None,
        dropout_rng: Optional[jax.random.PRNGKey] = None,
        gumble_rng: Optional[jax.random.PRNGKey] = None,
        train: bool = False,
    ) -> jnp.ndarray:
        """Decode the latent vector.

        Args:
            z (jnp.ndarray): the latent vector.
            params (Optional[FrozenDict], optional): the params of the model. Defaults to None.
            dropout_rng (Optional[jax.random.PRNGKey], optional): the dropout rng. Defaults to None.
            gumble_rng (Optional[jax.random.PRNGKey], optional): the gumbel rng. Defaults to None.
            train (bool, optional): Training or inference mode. Defaults to False.

        Returns:
            jnp.ndarray: the decoded image.
        """
        # Handle any PRNG if needed
        rngs = {"dropout": dropout_rng} if dropout_rng is not None else {}
        rngs["gumbel"] = gumble_rng if gumble_rng is not None else {}
        return self.module.apply(
            {"params": params or self.params},
            z,
            not train,
            rngs=rngs,
            method=self.module.decode,
        )

    def decode_code(
        self,
        indices: jnp.ndarray,
        z_shape: Tuple[int, ...],
        params: Optional[FrozenDict] = None,
    ) -> jnp.ndarray:
        """Decode the indices.

        Args:
            indices (jnp.ndarray): the indices.
            z_shape (Tuple[int, ...]): the shape of the latent vector.
            params (Optional[FrozenDict], optional): the params of the model. Defaults to None.

        Returns:
            jnp.ndarray: the decoded image from indices.
        """
        return self.module.apply(
            {"params": params or self.params},
            indices,
            z_shape,
            method=self.module.decode_code,
        )

    def update_temperature(
        self, temperature: float, params: Optional[FrozenDict] = None
    ) -> float:
        """Update the temperature of the model.
        Args:
            temperature (float): the temperature to update to.
            params (Optional[FrozenDict], optional): the params of the model. Defaults to None.
        Returns:
            float: the updated temperature.
        """
        new_temperature = self.module.apply(
            {"params": params or self.params},
            temperature,
            method=self.module.update_temperature,
        )
        return new_temperature

    def __call__(
        self,
        pixel_values: jnp.ndarray,
        params: Optional[FrozenDict] = None,
        dropout_rng: Optional[jax.random.PRNGKey] = None,
        gumble_rng: Optional[jax.random.PRNGKey] = None,
        train: bool = False,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, float, jnp.ndarray]:
        """Encode and decode the input.

        Args:
            pixel_values (jnp.ndarray): the input to the encoder.
            params (Optional[FrozenDict], optional): the params of the model. Defaults to None.
            dropout_rng (Optional[jax.random.PRNGKey], optional): the dropout rng. Defaults to None.
            gumble_rng (Optional[jax.random.PRNGKey], optional): the gumbel rng. Defaults to None.
                If gumble_rng is None then the defult rng is used and produce deterministic results.
            train (bool, optional): Training or inference mode. Defaults to False.

        Returns:
            Tuple[jnp.ndarray, jnp.ndarray, float, jnp.ndarray]:
                the encoded latent vector,
                the decoded image,
                the log prob of the latent vector,
                the indices of the latent vector.
        """
        # Check dtype
        pixel_values = (
            pixel_values.astype(self.dtype)
            if pixel_values.dtype != self.dtype
            else pixel_values
        )
        # Handle any PRNG if needed
        rngs = {"dropout": dropout_rng} if dropout_rng is not None else {}
        rngs["gumbel"] = gumble_rng if gumble_rng is not None else self.key
        return self.module.apply(
            {"params": params or self.params}, pixel_values, not train, rngs=rngs
        )


class VQModel(VQGANPreTrainedModel):
    """VQ-VAE model from pre-trained VQGAN."""

    module_class = VQModule
    config_class = VQGANConfig


class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator as in Pix2Pix
    See:
        https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py

    Arguments:
        ndf (int): the number of filters in the last conv layer
        n_layers (int): the number of conv layers in the discriminator
        output_dim (bool): dim of output the last channel of the discriminator
        dtype: the dtype of the computation (default: float32)
    """

    ndf: int
    n_layers: int
    output_dim: int
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool = True) -> jnp.ndarray:
        # input is bx256x256x(nc) return bx30x30x1
        x = nn.Conv(
            self.ndf,
            kernel_size=(4, 4),
            strides=(2, 2),
            padding=((1, 1), (1, 1)),
            dtype=self.dtype,
        )(x)
        x = nn.leaky_relu(x, negative_slope=0.2)
        # downsample
        for n in range(1, self.n_layers):
            nf_mult = min(2**n, 8)
            x = nn.Conv(
                self.ndf * nf_mult,
                kernel_size=(4, 4),
                strides=(2, 2),
                padding=((1, 1), (1, 1)),
                use_bias=False,
                dtype=self.dtype,
            )(x)
            x = nn.BatchNorm(use_running_average=not train, dtype=self.dtype)(x)
            x = nn.leaky_relu(x, negative_slope=0.2)

        # last downsample
        nf_mult = min(2**n, 8)
        x = nn.Conv(
            self.ndf * nf_mult,
            kernel_size=(4, 4),
            strides=(1, 1),
            padding=((1, 1), (1, 1)),
            use_bias=False,
            dtype=self.dtype,
        )(x)
        x = nn.BatchNorm(use_running_average=not train, dtype=self.dtype)(x)
        x = nn.leaky_relu(x, negative_slope=0.2)

        # output 1 channel prediction map
        logits = nn.Conv(
            self.output_dim,
            kernel_size=(4, 4),
            strides=(1, 1),
            padding=((1, 1), (1, 1)),
            dtype=self.dtype,
        )(x)
        return logits


class VQGanDiscriminator(FlaxPreTrainedModel):
    """VQGAN discriminator model.

    Arguments:
        module_class (nn.Module): the discriminator module class (NLayerDiscriminator).
        config_class (PretrainedConfig): configuration class to store the configuration of
            the model: (DiscConfig)
    """

    module_class: nn.Module = NLayerDiscriminator
    config_class: PretrainedConfig = DiscConfig

    def __init__(
        self,
        config: DiscConfig = DiscConfig(),
        input_shape: Tuple = (1, 256, 256, 3),
        seed: int = 0,
        dtype: jnp.dtype = jnp.float32,
        _do_init: bool = True,
        **kwargs,
    ):
        self._missing_keys: Set[str] = set()
        module = self.module_class(
            ndf=config.ndf,
            n_layers=config.n_layers,
            output_dim=config.output_last_dim,
            dtype=dtype,
            **kwargs,
        )
        super().__init__(
            config,
            module,
            input_shape=input_shape,
            seed=seed,
            dtype=dtype,
            _do_init=_do_init,
        )

    def init_weights(
        self,
        params_rng: jax.random.PRNGKey,
        input_shape: Tuple,
        params: FrozenDict = None,
    ) -> FrozenDict:
        # initialize model
        input_x = jnp.zeros(input_shape, dtype=self.dtype)
        random_params = self.module.init(params_rng, input_x, True)

        # If params provided find unitialized params and replace with provided params
        if params is not None:
            random_params = flatten_dict(unfreeze(random_params))
            params = flatten_dict(unfreeze(params))
            for missing_key in self._missing_keys:
                params[missing_key] = random_params[missing_key]
            self._missing_keys = set()
            return freeze(unflatten_dict(params))
        else:
            return random_params

    def __call__(
        self,
        input: jnp.ndarray,
        params: Optional[FrozenDict] = None,
        batch_stats: Optional[FrozenDict] = None,
        train: bool = False,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, float, jnp.ndarray]:
        # Handle any PRNG if needed
        if batch_stats is not None and params is not None:
            dict_params = {"params": params, "batch_stats": batch_stats}
        elif batch_stats is not None:
            dict_params = {"params": self.params["params"], "batch_stats": batch_stats}
        elif params is not None:
            dict_params = {"params": params, "batch_stats": self.params["batch_stats"]}
        else:
            dict_params = {
                "params": self.params["params"],
                "batch_stats": self.params["batch_stats"],
            }
        return self.module.apply(
            dict_params, input, train=train, mutable=["batch_stats"] if train else False
        )
