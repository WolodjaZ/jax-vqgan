from functools import partial
from typing import Any, Callable, Dict, Optional, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
from config import VQGANConfig
from einops import rearrange
from flax.core.frozen_dict import FrozenDict
from transformers import PretrainedConfig
from transformers.modeling_flax_utils import FlaxPreTrainedModel

from modules.config import VQGANConfig
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
        dtype (jnp.dtype): the dtype of the computation (default: float32).
    
    Config Attributes:
        n_embed (int) : number of embeddings.
        emb_dim (int): dimension of embedding.
    """
    config: VQGANConfig
    dtype: jnp.dtype = jnp.float32
    
    def setup(self):
        init_embbedings = jax.nn.initializers.uniform(scale=-1.0 / self.config.n_embed, dtype=self.dtype)
        self.embeddings = nn.Embed(self.config.n_embed,
                              self.config.embed_dim,
                              embedding_init=init_embbedings,
                              dtype=self.dtype)
    
    def __call__(self, z: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        # flatten z
        z_flatten = z.reshape(-1, self.config.embed_dim)
        
        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z
        emb_weights = self.variables["params"]["embedding"]["embedding"]
        distance = (jnp.sum(z_flatten**2, axis=1, keepdims=True) +
                    jnp.sum(emb_weights**2, axis=1) -
                    2 * jnp.dot(z_flatten, emb_weights.T))
        
        # get quantized latent vectors
        min_encoding_indices = jnp.argmin(distance, axis=1)
        z_q = self.embedding(min_encoding_indices).reshape(z.shape)
        
        # reshape to (batch, num_tokens)
        min_encoding_indices = min_encoding_indices.reshape(
            z.shape[0], -1)

        # compute the codebook_loss (q_loss) outside the model
        # here we return the embeddings and indices
        return z_q, min_encoding_indices

    def get_codebook_entry(self, indices: jnp.ndarray, shape: Optional[Tuple[int]] = None) -> jnp.ndarray:
        # indices are expected to be of shape (batch, num_tokens)
        # get quantized latent vectors
        z_q = self.embedding(indices)
        
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
        init_embbedings = jax.nn.initializers.uniform(scale=-1.0 / self.config.n_embed, dtype=self.dtype)
        self.embeddings = nn.Embed(self.config.n_embed,
                              self.config.embed_dim,
                              embedding_init=init_embbedings,
                              dtype=self.dtype)
    
    def __call__(self, z: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        # project z to get logits
        logits = self.proj(z)
        
        # given logits, sample from the Gumbel-Softmax distribution
        gumbel_rng = self.make_rng("gumbel")
        gumbels = jax.random.gumbel(gumbel_rng, logits.shape, dtype=self.dtype)
        indicies_prob = nn.softmax((logits + gumbels) / self.config.gumb_temp)
        
        # get quantized latent vectors
        emb_weights = self.variables["params"]["embedding"]["embedding"]
        z_q = jax.einsum("bhwp,pd->bhwd", indicies_prob, emb_weights).reshape(z.shape)
        
        # get indices [BxHxWxP] -> [BxHxW]
        indices = jnp.argmax(indicies_prob, axis=-1).reshape(z.shape[0], -1)

        # compute the codebook_loss (q_loss) outside the model
        # here we return the embeddings, indices and logits (for loss)
        return z_q, indices, logits

    def get_codebook_entry(self, indices: jnp.ndarray, shape: Tuple[int]) -> jnp.ndarray:
        # indices are expected to be of shape (batch, num_tokens)
        # get quantized latent vectors
        b, h, w, _ = shape
        assert h*w == indices.shape[1]
        indices = rearrange(indices, 'b (h w) -> b h w', b=b, h=h, w=w)
        one_hot_indices = jax.nn.one_hot(indices, self.config.n_embed, dtype=self.dtype)
        z_q = self.embedding(one_hot_indices)
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
        # Set activation function
        self.config.act_fn: Callable = ACTFUN[self.config.act_fn]
        # Encoder
        self.encoder = Encoder(self.config, dtype=self.dtype)
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
        self.decoder = Decoder(self.config, dtype=self.dtype)
        
    def encode(self, x: jnp.ndarray, deterministic: bool = True) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        # Encoder
        z = self.encoder(x, deterministic=deterministic)
        # Pre-quantizer
        z = self.pre_quantizer(z)
        # Quantizer
        z_q, indices, logits = self.quantizer(z) if self.config.use_gumbel else self.quantizer(z), None
        # Post-quantizer
        z_q = self.post_quantizer(z_q)
        return z_q, indices, logits
    
    def decode(self, z_q: jnp.ndarray, deterministic: bool = True) -> jnp.ndarray:
        # Post-quantizer
        z_q = self.post_quantizer(z_q)
        # Decoder
        x_recon = self.decoder(z_q, deterministic=deterministic)
        return x_recon
    
    def decode_code(self, code: jnp.ndarray, z_shape: Tuple[int]) -> jnp.ndarray:
        """Decode already created z_code"""
        z_q = self.quantizer.get_codebook_entry(code, shape=z_shape)
        x = self.decode(z_q)
        return x
    
    def __call__(self, x: jnp.ndarray, deterministic: bool = True) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        z_q, indices, logits = self.encode(x, deterministic=deterministic)
        x_recon = self.decode(z_q, deterministic=deterministic)
        return x_recon, z_q, indices, logits
    
class VQGANPreTrainedModel(FlaxPreTrainedModel):
    """An abstract class to handle weights initialization and a simple interface
    for downloading and loading pretrained models.

    Arguments:
        config_class (PretrainedConfig): a class derived from PretrainedConfig that defines the model's configuration.
        base_model_prefix (str): the string associated to the base model when downloading the model weights.
        module_class (nn.Module): a class derived from nn.Module that defines the model's core computation.
        
    """
    config_class: PretrainedConfig = VQGANConfig #TODO
    base_model_prefix: str = "model"
    module_class: nn.Module = None
    
    def __init__(
        self,
        config: VQGANConfig,
        input_shape: Tuple = (1, 256, 256, 3),
        seed: int = 0,
        dtype: jnp.dtype = jnp.float32,
        _do_init: bool = True,
        **kwargs,
    ):
        module = self.module_class(config=config, dtype=dtype, **kwargs)
        super().__init__(config, module, input_shape=input_shape, seed=seed, dtype=dtype, _do_init=_do_init)
    
    def init_model(self, rng: jax.random.PRNGKey,
                   input_shape: Tuple) -> FrozenDict:
        # initialize model
        input = jnp.zeros(input_shape, dtype=self.dtype)
        params_rng, dropout_rng = jax.random.split(rng)
        rngs = {"params": params_rng, "dropout": dropout_rng}
        
        return self.module.init(rngs, input)["params"]

    def encode(self, 
               input: jnp.ndarray, 
               params: Optional[Dict[str, Any]] = None,
               dropout_rng: Optional[jax.random.PRNGKey] = None,
               train: bool = False) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        # Handle any PRNG if needed
        rngs = {"dropout": dropout_rng} if dropout_rng is not None else {}
        return self.module.apply({"params": params or self.params},
                                jnp.array(input),
                                not train,
                                rngs=rngs,
                                method=self.module.encode)
    
    def decode(self,
                z: jnp.ndarray,
                params: Optional[Dict[str, Any]] = None,
                dropout_rng: Optional[jax.random.PRNGKey] = None,
                train: bool = False) -> jnp.ndarray:
        # Handle any PRNG if needed
        rngs = {"dropout": dropout_rng} if dropout_rng is not None else {}
        return self.module.apply({"params": params or self.params},
                                z,not train,
                                rngs=rngs,
                                method=self.module.decode)
        
    def decode_code(self, indices: jnp.ndarray, params: dict = None):
        return self.module.apply({"params": params or self.params},
                             indices,
                             self.config.z_shape,
                             method=self.module.decode_code)
    
    def __call__(self,
                input: jnp.ndarray,
                params: Optional[Dict[str, Any]] = None,
                dropout_rng: Optional[jax.random.PRNGKey] = None,
                train: bool = False) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        # Handle any PRNG if needed
        rngs = {"dropout": dropout_rng} if dropout_rng is not None else {}

        return self.module.apply(
            {"params": params or self.params},
            jnp.array(input),
            not train,
            rngs=rngs,
        )


class VQModel(VQGANPreTrainedModel):
    """VQ-VAE model from pre-trained VQGAN."""
    module_class = VQModule