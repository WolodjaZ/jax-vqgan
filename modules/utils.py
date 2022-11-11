from typing import Any, Dict, List, Optional, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image
from transformers import BatchFeature

IMAGENET_STANDARD_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STANDARD_STD = [0.229, 0.224, 0.225]


class DummyDataLoader:
    """A dummy data loader.
    Create dummy data of given shape and batches.
    End simulates pytorch dataloader.
    """

    def __init__(
        self,
        rng: jax.random.PRNGKey,
        dataset_shape: Tuple[int, int, int, int],
        batches: int,
        dtype: jnp.dtype,
    ) -> None:
        """Create a dummy data loader.
        Args:
            rng: Random number generator for creating random data.
            dataset_shape: Shape of the dataset.
            batches: Number of batches to create.
        """
        self.batches = batches
        self.dataset = jax.random.normal(rng, (batches,) + dataset_shape, dtype=dtype)

    def __len__(self) -> int:
        """Return the number of batches.
        Returns:
            int: Number of batches.
        """
        return self.batches

    def __iter__(self):
        """Return the iterator."""
        self._numb = 0
        return self

    def __next__(self) -> jnp.ndarray:
        """Return the next batch."""
        if self.batches > self._numb:
            i = self._numb
            self._numb += 1
            data = self.dataset[i]
            return data
        else:
            raise StopIteration


class VQGanImageProcessor:
    """
    Constructs a VQGan image processor.
    Args:
        do_resize (`bool`, *optional*, defaults to `True`):
            Whether to resize the image's (height, width) dimensions to the specified
            `(size["height"], size["width"])`. Can be overridden by the `do_resize` parameter in
            the `preprocess` method.
        size (`dict`, *optional*, defaults to `{"height": 256, "width": 256}`):
            Size of the output image after resizing. Can be overridden by the `size` parameter in
            the `preprocess` method.
        resample (`Image.Resampling`, *optional*, defaults to `Image.Resampling.BILINEAR`):
            Resampling filter to use if resizing the image. Can be overridden by the `resample`
            parameter in the `preprocess` method.
        do_rescale (`bool`, *optional*, defaults to `True`):
            Whether to rescale the image by the specified scale `rescale_factor`. Can be overridden
            by the `do_rescale` parameter in the `preprocess` method.
        rescale_factor (`int` or `float`, *optional*, defaults to `1/255`):
            Scale factor to use if rescaling the image. Can be overridden by the `rescale_factor`
            parameter in the `preprocess` method.
        do_normalize:
            Whether to normalize the image. Can be overridden by the `do_normalize` parameter in
            the `preprocess` method.
        image_mean (`float` or `List[float]`, *optional*, defaults to `IMAGENET_STANDARD_MEAN`):
            Mean to use if normalizing the image. This is a float or list of floats the length of
            the number of channels in the image. Can be overridden by the `image_mean` parameter
            in the `preprocess` method.
        image_std (`float` or `List[float]`, *optional*, defaults to `IMAGENET_STANDARD_STD`):
            Standard deviation to use if normalizing the image. This is a float or list of floats
            the length of the number of channels in the image. Can be overridden by the `image_std`
            parameter in the `preprocess` method.
        dtype (`jax.numpy.dtype`, *optional*, defaults to `jax.numpy.float32`):
    """

    model_input_names = ["pixel_values"]

    def __init__(
        self,
        do_resize: bool = True,
        size: Optional[Dict[str, int]] = None,
        resample: Image.Resampling = Image.Resampling.BILINEAR,
        do_rescale: bool = True,
        rescale_factor: Union[int, float] = 1 / 255,
        do_normalize: bool = True,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        size = size if size is not None else {"height": 256, "width": 256}
        self.do_resize = do_resize
        self.do_rescale = do_rescale
        self.do_normalize = do_normalize
        self.size = size
        self.resample = resample
        self.rescale_factor = rescale_factor
        self.image_mean = (
            image_mean if image_mean is not None else IMAGENET_STANDARD_MEAN
        )
        self.image_std = image_std if image_std is not None else IMAGENET_STANDARD_STD

    def resize(
        self,
        image: np.ndarray,
        size: Dict[str, int],
        resample: Image.Resampling = Image.Resampling.BILINEAR,
        data_format: Optional[str] = None,
        **kwargs,
    ) -> np.ndarray:
        """
        Resize an image to `(size["height"], size["width"])`.
        Args:
            image (`np.ndarray`):
                Image to resize.
            size (`Dict[str, int]`):
                Dictionary in the format `{"height": int, "width": int}` specifying the size
                of the output image.
            resample:
                `Image.Resampling` filter to use when resizing the image e.g.
                `Image.Resampling.BILINEAR`.
            data_format (`str`, *optional*):
                The channel dimension format for the output image. If unset, the channel
                dimension format of the input image is used. Can be one of:
                - `"channels_first"`: image in (num_channels, height, width) format.
                - `"channels_last"`: image in (height, width, num_channels) format.
        Returns:
            `np.ndarray`: The resized image.
        """
        if "height" not in size or "width" not in size:
            raise ValueError(
                "The `size` dictionary must contain the keys `height` and `width`. Got"
                f" {size.keys()}"
            )

        revert_format = False
        if data_format == "channels_first":
            revert_format = True
            image = np.transpose(image, (1, 2, 0))
        else:
            if image.shape[0] == 3:
                revert_format = True
                image = np.transpose(image, (1, 2, 0))

        pil_image = Image.fromarray(np.uint8(image))
        pil_image_resized = pil_image.resize(
            (size["width"], size["height"]), resample=resample
        )
        image_np = np.array(pil_image_resized)
        assert image_np.shape == (size["height"], size["width"], image.shape[-1])
        image_np = np.transpose(image_np, (2, 0, 1)) if revert_format else image_np
        return image_np

    def rescale(
        self,
        image: np.ndarray,
        scale: float,
        data_format: Optional[str] = None,
        **kwargs,
    ) -> np.ndarray:
        """
        Rescale an image by a scale factor. image = image * scale.
        Args:
            image (`np.ndarray`):
                Image to resize.
            scale (`float`):
                The scaling factor to rescale pixel values by.
            data_format (`str`, *optional*):
                The channel dimension format for the output image. If unset, the channel dimension
                format of the input image is used. Can be one of:
                - `"channels_first"`: image in (num_channels, height, width) format.
                - `"channels_last"`: image in (height, width, num_channels) format.
        Returns:
            `np.ndarray`: The resized image.
        """
        return image * scale

    def normalize(
        self,
        image: np.ndarray,
        mean: Union[float, List[float]],
        std: Union[float, List[float]],
        data_format: Optional[str] = None,
        **kwargs,
    ) -> np.ndarray:
        """
        Normalize an image. image = (image - image_mean) / image_std.
        Args:
            image (`np.ndarray`):
                Image to normalize.
            mean (`float` or `List[float]`):
                Image mean to use for normalization.
            std (`float` or `List[float]`):
                Image standard deviation to use for normalization.
            data_format (`str`, *optional*):
                The channel dimension format for the output image. If unset, the channel dimension
                format of the input image is used. Can be one of:
                - `"channels_first"`: image in (num_channels, height, width) format.
                - `"channels_last"`: image in (height, width, num_channels) format.
        Returns:
            `jnp.ndarray`: The normalized image.
        """
        if isinstance(mean, list):
            assert len(mean) == min(image.shape)
        if isinstance(std, list):
            assert len(std) == min(image.shape)

        revert_format = False
        if data_format == "channels_first":
            revert_format = True
            image = np.transpose(image, (1, 2, 0))
        else:
            if image.shape[0] == 3:
                revert_format = True
                image = np.transpose(image, (1, 2, 0))

        image_normalized = (image - mean) / std
        image_normalized = (
            np.transpose(image_normalized, (2, 0, 1))
            if revert_format
            else image_normalized
        )
        return image_normalized

    def preprocess(
        self,
        images: Union[Image.Image, np.ndarray, List[Image.Image], List[np.ndarray]],
        do_resize: Optional[bool] = None,
        size: Dict[str, int] = None,
        resample: Optional[Image.Resampling] = None,
        do_rescale: Optional[bool] = None,
        rescale_factor: Optional[float] = None,
        do_normalize: Optional[bool] = None,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        data_format: str = "channels_last",
        **kwargs,
    ) -> BatchFeature:
        """
        Preprocess an image or batch of images.
        Args:
            images (`Image.Image`, `np.ndarray`, `List[Image.Image]`, `List[np.ndarray]`):
                Image to preprocess.
            do_resize (`bool`, *optional*, defaults to `self.do_resize`):
                Whether to resize the image.
            size (`Dict[str, int]`, *optional*, defaults to `self.size`):
                Dictionary in the format `{"height": h, "width": w}` specifying the size of the
                output image after resizing.
            resample (`Image.Resampling` filter, *optional*, defaults to `self.resample`):
                `Image.Resampling` filter to use if resizing the image e.g.
                `Image.Resampling.BILINEAR`. Only has an effect if `do_resize` is set to `True`.
            do_rescale (`bool`, *optional*, defaults to `self.do_rescale`):
                Whether to rescale the image values between [0 - 1].
            rescale_factor (`float`, *optional*, defaults to `self.rescale_factor`):
                Rescale factor to rescale the image by if `do_rescale` is set to `True`.
            do_normalize (`bool`, *optional*, defaults to `self.do_normalize`):
                Whether to normalize the image.
            image_mean (`float` or `List[float]`, *optional*, defaults to `self.image_mean`):
                Image mean to use if `do_normalize` is set to `True`.
            image_std (`float` or `List[float]`, *optional*, defaults to `self.image_std`):
                Image standard deviation to use if `do_normalize` is set to `True`.
            data_format (`str`, *optional*, defaults to `channels_las`):):
                The channel dimension format for the output image. Can be one of:
                - `"channels_first"`: image in (num_channels, height, width) format.
                - `"channels_last"`: image in (height, width, num_channels) format.
                - Unset: Use the channel dimension format of the input image.
            Returns:
                `BatchFeature`: The preprocessed image(s).
        """
        assert data_format in ["channels_first", "channels_last"]
        do_resize = do_resize if do_resize is not None else self.do_resize
        do_rescale = do_rescale if do_rescale is not None else self.do_rescale
        do_normalize = do_normalize if do_normalize is not None else self.do_normalize
        resample = resample if resample is not None else self.resample
        rescale_factor = (
            rescale_factor if rescale_factor is not None else self.rescale_factor
        )
        image_mean = image_mean if image_mean is not None else self.image_mean
        image_std = image_std if image_std is not None else self.image_std

        size = size if size is not None else self.size

        # Batch of images and jnp.ndarray
        if isinstance(images, (list, tuple)):
            images = [np.array(image) for image in images]
        else:
            images = [np.array(images)]

        # if needed to rescale the image
        for img in images:
            assert (
                img.min() >= 0 and img.max() <= 255
            ), "Image values must be in [0 - 255] range."

        if do_resize and size is None:
            raise ValueError("Size must be specified if do_resize is True.")

        if do_rescale and rescale_factor is None:
            raise ValueError("Rescale factor must be specified if do_rescale is True.")

        if do_resize:
            images = [
                self.resize(image=image, size=size, resample=resample)
                for image in images
            ]

        if do_rescale:
            images = [
                self.rescale(image=image, scale=rescale_factor) for image in images
            ]

        if do_normalize:
            images = [
                self.normalize(image=image, mean=image_mean, std=image_std)
                for image in images
            ]

        if data_format == "channels_first":
            images = [jnp.transpose(image, (1, 2, 0)) for image in images]

        data = {"pixel_values": images}
        return BatchFeature(data=data, tensor_type="jax")


class VQGanFeatureExtractor(VQGanImageProcessor):
    """Extract features for VQGan from images.
    Extends VQGanImageProcessor only with call function to run preprocessing.
    """

    def __call__(self, *args: Any, **kwds: Any) -> BatchFeature:
        return self.preprocess(*args, **kwds)
