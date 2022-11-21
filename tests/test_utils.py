# flake8: noqa: E401
import tempfile
from typing import Tuple

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import tensorflow as tf  # noqa: E401
from jax.tree_util import tree_all, tree_map
from omegaconf import OmegaConf
from PIL import Image

from modules import config, utils

DATACONFIG_PATH = "tests/dataconfig_test.yaml"


@pytest.fixture()
def JAX_PRNG() -> Tuple[jax.random.PRNGKey, jnp.dtype]:
    """Seeding for random operations."""
    test_rng = jax.random.PRNGKey(42)
    dtype = jnp.float32
    return test_rng, dtype


def create_img(rng: jax.random.PRNGKey, reverted: bool) -> np.ndarray:
    """Create a random image.
    Args:
        rng (jax.random.PRNGKey): random number generator.
        reverted (bool): if True, image format is (C, H, W), else (H, W, C).
    Returns:
        image (np.ndarray): random image.
    """
    image = jax.random.normal(rng, (224, 224, 3))
    if reverted:
        jnp.transpose(image, (2, 0, 1))
    image = (image + 1) / 2 * 255
    image = jax.lax.clamp(0.0, image, 255.0)
    return np.uint8(np.asarray(image))


def test_seed():
    """Test the seed function."""
    utils.set_seed(42)
    assert tree_all(
        tree_map(lambda x, y: (x == y).all(), jax.random.PRNGKey(42), jax.random.PRNGKey(42))
    )


def test_DummyDataset(mocker):
    """Test the DummyDataset."""
    # load test data config
    utils.set_seed(42)
    dtype = jnp.float32
    cfg_omgega = OmegaConf.load(DATACONFIG_PATH)
    load_confg = OmegaConf.to_container(cfg_omgega)
    cfg = config.DataConfig(**load_confg)

    # Train dataset
    dataset_train_class = utils.DummyDataset(train=True, dtype=dtype, config=cfg)
    mock_transform = mocker.patch.object(
        dataset_train_class, "transforms", side_effect=lambda image: {"image": image}
    )
    assert issubclass(type(dataset_train_class), utils.BaseDataset)
    dataset_train = dataset_train_class.get_dataset()
    dataset_train = dataset_train.as_numpy_iterator()
    data = next(dataset_train)
    mock_transform.assert_called()
    assert data.shape == (cfg.size, cfg.size, 3)
    assert data.dtype == jnp.float32

    # Test dataset
    cfg.use_transforms = False
    dataset_test_class = utils.DummyDataset(train=False, dtype=dtype, config=cfg)
    mock_transform = mocker.patch.object(
        dataset_test_class, "transforms", side_effect=lambda image: {"image": image}
    )
    assert issubclass(type(dataset_test_class), utils.BaseDataset)
    dataset_test = dataset_test_class.get_dataset()
    dataset_test = dataset_test.as_numpy_iterator()
    data = next(dataset_test)
    mock_transform.assert_not_called()
    assert data.shape == (cfg.size, cfg.size, 3)
    assert data.dtype == jnp.float32

    # Clean up
    del dataset_train, dataset_test, cfg_omgega, load_confg, cfg


def test_reproducibility_Dataset(mocker):
    # load test data config
    import random  # noqa: F811, E401

    import numpy as np  # noqa: F811, E401
    import tensorflow as tf  # noqa: F811, E401

    seed = 42
    utils.set_seed(seed)
    dtype = jnp.float32
    cfg_omgega = OmegaConf.load(DATACONFIG_PATH)
    load_confg = OmegaConf.to_container(cfg_omgega)
    cfg = config.DataConfig(**load_confg)

    dataset_train_class = utils.DummyDataset(train=True, dtype=dtype, config=cfg)
    mocker.patch.object(
        dataset_train_class, "transforms", side_effect=lambda image: {"image": image}
    )
    dataset_first = dataset_train_class.get_dataset().as_numpy_iterator()

    import random  # noqa: F811, E401

    import numpy as np  # noqa: F811, E401
    import tensorflow as tf  # noqa: F811, E401

    utils.set_seed(seed)
    dataset_train_class = utils.DummyDataset(train=True, dtype=dtype, config=cfg)
    mocker.patch.object(
        dataset_train_class, "transforms", side_effect=lambda image: {"image": image}
    )
    dataset_second = dataset_train_class.get_dataset().as_numpy_iterator()

    for _ in range(3):
        try:
            assert np.all(next(dataset_first) == next(dataset_second))
        except StopIteration:
            break

    # Clean up
    del dataset_first, dataset_second, cfg_omgega, load_confg, cfg


def test_TensorflowDataset():
    """Test the TensorflowDataset."""
    # load test data config
    utils.set_seed(42)
    dtype = jnp.float32
    cfg_omgega = OmegaConf.load(DATACONFIG_PATH)
    load_confg = OmegaConf.to_container(cfg_omgega)
    load_confg["dataset_name"] = "cifar10"
    with tempfile.TemporaryDirectory() as dp:
        load_confg["dataset_root"] = dp
        cfg = config.DataConfig(**load_confg)

        # Train dataset
        dataset_train_class = utils.TensorflowDataset(train=True, dtype=dtype, config=cfg)
        assert issubclass(type(dataset_train_class), utils.BaseDataset)
        dataset_train = dataset_train_class.get_dataset()
        dataset_train = dataset_train.as_numpy_iterator()
        data = next(dataset_train)
        assert data.shape == (cfg.size, cfg.size, 3)
        assert data.dtype == jnp.float32

        # Test dataset
        cfg.use_transforms = False
        dataset_test_class = utils.TensorflowDataset(train=False, dtype=dtype, config=cfg)
        assert issubclass(type(dataset_test_class), utils.BaseDataset)
        dataset_test = dataset_test_class.get_dataset()
        dataset_test = dataset_test.as_numpy_iterator()
        data = next(dataset_test)
        assert data.shape == (cfg.size, cfg.size, 3)
        assert data.dtype == jnp.float32

    # Clean up
    del dataset_train, dataset_test, cfg_omgega, load_confg, cfg


def test_DataLoader():
    """Test the DataLoader."""
    # load test data config
    utils.set_seed(42)
    dtype = jnp.float32
    cfg_omgega = OmegaConf.load(DATACONFIG_PATH)
    load_confg = OmegaConf.to_container(cfg_omgega)
    cfg = utils.DataConfig(**load_confg)

    # create dummy dataset
    dataset_train_class = utils.DummyDataset(train=True, dtype=dtype, config=cfg)

    # create dataloader
    loader = utils.DataLoader(dataset_train_class, distributed=False)
    assert len(loader) == len(dataset_train_class) // cfg.train_params.batch_size
    in_loop = 0
    for x in loader():
        assert x.shape == (cfg.train_params.batch_size, cfg.size, cfg.size, 3)
        assert x.dtype == dtype
        assert in_loop < len(loader)
        in_loop += 1
    assert in_loop == len(loader)

    in_loop = 0
    for x in loader():
        assert x.shape == (cfg.train_params.batch_size, cfg.size, cfg.size, 3)
        assert x.dtype == dtype
        in_loop += 1

    assert in_loop == len(loader)

    # clean up
    del loader, dataset_train_class, cfg_omgega, load_confg, cfg


def test_VQGanImageProcessor():
    """Test the VQGanImageProcessor."""
    imageprocesser = utils.VQGanImageProcessor()
    assert imageprocesser is not None

    extractor = utils.VQGanFeatureExtractor()
    assert extractor is not None


def test_resize_VQGanImageProcessor():
    extractor = utils.VQGanImageProcessor()
    input = np.ones((256, 256, 3))
    input_resized = extractor.resize(input, size={"width": 512, "height": 512})
    assert input_resized.shape == (512, 512, 3)
    input_reverted = np.ones((3, 256, 256))
    input_reverted_resized = extractor.resize(input_reverted, size={"width": 512, "height": 512})
    assert input_reverted_resized.shape == (3, 512, 512)


@pytest.mark.parametrize(
    "rescale",
    [2.0, 1.0, 0.5],
)
def test_rescale_VQGanImageProcessor(rescale):
    """Test the rescale function."""
    extractor = utils.VQGanImageProcessor()
    input = np.ones((224, 224, 3))
    input_resized = extractor.rescale(input, rescale)
    assert input_resized.mean() == input.mean() * rescale
    input_reverted = np.ones((3, 224, 224))
    input_reverted_resized = extractor.rescale(input_reverted, rescale)
    assert input_reverted_resized.mean() == input_reverted.mean() * rescale


def test_normalize_VQGanImageProcessor():
    """Test the normalize function."""
    extractor = utils.VQGanImageProcessor()
    input = (jax.random.normal(jax.random.PRNGKey(0), (224, 224, 3)) + 1) / 2 * 255
    input = jax.lax.clamp(0.0, input, 255.0)
    input = np.asarray(input)
    input_normalized = extractor.normalize(
        input, mean=input.mean(axis=(0, 1)), std=input.std(axis=(0, 1))
    )
    assert input_normalized.shape == (224, 224, 3)
    assert input_normalized.mean() == pytest.approx(0.0, abs=0.1)
    assert input_normalized.std() == pytest.approx(1.0, abs=0.1)
    input = (jax.random.normal(jax.random.PRNGKey(0), (3, 224, 224)) + 1) / 2 * 255
    input = jax.lax.clamp(0.0, input, 255.0)
    input = np.asarray(input)
    input_normalized = extractor.normalize(
        input, mean=input.mean(axis=(1, 2)), std=input.std(axis=(1, 2))
    )
    assert input_normalized.shape == (3, 224, 224)
    assert input_normalized.mean() == pytest.approx(0.0, abs=0.1)
    assert input_normalized.std() == pytest.approx(1.0, abs=0.1)


@pytest.mark.parametrize(
    "pil_img, list_of_images, reverted_channels",
    [
        (True, False, False),
        (False, False, False),
        (True, True, False),
        (False, True, False),
        (False, True, True),
        (False, False, True),
    ],
)
def test_preprocess(pil_img, list_of_images, reverted_channels):
    """Test the preprocess function."""

    def create_img(rng: jax.random.PRNGKey, reverted: bool) -> np.ndarray:
        image = jax.random.normal(rng, (224, 224, 3))
        if reverted:
            jnp.transpose(image, (2, 0, 1))
        image = (image + 1) / 2 * 255
        image = jax.lax.clamp(0.0, image, 255.0)
        return np.uint8(np.asarray(image))

    splits = 4
    rng = jax.random.PRNGKey(0)
    if list_of_images:
        rngs = jax.random.split(rng, splits)
        images = [create_img(rng, reverted_channels) for rng in rngs]
        if pil_img:
            images = [Image.fromarray(image) for image in images]
    else:
        images = create_img(rng, reverted_channels)
        if pil_img:
            images = Image.fromarray(images)

    extractor = utils.VQGanImageProcessor()
    input_preprocessed = extractor.preprocess(images)
    assert "pixel_values" in input_preprocessed
    input_preprocessed = input_preprocessed["pixel_values"]
    assert input_preprocessed[0].shape == (256, 256, 3)

    # check do resize
    input_preprocessed = extractor.preprocess(images, do_resize=True)
    assert "pixel_values" in input_preprocessed
    input_preprocessed = input_preprocessed["pixel_values"]
    assert input_preprocessed[0].shape == (256, 256, 3)
    input_preprocessed = extractor.preprocess(images, do_resize=False)
    assert "pixel_values" in input_preprocessed
    input_preprocessed = input_preprocessed["pixel_values"]
    assert input_preprocessed[0].shape == (224, 224, 3)

    # check do rescale
    input_preprocessed = extractor.preprocess(images, do_rescale=True)
    assert "pixel_values" in input_preprocessed
    input_preprocessed = input_preprocessed["pixel_values"]
    assert input_preprocessed[0].shape == (256, 256, 3)
    assert input_preprocessed[0].mean() == pytest.approx(0.0, abs=0.3)
    input_preprocessed = extractor.preprocess(images, do_rescale=True, rescale_factor=1.0)
    assert "pixel_values" in input_preprocessed
    input_preprocessed = input_preprocessed["pixel_values"]
    assert input_preprocessed[0].shape == (256, 256, 3)
    assert input_preprocessed[0].mean() != pytest.approx(0.0, abs=0.3)
    input_preprocessed = extractor.preprocess(images, do_rescale=False)
    input_preprocessed = input_preprocessed["pixel_values"]
    assert input_preprocessed[0].shape == (256, 256, 3)
    assert input_preprocessed[0].mean() != pytest.approx(0.0, abs=0.1)
    # check do normalize
    input_preprocessed = extractor.preprocess(images, do_normalize=True)
    assert "pixel_values" in input_preprocessed
    input_preprocessed = input_preprocessed["pixel_values"]
    assert input_preprocessed[0].shape == (256, 256, 3)
    assert input_preprocessed[0].mean() == pytest.approx(0.0, abs=0.3)
    assert input_preprocessed[0].std() == pytest.approx(1.0, abs=0.3)
    input_preprocessed = extractor.preprocess(images, do_normalize=False)
    assert "pixel_values" in input_preprocessed
    input_preprocessed = input_preprocessed["pixel_values"]
    assert input_preprocessed[0].shape == (256, 256, 3)
    assert input_preprocessed[0].mean() != pytest.approx(0.0, abs=0.3)
    assert input_preprocessed[0].std() != pytest.approx(1.0, abs=0.3)


@pytest.mark.parametrize(
    "pil_img, list_of_images, reverted_channels",
    [
        (True, False, False),
        (False, False, False),
        (True, True, False),
        (False, True, False),
        (False, True, True),
        (False, False, True),
    ],
)
def test_VQGanFeatureExtractor(pil_img, list_of_images, reverted_channels):
    """Test the VQGanFeatureExtractor."""
    splits = 4
    rng = jax.random.PRNGKey(0)
    if list_of_images:
        rngs = jax.random.split(rng, splits)
        images = [create_img(rng, reverted_channels) for rng in rngs]
        if pil_img:
            images = [Image.fromarray(image) for image in images]
    else:
        images = create_img(rng, reverted_channels)
        if pil_img:
            images = Image.fromarray(images)

    extractor = utils.VQGanFeatureExtractor()
    input_preprocessed = extractor(images)
    assert "pixel_values" in input_preprocessed
    input_preprocessed = input_preprocessed["pixel_values"]
    assert input_preprocessed[0].shape == (256, 256, 3)


@pytest.mark.parametrize(
    "pil_img, list_of_images, reverted_channels",
    [
        (True, False, False),
        (False, False, False),
        (True, True, False),
        (False, True, False),
        (False, True, True),
        (False, False, True),
    ],
)
def test_pipeline(pil_img, list_of_images, reverted_channels):
    """Test the pipeline. Extracting features and then reconstructing the image."""
    splits = 4
    rng = jax.random.PRNGKey(0)
    if list_of_images:
        rngs = jax.random.split(rng, splits)
        images = [create_img(rng, reverted_channels) for rng in rngs]
        if pil_img:
            images = [Image.fromarray(image) for image in images]
    else:
        images = create_img(rng, reverted_channels)
        if pil_img:
            images = Image.fromarray(images)

    # Extractor
    extractor = utils.VQGanFeatureExtractor()
    input_preprocessed = extractor(images)
    assert input_preprocessed is not None
    assert "pixel_values" in input_preprocessed

    # Get model output
    # initalize model
    try:
        from modules.config import VQGANConfig
        from modules.vqgan import VQModel
    except ModuleNotFoundError:
        pytest.skip("Test requires VQModel and VQConfig implemented")

    config_vqgan = VQGANConfig(
        resolution=input_preprocessed["pixel_values"].shape[1],
    )
    assert config_vqgan is not None
    input_shape = input_preprocessed["pixel_values"].shape
    dtype = jnp.float32

    # Check initialization
    vqmodel = VQModel(config=config_vqgan, input_shape=input_shape, seed=0, dtype=dtype)
    assert vqmodel is not None

    # Check forward pass
    x_recon, _, _, _ = vqmodel(**input_preprocessed)
    assert x_recon is not None
    x_recon.shape == input_shape


@pytest.mark.parametrize(
    "size_img",
    [448, 256, 224, 128],
)
def test_pipeline_sizes(size_img):
    """Test the pipeline. Extracting features and then reconstructing the image.
    With different image sizes."""
    splits = 2
    rng = jax.random.PRNGKey(0)
    rngs = jax.random.split(rng, splits)
    images = [create_img(rng, False) for rng in rngs]

    # Extractor
    extractor = utils.VQGanFeatureExtractor(size={"width": size_img, "height": size_img})
    input_preprocessed = extractor(images)
    assert input_preprocessed is not None
    assert "pixel_values" in input_preprocessed
    assert input_preprocessed["pixel_values"].shape == (2, size_img, size_img, 3)

    # Get model output
    # initalize model
    try:
        from modules.config import VQGANConfig
        from modules.vqgan import VQModel
    except ModuleNotFoundError:
        pytest.skip("Test requires VQModel and VQConfig implemented")

    config_vqgan = VQGANConfig(
        resolution=input_preprocessed["pixel_values"].shape[1],
    )
    assert config_vqgan is not None
    input_shape = input_preprocessed["pixel_values"].shape
    print(input_shape)
    dtype = jnp.float32

    # Check initialization
    vqmodel = VQModel(config=config_vqgan, input_shape=input_shape, seed=0, dtype=dtype)
    assert vqmodel is not None

    # Check forward pass
    x_recon, _, _, _ = vqmodel(**input_preprocessed)
    assert x_recon is not None
    x_recon.shape == input_shape
