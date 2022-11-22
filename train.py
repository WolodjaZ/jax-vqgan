import logging
import resource

import hydra
from hydra.core.config_store import ConfigStore

from modules.config import LoadConfig
from modules.training import TrainerVQGan
from modules.utils import DataLoader, TensorflowDataset

cs = ConfigStore.instance()
cs.store(name="VQGAN_JAX", node=LoadConfig)
logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def train(cfg: LoadConfig):
    logging.info(f"Using config: {cfg}")  # TODO make it pretty to print
    # Prepare datasets and dataloaders
    logger.info("Preparing datasets and dataloaders 🚀")
    dataset_train_class = TensorflowDataset(train=True, dtype=cfg.train.dtype, config=cfg.data)
    dataset_test_class = TensorflowDataset(train=False, dtype=cfg.train.dtype, config=cfg.data)

    loader_train = DataLoader(dataset=dataset_train_class, distributed=cfg.train.distributed)
    loader_val = DataLoader(dataset=dataset_test_class, distributed=cfg.train.distributed)

    # Prepare model and trainer
    logger.info("Creating model and trainer 🚀")
    model = TrainerVQGan(module_config=cfg.train)

    # Start training
    model.train_model(loader_train, loader_val)


if __name__ == "__main__":
    # Please refer to modules.utils..py::TensorflowDataset for more details
    # about why we need to set this environment variable
    low, high = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (high, high))

    train()
