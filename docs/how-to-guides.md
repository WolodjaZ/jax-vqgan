# How-to guide

In the project we put every functionality changes into [yaml](https://en.wikipedia.org/wiki/YAML) files. Every aspect of tunning, dataset change, trainer change can be done with simply changing parameters in the yaml file. Our pipeline uses [hydra](https://hydra.cc) to load and manage pipeline with YAML structures. Firstly we will show you structure of the yaml file and than present you three sample yaml files used in this project. At the end we will also provide major modules used in the pipeline.

If you play to run `train.py` script your yaml file need to be in `conf` folder with the name `config.yaml` as hydra loads this file for the script. If you want to make changes just to them to the fail and for saving for now just rename it 🥸.

    your_project/
    │
    ├── conf/
    │   ├── config.yaml
    │   └── legacy.yaml
    │
    ├── modules/
    │   ├── __init__.py
    │   ├── config.py
    │   ├── losses.py
    │   ├── models.py
    │   ├── training.py
    │   ├── utils.py
    │   └── vqgan.py
    │
    └── train.py

## YAML Structures

Main yaml config file should have base structure `LoadConfig` defined in [`modules.config.py`](https://github.com/WolodjaZ/jax-vqgan/blob/e67c52cbb02fc39fbbfacc1ad06f40e6a7d53a79/modules/config.py#L238)

### LoadConfig

::: modules.config.LoadConfig
    options:
        show_root_heading: false

Config structure have two parameters `data` specifying dataset and dataloader parameters and `train` specifing architecture and training parameters.

### DataConfig

::: modules.config.DataConfig

`DataConfig` tells everything you need to know about downloading Tensorflow datasets and processing it. Agumentation information for the pipeline is based on [albumentations](https://albumentations.ai) framework and please refer to it for additional changes. This config relays on `train_params` and `test_params` telling about shuffling and batch size for train and test splits.

#### DataParams

::: modules.config.DataParams

### TrainConfig

::: modules.config.TrainConfig

`TrainConfig` is main config for setting trainer. The most important parameters are `model_name`, `save_dir`, `log_dir`, `dtype`, `seed`, `distributed`. `dtype`, `seed` and `distributed` are parameters also used in datasets (For now we support only false for `distributed`). `save_dir` and `log_dir` are paths for model checkpointing and tensorboard saving. `model_name` is the name of model which is referenced in saving and logging to tensorboard so you need to keep an eye on this parameter. `optimize` and `temp_scheduler` are parameters which are [instantiate](https://hydra.cc/docs/advanced/instantiate_objects/overview/) by hydra and for this we use objects from [`optax`](https://optax.readthedocs.io/en/latest/) (please refer to samples). `model_hparams` contains all the parameters for VQGAN module architecture and `disc_hparams` contains parameters for Discriminator.

### DiscConfig

::: modules.config.DiscConfig

### VQGANConfig

::: modules.config.VQGANConfig

`VQGANConfig` major parameter here is `use_gumbel`. VQGAN can be trained with [Gumbel-max Trick](https://arxiv.org/pdf/2110.01515.pdf) ([Original paper](https://arxiv.org/pdf/1611.01144.pdf)) which gives our bottleneck distribution on which we choose argmax for the code to assign.

## Samples

We provide three samples of data and train configs:
- [`config.yaml`](https://github.com/WolodjaZ/jax-vqgan/blob/main/conf/config.yaml) my training config on `imagenette` dataset.
- [`gumbel.yaml`](https://github.com/WolodjaZ/jax-vqgan/blob/main/conf/gumbel.yaml) official training config on `imagenet` dataset.
- [`imagenet.yaml`](https://github.com/WolodjaZ/jax-vqgan/blob/main/conf/imagenet.yaml) official training config with Gumble tick on `imagenet` dataset.

## Major Modules

Major Modules used in the pipeline are:

### TrainerVQGan

`TrainerVQGan` in [`modules.training`](https://github.com/WolodjaZ/jax-vqgan/blob/e67c52cbb02fc39fbbfacc1ad06f40e6a7d53a79/modules/training.py#L329), this modules responds for training VQGAN

::: modules.training.TrainerVQGan

### VQGANPreTrainedModel

`VQGANPreTrainedModel` in [`modules.vqgan`](https://github.com/WolodjaZ/jax-vqgan/blob/e67c52cbb02fc39fbbfacc1ad06f40e6a7d53a79/modules/vqgan.py#L322), response for VQ autoencoder architecture. This class is based on `FlaxPreTrainedModel` which gives ous abilities to push the architecture to Hugging Face Hub.

::: modules.vqgan.VQGANPreTrainedModel

### VQGanDiscriminator

`VQGanDiscriminator` in [`modules.vqgan`](https://github.com/WolodjaZ/jax-vqgan/blob/e67c52cbb02fc39fbbfacc1ad06f40e6a7d53a79/modules/vqgan.py#L613), response for Discriminator architecture. This class is based on `FlaxPreTrainedModel` which gives ous abilities to push the architecture to Hugging Face Hub.

::: modules.vqgan.VQGanDiscriminator

### TensorflowDataset

`TensorflowDataset` in [`modules.utils`](https://github.com/WolodjaZ/jax-vqgan/blob/e67c52cbb02fc39fbbfacc1ad06f40e6a7d53a79/modules/utils.py#L162), response for loading Tensorflow datasets and prepering them. This class is based on [`BaseDataset`](https://github.com/WolodjaZ/jax-vqgan/blob/e67c52cbb02fc39fbbfacc1ad06f40e6a7d53a79/modules/utils.py#L72)

::: modules.utils.TensorflowDataset

::: modules.utils.BaseDataset

### DataLoader

`DataLoader` in [`modules.utils`](https://github.com/WolodjaZ/jax-vqgan/blob/e67c52cbb02fc39fbbfacc1ad06f40e6a7d53a79/modules/utils.py#L185), responses for wraping datasets and creating batches. Similar to [Pytorch Dataloader](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader)

::: modules.utils.DataLoader
