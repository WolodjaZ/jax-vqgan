# jax-vqgan: Implementation of VQGAN in JAX/Flax.

This README is a very short. **To learn everything you need to know about this repo, see our [full documentation](https://wolodjaz.github.io/jax-vqgan/)**

[![pdm-managed](https://img.shields.io/badge/pdm-managed-blueviolet)](https://pdm.fming.dev) [![Docs](https://img.shields.io/badge/Docs-mkdocs-blue?style=flat)](https://squidfunk.github.io/mkdocs-material/) ![Github Actions](https://github.com/pdm-project/pdm/workflows/Tests/badge.svg)

If you want to start a project just install [pdm](https://pdm.fming.dev/latest/), init project and install dependencies and of you go. **Everything can be found in the documentation**.
## TODO list

- [ ] Full train and test pipeline 😛. I don't have a GPU for such a task so if anyone would like to lend me a GPU or try to do it themselves and give feedback and results I would be delighted 👌.
- [ ] Make documentation better
- [x] Add package and dependency manager [pdm](https://github.com/pdm-project/pdm)
- [x] ~~Check [jaxtyping](https://github.com/google/jaxtyping) if it time and memory optimize. If yes I think it would be great to add it.~~ I've decided not to use it because it won't provide much readability as I thought, and I don't think we need to provide more input validation in the architecture. However, it doesn't worsen performance, but I'm not fully convinced that it will be a big plus given that we will be dependent on this framework. If anyone has a different opinion pls share maybe I am wrong.
- [x] Make more tests to make coverage test percentage higher 😏. For now at least 90% is enough.
- [ ] Test distributed learning
- [ ] Add pip deployment
- [ ] Add hugging face deployment
- [ ] Optimize trainer
- [ ] Optimize JIT functions
- [ ] Optimize Datasets and Dataloaders
