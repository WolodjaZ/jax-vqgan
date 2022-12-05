# Welcome to VQGAN implementation in JAX/Flax

This site contains the project documentation for the `jax-vqgan`.

## What you need to know

[JAX](https://jax.readthedocs.io/en/latest/index.html) (Just After eXecution) is a recent machine/deep learning library developed by DeepMind and Google. Unlike Tensorflow, JAX is not an official Google product and is used for research purposes. The use of JAX is growing among the research community due to some really cool features. Additionally, the need to learn new syntax to use JAX is reduced by its NumPy-like syntax.

[Flax](https://flax.readthedocs.io/en/latest/index.html) is a high-performance neural network library for JAX that is designed for flexibility: Try new forms of training by forking an example and by modifying the training loop, not by adding features to a framework.

#TODO: @WolodjaZ nie gan architekturę tylko autoencoder tory dodatkowo korzysta z dyskryminatora z architektury Ganu + źródło

VQGAN (Vector Quantized Generative Adversarial Network): VQGAN is a GAN architecture, which can be used to learn and generate novel images based on previously seen data. It was first introduced for the paper [`Taming Transformers`](https://arxiv.org/abs/2012.09841) (2021). It works by first having image data directly input to a GAN to encode the feature map of the visual parts of the images. This image data is then vector quantized: a form of signal processing which encodes groupings of vectors into clusters accessible by a representative vector marking the centroid called a “codeword.” Once encoded, the vector quantized data is recorded as a dictionary of codewords, also known as a codebook. The codebook acts as an intermediate representation of the image data, which is then input as a sequence to a transformer. The transformer is then trained to model the composition of these encoded sequences as high resolution images as a generator.

![VQGAN](https://raw.githubusercontent.com/CompVis/taming-transformers/master/assets/teaser.png)

This project basically provides the implementation of VQGAN in JAX/Flax with the Trainer module, dataset loading and Tensorboard logging. In the future, we will try to add the ability to ship model into [Hugging Face](https://huggingface.co). This is my one of the first well done project, so if you have any advice how to improve or you see some problem, please, make an issue or pull request 😊(evil smiley face).

## Table Of Contents

The documentation follows the best practices for project documentation as described by Daniele Procida in the [Diátaxis documentation framework](https://diataxis.fr/) and consists of four separate parts and the [changelog](https://keepachangelog.com/en/1.0.0/):

1. [Tutorials](tutorials.md)
2. [How-To Guides](how-to-guides.md)
3. [Reference](reference.md)
4. [Explanation](explanation.md)
5. [Changelog](changelog.md)

## Acknowledgements

I want to thank me, myself and I 🥸. No, but honestly, I am thankful to the `Taming Transformers` authors for this marvelous architecture and this [repo](https://github.com/patil-suraj/vqgan-jax), on which I based my implementation (or stole some code, one can say 👤💰).
FURTHERMORE, YOU MUST WATCH THIS [TIKTOK USER](https://www.tiktok.com/@niebodieta?_t=8XZwt4OIP1q&_r=1)
AND LASTLY, I advise to listen to it, as it was my coding song for this project. Click on Captain Cat Sparrow made by [DALLE 2](https://openai.com/dall-e-2/) to listen it 🎧.
[![Captain Cat Sparrow](https://preview.redd.it/r70xbipvlgl91.jpg?width=640&crop=smart&auto=webp&s=5027a08a701c678299569207a2b9b964eb324f59)](https://www.youtube.com/watch?v=0C3zgYW_FAM "Island in The Sun - Click to Listen!")
