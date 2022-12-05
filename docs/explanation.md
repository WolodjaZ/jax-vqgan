## What it is

Well I already explained in main but ones again. This is JAX/Flax implementation of VQGAN architecture

[JAX](https://jax.readthedocs.io/en/latest/index.html) (Just After eXecution) is a recent machine/deep learning library developed by DeepMind and Google. Unlike Tensorflow, JAX is not an official Google product and is used for research purposes. The use of JAX is growing among the research community due to some really cool features. Additionally, the need to learn new syntax to use JAX is reduced by its NumPy-like syntax.

![JAX](https://geekflare.com/wp-content/uploads/2022/08/GoogleJAXFeatures.jpg)

[Flax](https://flax.readthedocs.io/en/latest/index.html) is a high-performance neural network library for JAX that is designed for flexibility: Try new forms of training by forking an example and modifying the training loop, not by adding features to a framework.

#TODO: @WolodjaZ add source

VQGAN (Vector Quantized Generative Adversarial Network): VQGAN is a GAN architecture which can be used to learn and generate novel images based on previously seen data. It was first introduced for [`Taming Transformers`](https://arxiv.org/abs/2012.09841) (2021). It works by first having image data directly input to a GAN to encode the feature map of the visual parts of the images. This image data is then vector quantized: a form of signal processing which encodes groupings of vectors into clusters accessible by a representative vector marking the centroid called a “codeword.” Once encoded, the vector quantized data is recorded as a dictionary of codewords, also known as a codebook. The codebook acts as an intermediate representation of the image data, which is then input as a sequence to a transformer. The transformer is then trained to model the composition of these encoded sequences as high resolution images as a generator.

![VQGAN](https://raw.githubusercontent.com/CompVis/taming-transformers/master/assets/teaser.png)

## Why

Well, I wrote this project to understand how VQGAN works and to train my skills in JAX/Flax. Additionally, I wanted to learn how to integrate with Hugging Face and make a cool professional project with all project developing aspects. I think that most research projects should be well documented and well written with testing and stuff because it makes it reproducible, readable, and infalliable.

## What to do with it

With this, you can use trained VQGAN architecture in JAX projects. You can also train it with your dataset. It is research project after all, so you can take it and develop your solutions based on my implementation. JAX/Flax is a fast and flexible library for developing DNN models. VQGAN is an encoder-decoder architecture, which is used in [Stable Diffusion](https://www.google.com/search?client=safari&rls=en&q=stable+diffusion&ie=UTF-8&oe=UTF-8), for example.
