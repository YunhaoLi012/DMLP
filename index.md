# DMLP
DMLP is a versatile Python library designed for the training, evaluation, and development of text diffusion models. This library focuses on an architecture synthesizing variational autoencoders (VAEs) with diffusion process, which enables text representation, reconstruction, and generation in one model. DMLP comes equipped with pre-defined functions and classes, offering users a platform to implement, experiment, and compare customized text diffusion models.

It provides:

- APIs for constructing and training/fine-tuning text diffusion model 
- Abstract classes for developing models in text diffusion


   
<!-- toc -->

- [Installation](#installation)
  - [On Linux](#on-linux)
- [Tutorial](#tutorial)
- [Report](#teport)
- [Documentation](#tocumentation)
- [Authors](#authors)

<!-- tocstop -->

## Introduction
----------------------
Diffusion models are generative models originally designed for image generation, interpolation, and reconstruction. Nowadays, combining with variational autoencoder (VAE), diffusion models are extended to natural language processing (NLP), and text-diffusion has become a novel challenging field to explore. In this capstone project, we will introduce Diffusion Model Learning Platform (DMLP), a versatile Python library designed for the training, evaluation, and development of text diffusion models.
DMLP is based on the Joint Autoencoding Diffusion (JEDI) proposed in "GENERATION, RECONSTRUCTION, REPRESENTATION
ALL-IN-ONE: A JOINT AUTOENCODING DIFFUSION MODEL". JEDI was an architecture designed for generation, reconstruction, and representation in image, text, and gene fields, and DMLP focuses on the text-diffusion component - generation and reconstruction of sentences.
DMLP modularizes the JEDI model and allows a flexible combination of different VAE models and diffusion process. Both a predefined JEDI structure and abstract models are provided for customized training tasks. Moreover, DMLP contains a complete training, evaluation, and saving pipeline that reduces implementation workload. The significance of DMLP lies in its versatility, fostering innovation and exploration in the evolving field of text-diffusion. Its comprehensive functions further streamlines the development process, making DMLP a resource for advancing research and applications in NLP.

## Installation
----------------------
### On Linux
```
pip install -i https://test.pypi.org/simple/ DMLP==0.0.1
```

## Tutorial
We provide a demo file in at https://github.com/YunhaoLi012/DMLP/blob/torchamp/tests/test_train.py . This script replicate the result of the following paper https://openreview.net/forum?id=bgIZDxd2bM . To run the code, simpily run the following command
```
CUDA_VISIBLE_DEVICES=0 torchrun test_train.py
```
Make sure you are in the folder which contains test_train.py file. 

In order to Use our library for new datasets/new tasks, here is what you need to do:

## Report
The full report can be accessed from: 

## Documentation
The full documentation can be accessed from: 
Here is an outline:

## Authors


