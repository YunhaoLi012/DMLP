---
layout: page
title: Tutorial
sitemap: false
---

## Installation
```
pip install DMLP
```

## Demo
We provided a [demo](https://github.com/YunhaoLi012/DMLP/blob/torchamp/tests/test_train.py) to demonstrate typical way of using our library to train text diffusion model from scratch. This script demonstrates the implementation of learning text reconstruction and generation using DMLP. It replicates the result of the the paper ["Generation, Reconstruction, Representation All-in-One: A Joint Autoencoding Diffusion Model"](https://openreview.net/forum?id=bgIZDxd2bM). To run the script, simpily run the following command after installing the library
```
CUDA_VISIBLE_DEVICES=0 torchrun test_train.py
```
Make sure you are in the folder which contains test_train.py file. 

## Tutorial
In order to use DMLP for new datasets/new tasks with default VAE_DDPM model, we need to first select the type of models they would like to use for autoencoder and diffusion model. We have provided a couple of options in ```DMLP.models.my_transformers``` module. Runs the following code to view the available options.

```
from DMLP.models.my_transformers import MODEL_CLASS
print(MODEL_CLASS)
```

###
