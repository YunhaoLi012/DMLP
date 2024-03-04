---
layout: page
title: Tutorial
sitemap: false
---

## Installation
```
pip install DMLP
```

## Tutorial
We provide a [demo file](https://github.com/YunhaoLi012/DMLP/blob/torchamp/tests/test_train.py). This script replicate the result of the the paper ["Generation, Reconstruction, Representation All-in-One: A Joint Autoencoding Diffusion Model"](https://openreview.net/forum?id=bgIZDxd2bM). To run the code, simpily run the following command
```
CUDA_VISIBLE_DEVICES=0 torchrun test_train.py
```
Make sure you are in the folder which contains test_train.py file. 

In order to Use our library for new datasets/new tasks, here is what you need to do: