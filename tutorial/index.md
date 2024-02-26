---
layout: page
title: Tutorial
sitemap: false
---

## Installation

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