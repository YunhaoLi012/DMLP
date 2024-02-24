# DMLP

---
authors:
  - Jieqi Liu jil146@ucsd.edu
  - Yunhao Li yul080@ucsd.edu
  - Zhiting Huzhh019@ucsd.edu
---

DMLP is a versatile Python library designed for the training, evaluation, and development of text diffusion models. This library focuses on an architecture synthesizing variational autoencoders (VAEs) with diffusion process, which enables text representation, reconstruction, and generation in one model. DMLP comes equipped with pre-defined functions and classes, offering users a platform to implement, experiment, and compare customized text diffusion models.

It provides:

- APIs for constructing and training/fine-tuning text diffusion model 
- Abstract classes for developing models in text diffusion


   
<!-- toc -->

- [Installation](#installation)
  - [On Linux](#on-linux)
- [Tutorial](#Tutorial)



<!-- tocstop -->

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