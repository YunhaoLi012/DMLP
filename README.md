# DMLP
DMLP is a python library for training diffusion model
- **Website: https://yunhaoli012.github.io/DMLP/** 
- **Documentation:** 
- **Mailing list:** 
- **Source code:** 
- **Contributing:** 
- **Bug reports:** 

It provides:

- APIs for constructing and training/fine-tuning text diffusion model 
- Abstract classes for developing models in text diffusion


   
<!-- toc -->

- [Installation](#installation)
- [Tutorial](#Tutorial)



<!-- tocstop -->

## Installation
----------------------
```
pip install DMLP
```

## Tutorial
We provide a demo file in at https://github.com/YunhaoLi012/DMLP/blob/torchamp/tests/test_train.py . This script replicate the result of the following paper https://openreview.net/forum?id=bgIZDxd2bM . To run the code, simpily run the following command after installing DMLP.
```
CUDA_VISIBLE_DEVICES=0 torchrun test_train.py
```
Make sure you are in the folder which contains test_train.py file. 

