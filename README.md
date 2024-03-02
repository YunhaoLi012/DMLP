# DMLP
DMLP is a python library for training diffusion model
- **Website:** 
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
  - [On Linux](#on-linux)
- [Tutorial](#Tutorial)



<!-- tocstop -->

## Installation
----------------------
### On Linux
```
pip install DMLP==1.0.0
```

## Tutorial
We provide a demo file in at https://github.com/YunhaoLi012/DMLP/blob/torchamp/tests/test_train.py . This script replicate the result of the following paper https://openreview.net/forum?id=bgIZDxd2bM . To run the code, simpily run the following command
```
CUDA_VISIBLE_DEVICES=0 torchrun test_train.py
```
Make sure you are in the folder which contains test_train.py file. 

