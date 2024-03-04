# DMLP
DMLP is a python library for training diffusion model
- **Website: https://yunhaoli12138.github.io/DMLP/** 
- **Documentation: https://yunhaoli12138.github.io/DMLP/docs/** 
- **Mailing list: yul080@ucsd.edu , jil146@ucsd.edu** 
- **Source code: https://github.com/YunhaoLi12138/DMLP** 
- **Contributing: YunhaoLi12138, DDDyylan** 
- **Bug reports: https://github.com/YunhaoLi12138/DMLP/issues** 

It provides:

- APIs for constructing and training/fine-tuning text diffusion model 
- Abstract classes for developing models in text diffusion


   


## Installation
```
pip install DMLP
```

## Tutorial
We provide a demo file in at https://github.com/YunhaoLi012/DMLP/blob/torchamp/tests/test_train.py . This script replicate the result of the following paper https://openreview.net/forum?id=bgIZDxd2bM . To run the code, simpily run the following command after installing DMLP.
```
CUDA_VISIBLE_DEVICES=0 torchrun test_train.py
```
Make sure you are in the folder which contains test_train.py file. 

