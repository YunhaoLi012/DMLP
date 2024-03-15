---
layout: page
title: Utils
description: >
  Page for utils modules documentation
hide_description: true
sitemap: false
---


The utils module provide a couple of helper function for training and evaluation.

- [ddpm_schedule](#ddpm_schedule)
- [random_init](#random_init)
- [sample_sequence_conditional](#sample_sequence_conditional)
- [save_checkpoint](#save_checkpoint)


## ddpm_schedule
```Function DMLP.utils.ddpm_schedule.ddpm_schedule(beta1:float, beta2: float, T:int) -> Dict[str, torch.Tensor]```
Generate parameters for diffusion/denosing

__Args__  
  beta1: hyperparameter for ddpm scheduler  
  beta2: hyperparameter for ddpm scheduler  
  T: number of steps for diffusion process  

__Return__  
  A dictionary of variables/parameters for diffusion/denoising process  

## random_init  
```Function DMLP.utils.random_init.random_init(model)```  
randomly initialize model weights  

__Args__  
  model: model we want to initialize weights  

__Return__  
  None

## sample_sequence_conditional
```Function DMLP.utils.sample.sample_sequence_conditional(model, length, context, past=None, num_samples=1, temperature=1,top_k=0, top_p=0.0,device='cpu', decoder_tokenizer=None, eos_id=50259, loss=False)```  
Generate text given a past token. If past is none, the function generate new sentence. If past is not none, the function 
reconstruct token based on past.

__Args__  
  model: model use to generate text  
  length: maximum sentence length  
  context: context token, usually bos token  
  past: latent representation of input text. If past is none, we are generating sentence with ddpm.  
  num_samples: number of sentence generate  
  temperature: temperature to normalized conditional probabilities  
  device: device for computation  
  decoder_tokenizer: tokenizer for selected decoder  
  eos_id: end of sentence id  
  loss: whether to calculate reconstruction loss

__Return__  
  Generated text tokens and reconstruction loss if specified in the argument.  

## save_checkpoint  
```Function DMLP.utils.save_checkpoint.save_checkpoint(model_vae, optimizer, global_step, parameter_name, output_dir, logger, ppl=False, ddpm=None, use_philly=False)```  
Save Model checkpoint to output directory based on best bleu score.  

__Args__ 
  model_vae: Variational Autoencoder  
  optimizer: optimizer used to train model  
  global_step: number of iteration trained  
  parameter_name: name of parameters to save  
  output_dir: directory to save checkpoints  
  logger: train logger  
  ddpm: DDPM  

__Return__  
  None