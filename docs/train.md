---
layout: page
title: Train
description: >
  Page for train modules documentation
hide_description: true
sitemap: false
---
- [Train](#train)
    - [train_vae_ddpm](#train_vae_ddpm)

- [Evaluation](#evaluation)
    - [calc_rec_lgy](#calc_rec_lgy)
    - [calc_ppl_lgy_ddpm](#calc_ppl_lgy_ddpm)


## Train
The training pipe line is writted in the function `train_vae_ddpm`. \
The function can be imported by `from DMLP.train.train_function import train_vae_ddpm`

### train_vae_ddpm
```
__Args__  
  local_rank: GPU device. This will be passed to the function automatically by CUDA_VISIBLE_DEVICE  
  world_size: the number of GPUs used
  model: vae_ddpm model  
  optimizer: optimizer for training
  train_dataloader: train_dataloader
  output_dir: directory to save checkpoints, example outputs, and logs
  batch_size: batch_size for logging; This will NOT affect dataloader
  condition_f=lambda x: False: A function selecting model parameters
  logging_steps = -1: logging step for logging and evaluation
  train_epoch = 20: Number of training epoch
  gradient_accumulation_steps = 1: gradient accumulation during training
  device = 'cpu': model device
  fp16=False: useless now; keep it false
  fp16_opt_level=None: useless now; leave it None
  learning_rate=9e-5: learning rate
  adam_epsilon=1e-5: eps for optimizer
  lr_end_multiplier= 0.01: An argument for `transformers.get_polynomial_decay_schedule_with_warmup`; please refer to their documentation
  power=3.0: An argument for `transformers.get_polynomial_decay_schedule_with_warmup`; please refer to their documentation
  warmup_steps=0: An argument for `transformers.get_polynomial_decay_schedule_with_warmup`; please refer to their documentation
  disable_bar=True: turn on or off tqdm bar
  max_grad_norm=1: paramter for `torch.nn.utils.clip_grad_norm_` to save sapce
  save=True: save checkpoint or not, True only if `evaluate_during_training=True`
  evaluate_during_training=False: evaluate model if True; False if no evaluation data
  eval_dataloader=None: evaluation dataloader
  sent_length=32: sentence length for generation
  model_id='gpt2': model to evaluate sentence generation perplexity; gpt by default
  ppl_eval=True: evaluate perplexity or not
```

## Evaluation

### calc_rec_lgy

### calc_ppl_lgy_ddpm