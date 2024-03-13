---
layout: page
title: Model
description: >
  Page for model modules documentation
hide_description: true
sitemap: false
---
Here we details functions and classes available in the model module of DMLP, describing their functionality 
and usage.

- [Abstract Models](#abstract-models)
    - [VAE abs](#vae-abs)
    - [DDPM abs](#ddpm-abs)
    - [VAE DDPM abs](#vae-ddpm-abs)
- [Models](#models)
    - [VAE](#vae)
    - [DDPM](#ddpm)
    - [VAE_DDPM](#vae-ddpm)
    - [TimeSiren](#timesiren)
    - [ResidualLinear](#residuallinear)
    - [LinearModel](#linearmodel)
    - [timestep_embedding](#timestep-embedding)
    - [TransformerNet](#transformernet)
    - [MLPSkipNet](#mlpskipnet)
    - [MLPNAct](#MLPNAct)
- [My Transformers](#my-transformer)

# Abstract Models
We provide abstract model classes to serve as the backbone for text diffusion model. We built our default models
upon these abstract classes. Users can also implement their own models with our abstract classes. Users can
import these classes with ```from DMLP.models.abstract_models import *```. The following section details each
of the abstract classes and abstract methods they contain.

### VAE abs <small> [[source]](https://github.com/YunhaoLi12138/DMLP/blob/main/DMLP/models/abstract_models.py)<small>
> ```Class DMLP.abstract_models.VAE_abs(encoder, decoder, *args, device=None,**kwargs) ``` 

Base class for variational auto-encoder. Your models should be a subclass of this class. 
> __Args__  
  encoder: model use to encode input text into latent space
  decoder: model use to decode latent space representation into text

```
from DMLP.models.abstract_models import VAE_Abs

class VAE(VAE_Abs):
    def __init__(self, encoder, decoder,  device=None):
        super(VAE, self).__init__(encoder, decoder, tokenizer_encoder, tokenizer_decoder, latent_size, output_dir,device=device)
        self.encoder = encoder
        self.decoder = decoder

        self.tokenizer_decoder = tokenizer_decoder
        self.tokenizer_encoder = tokenizer_encoder

        self.eos_token_id = tokenizer_decoder.convert_tokens_to_ids([tokenizer_decoder.eos_token])[0]
        self.pad_token_id = tokenizer_decoder.convert_tokens_to_ids([tokenizer_decoder.pad_token])[0]
        self.bos_token_id = tokenizer_decoder.convert_tokens_to_ids([tokenizer_decoder.bos_token])[0]
    
    def forward(self, inputs, labels):
        attention_mask = (inputs!=self.tokenizer_encoder.pad_token_id).float()

        out = self.encoder(inputs, attention_mask)
        out = self.decoder(input_ids=labels, past=latent_z, labels=labels, label_ignore=self.pad_token_id)
        return out
```



### DDPM abs <small>[[source]](https://github.com/YunhaoLi12138/DMLP/blob/main/DMLP/models/abstract_models.py)<small>
> ```Class DMLP.abstract_models.DDPM_abs(eps_model, betas, n_T, criterion, ddpm_schedule, *args, **kwargs) ```

Base class for Denoising Diffusion Probabilistic Model(DDPM). Your models should be a subclass of this class
> __Args__  
  eps_model: $$P_{\theta}$$, Model for backward denoising process, should be some types of neural network  
  betas: Parameters for ddpm scheduler  
  n_t: Number of steps for diffusion/denoising  
  criterion: Objective function for calculating the diffusion loss  
  ddpm_schedule: scheduler that Returns pre-computed schedules for DDPM sampling, training process. [reference](utils.md)

An example of usage can be found [here](https://github.com/YunhaoLi12138/DMLP/blob/main/DMLP/models/models.py)

### VAE DDPM abs
> ```Class DMLP.abstract_models.VAE_DDPM_Abs(model_vae, ddpm, ddpm_weight, *args, **kwargs) ```

Base class for the complete VAE_DDPM structure model. Combine initialized VAE and DDPM and form a new VAE_DDPM object.

> __Args__  
  model_vae: Initialized Variational Auto Encoder, should be a subclass of VAE_Abs  
  ddpm: Initialized DDPM, should be a subclass of DDPM_Abs
  ddpm_weight: hyperparameter $$\alpha$$ that adjust weight of ddpm loss in the total loss.  
  <div align="center"> $$\textbf{Loss} = \textbf{reconstruction loss} + \alpha \cdot \textbf{ddpm loss}$$</div>
```
from DMLP.models.abstract_models import VAE_DDPM_Abs

class VAE_DDPM(VAE_DDPM_Abs):
    def __init__(self, model_vae, ddpm, ddpm_weight) :
        super(VAE_DDPM, self).__init__(model_vae, ddpm, ddpm_weight)
        self.model_vae = model_vae
        self.ddpm = ddpm
        self.ddpm_weight = ddpm_weight

    def forward(self,inputs, labels): 
        
        loss_rec, loss_kl, loss, latent_z, mu = self.model_vae(inputs, labels)
        ddpm_loss, loss_weight = self.ddpm.forward(latent_z, mu)
        
        if self.ddpm_weight > 0:
            loss = (1/(loss_weight * self.ddpm.n_T)  * loss).mean() + self.ddpm_weight *ddpm_loss.mean()
        else:
            loss = loss.mean() + 0.0* ddpm_loss.mean()
        return loss_rec, loss_kl, loss, latent_z, mu, ddpm_loss, loss_weight
```


# Models
This module provides implementation of default models for users to use out of the box.

### VAE

### DDPM

### VAE DDPM

### Time Siren

### Residual Linear

### Linear Model

### Timestep Embedding

### Transformernet

### MLPSKipNet

### MLPNACT


# My Transformers