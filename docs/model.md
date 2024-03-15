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
    - [MLPSkipNet](#mlpskipnet)
- [My Transformers](#my-transformer)
    - [Model Class](#model-class)
# Abstract Models
We provide abstract model classes to serve as the backbone for text diffusion model. We built our default models
upon these abstract classes. Users can also implement their own models with our abstract classes. Users can
import these classes with ```from DMLP.models.abstract_models import *```. The following section details each
of the abstract classes and abstract methods they contain.

### VAE abs
> ```Class DMLP.abstract_models.VAE_abs(encoder, decoder, *args, device=None,**kwargs) ``` 

Base class for variational auto-encoder. Your models should be a subclass of this class. 
__Args__  
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



### DDPM abs
> ```Class DMLP.abstract_models.DDPM_abs(eps_model, betas, n_T, criterion, ddpm_schedule, *args, **kwargs) ```

Base class for Denoising Diffusion Probabilistic Model(DDPM). Your models should be a subclass of this class
__Args__  
  eps_model: $$P_{\theta}$$, Model for backward denoising process, should be some types of neural network  
  betas: Parameters for ddpm scheduler  
  n_t: Number of steps for diffusion/denoising  
  criterion: Objective function for calculating the diffusion loss  
  ddpm_schedule: scheduler that Returns pre-computed schedules for DDPM sampling, training process. [reference](utils.md)

An example of usage can be found [here](https://github.com/YunhaoLi12138/DMLP/blob/main/DMLP/models/models.py)

### VAE DDPM abs
> ```Class DMLP.abstract_models.VAE_DDPM_Abs(model_vae, ddpm, ddpm_weight, *args, **kwargs) ```

Base class for the complete VAE_DDPM structure model. Combine initialized VAE and DDPM and form a new VAE_DDPM object.
```
__Args__  
  model_vae: Initialized Variational Auto Encoder, should be a subclass of VAE_Abs  
  ddpm: Initialized DDPM, should be a subclass of DDPM_Abs
  ddpm_weight: hyperparameter $$\alpha$$ that adjust weight of ddpm loss in the total loss.  
  <div align="center"> $$\textbf{Loss} = \textbf{reconstruction loss} + \alpha \cdot \textbf{ddpm loss}$$</div>

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
> ```Class DMLP.models.models.VAE(encoder, decoder, tokenizer_encoder, tokenizer_decoder, latent_size, output_dir, device=None)```

Implementation of default variational autoencoder with transformer encoder and decoder. 

__Args__  
  encoder: model use to encode input text into latent space  
  decoder: model use to decode latent space representation into text  
  tokenizer_encoder: tokenizer for encoder to encode text to tokens  
  tokenizer_decoder: tokenizer for decoder to decode tokens to text  
  latent_size: hyperparmeters for latent size representation  
  output_dir: directory to save checkpoints   

__Variables__  
  ```reparametrized(mu, logvar, nsamples)```  sample from posterior Gaussian family
> __Parameters__:  
    mu: Tensor, Mean of gaussian distribution with shape (batch, nz)  
    logvar: Tensor, logvar of gaussian distibution with shape (batch, nz)  
> __Returns__:  
    Tensor, Sampled z with shape (batch,nz)  

```forward(inputs, labels)``` Define forward computation of VAE and compute reconstruction losses
> __Parameters__:  
    inputs: encoder tokenized text  
    labels: decoder toeknized text  
> __Return__:  
    loss_rec: Loss between generated text and target text
    loss_kl: KL distance between generated text and the diffusion model
    loss: loss_rec / sentence length
    latent_z: latent representation of input text
    mu: Tensor, Mean of gaussian distribution with shape (batch, nz)  


### DDPM
> ```Class DMLP.models.models.DDPM(eps_model, betas, n_T, criterion, ddpm_schedule)```
Implementation of default DDPM. 

__Args__  
  eps_model: $$P_{\theta}$$, Model for backward denoising process, should be some types of neural network  
  betas: Parameters for ddpm scheduler  
  n_t: Number of steps for diffusion/denoising  
  criterion: Objective function for calculating the diffusion loss  
  ddpm_schedule: scheduler that Returns pre-computed schedules for DDPM sampling, training process. [reference](utils.md)  

__Variables__  
  ```forward(x,mu)```  Makes forward diffusion x_t, and tries to guess epsilon value from x_t using eps_model.
> __Parameters__:  
  x: latent representation from encoder  
  mu: mean of latent representation distribution from encoder  
> __Return__:  
    loss: loss between ddpm prediction $$z_0$$ and actual $$x_0$$  

```sample(n_sample, size, device, fp16=False)```  Generate latent representation using denoising process  
> __Args__  
  n_sample: number of sentence to generate  
  size: length of the sentence  
  device:  GPU or CPU  
  fp16: whether compute at lower precision  

> __Return__  
    x_i: generated sentence latent representation

### VAE DDPM
> ```Class DMLP.models.models.VAE_DDPM(model_vae, ddpm, ddpm_weight)```  
Implementation of the Complete VAE_DDPM model.  

__Args__  
  model_vae: Variational Autoencoder  
  ddpm: DDPM  
  ddpm_weight: hyperparameter $$\alpha$$ that adjust weight of ddpm loss in the total loss.  
  <div align="center"> $$\textbf{Loss} = \textbf{reconstruction loss} + \alpha \cdot \textbf{ddpm loss}$$</div>  

__Variables__  
```forward(inputs, labels)```  Forward Computation of VAE_DDPM.  
> __Parameters__:  
  inputs: input text tokens  
  labels: output text tokens   
> __Return__:  
    All outputs from VAE and DDPM. For details check above description

### MLPSKipNet 
(TransformerNet, LinearModel, ResidualLinear has similar inputs)
> ```Class DMLP.models.models.MLPSkipNet(latent_dim)```  
Implementation of MLP with skip connection. 
Neural network that mimic $$q_{\theta}$$ in the forward process.

__Args__  
  latent_dim: latent representation size

__Variables__  
```forward(x,t,z_sem=None)``` Forward computation of MLPskipNet.  
> __Parameters__:  
  x: sampled x_t  
  t: number of diffusion steps  

> __Return__:  
  h: Latent space representation of generated text



# My Transformers
We provides 15 different transformers implementation as the options for VAE encoder/decoder. To access 
all models, import ```MODEL_CLASS``` from ```DMLP.models.my_transformers```.

### Model Class
```
MODEL_CLASS = {'BertForLatentConnector':BertForLatentConnector,
               'BertForLatentConnectorAVG':BertForLatentConnectorAVG,
               'BertForLatentConnectorNew':BertForLatentConnectorNew,
               'RobertaForLatentConnector':RobertaForLatentConnector,
               'RobertaForLatentConnectorNew':RobertaForLatentConnectorNew,
               'DebertaForLatentConnector':DebertaForLatentConnector,
               'DebertaForLatentConnectorNew':DebertaForLatentConnectorNew,
               'T5EncoderForLatentConnector':T5EncoderForLatentConnector,
               'GPT2ModelForVAE':GPT2ModelForVAE,
               'GPT2ForLatentConnector':GPT2ForLatentConnector,
               'GPT2ModelForVAENew':GPT2ModelForVAENew,
               'GPT2ForLatentConnectorNew':GPT2ForLatentConnectorNew,
               'GPT2ModelForVAENew2':GPT2ModelForVAENew2,
               'GPT2ForLatentConnectorNew2':GPT2ForLatentConnectorNew2,
               'AlbertForLatentConnector':AlbertForLatentConnector}
```

Our implementation based on 6 types of common used large language model: [BERT](https://huggingface.co/prajjwal1/bert-small), [RoBERTa](https://huggingface.co/docs/transformers/model_doc/roberta), [DeBERTa](https://huggingface.co/docs/transformers/en/model_doc/deberta), [T5](https://huggingface.co/docs/transformers/en/model_doc/t5), [GPT2](https://huggingface.co/openai-community/gpt2-xl), [ALBERT](https://huggingface.co/docs/transformers/en/model_doc/albert). Click 
through the link to read about how to download each pretrained model. Examples also provided in [here](../tutorial/index.md)
