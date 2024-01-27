# DMLP
A library for training diffusion model

#### Log
1. Perplexity: exponential of loss
original: calculated by model_ppl and tokenizer_ppl
idea: calculate by trained gpt2 and corresponding tokenizer (self.decoder and self.tokenizer_decoder)
question: why they used model_ppl, not even trained

2. apex optimizer and transformer optimizer
https://huggingface.co/transformers/v2.9.1/main_classes/optimizer_schedules.html
https://nvidia.github.io/apex/optimizers.html

More options can be provided according to the documentations

3. Abstract models:
abstract VAE
constructor,
forward
abstract DDPM
constructor
forward
sample
(in evaluation additonal sample options should be enabled)
abstract VAE_DDPM
constructor
forward

Need to check output from VAE_DDPM forward.
This must match train function