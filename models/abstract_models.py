from abc import ABC, abstractmethod
import torch.nn as nn

class VAE_Abs(ABC, nn.Module):

    @abstractmethod
    def __init__(self, encoder, decoder, tokenizer_encoder, tokenizer_decoder, latent_size, output_dir, device=None):
        super(VAE_Abs, self).__init__()
        pass

    @abstractmethod
    def forward(self, inputs, labels):
        pass

class DDPM_Abs(ABC, nn.Module):

    @abstractmethod
    def __init__(self, eps_model, betas, n_T, criterion, ddpm_schedule):
        super(DDPM_Abs, self).__init__()
        pass

    @abstractmethod
    def forward(self, x, mu):
        pass

    @abstractmethod
    def sample(self, n_sample, size, device, fp16=False):
        pass

class VAE_DDPM_Abs(ABC, nn.Module):

    @abstractmethod
    def __init__(self, model_vae, ddpm, ddpm_weight):
        super(VAE_DDPM_Abs, self).__init__()
        pass
    
    @abstractmethod
    def forward(self, inputs, labels):
        pass