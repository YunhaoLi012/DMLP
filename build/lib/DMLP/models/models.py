import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from DMLP.models.log_sum_exp import log_sum_exp

import numpy as np
import logging
from typing import Dict, Tuple
from DMLP.models.abstract_models import *

class VAE(VAE_Abs):
    def __init__(self, encoder, decoder, tokenizer_encoder, tokenizer_decoder, latent_size, output_dir, device=None):
        super(VAE, self).__init__(encoder, decoder, device=device)
        self.encoder = encoder
        self.decoder = decoder
        self.nz = latent_size
        self.output_dir = output_dir

        self.eos_token_id = tokenizer_decoder.convert_tokens_to_ids([tokenizer_decoder.eos_token])[0]
        self.pad_token_id = tokenizer_decoder.convert_tokens_to_ids([tokenizer_decoder.pad_token])[0]
        self.bos_token_id = tokenizer_decoder.convert_tokens_to_ids([tokenizer_decoder.bos_token])[0]
        self.tokenizer_decoder = tokenizer_decoder
        self.tokenizer_encoder = tokenizer_encoder
        # Standard Normal prior
        loc = torch.zeros(self.nz, device=device)
        scale = torch.ones(self.nz, device=device)
        self.prior = torch.distributions.normal.Normal(loc, scale)
    
    def reparameterize(self, mu, logvar, nsamples=1):
        """sample from posterior Gaussian family
        Args:
            mu: Tensor
                Mean of gaussian distribution with shape (batch, nz)
            logvar: Tensor
                logvar of gaussian distibution with shape (batch, nz)
        Returns: Tensor
            Sampled z with shape (batch, nsamples, nz)
        """
        batch_size, nz = mu.size()
        std = logvar.mul(0.5).exp()

        mu_expd = mu.unsqueeze(1).expand(batch_size, nsamples, nz)
        std_expd = std.unsqueeze(1).expand(batch_size, nsamples, nz)

        eps = torch.zeros_like(std_expd).normal_()

        return mu_expd + torch.mul(eps, std_expd)
    
    def forward(self, inputs, labels):

        attention_mask = (inputs!=self.tokenizer_encoder.pad_token_id).float()

        reconstruction_mask = (labels != self.tokenizer_decoder.pad_token_id).float()
        sent_length = torch.sum(reconstruction_mask, dim=1)

        outputs = self.encoder(inputs, attention_mask)
        pooled_hidden_fea = outputs[1]

        mu, logvar = self.encoder.linear(pooled_hidden_fea).chunk(2, -1)
        
        logvar = torch.log(torch.ones_like(logvar) * 0.008)
        loss_kl = 0.5 * (mu.pow(2) + logvar.exp() - logvar - 1).sum(dim=1)
        latent_z = self.reparameterize(mu, logvar, nsamples=1)
        latent_z = latent_z.squeeze(1)
        outputs = self.decoder(input_ids=labels, past=latent_z, labels=labels, label_ignore=self.pad_token_id)
        loss_rec = outputs[0]
        loss = loss_rec / sent_length
        return loss_rec, loss_kl, loss, latent_z, mu
    
class DDPM(DDPM_Abs):

    def __init__(self, eps_model, betas, n_T, criterion, ddpm_schedule):
        super(DDPM, self).__init__(eps_model, betas, n_T, criterion, ddpm_schedule)
        self.eps_model = eps_model

        # register_buffer allows us to freely access these tensors by name. It helps device placement.
        for k, v in ddpm_schedule(betas[0], betas[1], n_T).items():
            self.register_buffer(k, v)

        self.n_T = n_T
        self.criterion = criterion
    
    def forward(self, x, mu):
        """
        Makes forward diffusion x_t, and tries to guess epsilon value from x_t using eps_model.
        This implements Algorithm 1 in the paper.
        """

        _ts = torch.randint(1, self.n_T+1, (x.shape[0],)).to(device=x.device) # t ~ Uniform(0, n_T-1)  before: (1, n_T-1)
        eps = torch.randn_like(x)  # eps ~ N(0, 1)   
        x_t = (
            self.sqrtab[_ts, None] * x
            + self.sqrtmab[_ts, None] * eps
        )  # This is the x_t, which is sqrt(alphabar) x_0 + sqrt(1-alphabar) * eps
        # We should predict the "error term" from this x_t. Loss is what we return.
        eps_0 = self.eps_model(x_t, _ts / self.n_T)
        loss =  self.criterion(eps, eps_0).mean(1)  # compute loss 64 * 1
        mask_1 = (_ts == 1)
        if mask_1.any():
            loss_z0 = self.alpha_t[1]/self.beta_t[1] * self.criterion(mu, self.oneover_sqrta[1]*(x_t - self.sqrt_beta_t[1]*eps_0)).mean(1)
            loss = torch.where(mask_1,loss_z0, loss) 
        
        return loss,self.loss_weight[_ts, None]
    
    def sample(self, n_sample, size, device, fp16=False):
        dtype_ = torch.half if fp16 else torch.float
        x_i = torch.randn(n_sample, *size).to(device=device,dtype=dtype_)
        for i in range(self.n_T, 0, -1):
            z = torch.randn(n_sample, *size).to(device=device,dtype=dtype_)  if i > 1 else 0
            ts_ = torch.tensor(i).to(device=device,dtype=dtype_) / self.n_T
            ts_ = ts_.repeat(n_sample)
            eps = self.eps_model(x_i, ts_)
            x_i = (
                self.oneover_sqrta[i] * (x_i - eps * self.mab_over_sqrtmab[i]) # attributes from ddpm_schedule
                + self.sqrt_beta_t[i] * z
            )
        return x_i

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

class TimeSiren(nn.Module):
    def __init__(self, emb_dim: int) -> None:
        super(TimeSiren, self).__init__()
        self.emb_dim = emb_dim
        self.lin1 = nn.Linear(1, emb_dim, bias=False)
        self.lin2 = nn.Linear(emb_dim, emb_dim)
        self.act = nn.LeakyReLU()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(-1, 1)
        # x = torch.sin(self.lin1(x))
        # x = self.lin2(x)
        x = self.lin1(x)
        x = self.act(x)
        x = self.lin2(x)
        x = torch.sin(x)
        return x

blk_linear = lambda ic, oc: nn.Sequential(
    nn.Linear(ic, oc),
    nn.LeakyReLU(),
)

class ResidualLinear(nn.Module):
    def __init__(self, latent_dim=64):
        super(ResidualLinear, self).__init__()
        self.norm = nn.LayerNorm(latent_dim,1e-6)
        self.linear = blk_linear(latent_dim, latent_dim)
        self.dropout = nn.Dropout(0.1)
    def forward(self, x):
        y = self.norm(x)
        y = self.linear(y)
        # y = self.dropout(y)
        # y = x + y
        return y
class LinearModel(nn.Module):
    def __init__(self, latent_dim=64):
        super(LinearModel, self).__init__()
        self.timeembed = TimeSiren(latent_dim)
        self.linear = nn.Sequential(
            blk_linear(latent_dim, latent_dim*2 ),
            blk_linear(latent_dim * 2, latent_dim),
            nn.Linear(latent_dim,latent_dim)
        )
        # self.linear = nn.Sequential(
        #     ResidualLinear(),
        #     ResidualLinear(),
        #     nn.LayerNorm(latent_dim,eps=1e-6),
        #     # nn.Linear(latent_dim,latent_dim)
        # )

    def forward(self, x, t):
        temb = self.timeembed(t)
        return self.linear(x+temb)

def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.
    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) *
                   torch.arange(start=0, end=half, dtype=torch.float32) /
                   half).to(device=timesteps.device)
    args = timesteps[:, None] * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat(
            [embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding

class TransformerNet(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.num_latent = 4
        self.hidden_dim = 512
        self.linear_1 = nn.Linear(latent_dim, self.num_latent * self.hidden_dim)
        self.linear_2 = nn.Linear(self.num_latent * self.hidden_dim, latent_dim)
        self.time_embed_dim = 64
        num_layers = 12
        
        self.time_embed = nn.Sequential(
            nn.Linear(self.time_embed_dim, latent_dim),
            nn.SiLU(),
            nn.Linear( latent_dim,latent_dim)
        )
        self.transformer_encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=self.hidden_dim, nhead=8), num_layers=num_layers)
        
    def forward(self, x, t, z_sem=None):
        t = timestep_embedding(t, self.time_embed_dim).to(x.dtype)
        cond = self.time_embed(t)
        x = self.linear_1(x+cond)
        x = x.view(x.shape[0],self.num_latent,-1)
        # layers
        x = self.transformer_encoder(x).view(x.shape[0],-1)
        x = self.linear_2(x)
        return x
    
class MLPSkipNet(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        # time embedding
        # self.latent_dim=latent_dim
        # self.dae_mlp = nn.Sequential(
        #     nn.Linear(latent_dim, latent_dim//2),
        #     nn.SiLU(),
        #     nn.Linear(latent_dim//2,latent_dim)
        # )
        self.time_embed_dim = 64
        self.time_embed = nn.Sequential(
            nn.Linear(self.time_embed_dim, latent_dim),
            nn.SiLU(),
            nn.Linear(latent_dim,latent_dim)
        )
        # MLP layers
        self.activation = 'silu'
        use_norm = True
        num_layers =20
        num_hid_channels = 2048 # latent_dim * 4
        num_channels = latent_dim
        condition_bias=1
        dropout = 0
        self.skip_layers = list(range(1, num_layers))
        self.layers = nn.ModuleList([])
        for i in range(num_layers):
            if i == 0:
                act = self.activation
                norm = use_norm
                cond = True
                a, b = num_channels, num_hid_channels
                dropout = dropout
            elif i == num_layers - 1:
                act = 'none'
                norm = False
                cond = False
                a, b = num_hid_channels, num_channels
                dropout = dropout
            else:
                act = self.activation
                norm = use_norm
                cond = True
                a, b = num_hid_channels, num_hid_channels
                dropout = dropout

            if i in self.skip_layers:
                a += num_channels

            self.layers.append(
                MLPLNAct(
                    a,
                    b,
                    norm=norm,
                    activation=act,
                    cond_channels=num_channels,
                    use_cond=cond,
                    condition_bias=condition_bias,
                    dropout=dropout,
                ))
        self.last_act = nn.Identity()

    def forward(self, x, t, z_sem=None):
        # time embedding
        # t *= 2000
        
        t = timestep_embedding(t, self.time_embed_dim).to(x.dtype)
        cond = self.time_embed(t)
        h = x
        # except:
        #     cond = self.time_embed(t)
        #     h = x.float()
        if z_sem is not None:
            cond += z_sem
        # layers
        
        for i in range(len(self.layers)):
            if i in self.skip_layers:
                h = torch.cat([h, x], dim=1)
            h = self.layers[i].forward(x=h, cond=cond)
        h = self.last_act(h)
        return h

class MLPLNAct(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        norm: bool,
        activation: str,
        use_cond: bool,
        cond_channels: int,
        condition_bias: float = 0,
        dropout: float = 0,
    ):
        super().__init__()
        self.condition_bias = condition_bias
        self.use_cond = use_cond
        self.activation = activation
        self.linear = nn.Linear(in_channels, out_channels)
        if activation == 'silu':
            self.act = nn.SiLU()
        else:
            self.act = nn.Identity()
        if self.use_cond:
            self.linear_emb = nn.Linear(cond_channels, out_channels)
            self.cond_layers = nn.Sequential(self.act, self.linear_emb)
        if norm:
            self.norm = nn.LayerNorm(out_channels)
        else:
            self.norm = nn.Identity()

        if dropout > 0:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = nn.Identity()

        self.init_weights()

    def init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                if self.activation == 'silu':
                    nn.init.kaiming_normal_(module.weight,
                                             a=0,
                                            nonlinearity='relu')
                else:
                    pass

    def forward(self, x, cond=None):
        x = self.linear(x)
        if self.use_cond:
            # (n, c) or (n, c * 2)
            cond = self.cond_layers(cond)
            cond = (cond, None)

            # scale shift first
            x = x * (self.condition_bias + cond[0])
            if cond[1] is not None:
                x = x + cond[1]
            # then norm
            x = self.norm(x)
        else:
            # no condition
            x = self.norm(x)
        x = self.act(x)
        x = self.dropout(x)
        return x
