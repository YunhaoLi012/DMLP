import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import log_sum_exp

import pdb
import numpy as np
import logging
from typing import Dict, Tuple

logger = logging.getLogger(__name__)


def reparameterize(mu, logvar, nsamples=1):
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


class VAE(nn.Module):
    """VAE with normal prior"""

    def __init__(self, encoder, decoder, tokenizer_encoder, tokenizer_decoder, latent_size, output_dir, device=None): 
        """
        encoder: encoding model like BERT
        decoder: decoding model like GPT2
        tokenizer_encoder: such as bert-small tokenizer
        tokenizer_decoder: such as gpt2 tokenizer
        latent_size: dimension of latent variables
        device: None or gpu
        """
        super(VAE, self).__init__()
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

    def connect(self, bert_fea, nsamples=1):
        """
        Returns: Tensor1, Tensor2
            Tensor1: the tensor latent z with shape [batch, nsamples, nz]
            Tensor2: the tenor of KL for each x with shape [batch]
        """

        # (batch_size, nz)

        mean, logvar = self.encoder.linear(bert_fea).chunk(2, -1)
        # pdb.set_trace()
        # mean, logvar = mean.squeeze(0), logvar.squeeze(0)
        # z = mean
        # (batch, nsamples, nz)
        z = self.reparameterize(mean, logvar, nsamples)
        KL = 0.5 * (mean.pow(2) + logvar.exp() - logvar - 1).sum(dim=1)

        return z, KL

    def connect_deterministic(self, bert_fea, nsamples=1):
        """
        Returns: Tensor1, Tensor2
            Tensor1: the tensor latent z with shape [batch, nsamples, nz]
            Tensor2: the tenor of KL for each x with shape [batch]
        """

        # (batch_size, nz)

        mean, logvar = self.encoder.linear(bert_fea).chunk(2, -1)
        # pdb.set_trace()
        # mean, logvar = mean.squeeze(0), logvar.squeeze(0)

        logvar.fill_(.0)
        # (batch, nsamples, nz)
        z = self.reparameterize(mean, logvar, nsamples)
        KL = 0.5 * (mean.pow(2) + logvar.exp() - logvar - 1).sum(dim=1)

        return z, KL

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

    def forward(self, inputs, labels, std=False, return_z=False, return_mu=False):

        attention_mask = (inputs!=self.tokenizer_encoder.pad_token_id).float()

        reconstruction_mask = (labels != self.tokenizer_decoder.pad_token_id).float()  # 50257 is the padding token for GPT2
        sent_length = torch.sum(reconstruction_mask, dim=1)

        # with torch.no_grad():
        #     outputs = self.encoder(inputs, attention_mask)
        # pooled_out = self.encoder.pooler(outputs[2])
        # pooled_hidden_fea = pooled_out
        # outputs[0] 是所有hidden:  B x length x hidden
        # outputs[1] 是pooled out: B x hidden
        outputs = self.encoder(inputs, attention_mask)
        pooled_hidden_fea = outputs[1]


         #outputs[1]  # model outputs are always tuple in pytorch-transformers (see doc)
        mu, logvar = self.encoder.linear(pooled_hidden_fea).chunk(2, -1)
        
        logvar = torch.log(torch.ones_like(logvar) * 0.008)
        loss_kl = 0.5 * (mu.pow(2) + logvar.exp() - logvar - 1).sum(dim=1)
        latent_z = self.reparameterize(mu, logvar, nsamples=1)
        latent_z = latent_z.squeeze(1)
        outputs = self.decoder(input_ids=labels, past=latent_z, labels=labels, label_ignore=self.pad_token_id)
        loss_rec = outputs[0]
        loss = loss_rec / sent_length
        return loss_rec, loss_kl, loss, latent_z, mu


    def encode_x(self, inputs, repa=False):
        attention_mask = (inputs!=self.tokenizer_encoder.pad_token_id).float()
        outputs = self.encoder(inputs, attention_mask)
        pooled_hidden_fea = outputs[1]  # model outputs are always tuple in pytorch-transformers (see doc)

        # Connect hidden feature to the latent space
        mu, logvar = self.encoder.linear(pooled_hidden_fea).chunk(2, -1)
        if repa:
            latent_z = self.reparameterize(mu, logvar, nsamples=1)
            latent_z = latent_z.squeeze(1)
        else:
            latent_z = mu
        return latent_z

    def encoder_sample(self, bert_fea, nsamples):
        """sampling from the encoder
        Returns: Tensor1
            Tensor1: the tensor latent z with shape [batch, nsamples, nz]
        """

        # (batch_size, nz)

        mu, logvar = self.encoder.linear(bert_fea).chunk(2, -1)
        mu, logvar = mu.squeeze(0), logvar.squeeze(0)

        # (batch, nsamples, nz)
        z = self.reparameterize(mu, logvar, nsamples)

        return z, (mu, logvar)

    def encode_stats(self, x):
        """
        Returns: Tensor1, Tensor2
            Tensor1: the mean of latent z with shape [batch, nz]
            Tensor2: the logvar of latent z with shape [batch, nz]
        """

        return self.encoder.encode_stats(x)

    def decode(self, z, strategy, K=10):
        """generate samples from z given strategy
        Args:
            z: [batch, nsamples, nz]
            strategy: "beam" or "greedy" or "sample"
            K: the beam width parameter
        Returns: List1
            List1: a list of decoded word sequence
        """

        if strategy == "beam":
            return self.decoder.beam_search_decode(z, K)
        elif strategy == "greedy":
            return self.decoder.greedy_decode(z)
        elif strategy == "sample":
            return self.decoder.sample_decode(z)
        else:
            raise ValueError("the decoding strategy is not supported")

    # def decode_eval_greedy(self, x, z):
    #     #         n_sample, length = x.size()
    #     x_shape = list(x.size())
    #     z_shape = list(z.size())
    #     if len(z_shape) == 3:
    #         x = x.unsqueeze(1).repeat(1, z_shape[1], 1).contiguous().view(x_shape[0] * z_shape[1], x_shape[-1])
    #         z = z.contiguous().view(x_shape[0] * z_shape[1], z_shape[-1])
    #     batch_size = z.size()[0]
    #     decoded_batch = [[] for _ in range(batch_size)]
    #     x_ = torch.zeros_like(z[:, :1], dtype=torch.long) + self.bos_token_id
    #     #         for i in range(length):
    #     mask = torch.zeros_like(z[:, 0], dtype=torch.long) + 1
    #     length_c = 1
    #     end_symbol = torch.zeros_like(mask, dtype=torch.long) + self.eos_token_id
    #     while mask.sum().item() != 0 and length_c < 100:
    #         output = self.decoder(input_ids=x_, past=z)
    #         out_token = output[0][:, -1:].max(-1)[1]
    #         x_ = torch.cat((x_, out_token), -1)
    #         length_c += 1
    #         mask = torch.mul((out_token.squeeze(-1) != end_symbol), mask)
    #         for i in range(batch_size):
    #             #                 word = self.tokenizer_decoder.decode(out_token[i].tolist())
    #             if mask[i].item():
    #                 decoded_batch[i].append(self.tokenizer_decoder.
    #                                         decode(out_token[i].item()))
    #     #         out_tokens = x_[:,1:]
    #     for i in range(batch_size):
    #         decoded_batch[i] = ''.join(decoded_batch[i])

    #     return decoded_batch

    def decode_eval_greedy_tf(self, x, z):
        #         n_sample, length = x.size()
        x_shape = list(x.size())
        z_shape = list(z.size())
        if len(z_shape) == 3:
            x = x.unsqueeze(1).repeat(1, z_shape[1], 1).contiguous().view(x_shape[0] * z_shape[1], x_shape[-1])
            z = z.contiguous().view(x_shape[0] * z_shape[1], z_shape[-1])
        batch_size = z.size()[0]
        decoded_batch = [[] for _ in range(batch_size)]
        x_ = torch.zeros_like(z[:, :1], dtype=torch.long) + self.bos_token_id
        #         for i in range(length):
        mask = torch.zeros_like(z[:, 0], dtype=torch.long) + 1
        length_c = 1
        end_symbol = torch.zeros_like(mask, dtype=torch.long) + self.eos_token_id
        while mask.sum().item() != 0 and length_c < 100 and length_c <= x_shape[-1]:
            output = self.decoder(input_ids=x_, past=z)
            out_token = output[0][:, -1:].max(-1)[1]
            x_ = torch.cat((x_, x[:, length_c:length_c + 1]), -1)
            length_c += 1
            mask = torch.mul((out_token.squeeze(-1) != end_symbol), mask)
            for i in range(batch_size):
                #                 word = self.tokenizer_decoder.decode(out_token[i].tolist())
                if mask[i].item():
                    decoded_batch[i].append(self.tokenizer_decoder.
                                            decode(out_token[i].item()))
        #         out_tokens = x_[:,1:]
        for i in range(batch_size):
            decoded_batch[i] = ''.join(decoded_batch[i])
        return decoded_batch

    def reconstruct(self, x, decoding_strategy="greedy", K=5):
        """reconstruct from input x
        Args:
            x: (batch, *)
            decoding_strategy: "beam" or "greedy" or "sample"
            K: the beam width parameter
        Returns: List1
            List1: a list of decoded word sequence
        """
        z = self.sample_from_inference(x).squeeze(1)

        return self.decode(z, decoding_strategy, K)

    def log_probability(self, x, z):
        """Cross Entropy in the language case
        Args:
            x: (batch_size, seq_len)
            z: (batch_size, n_sample, nz)
        Returns:
            log_p: (batch_size, n_sample).
                log_p(x|z) across different x and z
        """
        #         outputs_ = self.decode_eval_gy(x,z)
        outputs = self.decoder(input_ids=x, past=z, labels=x, label_ignore=self.pad_token_id)
        loss_rec = outputs[0]
        return -loss_rec

    def log_probability_out(self, x, z):
        """Cross Entropy in the language case
        Args:
            x: (batch_size, seq_len)
            z: (batch_size, n_sample, nz)
        Returns:
            log_p: (batch_size, n_sample).
                log_p(x|z) across different x and z
        """
        #         outputs_ = self.decode_eval_gy(x,z)
        outputs = self.decoder(input_ids=x, past=z, labels=x, label_ignore=self.pad_token_id)
        return outputs

    def loss_iw(self, x0, x1, nsamples=50, ns=1):
        """
        Args:
            x: if the data is constant-length, x is the data tensor with
                shape (batch, *). Otherwise x is a tuple that contains
                the data tensor and length list
        Returns: Tensor1, Tensor2, Tensor3
            Tensor1: total loss [batch]
            Tensor2: reconstruction loss shape [batch]
            Tensor3: KL loss shape [batch]
        """

        # encoding into bert features
        bert_fea = self.encoder(x0)[1]
        # (batch_size, nz)

        mu, logvar = self.encoder.linear(bert_fea).chunk(2, -1)

        ##################
        # compute KL
        ##################
        # pdb.set_trace()
        KL = 0.5 * (mu.pow(2) + logvar.exp() - logvar - 1).sum(dim=1)

        # mu, logvar = mu.squeeze(0), logvar.squeeze(0)
        ll_tmp, rc_tmp = [], []
        for _ in range(int(nsamples / ns)):
            # (batch, nsamples, nz)
            z = self.reparameterize(mu, logvar, ns)
            # past = self.decoder.linear(z)
            past = z

            # [batch, nsamples]
            log_prior = self.eval_prior_dist(z)   # z corresponds to N(0,1) log prob
            log_gen = self.eval_cond_ll(x1, past)  # given z, the prob of x
            log_infer = self.eval_inference_dist(z, (mu, logvar))

            # pdb.set_trace()
            log_gen = log_gen.unsqueeze(0).contiguous().view(z.shape[0], -1)

            # pdb.set_trace()
            rc_tmp.append(log_gen)
            ll_tmp.append(log_gen + log_prior - log_infer)

        log_prob_iw = log_sum_exp(torch.cat(ll_tmp, dim=-1), dim=-1) - math.log(nsamples)
        log_gen_iw = torch.mean(torch.cat(rc_tmp, dim=-1), dim=-1)

        return log_prob_iw, log_gen_iw, KL

    def rec_sample(self, x0, x1, sample=False):
        bert_fea = self.encoder(x0)[1]
        mu, logvar = self.encoder.linear(bert_fea).chunk(2, -1)
        z = self.reparameterize(mu, logvar, 1)
        rec = 'rec'
        if sample:
            z = torch.tensor(np.random.normal(size=(z.size()[0], z.size()[-1])), dtype=torch.double).cuda().unsqueeze(1)
            x1 = torch.zeros_like(x1)
            rec = 'sample'
        decoded_batch = self.decode_eval_greedy(x1, z)
        # with open('/home/lptang/Optimus/samples/' + self.output_dir.split('/')[-1] + '.' + str(
        #         self.args.gloabl_step_eval) + '.' + rec, 'a+') as f:
        #     for sent in decoded_batch:
        #         f.write(sent + '\n')

    def nll_iw(self, x0, x1, nsamples, ns=1):
        """compute the importance weighting estimate of the log-likelihood
        Args:
            x0, x1:  two different tokenization results of x, where x is the data tensor with shape (batch, *). 
            nsamples: Int
                the number of samples required to estimate marginal data likelihood
        Returns: Tensor1
            Tensor1: the estimate of log p(x), shape [batch]
        """

        # compute iw every ns samples to address the memory issue
        # nsamples = 500, ns = 100
        # nsamples = 500, ns = 10

        # TODO: note that x is forwarded twice in self.encoder.sample(x, ns) and self.eval_inference_dist(x, z, param)
        # .      this problem is to be solved in order to speed up

        tmp = []
        for _ in range(int(nsamples / ns)):
            # [batch, ns, nz]

            # Chunyuan:
            # encoding into bert features
            pooled_hidden_fea = self.encoder(x0)[1]

            # param is the parameters required to evaluate q(z|x)
            z, param = self.encoder_sample(pooled_hidden_fea, ns)

            # [batch, ns]
            log_comp_ll = self.eval_complete_ll(x1, z)
            log_infer_ll = self.eval_inference_dist(z, param)

            tmp.append(log_comp_ll - log_infer_ll)

        ll_iw = log_sum_exp(torch.cat(tmp, dim=-1), dim=-1) - math.log(nsamples)

        return ll_iw

    def KL(self, x):
        _, KL = self.encode(x, 1)

        return KL

    def eval_prior_dist(self, zrange):
        """perform grid search to calculate the true posterior
        Args:
            zrange: tensor
                different z points that will be evaluated, with
                shape (k^2, nz), where k=(zmax - zmin)/space
        """

        # (k^2)
        return self.prior.log_prob(zrange).sum(dim=-1)

    def eval_complete_ll(self, x, z):
        """compute log p(z,x)
        Args:
            x: Tensor
                input with shape [batch, seq_len]
            z: Tensor
                evaluation points with shape [batch, nsamples, nz]
        Returns: Tensor1
            Tensor1: log p(z,x) Tensor with shape [batch, nsamples]
        """

        # [batch, nsamples]
        log_prior = self.eval_prior_dist(z)
        log_gen = self.eval_cond_ll(x, z)

        return log_prior + log_gen

    def eval_cond_ll(self, x, z):
        """compute log p(x|z)
        """
        x_shape = list(x.size())
        z_shape = list(z.size())
        if len(z_shape) == 3:
            x = x.unsqueeze(1).repeat(1, z_shape[1], 1).contiguous().view(x_shape[0] * z_shape[1], x_shape[-1])
            z = z.contiguous().view(x_shape[0] * z_shape[1], z_shape[-1])

        return self.log_probability(x, z)

    def eval_log_model_posterior(self, x, grid_z):
        """perform grid search to calculate the true posterior
         this function computes p(z|x)
        Args:
            grid_z: tensor
                different z points that will be evaluated, with
                shape (k^2, nz), where k=(zmax - zmin)/pace
        Returns: Tensor
            Tensor: the log posterior distribution log p(z|x) with
                    shape [batch_size, K^2]
        """
        try:
            batch_size = x.size(0)
        except:
            batch_size = x[0].size(0)

        # (batch_size, k^2, nz)
        grid_z = grid_z.unsqueeze(0).expand(batch_size, *grid_z.size()).contiguous()

        # (batch_size, k^2)
        log_comp = self.eval_complete_ll(x, grid_z)

        # normalize to posterior
        log_posterior = log_comp - log_sum_exp(log_comp, dim=1, keepdim=True)

        return log_posterior

    def sample_from_inference(self, x, nsamples=1):
        """perform sampling from inference net
        Returns: Tensor
            Tensor: samples from infernece nets with
                shape (batch_size, nsamples, nz)
        """
        bert_fea = self.encoder(x)[1]
        mu, logvar = self.encoder.linear(bert_fea).chunk(2, -1)
        #         z, _ = self.encoder.sample(x, nsamples)
        z = self.reparameterize(mu, logvar, nsamples)
        return z

    # def sample_from_posterior(self, x, nsamples):
    #     """perform MH sampling from model posterior
    #     Returns: Tensor
    #         Tensor: samples from model posterior with
    #             shape (batch_size, nsamples, nz)
    #     """

    #     # use the samples from inference net as initial points
    #     # for MCMC sampling. [batch_size, nsamples, nz]
    #     cur = self.encoder.sample_from_inference(x, 1)
    #     cur_ll = self.eval_complete_ll(x, cur)
    #     total_iter = self.args.mh_burn_in + nsamples * self.args.mh_thin
    #     samples = []
    #     for iter_ in range(total_iter):
    #         next = torch.normal(mean=cur,
    #                             std=cur.new_full(size=cur.size(), fill_value=self.args.mh_std))
    #         # [batch_size, 1]
    #         next_ll = self.eval_complete_ll(x, next)
    #         ratio = next_ll - cur_ll

    #         accept_prob = torch.min(ratio.exp(), ratio.new_ones(ratio.size()))

    #         uniform_t = accept_prob.new_empty(accept_prob.size()).uniform_()

    #         # [batch_size, 1]
    #         mask = (uniform_t < accept_prob).float()
    #         mask_ = mask.unsqueeze(2)

    #         cur = mask_ * next + (1 - mask_) * cur
    #         cur_ll = mask * next_ll + (1 - mask) * cur_ll

    #         if iter_ >= self.args.mh_burn_in and (iter_ - self.args.mh_burn_in) % self.args.mh_thin == 0:
    #             samples.append(cur.unsqueeze(1))

    #     return torch.cat(samples, dim=1)

    def calc_model_posterior_mean(self, x, grid_z):
        """compute the mean value of model posterior, i.e. E_{z ~ p(z|x)}[z]
        Args:
            grid_z: different z points that will be evaluated, with
                    shape (k^2, nz), where k=(zmax - zmin)/pace
            x: [batch, *]
        Returns: Tensor1
            Tensor1: the mean value tensor with shape [batch, nz]
        """

        # [batch, K^2]
        log_posterior = self.eval_log_model_posterior(x, grid_z)
        posterior = log_posterior.exp()

        # [batch, nz]
        return torch.mul(posterior.unsqueeze(2), grid_z.unsqueeze(0)).sum(1)

    def calc_infer_mean(self, x):
        """
        Returns: Tensor1
            Tensor1: the mean of inference distribution, with shape [batch, nz]
        """

        mean, logvar = self.encoder.forward(x)

        return mean

    def eval_inference_dist(self, z, param):
        """this function computes log q(z | x)
        Args:
            z: tensor
                different z points that will be evaluated, with
                shape [batch, nsamples, nz]
        Returns: Tensor1
            Tensor1: log q(z|x) with shape [batch, nsamples]
        """

        nz = z.size(2)
        mu, logvar = param

        # (batch_size, 1, nz)
        mu, logvar = mu.unsqueeze(1), logvar.unsqueeze(1)
        var = logvar.exp()

        # (batch_size, nsamples, nz)
        dev = z - mu

        # (batch_size, nsamples)
        log_density = -0.5 * ((dev ** 2) / var).sum(dim=-1) - \
                      0.5 * (nz * math.log(2 * math.pi) + logvar.sum(-1))

        return log_density

    # def calc_mi(self, test_data_batch, args):
    #     # calc_mi_v3
    #     import math
    #     from modules.utils import log_sum_exp

    #     mi = 0
    #     num_examples = 0

    #     mu_batch_list, logvar_batch_list = [], []
    #     neg_entropy = 0.
    #     for batch_data in test_data_batch:
    #         x0, _, _ = batch_data
    #         x0 = x0.to(args.device)

    #         # encoding into bert features
    #         bert_fea = self.encoder(x0)[1]

    #         (batch_size, nz)
    #         mu, logvar = self.encoder.linear(bert_fea).chunk(2, -1)

    #         x_batch, nz = mu.size()

    #         # print(x_batch, end=' ')

    #         num_examples += x_batch

    #         # E_{q(z|x)}log(q(z|x)) = -0.5*nz*log(2*\pi) - 0.5*(1+logvar).sum(-1)

    #         neg_entropy += (-0.5 * nz * math.log(2 * math.pi) - 0.5 * (1 + logvar).sum(-1)).sum().item()
    #         mu_batch_list += [mu.cpu()]
    #         logvar_batch_list += [logvar.cpu()]

    #     #             pdb.set_trace()

    #     neg_entropy = neg_entropy / num_examples
    #     ##print()

    #     num_examples = 0
    #     log_qz = 0.
    #     for i in range(len(mu_batch_list)):
    #         ###############
    #         # get z_samples
    #         ###############
    #         mu, logvar = mu_batch_list[i].cuda(), logvar_batch_list[i].cuda()

    #         # [z_batch, 1, nz]

    #         z_samples = self.reparameterize(mu, logvar, 1)

    #         z_samples = z_samples.view(-1, 1, nz)
    #         num_examples += z_samples.size(0)

    #         ###############
    #         # compute density
    #         ###############
    #         # [1, x_batch, nz]
    #         # mu, logvar = mu_batch_list[i].cuda(), logvar_batch_list[i].cuda()
    #         # indices = list(np.random.choice(np.arange(len(mu_batch_list)), 10)) + [i]
    #         indices = np.arange(len(mu_batch_list))
    #         mu = torch.cat([mu_batch_list[_] for _ in indices], dim=0).cuda()
    #         logvar = torch.cat([logvar_batch_list[_] for _ in indices], dim=0).cuda()
    #         x_batch, nz = mu.size()

    #         mu, logvar = mu.unsqueeze(0), logvar.unsqueeze(0)
    #         var = logvar.exp()

    #         # (z_batch, x_batch, nz)
    #         dev = z_samples - mu

    #         # (z_batch, x_batch)
    #         log_density = -0.5 * ((dev ** 2) / var).sum(dim=-1) - \
    #                       0.5 * (nz * math.log(2 * math.pi) + logvar.sum(-1))

    #         # log q(z): aggregate posterior
    #         # [z_batch]
    #         log_qz += (log_sum_exp(log_density, dim=1) - math.log(x_batch)).sum(-1)

    #     log_qz /= num_examples
    #     mi = neg_entropy - log_qz

    #     return mi

    def calc_au(self, eval_dataloader, delta=0.01):
        """compute the number of active units
        """
        cnt = 0
        for batch_data in eval_dataloader:

            x0, _, _ = batch_data
            x0 = x0.to(self.device)

            # encoding into bert features
            bert_fea = self.encoder(x0)[1]

            # (batch_size, nz)
            mean, logvar = self.encoder.linear(bert_fea).chunk(2, -1)

            if cnt == 0:
                means_sum = mean.sum(dim=0, keepdim=True)
            else:
                means_sum = means_sum + mean.sum(dim=0, keepdim=True)
            cnt += mean.size(0)

        # (1, nz)
        mean_mean = means_sum / cnt

        cnt = 0
        for batch_data in eval_dataloader:

            x0, _, _ = batch_data
            x0 = x0.to(self.device)

            # encoding into bert features
            bert_fea = self.encoder(x0)[1]

            # (batch_size, nz)
            mean, _ = self.encoder.linear(bert_fea).chunk(2, -1)

            if cnt == 0:
                var_sum = ((mean - mean_mean) ** 2).sum(dim=0)
            else:
                var_sum = var_sum + ((mean - mean_mean) ** 2).sum(dim=0)
            cnt += mean.size(0)

        # (nz)
        au_var = var_sum / (cnt - 1)

        return (au_var >= delta).sum().item(), au_var



# def ddpm_schedules_alpha_bar(choice: int, T: int) -> Dict[str, torch.Tensor]:
#     """
#     Returns pre-computed schedules for DDPM sampling, training process.
#     """

#     alphabar_t =  1 - torch.arange(0, T + 1, dtype=torch.float32)/ T 
#     beta_t = (beta2 - beta1) * torch.arange(0, T + 1, dtype=torch.float32) / T + beta1
    
#     beta_t = beta_t*0 + 0.008
#     # beta_t = 0.008
#     sqrt_beta_t = torch.sqrt(beta_t)
#     alpha_t = 1 - beta_t
#     log_alpha_t = torch.log(alpha_t)
#     alphabar_t = torch.cumsum(log_alpha_t, dim=0).exp()

#     sqrtab = torch.sqrt(alphabar_t)
#     sqrta = torch.sqrt(alpha_t)
#     oneover_sqrta = 1 / sqrta
#     mab = 1 - alphabar_t
#     sqrtmab = torch.sqrt(mab)
#     mab_over_sqrtmab_inv = (1 - alpha_t) / sqrtmab

#     sigma = sqrtmab / sqrtab
#     sigma_diff = sigma[1:] - sigma[:-1]
#     return {
#         "beta_t": beta_t,
#         "alpha_t": alpha_t,  # \alpha_t
#         "sqrta": sqrta,
#         "oneover_sqrta": oneover_sqrta,  # 1/\sqrt{\alpha_t}
#         "sqrt_beta_t": sqrt_beta_t,  # \sqrt{\beta_t}
#         "alphabar_t": alphabar_t,  # \bar{\alpha_t}
#         "sqrtab": sqrtab,  # \sqrt{\bar{\alpha_t}}
#         "mab": mab,
#         "sqrtmab": sqrtmab,  # \sqrt{1-\bar{\alpha_t}}
#         "mab_over_sqrtmab": mab_over_sqrtmab_inv,  # (1-\alpha_t)/\sqrt{1-\bar{\alpha_t}}
#         # "sigma" : sigma,
#         "diff_sigma": sigma_diff,
#     }

blk_linear = lambda ic, oc: nn.Sequential(
    nn.Linear(ic, oc),
    nn.LeakyReLU(),
)

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

from functions import ddpm_schedule

class DDPM(nn.Module):
    def __init__(
        self,
        eps_model: nn.Module,
        betas: Tuple[float, float],
        n_T: int,
        criterion: nn.Module = nn.L1Loss(),
    ) -> None:
        super(DDPM, self).__init__()
        self.eps_model = eps_model

        # register_buffer allows us to freely access these tensors by name. It helps device placement.
        for k, v in ddpm_schedule(betas[0], betas[1], n_T).items():
            self.register_buffer(k, v)

        self.n_T = n_T
        self.criterion = criterion

    def forward_new(self, x: torch.Tensor, mu) -> torch.Tensor:
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
            # 2. predict eps_0 from x_1
            # eps_0 = self.eps_model(x_t, _ts / self.n_T)
            # 2. predict x_0 from x_1
            # x_0 = self.oneover_sqrta[1] * (x_t - eps_0 * self.mab_over_sqrtmab[1]) + self.sqrt_beta_t[1] * eps
            # loss_z0 = self.alpha_t[1]/self.beta_t[1] * self.criterion(mu, self.oneover_sqrta[1]*       self.sqrt_beta_t[1]*eps_0).mean(1)
            loss_z0 = self.alpha_t[1]/self.beta_t[1] * self.criterion(mu, self.oneover_sqrta[1]*(x_t - self.sqrt_beta_t[1]*eps_0)).mean(1)
            
            # 3. calculate loss 
            # loss_z0 = self.criterion(x, x_0) # 1/self.beta_t[0] * 
            loss = torch.where(mask_1,loss_z0, loss) 
        
        return loss,self.loss_weight[_ts, None]

    # def forward(self, x: torch.Tensor) -> torch.Tensor:
    #     """
    #     Makes forward diffusion x_t, and tries to guess epsilon value from x_t using eps_model.
    #     This implements Algorithm 1 in the paper.
    #     """

    #     _ts = torch.randint(1, self.n_T+1, (x.shape[0],)).to(
    #         x.device
    #     )  # t ~ Uniform(0, n_T)
    #     eps = torch.randn_like(x)  # eps ~ N(0, 1)
    #     x_t = (
    #         self.sqrtab[_ts, None] * x
    #         + self.sqrtmab[_ts, None] * eps
    #     )  # This is the x_t, which is sqrt(alphabar) x_0 + sqrt(1-alphabar) * eps
    #     # We should predict the "error term" from this x_t. Loss is what we return.
    #     return self.criterion(eps, self.eps_model(x_t, _ts / self.n_T))


    # def forward_dae(self, x: torch.Tensor, z_sem) -> torch.Tensor:
    #     """
    #     Makes forward diffusion x_t, and tries to guess epsilon value from x_t using eps_model.
    #     This implements Algorithm 1 in the paper.
    #     """

    #     _ts = torch.randint(1, self.n_T+1, (x.shape[0],)).to(
    #         x.device
    #     )  # t ~ Uniform(0, n_T)
    #     eps = torch.randn_like(x)  # eps ~ N(0, 1)
    #     x_t = (
    #         self.sqrtab[_ts, None] * x
    #         + self.sqrtmab[_ts, None] * eps
    #     )  # This is the x_t, which is sqrt(alphabar) x_0 + sqrt(1-alphabar) * eps
    #     # We should predict the "error term" from this x_t. Loss is what we return.
    #     return self.criterion(eps, self.eps_model(x_t, _ts / self.n_T, self.eps_model.dae_mlp(z_sem)))

    def add_noise(self, x_i: torch.Tensor, T=None) -> torch.Tensor:
        """
        DDIM forward
        """
        # if if_mean:
        #     x_i = x_i + torch.randn_like(x_i) * self.sqrt_beta_t[1]
        if T == None:
            T = self.n_T
        with torch.no_grad():
            n_sample = x_i.size(0)
            for i in range(0, T):  # 0,T
                ts_ = torch.tensor(i).to(x_i.device) / self.n_T
                ts_ = ts_.repeat(n_sample)
                eps = self.eps_model(x_i, ts_)
                # x_i = self.sqrtab[i+1] * (x_i/self.sqrtab[i] + eps*self.diff_sigma[i])
                x_i = self.sqrta[i+1] * (x_i + (self.sqrtmab[i+1]/self.sqrta[i+1] - self.sqrtmab[i])*eps )
                # x_i = self.sqrta[i] * (x_i + (self.sqrtmab[i]/self.sqrta[i] - self.sqrtmab[i-1])*eps )
        return x_i

    def add_noise_ddpm(self, x_i: torch.Tensor, step=None):
        noise = torch.randn_like(x_i) 
        x_i = self.sqrtab[step]* x_i + self.sqrtmab[step] * noise
        return x_i

    def add_vpnoise(self, x_i):
        n_sample = x_i.size(0)
        for i in range(1, self.n_T):
            ts_ = torch.tensor(i).to(x_i.device) / self.n_T
            ts_ = ts_.repeat(n_sample)
            eps = self.eps_model(x_i, ts_)
            score = - eps / self.sqrtmab[i]
            x_i = x_i - 0.5*i/self.n_T * self.beta_t[i] * (x_i + score)

    def sample(self, n_sample: int, size, device) -> torch.Tensor:
        x_i = torch.randn(n_sample, *size).to(device)  # x_T ~ N(0, 1)
        # This samples accordingly to Algorithm 2. It is exactly the same logic.
        for i in range(self.n_T, 0, -1):
            z = torch.randn(n_sample, *size).to(device) if i > 1 else 0
            ts_ = torch.tensor(i).to(x_i.device) / self.n_T
            ts_ = ts_.repeat(n_sample)
            eps = self.eps_model(x_i, ts_)
            x_i = (
                self.oneover_sqrta[i] * (x_i - eps * self.mab_over_sqrtmab[i])
                + self.sqrt_beta_t[i] * z
            )
        return x_i
    
    def sample_new(self, n_sample: int, size, device, fp16=False) -> torch.Tensor:
        dtype_ = torch.half if fp16 else torch.float
        x_i = torch.randn(n_sample, *size).to(device=device,dtype=dtype_)  # x_T ~ N(0, 1)
        # This samples accordingly to Algorithm 2. It is exactly the same logic.
        for i in range(self.n_T, 0, -1):
            z = torch.randn(n_sample, *size).to(device=device,dtype=dtype_)  if i > 1 else 0
            ts_ = torch.tensor(i).to(device=device,dtype=dtype_) / self.n_T
            ts_ = ts_.repeat(n_sample)
            eps = self.eps_model(x_i, ts_)
            x_i = (
                self.oneover_sqrta[i] * (x_i - eps * self.mab_over_sqrtmab[i])
                + self.sqrt_beta_t[i] * z
            )
        # import ipdb
        # ipdb.set_trace()
        return x_i

    def sample_dae(self, z_sem) -> torch.Tensor:
        device = z_sem.device
        n_sample = z_sem.shape[0]
        size = (z_sem.shape[1],)
        x_i = torch.randn_like(z_sem)   # x_T ~ N(0, 1)
        # This samples accordingly to Algorithm 2. It is exactly the same logic.
        for i in range(self.n_T, 0, -1):
            z = torch.randn(n_sample, *size,dtype=z_sem.dtype).to(device) if i > 1 else 0
            ts_ = torch.tensor(i,dtype=z_sem.dtype).to(x_i.device) / self.n_T
            ts_ = ts_.repeat(n_sample)
            eps = self.eps_model(x_i, ts_, z_sem=z_sem)
            x_i = (
                self.oneover_sqrta[i] * (x_i - eps * self.mab_over_sqrtmab[i])
                + self.sqrt_beta_t[i] * z
            )
        return x_i

    def sample_cond(self, n_sample: int, size, device, classifier, y, scale=500, softmax_logits=False) -> torch.Tensor:
        x_i = torch.randn(n_sample, *size).to(device)  # x_T ~ N(0, 1)
        
        scale = -((y.view(-1,1) - 1) * (scale[0]-scale[1]) - scale[1] )
        classifier_scale = scale
        # This samples accordingly to Algorithm 2. It is exactly the same logic.
        with torch.no_grad():
            for i in range(self.n_T, 0, -1):
                z = torch.randn(n_sample, *size).to(device) if i > 1 else 0
                ts_ = torch.tensor(i).to(x_i.device) / self.n_T
                ts_ = ts_.repeat(n_sample)
                eps = self.eps_model(x_i, ts_)
                # with torch.enable_grad():
                #     x_in = z.detach().requires_grad_(True)
                #     logits = classifier.train_step(x_in, ts_)
                #     log_probs = F.log_softmax(logits, dim=-1)
                #     selected = log_probs[range(len(logits)), y.view(-1)]
                #     grad_z = torch.autograd.grad(selected.sum(),x_in)[0] * classifier_scale

                with torch.set_grad_enabled(True):
                    x_in = x_i.detach().requires_grad_(True)
                    logits = classifier.train_step(x_in, ts_)
                    if softmax_logits:
                        log_probs = F.log_softmax(logits, dim=-1)
                        selected = log_probs[range(len(logits)), y.view(-1)]
                        tmp = torch.autograd.grad(selected.sum(),x_in)[0]
                    else:  ### log_probs = energy
                        neg_energy = torch.gather(logits, 1, y.view(-1)[:,None]).squeeze() - logits.logsumexp(1)
                        tmp = torch.autograd.grad(neg_energy.sum(), x_in)[0]
                    grad_z = tmp * scale
                ##### DDIM 
                eps = eps - self.sqrtmab[i] * grad_z
                # eta = 0.0
                # sigma_ = eta * self.sqrtmab[i-1] / self.sqrtmab[i] * self.sqrt_beta_t[i]
                # x_i = ( self.oneover_sqrta[i] * (x_i - (self.sqrtmab[i] - self.sqrta[i] * torch.sqrt(self.mab[i-1] - sigma_**2)) * eps ) + sigma_ * z)
                # x_i = self.sqrtab[i-1] * ((x_i - self.sqrtmab[i] * eps )/ self.sqrtab[i]) + self.sqrtmab[i-1]* eps 
                ############  DDPM
                x_i = (
                    self.oneover_sqrta[i] * (x_i - eps * self.mab_over_sqrtmab[i]) + self.beta_t[i]*grad_z
                    + self.sqrt_beta_t[i] * z
                )
                #############
        return x_i

    def sample_cond_post(self,x_i, device, classifier, y, scale=[500,200], step=2000, softmax_logits=False) -> torch.Tensor:
        # x_i = torch.randn(n_sample, *size).to(device)  # x_T ~ N(0, 1)

        n_sample = x_i.shape[0]
        size = (x_i.shape[1],)
        scale = -((y.view(-1,1) - 1) * (scale[0]-scale[1]) - scale[1] )
        # This samples accordingly to Algorithm 2. It is exactly the same logic.
        with torch.no_grad():
            for i in range(self.n_T, 0, -1):
                z = torch.randn(n_sample, *size).to(device) if i > 1 else 0
                ts_ = torch.tensor(i).to(x_i.device) / self.n_T
                ts_ = ts_.repeat(n_sample)
                eps = self.eps_model(x_i, ts_)
                # with torch.enable_grad():
                #     x_in = z.detach().requires_grad_(True)
                #     logits = classifier.train_step(x_in, ts_)
                #     log_probs = F.log_softmax(logits, dim=-1)
                #     selected = log_probs[range(len(logits)), y.view(-1)]
                #     grad_z = torch.autograd.grad(selected.sum(),x_in)[0] * classifier_scale
                if i <= step:
                    with torch.set_grad_enabled(True):
                        x_in = x_i.detach().requires_grad_(True)
                        logits = classifier.train_step(x_in, ts_ * self.n_T)
                        if softmax_logits: ### log_probs = logsoftmax
                            log_probs = F.log_softmax(logits, dim=-1)
                            selected = log_probs[range(len(logits)), y.view(-1)]
                            tmp = torch.autograd.grad(selected.sum(),x_in)

                        else:  ### log_probs = energy
                            neg_energy = torch.gather(logits, 1, y.view(-1)[:,None]).squeeze() - logits.logsumexp(1)
                            tmp = torch.autograd.grad(neg_energy.sum(), x_in)
                        grad_z = tmp[0] * scale
                    eps = eps - self.sqrtmab[i] * grad_z
                eta = 0.0
                sigma_ = eta * self.sqrtmab[i-1] / self.sqrtmab[i] * self.sqrt_beta_t[i]
                # sigma_ = torch.sqrt(1- self.alphabar_t[i] / self.alphabar_t[i-1])
                # x_i = ( self.oneover_sqrta[i] * (x_i - (self.sqrtmab[i] - self.sqrta[i] * torch.sqrt(self.mab[i-1] - sigma_**2)) * eps ) + sigma_ * z)
                x_i = self.sqrtab[i-1] * ((x_i - self.sqrtmab[i] * eps )/ self.sqrtab[i]) + self.sqrtmab[i-1]* eps 
        return x_i
    def sample_one(self, n_sample: int, size, device, score_flag = 2, T=None, step=None, fp16=False) -> torch.Tensor:
        '''
        score_flag = 7:  DDIM 加速采样，step为需要采样的步数，T需要是step的倍数。(e.g., default T=2000 -> step=500)
        '''
        dtype_ = torch.half if fp16 else torch.float
        if T == None:
            T = self.n_T
        with torch.no_grad():
            x_i = torch.randn( n_sample, *size).to(device=device,dtype=dtype_)  # x_T ~ N(0, 1)
            # x_i = x_i.repeat(n_sample,1)
            # This samples accordingly to Algorithm 2. It is exactly the same logic.
            if score_flag == 4:  ## shperical interpolation
                z1 = torch.randn_like(x_i)
                theta = torch.arccos((x_i * z1).sum(1)/(torch.norm(z1,dim=1)*torch.norm(x_i,dim=1)))

                # theta = torch.arccos(torch.matmul(x_i,z1)/(torch.norm(z1)*torch.norm(x_i,dim=1)))
                sin_theta = torch.sin(theta)
                jj = 0.5
                tmp1 = torch.matmul(torch.diag(torch.sin((1-jj)*theta)/sin_theta) , z1)
                tmp2 = torch.matmul(torch.diag(torch.sin(jj*theta)/sin_theta) ,x_i)
                x_i = tmp1  +  tmp2
            if score_flag == 6:
                z1 = torch.randn_like(x_i).to(device=device,dtype=dtype_)
                x_i = (z1  +  x_i) * 0.5
            elif score_flag == 7:
                step_list = list(range(T,0,-T//step))
                cnt_step = 0
            for i in range(T, 0, -1):
                z = torch.randn(n_sample, *size).to(device=device,dtype=dtype_) if i > 1 else 0
                if score_flag != 7:
                    ts_ = torch.tensor(i).to(device=device,dtype=dtype_) / self.n_T
                    ts_ = ts_.repeat(n_sample)
                    eps = self.eps_model(x_i,ts_)
                # predict noise: epsilon( x_t, t)
                # score: - eps / self.sqrtmab
                
                if score_flag == 0: # score model
                    score = - eps / self.sqrtmab[i]
                    x_i = (
                        self.oneover_sqrta[i] * (x_i + self.beta_t[i] * score) + self.sqrt_beta_t[i] * z
                    )
                elif score_flag == 1: # DDPM
                    x_i = (
                        self.oneover_sqrta[i] * (x_i - eps * self.mab_over_sqrtmab[i])
                        + self.sqrt_beta_t[i] * z
                    )
                elif score_flag == 2: # DDIM
                    eta = 0.0
                    sigma_ = eta * self.sqrtmab[i-1] / self.sqrtmab[i] * self.sqrt_beta_t[i]
                    # sigma_ = torch.sqrt(1- self.alphabar_t[i] / self.alphabar_t[i-1])
                    x_i = ( self.oneover_sqrta[i] * (x_i - (self.sqrtmab[i] - self.sqrta[i] * torch.sqrt(self.mab[i-1] - sigma_**2)) * eps ) + sigma_ * z)
                    # x_i = self.oneover_sqrta[i] * (x_i - (self.sqrtmab[i] - self.sqrtab[i] * torch.sqrt(self.mab[i-1])) * eps ) 

                elif score_flag == 3: # VP-SDE -> ODE
                    x_i = x_i - 1./self.n_T * 0.5*self.beta_t[i]*(eps * self.sqrtmab[i] - x_i) 
                elif score_flag == 4: # DDIM with Interpolation
                    # z1 = torch.randn(*size).to(device) 
                    eta = 0.0
                    sigma_ = eta * self.sqrtmab[i-1] / self.sqrtmab[i] * self.sqrt_beta_t[i]
                    # sigma_ = torch.sqrt(1- self.alphabar_t[i] / self.alphabar_t[i-1])
                    x_i = ( self.oneover_sqrta[i] * (x_i - (self.sqrtmab[i] - self.sqrta[i] * torch.sqrt(self.mab[i-1] - sigma_**2)) * eps ) + sigma_ * z)

                elif score_flag == 6: # DDIM with linear Interpolation
                    eta = 0.0
                    sigma_ = eta * self.sqrtmab[i-1] / self.sqrtmab[i] * self.sqrt_beta_t[i]
                    # sigma_ = torch.sqrt(1- self.alphabar_t[i] / self.alphabar_t[i-1])
                    x_i = ( self.oneover_sqrta[i] * (x_i - (self.sqrtmab[i] - self.sqrta[i] * torch.sqrt(self.mab[i-1] - sigma_**2)) * eps ) + sigma_ * z)
                elif score_flag == 5: # VP  ODE
                    score = - eps / self.sqrtmab[i]
                    # x_i = x_i - 1/self.n_T * (-0.5 * self.beta_t[i] * (x_i + score))
                    x_i = (2 - torch.sqrt(1-self.beta_t[i])) * x_i + 0.5 * self.beta_t[i] * score
                elif score_flag == 7 and i in step_list[:-1]: # DDIM with less steps
                    ts_ = torch.tensor(i).to(x_i.device) / self.n_T
                    ts_ = ts_.repeat(n_sample)
                    eps = self.eps_model(x_i,ts_)
                    eta = 0.0
                    next_step = step_list[cnt_step+1]
                    # sigma_ = eta * self.sqrtmab[next_step] / self.sqrtmab[i] * self.sqrt_beta_t[i]
                    # sigma_ = torch.sqrt(1- self.alphabar_t[i] / self.alphabar_t[i-1])
                    # x_i = ( self.oneover_sqrta[i] * (x_i - (self.sqrtmab[i] - self.sqrta[i] * torch.sqrt(self.mab[next_step] )) * eps ))
                    x_i = self.sqrtab[next_step] * (x_i - self.sqrtmab[i]*eps)/self.sqrtab[i] + self.sqrtmab[next_step] * eps
                    cnt_step+= 1
        return x_i
    def sample_posterior(self, x_i, device, score_flag=2, T=None) -> torch.Tensor:

        # x_i = x_i.repeat(n_sample,1)
        # This samples accordingly to Algorithm 2. It is exactly the same logic.
        n_sample = x_i.size(0)
        if T == None:
            T = self.n_T
        for i in range(T, 0, -1):
            z = torch.randn_like(x_i).to(device) if i > 2 else 0
            ts_ = torch.tensor(i).to(x_i.device) / self.n_T
            ts_ = ts_.repeat(n_sample)
            eps = self.eps_model(x_i, ts_)
            # predict noise: epsilon( x_t, t)
            # score: - eps / self.sqrtmab
            # score_flag = 2
            if score_flag == 0: # score model
                score = - eps / self.sqrtmab[i]
                x_i = (
                    self.oneover_sqrta[i] * (x_i + self.beta_t[i] * score) + self.sqrt_beta_t[i] * z
                )
            elif score_flag == 1: # DDPM
                x_i = (
                    self.oneover_sqrta[i] * (x_i - eps * self.mab_over_sqrtmab[i])
                    + self.sqrt_beta_t[i] * z
                )
            elif score_flag == 2: # DDIM
                eta = 0.0
                sigma_ = eta * self.sqrtmab[i-1] / self.sqrtmab[i] * self.sqrt_beta_t[i]
                # sigma_ = torch.sqrt(1- self.alphabar_t[i] / self.alphabar_t[i-1])
                x_i = ( self.oneover_sqrta[i] * (x_i - (self.sqrtmab[i] - self.sqrta[i] * torch.sqrt(self.mab[i-1] - sigma_**2)) * eps ) + sigma_ * z)
            elif score_flag == 5: # VP  ODE
                score = - eps / self.sqrtmab[i]
                # x_i = x_i - 1/self.n_T * (-0.5 * self.beta_t[i] * (x_i + score))
                x_i = (2 - torch.sqrt(1-self.beta_t[i])) * x_i + 0.5 * self.beta_t[i] * score
        return x_i

    def em_sampler(self,n_sample, size, device='cuda'):
        t = torch.ones(n_sample, device=device) # initial t = 1
        init_x = torch.randn(n_sample, *size).to(device)

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

class VAE_DDPM(nn.Module):
    def __init__(self, model_vae, ddpm, ddpm_weight) :
        super(VAE_DDPM, self).__init__()
        self.model_vae = model_vae
        self.ddpm = ddpm
        self.ddpm_weight = ddpm_weight

    def forward(self,inputs, labels, std=False, return_z=False, return_mu=False): 
        
        loss_rec, loss_kl, loss, latent_z, mu = self.model_vae(inputs, labels, std=std, return_z=return_z, return_mu=return_mu)
        ddpm_loss, loss_weight = self.ddpm.forward_new(latent_z, mu)
        
        if self.ddpm_weight > 0:
            loss = (1/(loss_weight * self.ddpm.n_T)  * loss).mean() + self.ddpm_weight *ddpm_loss.mean()
        else:
            loss = loss.mean() + 0.0* ddpm_loss.mean()
        return loss_rec, loss_kl, loss, latent_z, mu, ddpm_loss, loss_weight
