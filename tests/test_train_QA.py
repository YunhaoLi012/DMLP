from datasets import load_dataset,concatenate_datasets, Dataset,DatasetDict, load_from_disk
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn
import torch
import os
import numpy as np

import torch.multiprocessing as mp
import time


from DMLP.models.my_transformers import MODEL_CLASS
from DMLP.models.models import VAE, DDPM, MLPSkipNet, TransformerNet,VAE_DDPM
from DMLP.train.reconstruction import *
from DMLP.utils.ddpm_schedule import ddpm_schedule
from DMLP.utils.random_init import weights_init_random
from DMLP.train.train_function import train_vae_ddpm
# import sys
# sys.path.append("/home/AD/yul080/DMLP/DMLP/train")
# from train_function import train_vae_ddpm



class MyCollator(object):
    def __init__(self, encoder_token, decoder_token):
        self.encoder_token = encoder_token
        self.decoder_token = decoder_token
    def __call__(self, batch):

        input_ids_bert = pad_sequence([torch.tensor(f['question'], dtype=torch.long) for f in batch],
                                  batch_first=True, padding_value=self.encoder_token)
        input_ids_gpt = pad_sequence([torch.tensor(f['answer'], dtype=torch.long) for f in batch],
                                    batch_first=True, padding_value=self.decoder_token)
        try:
            token_lengths = torch.tensor([[len(f['question']), len(f['answer'])] for f in batch],
                                        dtype=torch.long)
        except:
            token_lengths = torch.zeros((len(batch), 1091))
            for i in range(len(batch)):
                token_lengths[i, len(batch[i]['answer'])] = 1
        return (input_ids_bert, input_ids_gpt, token_lengths)

    

def condition_f(n):
        return ('linear' in n or 'wte' in n or 'decoder.transformer.h.0' in n or 'encoder' in n)

def main():
    batch_size = 8
    encoder_model_class = MODEL_CLASS['BertForLatentConnectorAVG']



    #initialize tokenizer and model
    print("initialize models")
    tokenizer_encoder = AutoTokenizer.from_pretrained("prajjwal1/bert-small")
    latent_size = 128
    model_encoder = encoder_model_class.from_pretrained("prajjwal1/bert-small", latent_size=latent_size,
                                                        pad_id=tokenizer_encoder.pad_token_id,local_files_only=False)


    decoder_model_class = MODEL_CLASS['GPT2ForLatentConnectorNew']
    tokenizer_decoder = AutoTokenizer.from_pretrained("gpt2-xl")
    model_decoder = decoder_model_class.from_pretrained("gpt2-xl", latent_size=latent_size,
                                                            latent_as_gpt_emb=True,
                                                            latent_as_gpt_memory=True,local_files_only=False)
    decoder_n_layer = model_decoder.transformer.config.n_layer
    model_decoder.transformer.change_order()

    special_tokens_dict = {'pad_token': '<PAD>', 'bos_token': '<BOS>', 'eos_token': '<EOS>' }
    num_added_toks = tokenizer_decoder.add_special_tokens(special_tokens_dict)
    model_decoder.resize_token_embeddings(len(tokenizer_decoder))
    bert_pad_token = tokenizer_encoder.pad_token_id
    gpt2_pad_token = tokenizer_decoder.pad_token_id


    my_collator = MyCollator(bert_pad_token, gpt2_pad_token)
    #download data
    print("download data")
    train_eval_dataset =load_from_disk("../Data/qaa")
    eval_dataloader =  DataLoader(train_eval_dataset['test'], num_workers=0, collate_fn=my_collator,batch_size=batch_size)
    train_dataloader = DataLoader(train_eval_dataset['train'], num_workers=0, collate_fn=my_collator, batch_size=batch_size)

    # output_dir = "/home/AD/yul080/runs_with_ddpm"
    output_dir = "../../ckpts/qa_without_ddpm"
    model_vae = VAE(model_encoder, model_decoder, tokenizer_encoder, tokenizer_decoder, latent_size, output_dir)
    # model_vae.apply(weights_init_random)
    # model_vae.to('cuda')   
    ddpm = DDPM(MLPSkipNet(latent_size), (1e-4, 0.02), 2000, nn.MSELoss(reduction='none'), ddpm_schedule)
    # ddpm.apply(weights_init_random)
    model = VAE_DDPM(model_vae, ddpm, 0.0)
    optimizer = torch.optim.Adam

    world_size = 1
    epochs =50

    print(world_size)
    start = time.time()
    args = (world_size,model, optimizer, train_dataloader,  output_dir, batch_size,condition_f, -1, epochs, 
        1,'cuda', True, None, 9e-5, 1e-5, 0.01, 3.0, 0, True, 1,True, True, eval_dataloader, 
          32, 'gpt2', True)
    print("start_training")
    mp.spawn(train_vae_ddpm,args = args,nprocs=world_size,join=True)
    end = time.time()
    print(f"training_done within {end-start} seconds")





if __name__ == "__main__":
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "10086"
    main()