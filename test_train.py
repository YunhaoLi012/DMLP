from datasets import load_dataset,concatenate_datasets, Dataset,DatasetDict
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch


from models.my_transformers import *
from models.models import VAE, DDPM, MLPSkipNet, TransformerNet,VAE_DDPM
from train.reconstruction import *
from functions import weights_init_rondom
from train import *

def collate(examples):
    # Convert to Tensors and build dataset

    input_ids_bert = pad_sequence([torch.tensor(f['bert_token'], dtype=torch.long) for f in examples],
                                  batch_first=True, padding_value=bert_pad_token)
    input_ids_gpt = pad_sequence([torch.tensor(f['gpt2_token'], dtype=torch.long) for f in examples],
                                 batch_first=True, padding_value=gpt2_pad_token)
    try:
        token_lengths = torch.tensor([[len(f['bert_token']), len(f['gpt2_token'])] for f in examples],
                                     dtype=torch.long)
    except:
        token_lengths = torch.zeros((len(examples), 1091))
        for i in range(len(examples)):
            token_lengths[i, len(examples[i]['gpt2_token'])] = 1
    return (input_ids_bert, input_ids_gpt, token_lengths)

def main():
    #download data
    print("download data")
    train_eval_dataset =load_dataset("guangyil/yelp_short_v2")
    eval_dataloader =  DataLoader(train_eval_dataset['test'], num_workers=0, collate_fn=collate,batch_size=64)
    train_dataloader = DataLoader(train_eval_dataset['train'], num_workers=0, collate_fn=collate, batch_size=64)
    encoder_model_class = MODEL_CLASS['BertForLatentConnectorAVG']

    #initialize tokenizer and model
    print("initialize models")
    tokenizer_encoder = AutoTokenizer.from_pretrained("prajjwal1/bert-small")
    latent_size = 64
    model_encoder = encoder_model_class.from_pretrained("prajjwal1/bert-small", latent_size=latent_size,
                                                        pad_id=tokenizer_encoder.pad_token_id,local_files_only=False)


    decoder_model_class = MODEL_CLASS['GPT2ForLatentConnectorNew']
    tokenizer_decoder = AutoTokenizer.from_pretrained("gpt2-xl")
    model_decoder = decoder_model_class.from_pretrained("gpt2-xl", latent_size=latent_size,
                                                            latent_as_gpt_emb=True,
                                                            latent_as_gpt_memory=True,local_files_only=False)
    decoder_n_layer = model_decoder.transformer.config.n_layer
    model_decoder.transformer.change_order()

    special_tokens_dict = {'pad_token': '<PAD>', 'bos_token': '<BOS>', 'eos_token': '<EOS>', }
    num_added_toks = tokenizer_decoder.add_special_tokens(special_tokens_dict)
    model_decoder.resize_token_embeddings(len(tokenizer_decoder))
    global bert_pad_token
    global gpt2_pad_token
    bert_pad_token = tokenizer_encoder.pad_token_id
    gpt2_pad_token = tokenizer_decoder.pad_token_id

    output_dir = "test"
    model_vae = VAE(model_encoder, model_decoder, tokenizer_encoder, tokenizer_decoder, latent_size, output_dir)
    model_vae.apply(weights_init_rondom)
    model_vae.to('cuda')   
    calc_rec_lgy(model_vae, tokenizer_encoder, tokenizer_decoder,eval_dataloader, "cuda", True, ns=1)
    ddpm = DDPM(eps_model=MLPSkipNet(latent_size), betas=(1e-4, 0.02), n_T=1000, criterion=nn.MSELoss(reduction='none'),)
    ddpm.apply(weights_init_rondom)
    model = VAE_DDPM(model_vae, ddpm,1.0 )

    print("start_training")
    train_vae_ddpm(model, train_dataloader, tokenizer_encoder, tokenizer_decoder, eval_dataloader, output_dir, condition_f=lambda x: False,
          checkpoint=None, local_rank = 6, batch_size = 32, eval_batch_size = 64,
          train_epoch = 20, gradient_accumulation_steps = 1, device = 'cuda',
          fp16=False, fp16_opt_level=None, learning_rate=9e-5, adam_epsilon=1e-5,
          lr_end_multiplier= 0.01, power=3.0, warmup_steps=0, 
          disable_bar=True, model_ppl=None, tokenizer_ppl=None, max_grad_norm=1, evaluate_during_training=False,
          no_save=True)
    print("training_done")


print("here")


if __name__ == "__main__":
    main()