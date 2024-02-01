from datasets import load_dataset,concatenate_datasets, Dataset,DatasetDict
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch


from models.my_transformers import *
from models.models import VAE, DDPM, MLPSkipNet, TransformerNet,VAE_DDPM
from train.reconstruction import *
from utils import weights_init_rondom, ddpm_schedule
from train import *


class MyCollator(object):
    def __init__(self, encoder_token, decoder_token):
        self.encoder_token = encoder_token
        self.decoder_token = decoder_token
    def __call__(self, batch):
        input_ids_bert = pad_sequence([torch.tensor(f['bert_token'], dtype=torch.long) for f in batch],
                                  batch_first=True, padding_value=self.encoder_token)
        input_ids_gpt = pad_sequence([torch.tensor(f['gpt2_token'], dtype=torch.long) for f in batch],
                                    batch_first=True, padding_value=self.decoder_token)
        try:
            token_lengths = torch.tensor([[len(f['bert_token']), len(f['gpt2_token'])] for f in batch],
                                        dtype=torch.long)
        except:
            token_lengths = torch.zeros((len(batch), 1091))
            for i in range(len(batch)):
                token_lengths[i, len(batch[i]['gpt2_token'])] = 1
        return (input_ids_bert, input_ids_gpt, token_lengths)

def main():
    batch_size = 8
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
    bert_pad_token = tokenizer_encoder.pad_token_id
    gpt2_pad_token = tokenizer_decoder.pad_token_id

    my_collator = MyCollator(bert_pad_token, gpt2_pad_token)
    #download data
    print("download data")
    train_eval_dataset =load_dataset("guangyil/yelp_short_v2")
    eval_dataloader =  DataLoader(train_eval_dataset['test'], num_workers=0, collate_fn=my_collator,batch_size=batch_size)
    train_dataloader = DataLoader(train_eval_dataset['train'], num_workers=0, collate_fn=my_collator, batch_size=batch_size)

    output_dir = "/data/jieqi/DMLP"
    model_vae = VAE(model_encoder, model_decoder, tokenizer_encoder, tokenizer_decoder, latent_size, output_dir)
    model_vae.apply(weights_init_rondom)
    # model_vae.to('cuda')   
    ddpm = DDPM(MLPSkipNet(latent_size), (1e-4, 0.02), 1000, nn.MSELoss(reduction='none'), ddpm_schedule)
    ddpm.apply(weights_init_rondom)
    model = VAE_DDPM(model_vae, ddpm,1.0 )

    print("start_training")
    train_vae_ddpm(model, eval_dataloader, output_dir,batch_size, condition_f=lambda x: False,
          local_rank = 5, train_epoch = 5, gradient_accumulation_steps = 1, device = 'cuda:5',
          fp16=False, fp16_opt_level=None, learning_rate=9e-5, adam_epsilon=1e-5,
          lr_end_multiplier= 0.01, power=3.0, warmup_steps=0, 
          disable_bar=True, max_grad_norm=1,save=True, evaluate_during_training=True, eval_dataloader=eval_dataloader)
    print("training_done")





if __name__ == "__main__":
    main()