---
layout: page
title: Tutorial
sitemap: false
---

## Installation
```
pip install DMLP
```

## Demo
We provided a [demo](https://github.com/YunhaoLi012/DMLP/blob/torchamp/tests/test_train.py) to demonstrate typical way of using our library to train text diffusion model from scratch. This script demonstrates the implementation of learning text reconstruction and generation using DMLP. It replicates the result of the the paper ["Generation, Reconstruction, Representation All-in-One: A Joint Autoencoding Diffusion Model"](https://openreview.net/forum?id=bgIZDxd2bM). To run the script, simpily run the following command after installing the library
```
CUDA_VISIBLE_DEVICES=0 torchrun test_train.py
```
Make sure you are in the folder which contains test_train.py file. 

## Tutorial
In order to use DMLP for new datasets/new tasks with default VAE_DDPM model, we need to first select the type of models they would like to use for autoencoder and diffusion model. We have provided a couple of options in ```DMLP.models.my_transformers``` module. Runs the following code to view the available options.

```
from DMLP.models.my_transformers import MODEL_CLASS
print(MODEL_CLASS)
```

Next we need to decide which model we want to use to construct VAE. In our example, we use ```BertForLatentConnectorAVG``` and ```GPT2ForLatentConnectorNew``` for encoder and decoder. Runs the following code to initialize and download pretrained model weigths from huggingface. We also need to select the corresponding tokenizer to tokenize our data base on the model selected. After constructing encoder and decoder, we will combine them to construct our variational autoencoder.

```
### Import DMLP modules
from DMLP.models.models import VAE, DDPM, MLPSkipNet, TransformerNet,VAE_DDPM

### Define Hyperparameter
latent_size = 128

### Select tokenizer
tokenizer_encoder = AutoTokenizer.from_pretrained("prajjwal1/bert-small")
tokenizer_decoder = AutoTokenizer.from_pretrained("gpt2-xl")

### Select/Initialize model class
encoder_model_class = MODEL_CLASS['BertForLatentConnectorAVG']
decoder_model_class = MODEL_CLASS['GPT2ForLatentConnectorNew']

### Download Model Weights
model_encoder = encoder_model_class.from_pretrained("prajjwal1/bert-small", latent_size=latent_size,
                                                        pad_id=tokenizer_encoder.pad_token_id,local_files_only=False)

model_decoder = decoder_model_class.from_pretrained("gpt2-xl", latent_size=latent_size,
                                                            latent_as_gpt_emb=True,
                                                            latent_as_gpt_memory=True,local_files_only=False)

### Construct VAE
model_vae = VAE(model_encoder, model_decoder, tokenizer_encoder, tokenizer_decoder, latent_size, output_dir)
```
We also need to decide the model structure for the backward diffusion process. We use MLP in our example, but you are free to use any other models. We also provided a default ddpm_scheduler that returns necessary parameters for forward diffusion process.

```
skipped_net_mlp = MLPSkipNet(latent_size)
ddpm = DDPM(skipped_net_mlp,(1e-4, 0.02), 2000, nn.MSELoss(reduction='none'), ddpm_schedule)
```

Combine VAE and DDPM to create text diffusion model.
```
model = VAE_DDPM(model_vae, ddpm, 10.0)
```

Next we will download preprocessed data for training. We use Yelp Review Dataset(Shen et al., 2017; Li et al., 2018) to train our model. 
```
### Define Collate function
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

bert_pad_token = tokenizer_encoder.pad_token_id
gpt2_pad_token = tokenizer_decoder.pad_token_id
my_collator = MyCollator(bert_pad_token, gpt2_pad_token)

### Download data and create dataloader
batch_size = 8
train_eval_dataset =load_dataset("guangyil/yelp_short_v2")
eval_dataloader =  DataLoader(train_eval_dataset['test'], num_workers=0, collate_fn=my_collator,batch_size=batch_size)
train_dataloader = DataLoader(train_eval_dataset['train'], num_workers=0, collate_fn=my_collator, batch_size=batch_size)
```

Now we have arrived to the final step where we can put everything together and start training. Import training function from DMLP train module.
```
from DMLP.train.train_function import train_vae_ddpm
### import torch multiprocessing module for multi-gpu training
import torch.multiprocessing as mp

### Define conditional function for checkpoint purpose
def condition_f(n):
        return ('linear' in n or 'wte' in n or 'decoder.transformer.h.0' in n or 'encoder' in n)

### Define optimizer and hyperparameters for training
optimizer = torch.optim.Adam
epochs = 2000
logging_steps = -1
gradient_accumulation_step = 1
device = "cuda"
fp16 = True
fp16_opt_level=None
condition_f = condition_f
lr = 9e-5
adam_epsilon = 1e-5
lr_end_multiplier = 0.01
power = 3.0
warmup_step=0
disable_bar = True
max_grad_norm = 1
save = True
evaluate_during_training = True
sent_length = 32
model_id = 'gpt2'
ppl_eval = True

#path for directory that saves training results
output_dir  = /path/to/checkpoint/dir

### Construct multiprocessing wrappers for multi-gpu training
world_size = 1 #num of gpu use for training
args = (world_size,model, optimizer, train_dataloader,  output_dir, batch_size,condition_f, logging_steps, epochs, 
        gradient_accumulation_step,device, fp16, fp16_opt_level, lr, adam_epsilon, lr_end_multiplier, power, warmup_step, disable_bar, max_grad_norm, save, evaluate_during_training, eval_dataloader, 
          sent_length, model_id, ppl_eval)
mp.spawn(train_vae_ddpm,args = args,nprocs=world_size,join=True)
        
```

We are now ready to train a text diffusion model for text reconstruction and generation. It will store trained model weights and loss history in the output directory.


### Tensorboard
We can also see visualization of evaluation metrics from tensorboard using following command
