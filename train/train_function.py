import torch
from tensorboardX import SummaryWriter
from transformers import AdamW 
from transformers import get_polynomial_decay_schedule_with_warmup
from tqdm import tqdm, trange
import apex
from apex import amp
import logging
from .reconstruction import *
from .generation import *
from functions import *


def train_vae_ddpm(model, train_dataloader,  output_dir, condition_f=lambda x: False,
                   local_rank = 0, logging_steps = -1,
          train_epoch = 20, gradient_accumulation_steps = 1, device = 'cpu',
          fp16=False, fp16_opt_level=None, learning_rate=9e-5, adam_epsilon=1e-5,
          lr_end_multiplier= 0.01, power=3.0, warmup_steps=0, 
          disable_bar=True, max_grad_norm=1):
    """ Train the model 
    condition_f: a function for linear warmup and decay
    evaluate_during_training: True only if using one GPU, or metrics may not average well
    model_ppl and tokenizer_ppl are required
    no_save: False if you want to save checkpoint
    """
    torch.cuda.set_device(local_rank) # set cuda to local rank; should be discouraged
    torch.cuda.empty_cache()

    if local_rank in [-1, 0]:
        tb_writer = SummaryWriter('./runs/' + output_dir)

    t_total = len(train_dataloader) // gradient_accumulation_steps * train_epoch
   
    model = model.to(device)
    para = [p for n, p in model.model_vae.named_parameters() if condition_f(n)]
    if model.ddpm_weight > 0:
        para.extend([p for n, p in model.ddpm.named_parameters()])
    if not fp16:
        optimizer_grouped_parameters = [
            {'params': para,
                'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=adam_epsilon)
    else:
        optimizer = apex.optimizers.FusedAdam(para, lr=learning_rate, eps=adam_epsilon)
        model, optimizer = amp.initialize(model, optimizer, opt_level=fp16_opt_level)
    parameter_name = []
    parameter_name.extend([n for n, _ in model.model_vae.named_parameters() if
                            condition_f(n)])
    parameter_name.extend([n for n, _ in model.ddpm.named_parameters()])
    scheduler = get_polynomial_decay_schedule_with_warmup(optimizer, warmup_steps, num_training_steps=t_total, \
                                                          lr_end=learning_rate*lr_end_multiplier, power=power)

    # multi-gpu training (should be after apex fp16 initialization)
    

    # Distributed training (should be after apex fp16 initialization)
    
    torch.distributed.init_process_group(backend='nccl',init_method='env://')
    # if local_rank != -1:
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank],
                                                          output_device=local_rank,
                                                          )

    # Train!
    # set_trace(term_size=(120,30))

    global_step = 0
    train_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()

    # model = model.module if hasattr(model,
    #                                         'module') else model  # Take care of distributed/parallel training
    
    train_iterator = trange(int(train_epoch), desc="Epoch", disable=disable_bar)
    model.eval()

    torch.distributed.barrier()

    if logging_steps == -1:
        logging_steps = len(train_dataloader) if len(train_dataloader)<2500 else 2500
    pbar_update = 100 if logging_steps > 1000 else logging_steps //5
    for epoch in train_iterator:
        # train_dataloader.reset()
        model.zero_grad()
        for idx_file in range(1):

            epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=False) 
            for step, batch in enumerate(epoch_iterator):
                # set_trace(term_size=(120,30))
                tokenized_text0, tokenized_text1, _ = batch
                inputs, labels = tokenized_text0, tokenized_text1
                labels = tokenized_text1

                tokenized_text1 = tokenized_text1.to(device)
                inputs = inputs.to(device)
                labels = labels.to(device)

                model.train()
                loss_rec, loss_kl, loss, latent_z, mu, ddpm_loss, loss_weight = model(inputs, labels, return_z=True)

                if train_step % 100 == 0:
                    if local_rank in [-1, 0]:
                        tb_writer.add_scalar('loss_rec_train', loss_rec.mean().item(), train_step)
                        tb_writer.add_scalar('loss_kl_train', loss_kl.mean().item(), train_step)
                        tb_writer.add_scalar('loss_train', loss.mean().item(), train_step)
                        tb_writer.add_scalar('lr_train', scheduler.get_last_lr()[0], train_step)
                        tb_writer.add_scalar('loss_ddpm_train', ddpm_loss.mean().item(), train_step)
                    torch.distributed.barrier()
                train_step += 1
                loss_rec = loss_rec.mean()  # mean() to average on multi-gpu parallel training
                loss_kl = loss_kl.mean()

                if train_step % pbar_update == 0:
                    epoch_iterator.set_description(
                        (
                            f'iter: {step + epoch * len(epoch_iterator)}; loss: {loss.item():.3f}; '
                            f'loss_rec: {loss_rec.item():.3f}; ddpm: {ddpm_loss.mean().item():.3f}; '
                        )
                    )

                if gradient_accumulation_steps > 1:
                    loss = loss / gradient_accumulation_steps

                if fp16:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()

                tr_loss += loss.item()
                if (step + 1) % gradient_accumulation_steps == 0:
                    if fp16:
                        torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), max_grad_norm)
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

                    optimizer.step()

                    scheduler.step()  # Update learning rate schedule

                    model.zero_grad()
                    
                    global_step += 1

                    torch.distributed.barrier()

    
    if local_rank in [-1, 0]:
        tb_writer.close()

    return global_step, tr_loss / global_step, optimizer