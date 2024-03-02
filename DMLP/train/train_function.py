import torch
from tensorboardX import SummaryWriter
from transformers import AdamW 
from transformers import get_polynomial_decay_schedule_with_warmup
from tqdm import tqdm, trange
# import apex
from torch.cuda.amp import GradScaler
# from apex import amp
import logging
from DMLP.train import *
from DMLP.utils.ddpm_schedule import *
from DMLP.utils.random_init import *
from DMLP.utils.sample import *
from DMLP.utils.save_checkpoint import *
import os

from DMLP.train.evaluation import evaluation



def train_vae_ddpm(local_rank, world_size, model, optimizer, train_dataloader,  output_dir, batch_size,condition_f=lambda x: False, logging_steps = -1,
          train_epoch = 20, gradient_accumulation_steps = 1, device = 'cpu',
          fp16=False, fp16_opt_level=None, learning_rate=9e-5, adam_epsilon=1e-5,
          lr_end_multiplier= 0.01, power=3.0, warmup_steps=0, 
          disable_bar=True, max_grad_norm=1,save=True, evaluate_during_training=False, eval_dataloader=None,
          sent_length=32, model_id='gpt2', ppl_eval=True):
    
    """ 
    Train the model 
    condition_f: a function for linear warmup and decay
    evaluate_during_training: True only if using one GPU, or metrics may not average well
    model_ppl and tokenizer_ppl are required
    save: True if you want to save checkpoint
    output_dir: provide absolute path to store the outputs
    device: device where the model store
    local_rank: list of gpus for parallel processing

    if evaluate_during_training: all related inputs should be given
    """
    # torch.cuda.set_device(local_rank) # set cuda to local rank; should be discouraged
    device = f"cuda:{local_rank}"
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    torch.cuda.empty_cache()
    logging.basicConfig(filename=output_dir+"/log.txt", level=logging.INFO)
    logger = logging.getLogger(__name__)
    writer = SummaryWriter(output_dir+"/tensorboard")

    t_total = len(train_dataloader) // gradient_accumulation_steps * train_epoch
   
    model = model.to(local_rank)
    para = [p for n, p in model.model_vae.named_parameters() if condition_f(n)]
    if model.ddpm_weight > 0:
        para.extend([p for n, p in model.ddpm.named_parameters()])
    if not fp16:
        # optimizer_grouped_parameters = [
        #     {'params': para,
        #         'weight_decay': 0.0}
        # ]
        optimizer = optimizer(para, lr=learning_rate, eps=adam_epsilon)
    else:
        optimizer = optimizer(para, lr=learning_rate, eps=adam_epsilon, fused=torch.float16)
        scaler = GradScaler()
    parameter_name = []
    parameter_name.extend([n for n, _ in model.model_vae.named_parameters() if
                            condition_f(n)])
    parameter_name.extend([n for n, _ in model.ddpm.named_parameters()])
    scheduler = get_polynomial_decay_schedule_with_warmup(optimizer, warmup_steps, num_training_steps=t_total, \
                                                          lr_end=learning_rate*lr_end_multiplier, power=power)

    # multi-gpu training (should be after apex fp16 initialization)
    

    # Distributed training (should be after apex fp16 initialization)
    
    torch.distributed.init_process_group(backend='nccl',init_method='env://',rank = local_rank,world_size = world_size)
    # if local_rank != -1:
    model = torch.nn.parallel.DataParallel(model, device_ids=[local_rank])

    # Train!
    logger.info("***** Running training *****")
    # logger.info("  Num examples = %d", train_dataloader.num_examples)
    logger.info("  Num Epochs = %d", train_epoch)
    logger.info("  Instantaneous batch size per GPU = %d", batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                batch_size * gradient_accumulation_steps * (
                    torch.distributed.get_world_size() if local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)


    global_step = 0
    train_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()

    # model = model.module if hasattr(model,
    #                                         'module') else model  # Take care of distributed/parallel training
    
    train_iterator = trange(int(train_epoch), desc="Epoch", disable=disable_bar)

    torch.distributed.barrier()

    if logging_steps == -1:
        logging_steps = len(train_dataloader) if len(train_dataloader)<2500 else 2500
    pbar_update = 100 if logging_steps > 1000 else logging_steps //5

    max_bleu_score = 0

    for epoch in train_iterator:
        # train_dataloader.reset()
        model.zero_grad()
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=False) 
        for step, batch in enumerate(epoch_iterator):
            # set_trace(term_size=(120,30))
            tokenized_text0, tokenized_text1, _ = batch
            inputs, labels = tokenized_text0, tokenized_text1

            tokenized_text1 = tokenized_text1.to(local_rank)
            inputs = inputs.to(local_rank)
            labels = labels.to(local_rank)

            model.train()
            if fp16:
                with torch.autocast(device_type="cuda",dtype=torch.float16):
                    loss_rec, loss_kl, loss, latent_z, mu, ddpm_loss, loss_weight = model(inputs, labels)
            else:
                loss_rec, loss_kl, loss, latent_z, mu, ddpm_loss, loss_weight = model(inputs, labels)
            if train_step % 100 == 0:
                writer.add_scalar('loss_rec_train', loss_rec.mean().item(), train_step)
                writer.add_scalar('loss_kl_train', loss_kl.mean().item(), train_step)
                writer.add_scalar('loss_train', loss.mean().item(), train_step)
                writer.add_scalar('lr_train', scheduler.get_last_lr()[0], train_step)
                writer.add_scalar('loss_ddpm_train', ddpm_loss.mean().item(), train_step)
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
                logger.info(
                    (
                    f'iter: {step + epoch * len(epoch_iterator)}; lr_train: {scheduler.get_last_lr()[0]}; '
                    f'loss: {loss.item():.3f}; loss_rec_train: {loss_rec.item():.3f}; '
                    f'loss_kl_train: {loss_kl.mean().item()}; ddpm: {ddpm_loss.mean().item():.3f}; '
                    )
                )

            if gradient_accumulation_steps > 1:
                loss = loss / gradient_accumulation_steps

            if fp16:
                scaler.scale(torch.mean(loss)).backward()

            else:
                loss.backward()
            
            # loss = torch.mean(loss)
            
            writer.add_scalar('lr', scheduler.get_last_lr()[0], global_step)
            writer.add_scalar('loss', (tr_loss - logging_loss) / logging_steps, global_step)
            tr_loss += loss.item()
            if (step + 1) % gradient_accumulation_steps == 0:
                if fp16:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

                scaler.step(optimizer)
                scaler.update()
                scheduler.step()  # Update learning rate schedule

                model.zero_grad()
                
                global_step += 1

              
                if evaluate_during_training and \
                    logging_steps > 0 and global_step % logging_steps == 0 and eval_dataloader != None:
                    
                    model.eval()
                    with torch.no_grad():
                        if fp16:
                            with torch.autocast(device_type="cuda",dtype=torch.float16):
                                loss_rec, loss_kl, loss, latent_z, mu, ddpm_loss, loss_weight = model(inputs, labels)
                                results = evaluation(model.module, eval_dataloader, device, disable_bar, \
                                        output_dir=output_dir, sent_length=sent_length, fp16=fp16, model_id=model_id, ppl_eval=ppl_eval)
                        else:
                            loss_rec, loss_kl, loss, latent_z, mu, ddpm_loss, loss_weight = model(inputs, labels)
                            results = evaluation(model.module, eval_dataloader, device, disable_bar, \
                                output_dir=output_dir, sent_length=sent_length, fp16=fp16, model_id=model_id, ppl_eval=ppl_eval)
                            
                    for key, value in results.items():
                        writer.add_scalar('eval_{}'.format(key), value, global_step)
                        logger.info('eval_{} {}'.format(key, value))
                    if results['bleu'] > max_bleu_score:
                        max_bleu_score = results['bleu']
                        if save:
                            save_checkpoint(model.module.model_vae, optimizer, global_step, parameter_name, output_dir, logger, ppl=True, ddpm=model.module.ddpm)
                            
                torch.distributed.barrier()

    writer.close()
    return global_step, tr_loss / global_step, optimizer


