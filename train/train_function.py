import torch
from tensorboardX import SummaryWriter
from transformers import AdamW 
from transformers import get_polynomial_decay_schedule_with_warmup
from tqdm import tqdm, trange
import apex
from apex import amp
import logging
from accelerate.utils import set_seed
from reconstruction import *
from generation import *
from functions import save_checkpoint


def train_vae_ddpm(model, train_dataloader, encoder_tokenizer, decoder_tokenizer, 
          table_name, eval_dataloader, output_dir, condition_f=lambda x: False,
          checkpoint=None, local_rank = -1, batch_size = 32, eval_batch_size = 32,
          train_epoch = 20, gradient_accumulation_steps = 1, device = 'cpu',
          fp16=False, fp16_opt_level=None, learning_rate=9e-5, adam_epsilon=1e-5,
          lr_end_multiplier= 0.01, power=3.0, warmup_steps=0, random_state=False, seed=44,
          disable_bar=True, model_ppl=None, tokenizer_ppl=None, max_grad_norm=1, evaluate_during_training=False,
          no_save=True):
    """ Train the model 
    condition_f: a function for linear warmup and decay
    evaluate_during_training: True only if using one GPU, or metrics may not average well
    model_ppl and tokenizer_ppl are required
    no_save: False if you want to save checkpoint
    """
    logger = logging.getLogger(__name__)

    torch.cuda.set_device(local_rank)
    torch.cuda.empty_cache()

    if local_rank in [-1, 0]:
        tb_writer = SummaryWriter('./runs/' + output_dir)

    train_batch_size = batch_size
    t_total = len(train_dataloader) // gradient_accumulation_steps * train_epoch
    # Prepare optimizer and schedule (linear warmup and decay)
    # if args.fix_model == 84:
    #     def condition_f(n):
    #         return ('linear' in n or 'wte' in n or 'decoder.transformer.h.0' in n or 'encoder' in n)
    
    # elif args.fix_model == 841:
    #     def condition_f(n):
    #         return ('linear' in n or 'lora' in n or 'encoder' in n)
    # model_encoder, model_decoder, model_connector = model_vae.encoder,  model_vae.decoder, model_vae.linear
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
        # ddpm = amp.initialize(ddpm, opt_level=args.fp16_opt_level)
    parameter_name = []
    parameter_name.extend([n for n, _ in model.model_vae.named_parameters() if
                            condition_f(n)])
    parameter_name.extend([n for n, _ in model.ddpm.named_parameters()])
    scheduler = get_polynomial_decay_schedule_with_warmup(optimizer, warmup_steps, num_training_steps=t_total, \
                                                          lr_end=learning_rate*lr_end_multiplier, power=power)

    # multi-gpu training (should be after apex fp16 initialization)
    

    # Distributed training (should be after apex fp16 initialization)
    
    # if args.local_rank != -1:
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank],
                                                          output_device=local_rank,
                                                          )

    # Train!
    # set_trace(term_size=(120,30))
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
    if random_state:
        set_seed(seed)
    model.eval()
    if local_rank==0:
        with torch.no_grad():
            result_new = calc_rec_lgy(model.model_vae, encoder_tokenizer, decoder_tokenizer, eval_dataloader, device, disable_bar, ns=100)
            # result_new.update(evaluate(args, model.module.model_vae, encoder_tokenizer, decoder_tokenizer, table_name,eval_dataloader))
            for key, value in result_new.items():
                logger.info('eval_%s:%f',key,value)
                tb_writer.add_scalar('eval_{}'.format(key), value, global_step)
            results_new = calc_ppl_lgy_ddpm(
                            model.model_vae, encoder_tokenizer, decoder_tokenizer, ns=1,
                            ddpm=model.ddpm, model_ppl=model_ppl, tokenizer_ppl=tokenizer_ppl, fp16=fp16
                        )
            for key, value in result_new.items():
                logger.info('eval_%s:%f',key,value)
                tb_writer.add_scalar('eval_DDPM_{}'.format(key), value, global_step)

        logger.info('\nBLEU is %f\n"', result_new['bleu'])
        for key, value in results_new.items():
            tb_writer.add_scalar('eval_DDPM_{}'.format(key), value, global_step)
            logger.info("DDPM_%s:%s",str(key),str(value))
        logger.info('\nBLEU is %f\n"', result_new['bleu'])
        torch.cuda.empty_cache()
    torch.distributed.barrier()

    best_bleu = 0
    best_ppl = 100
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

                    if local_rank in [0] and logging_steps > 0 and global_step % logging_steps == 0:
                        # Log metrics
                        if local_rank == 0 and evaluate_during_training:  # Only evaluate when single GPU otherwise metrics may not average well
                            #                         args.per_gpu_eval_batch_size = args.per_gpu_eval_batch_size // 2
                            model.eval()
                            with torch.no_grad():
                                results_new = calc_ppl_lgy_ddpm(
                                    model.model_vae, encoder_tokenizer, decoder_tokenizer, ns=1,
                                    ddpm=model.ddpm, model_ppl=model_ppl, tokenizer_ppl=tokenizer_ppl, fp16=fp16
                                )
                                for key, value in results_new.items():
                                    logger.info("DDPM_"+key+": %s",str(results_new[key]))
                                    tb_writer.add_scalar('eval_{}'.format("DDPM_"+key), value, global_step)
                                results = calc_rec_lgy(model.model_vae, encoder_tokenizer, decoder_tokenizer, eval_dataloader, device, disable_bar, ns=100)

                            for key, value in results.items():
                                tb_writer.add_scalar('eval_{}'.format(key), value, global_step)
                        
                        tb_writer.add_scalar('lr', scheduler.get_last_lr()[0], global_step)
                        tb_writer.add_scalar('loss', (tr_loss - logging_loss) / logging_steps, global_step)
                        logging_loss = tr_loss
                        if results['bleu'] >= best_bleu:
                            best_bleu = results['bleu']
                            if not no_save:
                                save_checkpoint(model.model_vae, optimizer, global_step, parameter_name, output_dir, local_rank, logger, ppl=True, ddpm=model.ddpm)
                        if 12 < results_new['ppl'] < best_ppl and results_new['norm_z'] < 12 and global_step > 2 * logging_steps:
                            best_ppl = results_new['ppl']
                            if not no_save:
                                tb_writer.add_scalar('eval_best_ppl', best_ppl, global_step)
                                tb_writer.add_scalar('eval_best_bleu', results['bleu'], global_step)
                                save_checkpoint(model.model_vae, optimizer, global_step, parameter_name, output_dir, local_rank, logger, ppl=True, ddpm=model.ddpm)
                        logger.info("Current Path is %s", output_dir)
                    torch.distributed.barrier()

    results = calc_rec_lgy(model.model_vae, encoder_tokenizer, decoder_tokenizer, eval_dataloader, device, disable_bar, ns=100)

    if local_rank in [-1, 0]:
        tb_writer.close()

    return global_step, tr_loss / global_step, optimizer