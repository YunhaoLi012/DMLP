import torch
from DMLP.utils.sample import sample_sequence_conditional
from tqdm import tqdm, trange
import numpy as np
from transformers import AutoTokenizer, GPT2TokenizerFast
from transformers import GPT2LMHeadModel as GPT2_

def calc_ppl_lgy_ddpm(model_vae, decoder_tokenizer, ns=1, sent_length=32,
                      ddpm=None, device='cpu', output_dir = "output.txt", disable_bar=True, fp16=False, model_id='gpt2', ppl_eval=True):
    generate_text = []
    bz = 100
    num_epoch = ns
    context_tokens = decoder_tokenizer.encode(decoder_tokenizer.bos_token)
    
    loss_list = []
    for _ in trange(num_epoch, desc="Evaluating PPL", disable=disable_bar):
        with torch.no_grad():
            latent_z = ddpm.sample(bz,(model_vae.nz,), device, fp16=fp16)
            loss = True
            out = sample_sequence_conditional(
                model=model_vae.decoder,
                context=context_tokens,
                past=latent_z,
                length=sent_length,
                num_samples=latent_z.size(0),
                device=device,
                decoder_tokenizer=decoder_tokenizer,
                eos_id=decoder_tokenizer.eos_token_id,
                loss=loss
            )
            if loss:
                loss_ = round(-np.mean(out[1]),3)
                loss_list.append(loss_)
                out = out[0]
        for i in range(latent_z.size(0)):
            text_x1 = decoder_tokenizer.decode(out[i, :].tolist(), clean_up_tokenization_spaces=False).split(decoder_tokenizer.eos_token)[
                0].replace(decoder_tokenizer.bos_token, '').strip()
            text_x1 = ' '.join(text_x1.split())
            generate_text.append(text_x1 + '\n')

    with open(output_dir+"/output.txt",'w') as f:
        f.write(''.join(generate_text))
    
    if ppl_eval:
        if model_id == 'gpt2':
            model_ppl = GPT2_.from_pretrained(model_id,local_files_only=False).to(device)
            tokenizer_ppl = GPT2TokenizerFast.from_pretrained(model_id,local_files_only=False)
        else:
            model_ppl = GPT2_.from_pretrained(model_id,local_files_only=False).to(device)
            tokenizer_ppl = GPT2TokenizerFast.from_pretrained(model_id,local_files_only=False)

        encodings = tokenizer_ppl('\n\n'.join(generate_text), return_tensors='pt')
        max_length = model_ppl.config.n_positions
        stride = 512

        nlls = []
        for i in range(0, encodings.input_ids.size(1), stride):
            begin_loc = max(i + stride - max_length, 0)
            end_loc = min(i + stride, encodings.input_ids.size(1))
            trg_len = end_loc - i  # may be different from stride on last loop
            input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
            target_ids = input_ids.clone()
            target_ids[:, :-trg_len] = -100

            with torch.no_grad():
                outputs = model_ppl(input_ids, labels=target_ids)
                neg_log_likelihood = outputs[0] * trg_len

            nlls.append(neg_log_likelihood)
        ppl = torch.exp(torch.stack(nlls).sum() / end_loc)

    list_of_references = []
    len_list = []
    for jj, line in enumerate(generate_text):
        if jj < 10:
            if jj == 0: 
                print('\n\n')
            print(line)
        split = line.strip().split(' ')
        list_of_references.append(split)
        len_list.append(len(split))
    sbleu = []
    num_all = len(list_of_references)
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    for i in range(num_all):
        refs = [list_of_references[j] for j in range(num_all) if i != j]
        bleu_ = sentence_bleu(refs, list_of_references[i], smoothing_function=SmoothingFunction().method1)
        sbleu.append(bleu_ * 100)
    score = np.mean(sbleu)

    len_mean = np.mean(len_list)
    norm_z = latent_z.norm(dim=-1).mean().item()

    if ppl_eval:
        return {'ppl': ppl, 'sbleu': round(score, 2), 'length': round(len_mean, 2), 'norm_z': norm_z,
                'ppl_sbleu': ppl + round(score, 2)}
    else:
        return {'sbleu': round(score, 2), 'length': round(len_mean, 2), 'norm_z': norm_z}