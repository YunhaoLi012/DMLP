import torch
from nltk.translate.bleu_score import corpus_bleu
from modules.sample import sample_sequence_conditional
from collections import defaultdict
from tqdm import tqdm





def calc_rec_lgy(model_vae, encoder_tokenizer, decoder_tokenizer,eval_dataloader, device, disable_bar, ns=1):

    # eval_dataloader = build_dataload_and_cache_examples(args, [encoder_tokenizer, decoder_tokenizer], evaluate=True)
    count = 0
    result = defaultdict(str)
    ref = []
    cand = []
    for batch in tqdm(eval_dataloader, desc="Evaluating recontruction", disable=disable_bar):
        x0, x1, x_lengths = batch

        max_len_values, _ = x_lengths.max(0)
        x0 = x0[:, :max_len_values[0]]
        x1 = x1[:, :max_len_values[1]]
        x0 = x0.to(device)
        x1 = x1.to(device)
        x_lengths = x_lengths.to(device)
        context_tokens = decoder_tokenizer.encode(decoder_tokenizer.bos_token)

        with torch.no_grad():
            # text_x0 = encoder_tokenizer.decode(x0[0,:x_lengths[0,0]].tolist(), clean_up_tokenization_spaces=True)[0]
            # result["INPUT TEXT " + str(count)].append(text_x0)
            attention_mask = (x0 != encoder_tokenizer.pad_token_id).float()

            pooled_hidden_fea = model_vae.encoder(x0, attention_mask)[1]

            # Connect hidden feature to the latent space
            # latent_z, loss_kl = model_vae.connect(pooled_hidden_fea)
            mean, logvar = model_vae.encoder.linear(pooled_hidden_fea).chunk(2, -1)
            # latent_z = model_vae.reparameterize(mean, logvar, nsamples=1).squeeze(1)
            latent_z = mean.squeeze(1)

            past = latent_z
            out = sample_sequence_conditional(
                model=model_vae.decoder,
                context=context_tokens,
                past=past,
                length=x_lengths[0, 1],  # Chunyuan: Fix length; or use <EOS> to complete a sentence
                num_samples=latent_z.size(0),
                device=device,
                decoder_tokenizer=decoder_tokenizer,
                eos_id=model_vae.eos_token_id
            )

        for i in range(latent_z.size(0)):
            text_x0_ = decoder_tokenizer.decode(x1[i, :].tolist(), clean_up_tokenization_spaces=False).split(decoder_tokenizer.eos_token)[
                0].replace(decoder_tokenizer.bos_token, '').strip()
            text_x0_ = text_x0_.split()
            text_x1 = decoder_tokenizer.decode(out[i, :].tolist(), clean_up_tokenization_spaces=False).split(decoder_tokenizer.eos_token)[
                0].replace(decoder_tokenizer.bos_token, '').strip()
            text_x1 = text_x1.split()

            count += 1
            ref.append([text_x0_])
            cand.append(text_x1)

        if count > 1000:
            break
    bleu = corpus_bleu(ref, cand) * 100
    # output_eval_file = os.path.join(args.output_dir, "eval_results_bleu.txt")
    # if not os.path.exists(args.output_dir):
    #     os.makedirs(args.output_dir)
    # with open(output_eval_file, "w") as writer:
    #     writer.write("%s = %s\n" % ('bleu', str(bleu)))
    return {'bleu': bleu}

