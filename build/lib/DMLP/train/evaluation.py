from DMLP.train.reconstruction import calc_rec_lgy
from DMLP.train.generation import calc_ppl_lgy_ddpm

def evaluation(model, eval_dataloader, device, disable_bar, ns=1, \
               sent_length=32, output_dir="output.txt", fp16=False, model_id='gpt2', ppl_eval=True):

    rec_bleu = calc_rec_lgy(model.model_vae, 
                 model.model_vae.tokenizer_encoder, 
                 model.model_vae.tokenizer_decoder,
                 eval_dataloader,
                 device,
                 disable_bar)
    
    generation_eval = calc_ppl_lgy_ddpm(model.model_vae,
                                        model.model_vae.tokenizer_decoder,
                                        ns=ns,
                                        sent_length=sent_length,
                                        ddpm=model.ddpm,
                                        device=device,
                                        output_dir=output_dir,
                                        disable_bar=disable_bar,
                                        fp16=fp16,
                                        model_id=model_id,
                                        ppl_eval=ppl_eval)
    generation_eval['bleu'] = rec_bleu['bleu']
    return generation_eval