import torch
import torch.nn.functional as F

def sample_sequence_conditional(model, length, context, past=None, num_samples=1, temperature=1, top_k=0, top_p=0.0,
                                device='cpu', decoder_tokenizer=None, eos_id=50259, loss=False):
    context = torch.tensor(context, dtype=torch.long, device=device)
    context = context.unsqueeze(0).repeat(num_samples, 1)
    generated = context
    # sets = set()
    bz = context.size(0)
    t1, t2, t3 = 0, 0, 0
    alist = list(range(bz))
    with torch.no_grad():
        for ii in range(length):
            inputs = {'input_ids': generated, 'past': past}
            outputs = model(
                **inputs)  # Note: we could also use 'past' with GPT-2/Transfo-XL/XLNet (cached hidden-states)
            next_token_logits = outputs[0][:, -1, :] / temperature
            # next_token = torch.multinomial(F.softmax(next_token_logits, dim=-1), num_samples=1)
            next_token = F.softmax(next_token_logits, dim=-1).max(-1, keepdim=True)[1]
            generated = torch.cat((generated, next_token), dim=1)
            tmp = (next_token.squeeze() == eos_id)
            if ii == 0:
                tmp22 = torch.zeros_like(tmp, device=device)
            tmp22 = torch.logical_or(tmp22, tmp)
            if False not in tmp22:
                break
        if loss:
            outputs = model(input_ids=generated, past=past, labels=generated,
                            label_ignore=decoder_tokenizer.pad_token_id)
            rec_loss = (-outputs[0]).tolist()
            return generated, rec_loss
    return generated
