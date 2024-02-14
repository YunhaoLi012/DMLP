import torch.nn.init as init
def weights_init_random(model):
    model = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
    model_state_dict = model.state_dict()
    for key in model_state_dict:
        #         pdb.set_trace()
        if 'encoder' in key:
            init.normal_(model_state_dict[key].data)