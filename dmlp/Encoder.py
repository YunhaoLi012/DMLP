import transformers

import torch
import torch.nn as nn


class Encoder(nn.Module):
    """
    Encoder
    """

    def __init__(self,
                 modelname = 'torch_embedding', pretrained = True):
        super().__init__()

        if modelname == "torch_embedding":
            self.encoder = nn.Embedding()