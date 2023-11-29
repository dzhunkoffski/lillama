import torch
from torch import nn
from torch import Tensor

class PositionalEncoding(nn.Module):

    def __init__(self, embed_dim, max_len: int = 5000):
        """
        Inputs
            embed_dim - Hidden dimensionality of the input.
            max_len - Maximum length of a sequence to expect.
        """
        super().__init__()
        # here should be a tensor of size (1, max_len, embed_dim), dummy dimension is needed for proper addition

        pe = torch.arange(max_len).float()
        pe = pe.repeat(embed_dim, 1).t()
        pe = pe.unsqueeze(0)

        sin_seq = (torch.arange(start=0, end=embed_dim, step=2) / embed_dim).float()
        cos_seq = ((torch.arange(start=1, end=embed_dim, step=2) - 1) / embed_dim).float()

        sin_seq = torch.pow(
            torch.tensor([10000] * len(sin_seq)),
            sin_seq
        )
        cos_seq = torch.pow(
            torch.tensor([10000] * len(cos_seq)),
            cos_seq
        )
        pe[0, :, 0::2] /= sin_seq
        pe[0, :, 0::2] = torch.sin(pe[0, :, 0::2])
        pe[0, :, 1::2] /= cos_seq
        pe[0, :, 1::2] = torch.cos(pe[0, :, 1::2])

        # register_buffer => Tensor which is not a parameter, but should be part of the modules state.
        # Used for tensors that need to be on the same device as the module.
        # persistent=False tells PyTorch to not add the buffer to the state dict (e.g. when we save the model)
        self.register_buffer('pe', pe, persistent=False)

    def forward(self, x):
        seq_len = x.size()[1]
        x = x + self.pe[:, :seq_len, :]
        return x

