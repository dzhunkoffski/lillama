import torch
from torch import nn

from torch import Tensor

@torch.no_grad()
def generate_nucleus(model, tokenizer, batch_size: int, device, prefix: Tensor = None, max_len=100, nucleus=0.9, vocab_size: int = 1000):
    """
    Samples output sequence from probability distribution obtained by model

    :params
        model: predict next token for the whole batch of sequences
        tokenizer: tokenizer for the model and [BOS] token
        batch_size: number of sequence
        prefix: Tensor of tokens with shape: [batch_size, seq_len]
        max_len: max length of predicted sequence
        nucleus: parameter of nucleus sampling

    :return
        the Tensor of tokens of shape: [batch_size, max_len]
    """
    
    prefix = torch.stack(batch_size * [prefix], dim=0)
    prefix = prefix.to(device)

    for i in range(max_len):
        next_token = model.get_next_token(prefix)
        ordered_probs, ordered_indexes = torch.topk(next_token, vocab_size, largest=True, sorted=True)
        probs_cumsum = torch.cumsum(ordered_probs, dim=-1)
        tokens = []
        for batch_ix in range(batch_size):
            ordered_ixs_threshold = torch.argwhere(probs_cumsum[batch_ix] > nucleus).squeeze()[0]
            sampled_ix = torch.multinomial(ordered_probs[batch_ix][:ordered_ixs_threshold+1], 1)
            tokens.append(ordered_indexes[batch_ix][sampled_ix])
        tokens = torch.tensor(tokens).unsqueeze(-1).to(device)
        prefix = torch.cat([prefix, tokens], dim=1)

    return prefix