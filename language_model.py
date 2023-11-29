import torch
from torch import nn
import torch.nn.functional as F

import numpy as np

from utils import PositionalEncoding

class DecoderBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float, feedforward_dim: int, activasion):
        super().__init__()
        self.Q_proj = nn.Linear(embed_dim, embed_dim)
        self.K_proj = nn.Linear(embed_dim, embed_dim)
        self.V_proj = nn.Linear(embed_dim, embed_dim)

        self.masked_multihead_attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True
        )
        self.lay_norm1 = nn.LayerNorm(embed_dim)
        self.feedforward = nn.Sequential(
            nn.Linear(embed_dim, feedforward_dim),
            activasion(),
            nn.Linear(feedforward_dim, embed_dim)
        )
        self.dropout = nn.Dropout(p=dropout)
        self.lay_norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        causal_mask = nn.Transformer.generate_square_subsequent_mask(
            sz=x.size()[1], device=x.get_device()
        )

        output = self.lay_norm1(x)
        query = self.Q_proj(output)
        key = self.K_proj(output)
        value = self.V_proj(output)
        output, _ = self.masked_multihead_attention(query, key, value, attn_mask=causal_mask)
        output = self.dropout(output)
        x = x + output
        
        output = self.lay_norm2(x)
        output = self.feedforward(output)
        output = self.dropout(output)
        x = x + output
        return x

class TransformerDecoder(nn.Module):
    def __init__(
            self, max_len: int, pad_idx: int, vocab_size: int, 
            num_layers: int, embed_dim: int, num_heads: int, dropout: float, 
            feedforward_dim: int, activasion = nn.ReLU):
        super().__init__()
        self.num_layers = num_layers
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size, 
            embedding_dim=embed_dim,
            padding_idx=pad_idx
        )
        self.pos_encoder = PositionalEncoding(embed_dim=embed_dim, max_len=max_len)
        self.decoder = nn.ModuleList([
            DecoderBlock(
                embed_dim=embed_dim, num_heads=num_heads, dropout=dropout, feedforward_dim=feedforward_dim,
                activasion=activasion
            ) for _ in range(num_layers)
        ])
    
    def forward(self, x):
        x = self.embedding(x)
        x = self.pos_encoder(x)
        for lay_ix in range(self.num_layers):
            x = self.decoder[lay_ix](x)
        return x

class LanguageModel(nn.Module):
    def __init__(
            self, embed_dim: int, vocab_size: int, max_len: int, pad_idx: int, 
            num_layers: int, num_heads: int, dropout: float, feedforward_dim: int):
        super().__init__()
        self.transformer = TransformerDecoder(
            max_len=max_len,
            pad_idx=pad_idx,
            vocab_size=vocab_size,
            num_layers=num_layers,
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            feedforward_dim=feedforward_dim,
            activasion=nn.ReLU
        )
        self.linear = nn.Linear(embed_dim, vocab_size)
    
    def forward(self, x):
        x = self.transformer(x)
        x = self.linear(x)
        return x
    
    @torch.no_grad()
    def get_next_token(self, prefix) -> torch.Tensor:
        """ 
        Predict text token for given prefix.

        :params
            prefix -- tensor of shape [batch_size, prefix_len]
        
        :returns: 
            probabilities of next token, 
        """
        prob = F.softmax(self.forward(prefix)[:, -1, :], dim=1)
        return prob
    
    def __str__(self):
        """
        Model prints with number of trainable parameters
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + "\nTrainable parameters: {}".format(params)
