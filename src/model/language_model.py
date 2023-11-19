import torch
from torch import nn

from src.model.utils import PositionalEncoding

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
        query = self.Q_proj(x)
        key = self.K_proj(x)
        value = self.V_proj(x)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(
            sz=x.size()[1], device=x.get_device()
        )

        # print(causal_mask.size())
        # print(query.size())
        # print(key.size())
        # print(value.size())

        output, _ = self.masked_multihead_attention(query, key, value, attn_mask=causal_mask)
        output = self.dropout(output)
        output = x + output
        x = self.lay_norm1(output)

        output = self.feedforward(x)
        output = self.dropout(output)

        output = x + output
        x = self.lay_norm2(output)
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
