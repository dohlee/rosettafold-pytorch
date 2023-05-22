import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl


def sinusoidal_positional_encoding(max_len, d_emb):
    # Sinusoidal positional encoding
    # PE(pos, 2i) = sin(pos / 10000^(2i/d_emb))
    # PE(pos, 2i+1) = cos(pos / 10000^(2i/d_emb))
    pos_enc = torch.zeros(max_len, d_emb)
    pos = torch.arange(0, max_len).view(-1, 1)
    denom = torch.exp(math.log(10000.0) * torch.arange(0, d_emb, 2) / d_emb)

    pos_enc[:, 0::2] = torch.sin(pos / denom)
    pos_enc[:, 1::2] = torch.cos(pos / denom)
    return pos_enc


class MSAEmbedding(nn.Module):
    def __init__(self, d_input=21, d_emb=64, max_len=5000, p_pe_drop=0.1):
        super().__init__()
        self.to_embedding = nn.Embedding(d_input, d_emb)

        self.pos_enc = sinusoidal_positional_encoding(max_len, d_emb)
        self.pos_enc_drop = nn.Dropout(p_pe_drop)

        self.query_enc = nn.Embedding(2, d_emb)  # 0: query, 1: targets

    def forward(self, x):
        query_idx = torch.ones(x.size(-2), 1).long()
        query_idx[0] = 0

        pe, qe = self.pos_enc_drop(self.pos_enc), self.query_enc(query_idx)

        # x : B, N, L
        x = self.to_embedding(x) + pe + qe
        return x  # B, N, L, d_emb


class RoseTTAFold(pl.LightningModule):
    def __init__(
        self,
        d_input=21,
        d_emb=64,
        max_len=5000,
        p_pe_drop=0.1,
    ):
        super().__init__()

        self.msa_emb = MSAEmbedding(
            d_input=d_input, d_emb=d_emb, max_len=max_len, p_pe_drop=p_pe_drop
        )

    def forward(self, msa, distogram):
        msa = self.msa_emb(msa)

        pass

    def training_step(self, batch, batch_idx):
        pass

    def validation_step(self, batch, batch_idx):
        pass

    def test_step(self, batch, batch_idx):
        pass


if __name__ == "__main__":
    msa = torch.randint(0, 21, (1, 10, 5000))
    msa_emb = MSAEmbedding()

    print(msa_emb(msa).shape)
