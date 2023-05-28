import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from einops import rearrange
from einops.layers.torch import Rearrange
from performer_pytorch import SelfAttention as PerformerSelfAttention


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x


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


class PositionWiseWeightFactor(nn.Module):
    def __init__(self, d_emb=384, n_heads=12, p_dropout=0.1):
        super().__init__()

        assert (
            d_emb % n_heads == 0
        ), f"[{self.__class__.__name__}]: d_emb ({d_emb}) must be divisible by n_heads ({n_heads})."

        self.d_head = d_emb // n_heads
        self.scale = self.d_head ** (-0.5)

        self.to_q = nn.Sequential(
            nn.Linear(d_emb, d_emb),  # IDEA: maybe we can use LinearNoBias here.
            Rearrange("b 1 l (h d) -> b l h 1 d", h=n_heads),
        )
        self.to_k = nn.Sequential(
            nn.Linear(d_emb, d_emb),  # IDEA: maybe we can use LinearNoBias here.
            Rearrange("b N l (h d) -> b l h N d", h=n_heads),
        )
        self.dropout = nn.Dropout(p_dropout)

    def forward(self, msa_emb):
        """msa : (B, N, L, d_emb)"""
        query_seq = msa_emb[:, 0].unsqueeze(1)  # Take the first sequence as query.

        q = self.to_q(query_seq) * self.scale
        k = self.to_k(msa_emb)

        logits = torch.einsum("b l h q d, b l h n d -> b l h q n", q, k)
        # IDEA: maybe we can use dropout here at logits to make weights sum to 1.

        att = logits.softmax(dim=-1)
        att = rearrange(att, "b l h 1 N -> b N h l 1")
        return self.dropout(att)


class SoftTiedAttentionOverResidues(nn.Module):
    def __init__(self, d_emb=384, n_heads=12, p_dropout=0.1):
        super().__init__()
        assert (
            d_emb % n_heads == 0
        ), f"[{self.__class__.__name__}]: d_emb ({d_emb}) must be divisible by n_heads ({n_heads})."

        self.n_heads = n_heads
        self.d_head = d_emb // n_heads
        self.scale = self.d_head ** (-0.5)

        self.poswise_weight = PositionWiseWeightFactor(d_emb, n_heads, p_dropout)

        # IDEA: maybe we can use LinearNoBias for the projections below.
        self.to_q = nn.Linear(d_emb, d_emb)
        self.to_k = nn.Linear(d_emb, d_emb)
        self.to_v = nn.Linear(d_emb, d_emb)
        self.to_out = nn.Linear(d_emb, d_emb)
        self.dropout = nn.Dropout(p_dropout)

    def forward(self, x):
        """x : (B, N, L, d_emb)"""
        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)

        q, k, v = map(
            lambda t: Rearrange("b n l (h d) -> b n h l d", h=self.n_heads)(t),
            (q, k, v),
        )
        # poswise_weight : (b n h l 1)
        q = q * self.poswise_weight(x) * self.scale

        logits = torch.einsum("b n h i d, b n h j d -> b n h i j", q, k)
        att = logits.softmax(dim=-1)

        out = torch.einsum("b n h i j, b n h j d -> b n h i d", att, v)
        out = rearrange(out, "b n h l d -> b n l (h d)")
        out = self.to_out(out)

        return self.dropout(out)


class FeedForward(nn.Module):
    def __init__(self, d_emb, d_ff, p_dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_emb, d_ff),
            nn.ReLU(),
            nn.Dropout(p_dropout),
            nn.Linear(d_ff, d_emb),
        )

    def forward(self, x):
        return self.net(x)


class EncoderLayer(nn.Module):
    def __init__(
        self,
        d_emb=384,
        d_ff=384 * 4,
        n_heads=12,
        p_dropout=0.1,
        tied=False,
        performer=False,
        performer_kws={},
    ):
        super().__init__()

        # Define attention operation
        if tied:
            self.attn = SoftTiedAttentionOverResidues(
                d_emb=d_emb, n_heads=n_heads, p_dropout=p_dropout
            )
        elif performer:
            self.attn = PerformerSelfAttention(
                dim=d_emb, heads=n_heads, dropout=p_dropout,
                generalized_attention=True, **performer_kws
            )
        else:
            raise NotImplementedError

        # Define layer
        self.layer = nn.Sequential(
            Residual(
                nn.Sequential(
                    Rearrange('b n l d -> (b n) l d') if not tied else nn.Identity(),
                    nn.LayerNorm(d_emb),
                    self.attn,
                    nn.Dropout(p_dropout),
                    Rearrange('(b n) l d -> b n l d', n=n_heads) if not tied else nn.Identity(),
                ),
            ),
            Residual(
                nn.Sequential(
                    nn.LayerNorm(d_emb),
                    FeedForward(d_emb, d_ff, p_dropout=p_dropout),
                    nn.Dropout(p_dropout),
                )
            )
        )

    def forward(self, x):
        return self.layer(x)


class MSAUpdateUsingSelfAttention(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        pass


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
