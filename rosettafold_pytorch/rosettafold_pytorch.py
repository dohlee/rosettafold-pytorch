import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange
from performer_pytorch import SelfAttention as PerformerSelfAttention


class Residual(nn.Module):
    def __init__(self, fn, p_dropout=None):
        super().__init__()
        self.fn = fn
        self.dropout = nn.Dropout(p_dropout) if p_dropout is not None else None

    def forward(self, x):
        if self.dropout is not None:
            return self.dropout(self.fn(x)) + x
        else:
            return self.fn(x) + x


class ColWise(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        b, n = x.size(0), x.size(1)
        x = rearrange(x, "b n l d -> (b n) l d")
        x = self.fn(x)
        x = rearrange(x, "(b n) l d -> b n l d", b=b, n=n)
        return x


class RowWise(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        b, n = x.size(0), x.size(1)
        x = rearrange(x, "b n l d -> (b l) n d")
        x = self.fn(x)
        x = rearrange(x, "(b l) n d -> b n l d", b=b, n=n)
        return x


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
        self.tied = tied

        # Define attention operation
        if self.tied:
            self.attn = SoftTiedAttentionOverResidues(
                d_emb=d_emb, n_heads=n_heads, p_dropout=p_dropout
            )
        elif performer:
            self.attn = PerformerSelfAttention(
                dim=d_emb,
                heads=n_heads,
                dropout=p_dropout,
                **performer_kws,
            )
        else:
            raise NotImplementedError

        # Define layer
        self.att = Residual(
            nn.Sequential(
                nn.LayerNorm(d_emb),
                self.attn,
                nn.Dropout(p_dropout),
            ),
        )
        self.ff = Residual(
            nn.Sequential(
                nn.LayerNorm(d_emb),
                FeedForward(d_emb, d_ff, p_dropout=p_dropout),
                nn.Dropout(p_dropout),
            )
        )

    def forward(self, x):
        N = x.size(1)  # Number of sequences

        if not self.tied:
            x = rearrange(x, "b n l d -> (b n) l d")

        x = self.att(x)

        if not self.tied:
            x = rearrange(x, "(b n) l d -> b n l d", n=N)

        return self.ff(x)


class MSAUpdateUsingSelfAttention(nn.Module):
    def __init__(
        self,
        d_emb=384,
        d_ff=384 * 4,
        n_heads=12,
        p_dropout=0.1,
        performer_kws={},
    ):
        super().__init__()

        self.layer = nn.Sequential(
            EncoderLayer(
                d_emb=d_emb,
                d_ff=d_ff,
                n_heads=n_heads,
                p_dropout=p_dropout,
                tied=True,
                performer=False,
            ),
            Rearrange("b n l d -> b l n d"),
            EncoderLayer(
                d_emb=d_emb,
                d_ff=d_ff,
                n_heads=n_heads,
                p_dropout=p_dropout,
                tied=False,
                performer=True,
                performer_kws=performer_kws,
            ),
            Rearrange("b l n d -> b n l d"),
        )

    def forward(self, x):
        return self.layer(x)


class OuterProductMean(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.to_out = nn.Sequential(
            nn.LayerNorm(in_features**2), nn.Linear(in_features**2, out_features)
        )

    def forward(self, x, y=None):
        y = x if y is None else y

        x = torch.einsum("b n i u, b n j v -> b i j u v", x, y)  # Outer product mean
        x = rearrange(x, "b i j u v -> b i j (u v)")

        return self.to_out(x)


class PairUpdateWithMSA(nn.Module):
    def __init__(self, d_emb, d_proj, d_pair, n_heads, p_dropout=0.1):
        super().__init__()

        self.proj_msa = nn.Sequential(
            nn.LayerNorm(d_emb),
            nn.Linear(d_emb, d_proj),
            nn.LayerNorm(d_proj),
        )
        self.poswise_weight = PositionWiseWeightFactor(d_proj, 1, p_dropout)

        self.outer_product_mean = OuterProductMean(d_proj, d_pair)
        self.ln_coevol_feat = nn.LayerNorm(d_pair)
        self.ln_pair = nn.LayerNorm(d_pair)

        d_feat_full = d_pair * 2 + d_proj * 4 + n_heads
        self.resnet = nn.Sequential(
            nn.Linear(d_feat_full, d_pair),
            Residual(
                nn.Sequential(
                    Rearrange("b l1 l2 d -> b d l1 l2"),
                    nn.Conv2d(
                        d_pair, d_pair, kernel_size=3, padding="same", bias=False
                    ),
                    nn.InstanceNorm2d(d_pair, affine=True, eps=1e-6),
                    nn.ELU(),
                    nn.Dropout(p_dropout),
                    nn.Conv2d(
                        d_pair, d_pair, kernel_size=3, padding="same", bias=False
                    ),
                    nn.InstanceNorm2d(d_pair, affine=True, eps=1e-6),
                    Rearrange("b d l1 l2 -> b l1 l2 d"),
                )
            ),
            # (b l1 l2 d_pair)
            nn.ELU(),
        )

    def forward(self, msa, pair, att):
        L = msa.size(2)  # Length of sequences in MSA
        msa_proj = self.proj_msa(msa)  # (b N l d)

        w = self.poswise_weight(msa_proj)
        w = rearrange(w, "b n 1 l 1 -> b n l 1")  # (b N l 1)

        msa_proj_weighted = msa_proj * w

        coevol_feat = self.outer_product_mean(msa_proj, msa_proj_weighted)
        coevol_feat = self.ln_coevol_feat(coevol_feat)

        msa_1d = torch.cat(
            [
                reduce(msa_proj, "b n l d -> b l d", "sum"),  # IDEA: mean-reduction?
                msa_proj[:, 0],  # MSA embeddings for query sequence
            ],
            dim=-1,
        )

        msa_rowwise_tiled_feat = repeat(msa_1d, "b l1 d -> b l1 l2 d", l2=L)
        msa_colwise_tiled_feat = repeat(msa_1d, "b l1 d -> b l2 l1 d", l2=L)

        feat = torch.cat(
            [
                coevol_feat,  # (b l l d_pair)
                msa_rowwise_tiled_feat,  # (b l l d_proj * 2)
                msa_colwise_tiled_feat,  # (b l l d_proj * 2)
                self.ln_pair(pair),  # (b l l d_pair)
                att,  # (b l l n_heads)
            ],
            dim=-1,
        )

        return self.resnet(feat)


class PairUpdateWithAxialAttention(nn.Module):
    def __init__(self, d_pair, d_ff, n_heads, p_dropout, performer_kws={}):
        super().__init__()

        self.row_attn = PerformerSelfAttention(
            dim=d_pair,
            heads=n_heads,
            dropout=p_dropout,
            generalized_attention=True,
            **performer_kws,
        )
        self.col_attn = PerformerSelfAttention(
            dim=d_pair,
            heads=n_heads,
            dropout=p_dropout,
            generalized_attention=True,
            **performer_kws,
        )
        self.ff = FeedForward(d_pair, d_ff, p_dropout)

        self.layer = nn.Sequential(
            Residual(nn.Sequential(nn.LayerNorm(d_pair), RowWise(self.row_attn))),
            Residual(nn.Sequential(nn.LayerNorm(d_pair), ColWise(self.col_attn))),
            Residual(nn.Sequential(nn.LayerNorm(d_pair), self.ff)),
        )

    def forward(self, x):
        return self.layer(x)


class Symmetrization(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        xt = rearrange(x, "b i j d -> b j i d")
        return 0.5 * (x + xt)


class MSAUpdateWithPair(nn.Module):
    def __init__(self, d_emb, d_pair, n_heads, p_dropout=0.1):
        super().__init__()

        self.pair2att = nn.Sequential(
            Symmetrization(),
            nn.LayerNorm(d_pair),
            nn.Linear(d_pair, n_heads),
            nn.Dropout(p_dropout),
            Rearrange("b i j h -> b () h i j"),
            nn.Softmax(dim=-1),
        )

        self.msa2value = nn.Sequential(
            nn.LayerNorm(d_emb),
            nn.Linear(d_emb, d_emb),
            Rearrange("b n j (h d) -> b n h j d", h=n_heads),
        )

        self.ff = Residual(
            nn.Sequential(
                nn.LayerNorm(d_emb),
                FeedForward(d_emb, d_emb, p_dropout),
            ),
            p_dropout=p_dropout,
        )

        self.dropout1 = nn.Dropout(p_dropout)

    def forward(self, msa, pair):
        value = self.msa2value(msa)
        att = self.pair2att(pair)

        updated = self.dropout(
            torch.einsum("b ... h i j, b n h j d -> b n h i d", att, value)
        )

        return self.ff(msa + updated)


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
