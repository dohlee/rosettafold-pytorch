import math

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange
from performer_pytorch import SelfAttention as PerformerSelfAttention
from se3_transformer_pytorch import SE3Transformer


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


class MsaEmbedding(nn.Module):
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


class MsaUpdateUsingSelfAttention(nn.Module):
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

        n = x.size(1)
        x = (
            torch.einsum("b n i u, b n j v -> b i j u v", x, y) / n
        )  # Outer product mean
        x = rearrange(x, "b i j u v -> b i j (u v)")

        return self.to_out(x)


class PairUpdateWithMsa(nn.Module):
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


class GraphTransformer(nn.Module):
    def __init__(self, d_node_in, d_node_out, d_edge, n_heads, p_dropout=0.15):
        super().__init__()
        self.scale = d_node_out ** (-0.5)

        # These linear weights should be initialized with
        # kaiming_uniform(weight, fan=in_channels, a=math.sqrt(5))
        self.node_update = nn.Linear(d_node_in, d_node_out * n_heads, bias=True)

        self.node_to_q = nn.Linear(d_node_in, d_node_out * n_heads, bias=True)
        self.node_to_k = nn.Linear(d_node_in, d_node_out * n_heads, bias=True)
        self.node_to_v = nn.Linear(d_node_in, d_node_out * n_heads, bias=True)

        self.edge_emb = nn.Linear(d_edge, d_node_out * n_heads, bias=False)

        self.att_dropout = nn.Dropout(p_dropout)

        self.n_heads = n_heads

    def forward(self, node_feat, edge_feat, edge_mask):
        """node_feat, (b l d_node_in)
        edge_feat, (b l l d_edge)
        edge_mask, (b l l): True (1) if eij exists, else False (0)
        """
        q = self.node_to_q(node_feat)
        k = self.node_to_k(node_feat)
        v = self.node_to_v(node_feat)
        q, k, v = map(
            lambda t: rearrange(t, "b l (h d) -> b h l d", h=self.n_heads), (q, k, v)
        )

        e = self.edge_emb(edge_feat)
        e = rearrange(e, "b i j (h d) -> b h i j d", h=self.n_heads)

        logit = torch.einsum("b h i d, b h j d -> b h i j", q, k)
        logit += torch.einsum("b h i d, b h i j d -> b h i j", q, e)

        att = logit * self.scale

        if edge_mask is not None:
            edge_mask = (1.0 - edge_mask) * (-1e9)
            edge_mask = rearrange(edge_mask, "b i j -> b () i j")
            att += edge_mask

        att = att.softmax(dim=-1)  # b h i j
        att = self.att_dropout(att)

        updated = torch.einsum("b h i j, b h j d -> b h i d", att, v)
        updated += torch.einsum("b h i j, b h i j d -> b h i d", att, e)
        updated = rearrange(updated, "b h i d -> b i (h d)")

        return self.node_update(node_feat) + updated


class GraphTransformerBlock(nn.Module):
    def __init__(self, d_node_in, d_node_out, d_edge, n_heads, p_dropout=0.15):
        super().__init__()

        self.attn = GraphTransformer(d_node_in, d_node_out, d_edge, n_heads, p_dropout)
        self.ln = nn.LayerNorm(d_node_out * n_heads)
        self.to_out = nn.Sequential(
            nn.Linear(d_node_out * n_heads, d_node_in), nn.ELU()
        )

    def forward(self, node_feat, edge_feat, edge_mask):
        return (
            self.to_out(self.ln(self.attn(node_feat, edge_feat, edge_mask))) + node_feat
        )


class InitialCoordGenerationWithMsaAndPair(nn.Module):
    def __init__(
        self, d_emb, d_pair, d_node=64, d_edge=64, n_heads=4, n_layers=4, p_dropout=0.1
    ):
        super().__init__()

        self.ln_msa = nn.LayerNorm(d_emb)
        self.ln_pair = nn.LayerNorm(d_pair)
        self.poswise_weight = PositionWiseWeightFactor(d_emb, 1, p_dropout)

        self.node_embed = nn.Sequential(
            nn.Linear(d_emb + 21, d_node),
            nn.ELU(),
        )

        self.edge_embed = nn.Sequential(
            nn.Linear(d_pair + 1, d_edge),
            nn.ELU(),
        )

        self.blocks = [
            GraphTransformerBlock(d_node, d_node, d_edge, n_heads, p_dropout)
            for _ in range(n_layers)
        ]

        self.to_out = nn.Linear(d_node, 9)

    def _sequence_separation_matrix(self, aa_pos):
        """Featurize the distance between amino acid sequence.
        Given two amino acid positions, i and j, the sequence separation feature
        is defined as sign(j - i) * log(|j - i| + 1)

        Args:
            aa_pos, (b l): the position of amino acids in the sequence.

        Returns:
            (b l l 1): sequence separation feature.
        """
        dist = aa_pos.unsqueeze(-1) - aa_pos.unsqueeze(-2)  # All-pairwise diff
        dist = torch.sign(dist) * torch.log(torch.abs(dist) + 1)

        return dist.clamp(0.0, 5.5).unsqueeze(-1)

    def forward(self, msa, pair, seq_onehot, aa_pos):
        """
        Args:
            msa, (b n l d): MSA features.
            pair, (b l l d): Pairwise features.
            seq_onehot, (b l 21): One-hot encoding of the query sequence.
            aa_pos, (b l): the position of amino acids in the sequence.
        """

        msa = self.ln_msa(msa)  # (b N l d)
        pair = self.ln_pair(pair)  # (b l l d)

        # Compute the position-wise weight factor.
        w = self.poswise_weight(msa)
        w = rearrange(w, "b n 1 l 1 -> b n l 1")  # (b N l 1)

        # Compute the node feature.
        node = torch.cat([(msa * w).sum(dim=1), seq_onehot], dim=-1)
        node = self.node_embed(node)

        # Attach sequence separation feature to pair feature.
        edge = torch.cat([pair, self._sequence_separation_matrix(aa_pos)], dim=-1)
        edge = self.edge_embed(edge)

        for block in self.blocks:
            node = block(node, edge, edge_mask=None)  # Fully connected graph

        return rearrange(self.to_out(node), "b l (a xyz) -> b l a xyz", a=3, xyz=3)


class CoordUpdateWithMsaAndPair(nn.Module):
    def __init__(self, d_emb, d_pair, d_node, d_edge, p_dropout=0.1):
        super().__init__()

        self.ln_msa = nn.LayerNorm(d_emb)
        self.ln_pair = nn.LayerNorm(d_pair)
        self.poswise_weight = PositionWiseWeightFactor(d_emb, 1, p_dropout)

        self.node_embed = nn.Sequential(
            nn.Linear(d_emb + 21, d_node),
            nn.ELU(),
        )

        self.edge_embed = nn.Sequential(
            nn.Linear(d_pair + 1, d_edge),
            nn.ELU(),
        )
        # SE(3) equivariant GCN with attention
        # def __init__(self, num_layers=2, num_channels=32, num_degrees=3, n_heads=4, div=4,
        #              si_m='1x1', si_e='att',
        #              l0_in_features=32, l0_out_features=32,
        #              l1_in_features=3, l1_out_features=3,
        #              num_edge_features=32, x_ij=None):

        self.se3_transformer = SE3Transformer()

    def forward(self, xyz, msa, pair, seq_onehot):
        msa = self.ln_msa(msa)  # (b N l d)
        pair = self.ln_pair(pair)  # (b l l d)

        # Compute the position-wise weight factor.
        w = self.poswise_weight(msa)
        w = rearrange(w, "b n 1 l 1 -> b n l 1")  # (b N l 1)

        # Compute the node feature.
        node = torch.cat([(msa * w).sum(dim=1), seq_onehot], dim=-1)
        node = self.node_embed(node)

        # Attach sequence separation feature to pair feature.
        edge = self.edge_embed(pair)


class RoseTTAFold(pl.LightningModule):
    def __init__(
        self,
        d_input=21,
        d_emb=64,
        max_len=5000,
        p_pe_drop=0.1,
    ):
        super().__init__()

        self.msa_emb = MsaEmbedding(
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
    msa_emb = MsaEmbedding()

    print(msa_emb(msa).shape)
