import math

import dgl
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange
from performer_pytorch import SelfAttention as PerformerSelfAttention

from .resnet import ResNet
from .se3_modules import SE3Transformer

N_IDX, CA_IDX, C_IDX = 0, 1, 2


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


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, dim, max_len, p_dropout=0.1):
        super().__init__()
        self.dim = dim
        self.max_len = max_len

        self.pos_enc = torch.zeros(max_len, dim)
        denom = torch.exp(math.log(10000.0) * torch.arange(0, dim, 2) / dim)

        pos = torch.arange(0, max_len).view(-1, 1)
        self.pos_enc[:, 0::2] = torch.sin(pos / denom)
        self.pos_enc[:, 1::2] = torch.cos(pos / denom)

        self.dropout = nn.Dropout(p_dropout)

    def forward(self, x, aa_idx):
        pe = torch.stack([self.pos_enc[idx] for idx in aa_idx])
        pe = rearrange(pe, "b l d -> b () l d")

        return self.dropout(x + pe)


class SinusoidalPositionalEncoding2D(nn.Module):
    def __init__(self, dim, max_len, p_dropout=0.1):
        super().__init__()

        dim_half = dim // 2
        self.max_len = max_len

        self.pos_enc = torch.zeros(max_len, dim_half)
        denom = torch.exp(math.log(10000.0) * torch.arange(0, dim_half, 2) / dim_half)

        pos = torch.arange(0, max_len).view(-1, 1)
        self.pos_enc[:, 0::2] = torch.sin(pos / denom)
        self.pos_enc[:, 1::2] = torch.cos(pos / denom)

        self.dropout = nn.Dropout(p_dropout)

    def forward(self, x, aa_idx):
        L = aa_idx.size(1)
        # aa_idx : bsz x L
        pe_half = torch.stack([self.pos_enc[idx] for idx in aa_idx])  # bsz x L x dim_half

        pe_rowwise = repeat(pe_half, "b l d -> b l k d", k=L)
        pe_colwise = repeat(pe_half, "b l d -> b k l d", k=L)

        return x + torch.cat([pe_rowwise, pe_colwise], dim=-1)


class MsaEmbedding(nn.Module):
    def __init__(self, d_input=21, d_msa=384, max_len=260, p_pe_drop=0.1):
        super().__init__()
        self.to_embedding = nn.Embedding(d_input, d_msa)

        self.pos_enc = SinusoidalPositionalEncoding(d_msa, max_len, p_pe_drop)
        self.query_enc = nn.Embedding(2, d_msa)  # 0: query, 1: targets

    def forward(self, x, aa_idx):
        query_idx = torch.ones(x.size(-2), 1).long()
        query_idx[0] = 0

        # x : B, N, L
        x = self.pos_enc(self.to_embedding(x), aa_idx) + self.query_enc(query_idx)
        return x  # B, N, L, d_msa


class PairEmbedding(nn.Module):
    def __init__(
        self,
        d_input=21,
        d_pair=288,
        max_len=260,
        p_pe_drop=0.1,
        use_template=False,
        d_template=64,
    ):
        super().__init__()
        self.half_d_pair = d_pair // 2

        self.embed_seq = nn.Embedding(d_input, self.half_d_pair)

        self.pos_enc = SinusoidalPositionalEncoding2D(d_pair, max_len, p_pe_drop)

        self.use_template = use_template
        if self.use_template:
            self.ln_template = nn.LayerNorm(d_template)
            self.proj = nn.Linear(d_pair + d_template + 1, d_pair)
        else:
            self.proj = nn.Linear(d_pair + 1, d_pair)

    def forward(self, seq, aa_idx, template=None):
        if not self.use_template and template is not None:
            raise ValueError(
                f"[{self.__class__.__name__}]: template is not None but use_template is False"
            )

        L = seq.size(-1)

        seq_emb = self.embed_seq(seq)  # B, L, d_pair/2
        left_seq_emb = repeat(seq_emb, "b l d -> b k l d", k=L)
        right_seq_emb = repeat(seq_emb, "b l d -> b l k d", k=L)
        seq_sep = self._sequence_separation_matrix(aa_idx)

        if self.use_template:
            x = torch.cat(
                [
                    left_seq_emb,
                    right_seq_emb,
                    seq_sep,
                    self.ln_template(template),
                ],
                dim=-1,
            )
        else:
            x = torch.cat([left_seq_emb, right_seq_emb, seq_sep], dim=-1)  # B, L, L, d_pair+1

        x = self.proj(x)  # B, L, L, d_pair

        return self.pos_enc(x, aa_idx)

    def _sequence_separation_matrix(self, aa_idx):
        dist = aa_idx.unsqueeze(-1) - aa_idx.unsqueeze(-2)  # All-pairwise diff
        dist = torch.log(torch.abs(dist) + 1)

        return rearrange(dist, "b i j -> b i j ()")


class PositionWiseWeightFactor(nn.Module):
    def __init__(self, d_msa=384, n_heads=12, p_dropout=0.1):
        super().__init__()

        assert (
            d_msa % n_heads == 0
        ), f"[{self.__class__.__name__}]: d_msa ({d_msa}) must be divisible by n_heads ({n_heads})."

        self.d_head = d_msa // n_heads
        self.scale = self.d_head ** (-0.5)

        self.to_q = nn.Sequential(
            nn.Linear(d_msa, d_msa),  # IDEA: maybe we can use LinearNoBias here.
            Rearrange("b 1 l (h d) -> b l h 1 d", h=n_heads),
        )
        self.to_k = nn.Sequential(
            nn.Linear(d_msa, d_msa),  # IDEA: maybe we can use LinearNoBias here.
            Rearrange("b N l (h d) -> b l h N d", h=n_heads),
        )
        self.dropout = nn.Dropout(p_dropout)

    def forward(self, msa_emb):
        """msa : (B, N, L, d_msa)"""
        query_seq = msa_emb[:, 0].unsqueeze(1)  # Take the first sequence as query.

        q = self.to_q(query_seq) * self.scale
        k = self.to_k(msa_emb)

        logits = torch.einsum("b l h q d, b l h n d -> b l h q n", q, k)
        # IDEA: maybe we can use dropout here at logits to make weights sum to 1.

        att = logits.softmax(dim=-1)
        att = rearrange(att, "b l h 1 N -> b N h l 1")
        return self.dropout(att)


class SoftTiedAttentionOverResidues(nn.Module):
    def __init__(self, d_msa=384, n_heads=12, p_dropout=0.1, return_att=False):
        super().__init__()
        assert (
            d_msa % n_heads == 0
        ), f"[{self.__class__.__name__}]: d_msa ({d_msa}) must be divisible by n_heads ({n_heads})."

        self.n_heads = n_heads
        self.d_head = d_msa // n_heads
        self.scale = self.d_head ** (-0.5)
        self.return_att = return_att

        self.poswise_weight = PositionWiseWeightFactor(d_msa, n_heads, p_dropout)

        # IDEA: maybe we can use LinearNoBias for the projections below.
        self.to_q = nn.Linear(d_msa, d_msa)
        self.to_k = nn.Linear(d_msa, d_msa)
        self.to_v = nn.Linear(d_msa, d_msa)
        self.to_out = nn.Linear(d_msa, d_msa)
        self.dropout = nn.Dropout(p_dropout)

    def forward(self, x):
        """x : (B, N, L, d_msa)"""
        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)

        q, k, v = map(
            lambda t: Rearrange("b n l (h d) -> b n h l d", h=self.n_heads)(t),
            (q, k, v),
        )
        # poswise_weight : (b n h l 1)
        q = q * self.poswise_weight(x) * self.scale

        logits = torch.einsum("b n h i d, b n h j d -> b h i j", q, k)
        att = logits.softmax(dim=-1)

        out = torch.einsum("b h i j, b n h j d -> b n h i d", att, v)
        out = rearrange(out, "b n h l d -> b n l (h d)")
        out = self.to_out(out)

        if self.return_att:
            # symmetrize attention
            att = (att + rearrange(att, "b h i j -> b h j i")) * 0.5
            att = rearrange(att, "b h i j -> b i j h")
            return self.dropout(out), att
        else:
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
        d_msa=384,
        d_ff=384 * 4,
        n_heads=12,
        p_dropout=0.1,
        tied=False,
        performer=False,
        performer_kws={},
        return_att=False,
    ):
        super().__init__()
        self.tied = tied
        self.return_att = return_att

        # define attention operation
        if self.tied:
            self.attn = SoftTiedAttentionOverResidues(
                d_msa=d_msa,
                n_heads=n_heads,
                p_dropout=p_dropout,
                return_att=return_att,
            )
        elif performer:
            if return_att:
                raise NotImplementedError(
                    "PerformerSelfAttention does not support return_att."
                )
            self.attn = PerformerSelfAttention(
                dim=d_msa,
                heads=n_heads,
                dropout=p_dropout,
                **performer_kws,
            )
        else:
            raise NotImplementedError

        # define layers
        self.ln = nn.LayerNorm(d_msa)
        self.dropout = nn.Dropout(p_dropout)

        self.ff = Residual(
            nn.Sequential(
                nn.LayerNorm(d_msa),
                FeedForward(d_msa, d_ff, p_dropout=p_dropout),
                nn.Dropout(p_dropout),
            )
        )

    def forward(self, x):
        N = x.size(1)  # Number of sequences

        if not self.tied:
            x = rearrange(x, "b n l d -> (b n) l d")

        orig = x
        x = self.ln(x)
        if self.return_att:
            x, att = self.attn(x)
        else:
            x = self.attn(x)
        x = orig + self.dropout(x)

        if not self.tied:
            x = rearrange(x, "(b n) l d -> b n l d", n=N)

        if self.return_att:
            return self.ff(x), att
        else:
            return self.ff(x)


class MsaUpdateUsingSelfAttention(nn.Module):
    def __init__(
        self,
        d_msa=384,
        d_ff=384 * 4,
        n_heads=12,
        p_dropout=0.1,
        n_encoder_layers=4,
        performer_kws={},
    ):
        super().__init__()

        self.residue_wise_encoder_layers = nn.ModuleList(
            [
                EncoderLayer(
                    d_msa=d_msa,
                    d_ff=d_ff,
                    n_heads=n_heads,
                    p_dropout=p_dropout,
                    tied=True,
                    performer=False,
                    return_att=True,
                )
                for _ in range(n_encoder_layers)
            ]
        )

        self.sequence_wise_encoder_layers = nn.ModuleList(
            [
                EncoderLayer(
                    d_msa=d_msa,
                    d_ff=d_ff,
                    n_heads=n_heads,
                    p_dropout=p_dropout,
                    tied=False,
                    performer=True,
                    performer_kws=performer_kws,
                )
                for _ in range(n_encoder_layers)
            ]
        )

    def forward(self, x):
        for layer in self.residue_wise_encoder_layers:
            x, att = layer(x)

        x = rearrange(x, "b n l d -> b l n d")

        for layer in self.sequence_wise_encoder_layers:
            x = layer(x)

        x = rearrange(x, "b l n d -> b n l d")
        return x, att


class OuterProductMean(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.to_out = nn.Sequential(
            nn.LayerNorm(in_features**2), nn.Linear(in_features**2, out_features)
        )

    def forward(self, x, y=None):
        y = x if y is None else y

        # here we use outer product sum instead of mean,
        # since we assume that y is already weighted by attention
        x = torch.einsum("b n i u, b n j v -> b i j u v", x, y)
        x = rearrange(x, "b i j u v -> b i j (u v)")

        return self.to_out(x)


class PairUpdateWithMsa(nn.Module):
    def __init__(self, d_msa, d_proj, d_pair, n_heads, p_dropout=0.1):
        super().__init__()

        self.proj_msa = nn.Sequential(
            nn.LayerNorm(d_msa),
            nn.Linear(d_msa, d_proj),
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
                    nn.Conv2d(d_pair, d_pair, kernel_size=3, padding="same", bias=False),
                    nn.InstanceNorm2d(d_pair, affine=True, eps=1e-6),
                    nn.ELU(),
                    nn.Dropout(p_dropout),
                    nn.Conv2d(d_pair, d_pair, kernel_size=3, padding="same", bias=False),
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


class PairUpdateWithAxialAttentionLayer(nn.Module):
    def __init__(self, d_pair, d_ff, n_heads, p_dropout, performer_kws):
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


class PairUpdateWithAxialAttention(nn.Module):
    def __init__(self, d_pair, d_ff, n_heads, p_dropout, n_encoder_layers, performer_kws={}):
        super().__init__()

        self.layers = nn.ModuleList(
            [
                PairUpdateWithAxialAttentionLayer(
                    d_pair, d_ff, n_heads, p_dropout, performer_kws
                )
                for _ in range(n_encoder_layers)
            ]
        )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class Symmetrization(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        xt = rearrange(x, "b i j d -> b j i d")
        return 0.5 * (x + xt)


class MsaUpdateWithPairLayer(nn.Module):
    def __init__(self, d_msa, d_pair, n_heads, p_dropout=0.1):
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
            nn.LayerNorm(d_msa),
            nn.Linear(d_msa, d_msa),
            Rearrange("b n j (h d) -> b n h j d", h=n_heads),
        )

        self.ff = Residual(
            nn.Sequential(
                nn.LayerNorm(d_msa),
                FeedForward(d_msa, d_msa, p_dropout),
            ),
            p_dropout=p_dropout,
        )

        self.dropout = nn.Dropout(p_dropout)

    def forward(self, msa, pair):
        value = self.msa2value(msa)
        att = self.pair2att(pair)

        updated = self.dropout(torch.einsum("b ... h i j, b n h j d -> b n h i d", att, value))
        updated = rearrange(updated, "b n h i d -> b n i (h d)")

        return self.ff(msa + updated)


class MsaUpdateWithPair(nn.Module):
    def __init__(self, d_msa, d_pair, n_heads, n_encoder_layers=4, p_dropout=0.1):
        super().__init__()

        self.encoder_layers = [
            MsaUpdateWithPairLayer(d_msa, d_pair, n_heads, p_dropout)
            for _ in range(n_encoder_layers)
        ]

    def forward(self, msa, pair):
        for layer in self.encoder_layers:
            msa = layer(msa, pair)
        return msa


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
        self.to_out = nn.Sequential(nn.Linear(d_node_out * n_heads, d_node_in), nn.ELU())

    def forward(self, node_feat, edge_feat, edge_mask):
        return self.to_out(self.ln(self.attn(node_feat, edge_feat, edge_mask))) + node_feat


class InitialCoordGenerationWithMsaAndPair(nn.Module):
    def __init__(
        self, d_msa, d_pair, d_node=64, d_edge=64, n_heads=4, n_layers=4, p_dropout=0.1
    ):
        super().__init__()

        self.ln_msa = nn.LayerNorm(d_msa)
        self.ln_pair = nn.LayerNorm(d_pair)
        self.poswise_weight = PositionWiseWeightFactor(d_msa, 1, p_dropout)

        self.node_embed = nn.Sequential(
            nn.Linear(d_msa + 21, d_node),
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

    def forward(self, msa, pair, seq_onehot, aa_idx):
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
        edge = torch.cat([pair, self._sequence_separation_matrix(aa_idx)], dim=-1)
        edge = self.edge_embed(edge)

        for block in self.blocks:
            node = block(node, edge, edge_mask=None)  # Fully connected graph

        return rearrange(self.to_out(node), "b l (a xyz) -> b l a xyz", a=3, xyz=3)

    def _sequence_separation_matrix(self, aa_idx):
        """Featurize the distance between amino acid sequence.
        Given two amino acid positions, i and j, the sequence separation feature
        is defined as sign(j - i) * log(|j - i| + 1)

        Args:
            aa_pos, (b l): the position of amino acids in the sequence.

        Returns:
            (b l l 1): sequence separation feature.
        """
        dist = aa_idx.unsqueeze(-1) - aa_idx.unsqueeze(-2)  # All-pairwise diff
        dist = torch.sign(dist) * torch.log(torch.abs(dist) + 1)

        return dist.clamp(0.0, 5.5).unsqueeze(-1)


class CoordUpdateWithMsaAndPair(nn.Module):
    def __init__(self, d_msa, d_pair, d_node, d_edge, d_state, n_neighbors, p_dropout=0.1):
        super().__init__()

        self.n_neighbors = n_neighbors

        self.ln_msa = nn.LayerNorm(d_msa)
        self.ln_pair = nn.LayerNorm(d_pair)
        self.poswise_weight = PositionWiseWeightFactor(d_msa, 1, p_dropout)

        self.node_embed = nn.Sequential(
            nn.Linear(d_msa + 21, d_node),
            nn.ELU(),
            nn.LayerNorm(d_node),
        )

        self.edge_embed = nn.Sequential(
            nn.Linear(d_pair, d_edge),
            nn.ELU(),
            nn.LayerNorm(d_edge),
        )

        self.se3_transformer = SE3Transformer(
            num_layers=2,
            num_channels=16,
            n_heads=4,
            num_degrees=2,
            l0_in_features=d_node,
            l1_in_features=3,
            l0_out_features=d_state,
            l1_out_features=3,
            num_edge_features=d_edge,
        )

    def forward(self, xyz, msa, pair, aa_idx, seq_onehot):
        bsz = xyz.size(0)

        msa = self.ln_msa(msa)  # (b N l d)
        pair = self.ln_pair(pair)  # (b l l d)

        # Compute the position-wise weight factor.
        w = self.poswise_weight(msa)
        w = rearrange(w, "b n 1 l 1 -> b n l 1")  # (b N l 1)

        # Compute the node feature.
        node = torch.cat([(msa * w).sum(dim=1), seq_onehot], dim=-1)
        node = self.node_embed(node)  # (b l d_node)

        # Attach sequence separation feature to pair feature.
        edge = self.edge_embed(pair)  # (b l d_edge)

        G = self._knn_graph(xyz, edge, aa_idx, n_neighbors=self.n_neighbors)

        type0_feat = rearrange(node, "b l d -> (b l) d ()")

        type1_feat = xyz - xyz[:, :, CA_IDX].unsqueeze(-2)
        type1_feat = rearrange(type1_feat, "b l a xyz -> (b l) a xyz")

        out = self.se3_transformer(G, type0_feat, type1_feat)
        state, displacement = out["0"], out["1"]

        state = rearrange(state, "(b l) d () -> b l d", b=bsz)
        displacement = rearrange(displacement, "(b l) a xyz -> b l a xyz", b=bsz)

        ca_xyz = xyz[:, :, CA_IDX] + displacement[:, :, CA_IDX]
        n_xyz = ca_xyz + displacement[:, :, N_IDX]
        c_xyz = ca_xyz + displacement[:, :, C_IDX]
        xyz = torch.stack([n_xyz, ca_xyz, c_xyz], dim=2)

        return state, xyz

    def _knn_graph(self, xyz, edge, idx, n_neighbors=64, kmin=9):
        """
        Adopted from
        https://github.com/RosettaCommons/RoseTTAFold/blob/main/network/Attention_module_w_str.py#L19
        """
        B, L = xyz.shape[:2]
        device = xyz.device

        # pairwise distance between CA atoms
        # NOTE: self-interactions are excluded
        pdist = torch.cdist(xyz[:, :, CA_IDX], xyz[:, :, CA_IDX])
        pdist += torch.eye(L, device=device).unsqueeze(0) * 1e3

        # 1D-distance between residues in amino acid sequence
        sep = idx[:, None, :] - idx[:, :, None]
        sep = sep.abs() + torch.eye(L, device=device).unsqueeze(0) * 999.9

        # get top_k neighbors
        n_neighbors = min(n_neighbors, L)

        # E_idx: (B, L, n_neighbors)
        _, neighbor_idx = torch.topk(pdist, n_neighbors, largest=False)

        adj = torch.zeros((B, L, L), device=device)
        adj = adj.scatter(dim=2, index=neighbor_idx, value=1.0)

        # put an edge if any of the 3 conditions are met:
        #   1) |i-j| <= kmin (connect sequentially adjacent residues)
        #   2) top_k neighbors
        cond = torch.logical_or(adj > 0.0, sep < kmin)
        b, i, j = torch.where(cond)

        edge_idx = (b * L + i, b * L + j)
        G = dgl.graph(edge_idx, num_nodes=B * L).to(device)

        # no gradient through basis function
        G.edata["d"] = (xyz[b, j, CA_IDX] - xyz[b, i, CA_IDX]).detach()
        G.edata["w"] = edge[b, i, j]

        return G


class MsaUpdateWithPairAndCoord(nn.Module):
    def __init__(
        self, d_msa, d_state, d_trfm_inner, d_ff, distance_bins=[8, 12, 16, 20], p_dropout=0.1
    ):
        super().__init__()

        self.distance_bins = distance_bins
        self.n_heads = len(self.distance_bins)

        self.scale = (d_state // self.n_heads) ** -0.5

        self.ln_msa = nn.LayerNorm(d_msa)
        self.ln_state = nn.LayerNorm(d_state)

        self.to_q = nn.Linear(d_state, d_trfm_inner * self.n_heads)
        self.to_k = nn.Linear(d_state, d_trfm_inner * self.n_heads)
        self.to_v = nn.Linear(d_msa, d_msa)

        self.ln_out = nn.LayerNorm(d_msa)
        self.to_out = Residual(
            nn.Sequential(
                nn.LayerNorm(d_msa),
                FeedForward(d_msa, d_ff, p_dropout),
            )
        )

    def forward(self, xyz, state, msa):
        state = self.ln_state(state)
        msa = self.ln_msa(msa)

        q = self.to_q(state)
        k = self.to_k(state)
        v = self.to_v(msa)

        # compose attention mask according to the Euclidean distance
        # between atoms
        pdist = torch.cdist(xyz[:, :, CA_IDX], xyz[:, :, CA_IDX])
        att_mask = torch.cat(
            [(pdist < dist_thresh).unsqueeze(1).float() for dist_thresh in self.distance_bins],
            dim=1,
        )

        q = rearrange(q, "b l (h d) -> b h l d", h=self.n_heads)
        k = rearrange(k, "b l (h d) -> b h l d", h=self.n_heads)
        v = rearrange(v, "b n l (h d) -> b h n l d", h=self.n_heads)

        q = q * self.scale

        logits = torch.einsum("b h i d, b h j d -> b h i j", q, k) + (1.0 - att_mask) * -1e9
        att = logits.softmax(dim=-1)

        out = torch.einsum("b h i j, b h n j d -> b h n i d", att, v)
        out = rearrange(out, "b h n l d -> b n l (h d)")
        msa = msa + self.ln_out(out)

        return self.to_out(msa)


class TwoTrackBlock(nn.Module):
    def __init__(self, d_msa, d_pair, n_encoder_layers, p_dropout=0.1):
        """
        n_encoder_layers: number of encoder layers for modules using transformer encoders at their core.
        """
        super().__init__()

        self.msa_update_using_self_att = MsaUpdateUsingSelfAttention(
            d_msa=d_msa,
            d_ff=d_msa * 4,
            n_heads=12,
            n_encoder_layers=n_encoder_layers,
            p_dropout=p_dropout,
        )

        self.pair_update_with_msa = PairUpdateWithMsa(
            d_pair=d_pair,
            n_heads=12,
            d_msa=d_msa,
            d_proj=32,
        )

        self.pair_update_with_axial_attention = PairUpdateWithAxialAttention(
            d_pair=d_pair,
            d_ff=d_pair * 4,
            n_heads=8,
            p_dropout=p_dropout,
            n_encoder_layers=n_encoder_layers,
            performer_kws={},
        )

        self.msa_update_with_pair = MsaUpdateWithPair(
            d_msa=d_msa,
            d_pair=d_pair,
            n_heads=4,
            n_encoder_layers=n_encoder_layers,
            p_dropout=p_dropout,
        )

    def forward(self, msa, pair):
        msa, att = self.msa_update_using_self_att(msa)
        pair = self.pair_update_with_msa(msa, pair, att)
        pair = self.pair_update_with_axial_attention(pair)
        msa = self.msa_update_with_pair(msa, pair)

        return msa, pair


class ThreeTrackBlock(nn.Module):
    def __init__(
        self, d_msa, d_pair, d_node, d_edge, d_state, n_encoder_layers, n_neighbors, p_dropout
    ):
        """
        n_encoder_layers: number of encoder layers for modules using transformer encoders at their core.
        """
        super().__init__()

        # d_msa = 384
        # d_pair = 288
        # d_node = 64
        # d_edge = 64
        # d_state = 32

        self.msa_update_using_self_att = MsaUpdateUsingSelfAttention(
            d_msa=d_msa,
            d_ff=d_msa * 4,
            n_heads=12,
            n_encoder_layers=n_encoder_layers,
            p_dropout=p_dropout,
        )

        self.pair_update_with_msa = PairUpdateWithMsa(
            d_pair=d_pair,
            n_heads=12,
            d_msa=d_msa,
            d_proj=32,
        )

        self.pair_update_with_axial_attention = PairUpdateWithAxialAttention(
            d_pair=d_pair,
            d_ff=d_pair * 4,
            n_heads=8,
            p_dropout=p_dropout,
            n_encoder_layers=n_encoder_layers,
            performer_kws={},
        )

        self.msa_update_with_pair = MsaUpdateWithPair(
            d_msa=d_msa,
            d_pair=d_pair,
            n_heads=4,
            n_encoder_layers=n_encoder_layers,
            p_dropout=0.1,
        )

        self.coord_update_with_msa_and_pair = CoordUpdateWithMsaAndPair(
            d_msa=d_msa,
            d_pair=d_pair,
            d_node=d_node,
            d_edge=d_edge,
            d_state=d_state,
            n_neighbors=n_neighbors,
            p_dropout=p_dropout,
        )

        self.msa_update_with_pair_and_coord = MsaUpdateWithPairAndCoord(
            d_msa=d_msa,
            d_state=d_state,
            d_trfm_inner=32,
            d_ff=d_msa * 4,
            distance_bins=[8, 12, 16, 20],
            p_dropout=p_dropout,
        )

    def forward(self, msa, pair, xyz, seq_onehot, aa_idx):
        msa, att = self.msa_update_using_self_att(msa)
        pair = self.pair_update_with_msa(msa, pair, att)
        pair = self.pair_update_with_axial_attention(pair)
        msa = self.msa_update_with_pair(msa, pair)

        state, xyz = self.coord_update_with_msa_and_pair(xyz, msa, pair, aa_idx, seq_onehot)
        msa = self.msa_update_with_pair_and_coord(xyz, state, msa)

        return msa, pair, xyz


class FinalBlock(nn.Module):
    def __init__(
        self,
        d_msa,
        d_pair,
        d_node,
        d_edge,
        d_state,
        n_encoder_layers,
        p_dropout,
        n_neighbors=32,
    ):
        """
        n_encoder_layers: number of encoder layers for modules using transformer encoders at their core.
        """
        super().__init__()

        # d_msa = 384
        # d_pair = 288
        # d_node = 64
        # d_edge = 64
        # d_state = 32

        self.msa_update_using_self_att = MsaUpdateUsingSelfAttention(
            d_msa=d_msa,
            d_ff=d_msa * 4,
            n_heads=12,
            n_encoder_layers=n_encoder_layers,
            p_dropout=p_dropout,
        )

        self.pair_update_with_msa = PairUpdateWithMsa(
            d_pair=d_pair,
            n_heads=12,
            d_msa=d_msa,
            d_proj=32,
        )

        self.pair_update_with_axial_attention = PairUpdateWithAxialAttention(
            d_pair=d_pair,
            d_ff=d_pair * 4,
            n_heads=8,
            p_dropout=p_dropout,
            n_encoder_layers=n_encoder_layers,
            performer_kws={},
        )

        self.msa_update_with_pair = MsaUpdateWithPair(
            d_msa=d_msa,
            d_pair=d_pair,
            n_heads=4,
            n_encoder_layers=n_encoder_layers,
            p_dropout=0.1,
        )

        self.coord_update_with_msa_and_pair = CoordUpdateWithMsaAndPair(
            d_msa=d_msa,
            d_pair=d_pair,
            d_node=d_node,
            d_edge=d_edge,
            d_state=d_state,
            n_neighbors=n_neighbors,
            p_dropout=p_dropout,
        )

        self.plddt_head = nn.Linear(d_state, 1)

    def forward(self, msa, pair, xyz, seq_onehot, aa_idx):
        msa, att = self.msa_update_using_self_att(msa)
        pair = self.pair_update_with_msa(msa, pair, att)
        pair = self.pair_update_with_axial_attention(pair)
        msa = self.msa_update_with_pair(msa, pair)

        state, xyz = self.coord_update_with_msa_and_pair(xyz, msa, pair, aa_idx, seq_onehot)

        plddt = self.plddt_head(state)
        plddt = rearrange(plddt, "b l () -> b l")

        return msa, pair, xyz, plddt


class PredictionHead(nn.Module):
    def __init__(self, in_channels, n_res_blocks, p_dropout):
        super().__init__()

        intermediate_channels = in_channels
        self.proj = nn.Sequential(
            nn.LayerNorm(in_channels),
            nn.Linear(in_channels, intermediate_channels),
            nn.Dropout(p_dropout),
            Rearrange("b i j c -> b c i j"),
        )

        self.dist_head = nn.Sequential(
            ResNet(n_res_blocks, in_channels, intermediate_channels, 37, p_dropout=p_dropout),
            Rearrange("b c i j -> b i j c"),
        )
        self.omega_head = nn.Sequential(
            ResNet(n_res_blocks, in_channels, intermediate_channels, 37, p_dropout=p_dropout),
            Rearrange("b c i j -> b i j c"),
        )
        self.theta_head = nn.Sequential(
            ResNet(n_res_blocks, in_channels, intermediate_channels, 37, p_dropout=p_dropout),
            Rearrange("b c i j -> b i j c"),
        )
        self.phi_head = nn.Sequential(
            ResNet(n_res_blocks, in_channels, intermediate_channels, 19, p_dropout=p_dropout),
            Rearrange("b c i j -> b i j c"),
        )

    def forward(self, pair):
        pair = self.proj(pair)

        logits = {}
        # theta, phi
        logits["theta"] = self.theta_head(pair)
        logits["phi"] = self.phi_head(pair)

        # dist, omega should be predicted with symmetrized pair embedding
        pair_symmetrized = (pair + rearrange(pair, "b c i j -> b c j i")) * 0.5
        logits["dist"] = self.dist_head(pair_symmetrized)
        logits["omega"] = self.omega_head(pair_symmetrized)

        return logits


class RoseTTAFold(pl.LightningModule):
    def __init__(
        self,
        d_input=21,
        d_msa=384,
        d_pair=288,
        d_node=64,
        d_edge=64,
        d_state=32,
        n_two_track_blocks=3,
        n_three_track_blocks=4,
        n_encoder_layers=4,
        max_len=5000,
        n_neighbors=[128, 128, 64, 64, 64],
        p_dropout=0.1,
        use_template=False,
    ):
        super().__init__()

        # parameters
        self.d_msa = d_msa
        self.d_pair = d_pair
        self.d_node = d_node
        self.d_edge = d_edge
        self.d_state = d_state
        self.n_two_track_blocks = n_two_track_blocks
        self.n_three_track_blocks = n_three_track_blocks
        self.n_encoder_layers = n_encoder_layers
        self.use_template = use_template

        self.msa_emb = MsaEmbedding(
            d_input=d_input,
            d_msa=d_msa,
            max_len=max_len,
            p_pe_drop=p_dropout,
        )

        self.pair_emb = PairEmbedding(
            d_input=d_input,
            d_pair=d_pair,
            max_len=max_len,
            use_template=use_template,
            p_pe_drop=p_dropout,
        )

        self.two_track_blocks = nn.ModuleList(
            [
                TwoTrackBlock(
                    d_msa,
                    d_pair,
                    n_encoder_layers=n_encoder_layers,
                    p_dropout=p_dropout,
                )
                for _ in range(n_two_track_blocks)
            ]
        )

        self.initial_coord_generation_with_msa_and_pair = InitialCoordGenerationWithMsaAndPair(
            d_msa=d_msa,
            d_pair=d_pair,
            d_node=d_node,
            d_edge=d_edge,
            n_heads=4,
            n_layers=4,
            p_dropout=p_dropout,
        )

        self.three_track_blocks = nn.ModuleList(
            [
                ThreeTrackBlock(
                    d_msa,
                    d_pair,
                    d_node,
                    d_edge,
                    d_state,
                    n_encoder_layers=n_encoder_layers,
                    n_neighbors=n_neighbors[i],
                    p_dropout=p_dropout,
                )
                for i in range(n_three_track_blocks - 1)
            ]
        )

        self.final_block = FinalBlock(
            d_msa,
            d_pair,
            d_node,
            d_edge,
            d_state,
            n_encoder_layers=n_encoder_layers,
            n_neighbors=32,  # fixed
            p_dropout=p_dropout,
        )

        self.prediction_head = PredictionHead(
            in_channels=d_pair, n_res_blocks=4, p_dropout=p_dropout
        )

    def forward(self, msa, seq, aa_idx):
        msa = self.msa_emb(msa, aa_idx)
        pair = self.pair_emb(seq, aa_idx)
        seq_onehot = F.one_hot(seq, num_classes=21).float()

        for two_track_block in self.two_track_blocks:
            msa, pair = two_track_block(msa, pair)

        xyz = self.initial_coord_generation_with_msa_and_pair(msa, pair, seq_onehot, aa_idx)

        for three_track_block in self.three_track_blocks:
            msa, pair, xyz = three_track_block(msa, pair, xyz, seq_onehot, aa_idx)

        msa, pair, xyz, plddt = self.final_block(msa, pair, xyz, seq_onehot, aa_idx)
        logits = self.prediction_head(pair)

        return logits, xyz, plddt

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
