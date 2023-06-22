import warnings

import pytest
import torch
import torch.nn.functional as F

warnings.filterwarnings("ignore")

from rosettafold_pytorch.rosettafold_pytorch import (
    EncoderLayer,
    MsaEmbedding,
    MsaUpdateUsingSelfAttention,
    OuterProductMean,
    PairUpdateWithAxialAttention,
    PairUpdateWithMsa,
    PositionWiseWeightFactor,
    SoftTiedAttentionOverResidues,
    Symmetrization,
    MsaUpdateWithPair,
    GraphTransformer,
    InitialCoordGenerationWithMsaAndPair,
    CoordUpdateWithMsaAndPair,
    MsaUpdateWithPairAndCoord,
    TwoTrackBlock,
    ThreeTrackBlock,
    RoseTTAFold,
)


# MSAEmbedding
def test_MsaEmbedding_init():
    msa_emb = MsaEmbedding(d_input=21, d_emb=64, max_len=5000, p_pe_drop=0.1)
    assert msa_emb is not None


def test_MsaEmbedding_positional_encoding_is_sinusoidal():
    msa_emb = MsaEmbedding()

    s = msa_emb.pos_enc[:, 0::2].square() + msa_emb.pos_enc[:, 1::2].square()
    assert torch.isclose(s, torch.tensor([1.0])).all()


def test_MsaEmbedding_shape():
    bsz, n_seq, max_len = 4, 10, 5000

    msa = torch.randint(0, 21, (bsz, n_seq, max_len))
    msa_emb = MsaEmbedding(d_input=21, d_emb=64, max_len=5000, p_pe_drop=0.1)

    assert msa_emb(msa).shape == (bsz, n_seq, max_len, 64)


# PositionWiseWeightFactor
def test_PositionWiseWeightFactor_init():
    d_emb, n_heads = 64, 4

    pos_wise_weight_factor = PositionWiseWeightFactor(
        d_emb=d_emb, n_heads=n_heads, p_dropout=0.1
    )
    assert pos_wise_weight_factor is not None


def test_PositionWiseWeightFactor_init_errors_when_d_emb_is_not_divisible_by_n_heads():
    d_emb, n_heads = 64, 3

    with pytest.raises(AssertionError):
        PositionWiseWeightFactor(d_emb=d_emb, n_heads=n_heads, p_dropout=0.1)


def test_PositionWiseWeightFactor_shape():
    bsz, n_seq, max_len = 4, 10, 5000
    d_emb, n_heads, max_len = 64, 4, 32

    msa = torch.randint(0, 21, (bsz, n_seq, max_len))
    msa_embedder = MsaEmbedding(d_input=21, d_emb=d_emb, max_len=max_len, p_pe_drop=0.1)
    msa_emb = msa_embedder(msa)

    pos_wise_weight_factor = PositionWiseWeightFactor(
        d_emb=d_emb, n_heads=n_heads, p_dropout=0.1
    )

    assert pos_wise_weight_factor(msa_emb).shape == (bsz, n_seq, n_heads, max_len, 1)


def test_PositionWiseWeightFactor_sums_to_1():
    bsz, n_seq, max_len = 4, 10, 5000
    d_emb, n_heads, max_len = 64, 4, 32

    msa = torch.randint(0, 21, (bsz, n_seq, max_len))
    msa_embedder = MsaEmbedding(d_input=21, d_emb=d_emb, max_len=max_len, p_pe_drop=0.1)
    msa_emb = msa_embedder(msa)

    pos_wise_weight_factor = PositionWiseWeightFactor(
        d_emb=d_emb,
        n_heads=n_heads,
        p_dropout=0.0,  # Make sure that no dropout is applied in this test
    )

    assert (
        pos_wise_weight_factor(msa_emb)
        .sum(dim=1)
        .squeeze()
        .allclose(torch.ones((bsz, n_heads, max_len)))
    )


# SoftTiedAttentionOverResidues
def test_SoftTiedAttentionOverResidues_init():
    bsz, n_seq, max_len = 4, 10, 5000
    d_emb, n_heads, max_len = 64, 4, 32

    msa = torch.randint(0, 21, (bsz, n_seq, max_len))
    msa_embedder = MsaEmbedding(d_input=21, d_emb=d_emb, max_len=max_len, p_pe_drop=0.1)

    att = SoftTiedAttentionOverResidues(
        d_emb=d_emb,
        n_heads=n_heads,
        p_dropout=0.0,  # Make sure that no dropout is applied in this test
    )

    assert att is not None


def test_SoftTiedAttentionOverResidues_shape():
    bsz, n_seq, max_len = 4, 10, 5000
    d_emb, n_heads, max_len = 64, 4, 32

    msa = torch.randint(0, 21, (bsz, n_seq, max_len))
    msa_embedder = MsaEmbedding(d_input=21, d_emb=d_emb, max_len=max_len, p_pe_drop=0.1)
    msa_emb = msa_embedder(msa)

    att = SoftTiedAttentionOverResidues(
        d_emb=d_emb,
        n_heads=n_heads,
        p_dropout=0.0,  # Make sure that no dropout is applied in this test
    )

    assert att(msa_emb).shape == (bsz, n_seq, max_len, d_emb)


# EncoderLayer
def test_EncoderLayer_tied_init():
    d_emb, n_heads = 64, 4
    d_ff = d_emb * 4

    enc = EncoderLayer(d_emb=d_emb, d_ff=d_ff, n_heads=n_heads, p_dropout=0.1, tied=True)
    assert enc is not None


def test_EncoderLayer_performer_init():
    d_emb, n_heads = 64, 4
    d_ff = d_emb * 4

    enc = EncoderLayer(d_emb=d_emb, d_ff=d_ff, n_heads=n_heads, p_dropout=0.1, performer=True)
    assert enc is not None


def test_EncoderLayer_tied_shape():
    bsz, n_seq, max_len = 4, 10, 64
    d_emb, n_heads = 16, 2
    d_ff = d_emb * 4

    msa = torch.randint(0, 21, (bsz, n_seq, max_len))
    msa_embedder = MsaEmbedding(d_input=21, d_emb=d_emb, max_len=max_len, p_pe_drop=0.1)
    msa_emb = msa_embedder(msa)

    enc = EncoderLayer(d_emb=d_emb, d_ff=d_ff, n_heads=n_heads, p_dropout=0.1, tied=True)

    assert enc(msa_emb).shape == (bsz, n_seq, max_len, d_emb)


def test_EncoderLayer_performer_shape():
    bsz, n_seq, max_len = 4, 10, 64
    d_emb, n_heads = 16, 2
    d_ff = d_emb * 4

    msa = torch.randint(0, 21, (bsz, n_seq, max_len))
    msa_embedder = MsaEmbedding(d_input=21, d_emb=d_emb, max_len=max_len, p_pe_drop=0.1)
    msa_emb = msa_embedder(msa)

    enc = EncoderLayer(d_emb=d_emb, d_ff=d_ff, n_heads=n_heads, p_dropout=0.1, performer=True)

    assert enc(msa_emb).shape == (bsz, n_seq, max_len, d_emb)


# MsaUpdateUsingSelfAttention
def test_MsaUpdateUsingSelfAttention_init():
    d_emb, n_heads = 16, 2
    d_ff = d_emb * 4

    msa_update = MsaUpdateUsingSelfAttention(
        d_emb=d_emb, d_ff=d_ff, n_heads=n_heads, p_dropout=0.1
    )

    assert msa_update is not None


def test_MsaUpdateUsingSelfAttention_shape():
    bsz, n_seq, max_len = 4, 10, 64
    d_emb, n_heads = 16, 2
    d_ff = d_emb * 4

    msa = torch.randint(0, 21, (bsz, n_seq, max_len))
    msa_embedder = MsaEmbedding(d_input=21, d_emb=d_emb, max_len=max_len, p_pe_drop=0.1)
    msa_emb = msa_embedder(msa)

    msa_update = MsaUpdateUsingSelfAttention(
        d_emb=d_emb, d_ff=d_ff, n_heads=n_heads, p_dropout=0.1
    )

    x, att = msa_update(msa_emb)
    assert x.shape == (bsz, n_seq, max_len, d_emb)
    assert att.shape == (bsz, max_len, max_len, n_heads)


# OuterProductMean
def test_OuterProductMean_init():
    d_emb = 16
    out_features = 32

    outer_product_mean = OuterProductMean(in_features=d_emb, out_features=out_features)

    assert outer_product_mean is not None


def test_OuterProductMean_shape():
    bsz, n_seq, max_len = 4, 10, 64
    d_emb = 16
    out_features = 32

    msa = torch.randint(0, 21, (bsz, n_seq, max_len))
    msa_embedder = MsaEmbedding(d_input=21, d_emb=d_emb, max_len=max_len, p_pe_drop=0.1)
    msa_emb = msa_embedder(msa)

    outer_product_mean = OuterProductMean(in_features=d_emb, out_features=out_features)

    assert outer_product_mean(msa_emb, msa_emb).shape == (
        bsz,
        max_len,
        max_len,
        out_features,
    )


# PairUpdateWithMSA
def test_PairUpdateWithMsa_init():
    d_emb, d_proj, d_pair, n_heads = 16, 4, 4, 2

    pair_update = PairUpdateWithMsa(
        d_emb=d_emb, d_proj=d_proj, d_pair=d_pair, n_heads=n_heads, p_dropout=0.1
    )

    assert pair_update is not None


def test_PairUpdateWithMsa_shape():
    bsz, n_seq, max_len = 4, 10, 64
    n_heads = 2
    d_emb = 64

    msa = torch.randint(0, 21, (bsz, n_seq, max_len))
    msa_embedder = MsaEmbedding(d_input=21, d_emb=d_emb, max_len=max_len, p_pe_drop=0.1)
    msa_emb = msa_embedder(msa)

    d_proj, d_pair = 16, 32

    pair_update = PairUpdateWithMsa(
        d_emb=d_emb, d_proj=d_proj, d_pair=d_pair, n_heads=n_heads, p_dropout=0.1
    )

    pair = torch.randn(bsz, max_len, max_len, d_pair)
    att = torch.randn(bsz, max_len, max_len, n_heads)

    assert pair_update(msa_emb, pair, att).shape == (bsz, max_len, max_len, d_pair)


# PairUpdateWithAxialAttention
def test_PairUpdateWithAxialAttention_init():
    d_pair, n_heads = 4, 2

    pair_update = PairUpdateWithAxialAttention(
        d_pair=d_pair,
        d_ff=d_pair * 4,
        n_heads=n_heads,
        p_dropout=0.1,
    )

    assert pair_update is not None


def test_PairUpdateWithAxialAttention_shape():
    bsz, d_pair, n_heads, max_len = 4, 16, 4, 64

    pair_update = PairUpdateWithAxialAttention(
        d_pair=d_pair,
        d_ff=d_pair * 4,
        n_heads=n_heads,
        p_dropout=0.1,
    )

    pair = torch.randn(bsz, max_len, max_len, d_pair)
    assert pair_update(pair).shape == (bsz, max_len, max_len, d_pair)


# Symmetrization
def test_Symmetrization():
    sym = Symmetrization()
    assert sym is not None

    x = torch.randn(4, 10, 10, 8)
    x_sym = sym(x)
    assert x_sym.shape == (4, 10, 10, 8)
    assert (x_sym == x_sym.transpose(-2, -3)).all()


# MsaUpdateWithPair
def test_MSAUpdateWithPair_init():
    d_emb, d_pair, n_heads = 16, 4, 2

    msa_update = MsaUpdateWithPair(d_emb=d_emb, d_pair=d_pair, n_heads=n_heads, p_dropout=0.1)

    assert msa_update is not None


def test_MSAUpdateWithPair_shape():
    d_emb, d_pair, n_heads = 16, 4, 2
    bsz, n_seq, max_len = 4, 10, 64

    msa = torch.randint(0, 21, (bsz, n_seq, max_len))
    msa_embedder = MsaEmbedding(d_input=21, d_emb=d_emb, max_len=max_len, p_pe_drop=0.1)
    msa = msa_embedder(msa)

    pair = torch.randn(bsz, max_len, max_len, d_pair)
    msa_update = MsaUpdateWithPair(d_emb=d_emb, d_pair=d_pair, n_heads=n_heads, p_dropout=0.1)

    assert msa_update is not None
    assert msa_update(msa, pair).shape == (bsz, n_seq, max_len, d_emb)


# GraphTransformer
def test_GraphTransformer_init():
    d_node_in = 16
    d_node_out = 16
    d_edge = 16
    n_heads = 4

    graph_transformer = GraphTransformer(
        d_node_in=d_node_in,
        d_node_out=d_node_out,
        d_edge=d_edge,
        n_heads=n_heads,
        p_dropout=0.15,
    )

    assert graph_transformer is not None


def test_GraphTransformer_shape():
    d_node_in = 16
    d_node_out = 16
    d_edge = 16
    n_heads = 4

    graph_transformer = GraphTransformer(
        d_node_in=d_node_in,
        d_node_out=d_node_out,
        d_edge=d_edge,
        n_heads=n_heads,
        p_dropout=0.15,
    )

    """node_feat, (b l d_node_in)
        edge_feat, (b l l d_edge)
        edge_mask, (b l l): True (1) if eij exists, else False (0)
        """
    bsz, l, d_node = 4, 32, 16
    d_edge = 16

    node_feat = torch.randn(bsz, l, d_node)
    edge_feat = torch.randn(bsz, l, l, d_edge)
    edge_mask = torch.randint(0, 2, (bsz, l, l)).float()

    assert graph_transformer(node_feat, edge_feat, edge_mask).shape == (
        bsz,
        l,
        d_node_out * n_heads,
    )


# InitialCoordGenerationWithMsaAndPair
def test_InitialCoordGenerationWithMsaAndPair_init():
    d_emb = 16
    d_pair = 16
    n_heads = 4
    n_layers = 2

    initial_coord_generation = InitialCoordGenerationWithMsaAndPair(
        d_emb=d_emb,
        d_pair=d_pair,
        n_heads=n_heads,
        n_layers=n_layers,
        p_dropout=0.1,
    )

    assert initial_coord_generation is not None


def test_InitialCoordGenerationWithMsaAndPair_shape():
    d_emb = 16
    d_pair = 16
    n_heads = 4
    n_layers = 2

    initial_coord_generation = InitialCoordGenerationWithMsaAndPair(
        d_emb=d_emb,
        d_pair=d_pair,
        n_heads=n_heads,
        n_layers=n_layers,
        p_dropout=0.1,
    )

    bsz, n_seq, max_len = 4, 10, 64
    msa = torch.randint(0, 21, (bsz, n_seq, max_len))
    msa_embedder = MsaEmbedding(d_input=21, d_emb=d_emb, max_len=max_len, p_pe_drop=0.1)
    msa_emb = msa_embedder(msa)

    pair_emb = torch.randn(bsz, max_len, max_len, d_pair)
    seq_onehot = F.one_hot(torch.randint(0, 21, (bsz, max_len)), num_classes=21).float()

    aa_pos = torch.randint(0, max_len, (bsz, max_len))

    assert initial_coord_generation(msa_emb, pair_emb, seq_onehot, aa_pos).shape == (
        bsz,
        max_len,
        3,
        3,
    )


# CoordUpdateWithMsaAndPair
def test_CoordUpdateWithMsaAndPair_init():
    d_emb, d_pair = 16, 16
    d_node, d_edge = 8, 8
    d_state, n_neighbors = 8, 4

    coord_update = CoordUpdateWithMsaAndPair(
        d_emb=d_emb,
        d_pair=d_pair,
        d_node=d_node,
        d_edge=d_edge,
        d_state=d_state,
        n_neighbors=n_neighbors,
    )
    assert coord_update is not None


def test_CoordUpdateWithMsaAndPair_shape():
    d_emb, d_pair = 16, 16
    d_node, d_edge = 8, 8
    d_state, n_neighbors = 8, 4

    coord_update = CoordUpdateWithMsaAndPair(
        d_emb=d_emb,
        d_pair=d_pair,
        d_node=d_node,
        d_edge=d_edge,
        d_state=d_state,
        n_neighbors=n_neighbors,
    )
    assert coord_update is not None

    bsz, n_seq, max_len = 4, 10, 64

    xyz = torch.randn(bsz, max_len, 3, 3)
    msa = torch.randn(bsz, n_seq, max_len, d_emb)
    pair = torch.randn(bsz, max_len, max_len, d_pair)
    aa_idx = torch.randint(0, max_len, (bsz, max_len))
    seq_onehot = F.one_hot(torch.randint(0, 21, (bsz, max_len)), num_classes=21).float()

    state, xyz = coord_update(xyz, msa, pair, aa_idx, seq_onehot)
    assert state.shape == (bsz, max_len, d_state)
    assert xyz.shape == (bsz, max_len, 3, 3)


# MsaUpdateWithPairAndCoord
def test_MsaUpdateWithPairAndCoord_init():
    d_emb, d_state = 16, 8
    d_trfm_inner, d_ff = 32, 64
    distance_bins = [8, 12, 16, 20]

    msa_update = MsaUpdateWithPairAndCoord(
        d_emb=d_emb,
        d_state=d_state,
        d_trfm_inner=d_trfm_inner,
        d_ff=d_ff,
        distance_bins=distance_bins,
        p_dropout=0.1,
    )
    assert msa_update is not None


def test_MsaUpdateWithPairAndCoord_shape():
    d_emb, d_state = 16, 8
    d_trfm_inner, d_ff = 32, 64
    distance_bins = [8, 12, 16, 20]

    msa_update = MsaUpdateWithPairAndCoord(
        d_emb=d_emb,
        d_state=d_state,
        d_trfm_inner=d_trfm_inner,
        d_ff=d_ff,
        distance_bins=distance_bins,
        p_dropout=0.1,
    )
    assert msa_update is not None

    bsz, n_seq, max_len = 4, 10, 64

    xyz = torch.randn(bsz, max_len, 3, 3)
    state = torch.randn(bsz, max_len, d_state)
    msa = torch.randn(bsz, n_seq, max_len, d_emb)

    msa = msa_update(xyz, state, msa)
    assert msa.shape == (bsz, n_seq, max_len, d_emb)


# TwoTrackBlock
def test_TwoTrackBlock_init():
    p_dropout = 0.1

    block = TwoTrackBlock(p_dropout=p_dropout)

    assert block is not None


def test_TwoTrackBlock_shape():
    d_emb, d_pair = 384, 288
    bsz, n_seq, max_len = 4, 10, 64

    msa = torch.randn(bsz, n_seq, max_len, d_emb)
    pair = torch.randn(bsz, max_len, max_len, d_pair)

    block = TwoTrackBlock(p_dropout=0.1)
    msa, pair = block(msa, pair)

    assert msa.shape == (bsz, n_seq, max_len, d_emb)
    assert pair.shape == (bsz, max_len, max_len, d_pair)


# ThreeTrackBlock
def test_ThreeTrackBlock_init():
    p_dropout = 0.1
    n_neighbors = 128

    block = ThreeTrackBlock(
        n_neighbors=n_neighbors,
        p_dropout=p_dropout,
    )

    assert block is not None


def test_ThreeTrackBlock_shape():
    d_emb, d_pair = 384, 288
    bsz, n_seq, max_len = 4, 10, 64
    n_neighbors, p_dropout = 128, 0.1

    msa = torch.randn(bsz, n_seq, max_len, d_emb)
    pair = torch.randn(bsz, max_len, max_len, d_pair)
    xyz = torch.randn(bsz, max_len, 3, 3)
    aa_idx = torch.randint(0, max_len, (bsz, max_len))
    seq_onehot = F.one_hot(torch.randint(0, 21, (bsz, max_len)), num_classes=21).float()

    block = ThreeTrackBlock(
        n_neighbors=n_neighbors,
        p_dropout=p_dropout,
    )
    msa, pair, xyz = block(msa, pair, xyz, seq_onehot, aa_idx)

    assert msa.shape == (bsz, n_seq, max_len, d_emb)
    assert pair.shape == (bsz, max_len, max_len, d_pair)
    assert xyz.shape == (bsz, max_len, 3, 3)
