import pytest
import torch
import warnings

warnings.filterwarnings("ignore")

from rosettafold_pytorch.rosettafold_pytorch import (
    MSAEmbedding,
    PositionWiseWeightFactor,
    SoftTiedAttentionOverResidues,
    EncoderLayer,
    MSAUpdateUsingSelfAttention,
    OuterProductMean,
    PairUpdateWithMSA,
)


# MSAEmbedding
def test_MSAEmbedding_init():
    msa_emb = MSAEmbedding(d_input=21, d_emb=64, max_len=5000, p_pe_drop=0.1)
    assert msa_emb is not None


def test_MSAEmbedding_positional_encoding_is_sinusoidal():
    msa_emb = MSAEmbedding()

    s = msa_emb.pos_enc[:, 0::2].square() + msa_emb.pos_enc[:, 1::2].square()
    assert torch.isclose(s, torch.tensor([1.0])).all()


def test_MSAEmbedding_shape():
    bsz, n_seq, max_len = 4, 10, 5000

    msa = torch.randint(0, 21, (bsz, n_seq, max_len))
    msa_emb = MSAEmbedding(d_input=21, d_emb=64, max_len=5000, p_pe_drop=0.1)

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
        pos_wise_weight_factor = PositionWiseWeightFactor(
            d_emb=d_emb, n_heads=n_heads, p_dropout=0.1
        )


def test_PositionWiseWeightFactor_shape():
    bsz, n_seq, max_len = 4, 10, 5000
    d_emb, n_heads, max_len = 64, 4, 32

    msa = torch.randint(0, 21, (bsz, n_seq, max_len))
    msa_embedder = MSAEmbedding(d_input=21, d_emb=d_emb, max_len=max_len, p_pe_drop=0.1)
    msa_emb = msa_embedder(msa)

    pos_wise_weight_factor = PositionWiseWeightFactor(
        d_emb=d_emb, n_heads=n_heads, p_dropout=0.1
    )

    assert pos_wise_weight_factor(msa_emb).shape == (bsz, n_seq, n_heads, max_len, 1)


def test_PositionWiseWeightFactor_sums_to_1():
    bsz, n_seq, max_len = 4, 10, 5000
    d_emb, n_heads, max_len = 64, 4, 32

    msa = torch.randint(0, 21, (bsz, n_seq, max_len))
    msa_embedder = MSAEmbedding(d_input=21, d_emb=d_emb, max_len=max_len, p_pe_drop=0.1)
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
    msa_embedder = MSAEmbedding(d_input=21, d_emb=d_emb, max_len=max_len, p_pe_drop=0.1)
    msa_emb = msa_embedder(msa)

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
    msa_embedder = MSAEmbedding(d_input=21, d_emb=d_emb, max_len=max_len, p_pe_drop=0.1)
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

    enc = EncoderLayer(
        d_emb=d_emb, d_ff=d_ff, n_heads=n_heads, p_dropout=0.1, tied=True
    )
    assert enc is not None


def test_EncoderLayer_performer_init():
    d_emb, n_heads = 64, 4
    d_ff = d_emb * 4

    enc = EncoderLayer(
        d_emb=d_emb, d_ff=d_ff, n_heads=n_heads, p_dropout=0.1, performer=True
    )
    assert enc is not None


def test_EncoderLayer_tied_shape():
    bsz, n_seq, max_len = 4, 10, 64
    d_emb, n_heads = 16, 2
    d_ff = d_emb * 4

    msa = torch.randint(0, 21, (bsz, n_seq, max_len))
    msa_embedder = MSAEmbedding(d_input=21, d_emb=d_emb, max_len=max_len, p_pe_drop=0.1)
    msa_emb = msa_embedder(msa)

    enc = EncoderLayer(
        d_emb=d_emb, d_ff=d_ff, n_heads=n_heads, p_dropout=0.1, tied=True
    )

    assert enc(msa_emb).shape == (bsz, n_seq, max_len, d_emb)


def test_EncoderLayer_performer_shape():
    bsz, n_seq, max_len = 4, 10, 64
    d_emb, n_heads = 16, 2
    d_ff = d_emb * 4

    msa = torch.randint(0, 21, (bsz, n_seq, max_len))
    msa_embedder = MSAEmbedding(d_input=21, d_emb=d_emb, max_len=max_len, p_pe_drop=0.1)
    msa_emb = msa_embedder(msa)

    enc = EncoderLayer(
        d_emb=d_emb, d_ff=d_ff, n_heads=n_heads, p_dropout=0.1, performer=True
    )

    assert enc(msa_emb).shape == (bsz, n_seq, max_len, d_emb)


# MSAUpdateUsingSelfAttention
def test_MSAUpdateUsingSelfAttention_init():
    d_emb, n_heads = 16, 2
    d_ff = d_emb * 4

    msa_update = MSAUpdateUsingSelfAttention(
        d_emb=d_emb, d_ff=d_ff, n_heads=n_heads, p_dropout=0.1
    )

    assert msa_update is not None


def test_MSAUpdateUsingSelfAttention_shape():
    bsz, n_seq, max_len = 4, 10, 64
    d_emb, n_heads = 16, 2
    d_ff = d_emb * 4

    msa = torch.randint(0, 21, (bsz, n_seq, max_len))
    msa_embedder = MSAEmbedding(d_input=21, d_emb=d_emb, max_len=max_len, p_pe_drop=0.1)
    msa_emb = msa_embedder(msa)

    msa_update = MSAUpdateUsingSelfAttention(
        d_emb=d_emb, d_ff=d_ff, n_heads=n_heads, p_dropout=0.1
    )

    assert msa_update(msa_emb).shape == (bsz, n_seq, max_len, d_emb)


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
    msa_embedder = MSAEmbedding(d_input=21, d_emb=d_emb, max_len=max_len, p_pe_drop=0.1)
    msa_emb = msa_embedder(msa)

    outer_product_mean = OuterProductMean(in_features=d_emb, out_features=out_features)

    assert outer_product_mean(msa_emb, msa_emb).shape == (
        bsz,
        max_len,
        max_len,
        out_features,
    )


# PairUpdateWithMSA
def test_PairUpdateWithMSA_init():
    d_emb, d_proj, d_pair, n_heads = 16, 4, 4, 2

    pair_update = PairUpdateWithMSA(
        d_emb=d_emb, d_proj=d_proj, d_pair=d_pair, n_heads=n_heads, p_dropout=0.1
    )

    assert pair_update is not None


def test_PairUpdateWithMSA_shape():
    bsz, n_seq, max_len = 4, 10, 64
    n_heads = 2
    d_emb = 64

    msa = torch.randint(0, 21, (bsz, n_seq, max_len))
    msa_embedder = MSAEmbedding(d_input=21, d_emb=d_emb, max_len=max_len, p_pe_drop=0.1)
    msa_emb = msa_embedder(msa)

    d_proj, d_pair = 16, 32

    pair_update = PairUpdateWithMSA(
        d_emb=d_emb, d_proj=d_proj, d_pair=d_pair, n_heads=n_heads, p_dropout=0.1
    )

    pair = torch.randn(bsz, max_len, max_len, d_pair)
    att = torch.randn(bsz, max_len, max_len, n_heads)

    assert pair_update(msa_emb, pair, att).shape == (bsz, max_len, max_len, d_pair)
