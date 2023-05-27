import pytest
import torch
import warnings

warnings.filterwarnings("ignore")

from rosettafold_pytorch.rosettafold_pytorch import (
    MSAEmbedding,
    PositionWiseWeightFactor,
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

    assert pos_wise_weight_factor(msa_emb).shape == (bsz, max_len, n_heads, 1, n_seq)


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
        .sum(dim=-1)
        .squeeze()
        .allclose(torch.ones((bsz, max_len, n_heads)))
    )
