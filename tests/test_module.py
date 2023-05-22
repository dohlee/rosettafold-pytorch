import pytest
import torch
import warnings

warnings.filterwarnings("ignore")

from rosettafold_pytorch.rosettafold_pytorch import MSAEmbedding

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
