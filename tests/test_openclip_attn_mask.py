from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
from torch import nn

from tooncrafter_animator.inference.adapter import ensure_vendor_on_path


def _causal_mask(n_ctx: int = 77) -> torch.Tensor:
    mask = torch.empty(n_ctx, n_ctx).fill_(float("-inf"))
    return mask.triu_(1)


def test_wrong_nld_to_lnd_permute_raises_77_vs_1x1() -> None:
    """The frozen-EXE failure: newer OpenCLIP MHA is batch_first (NLD).

    FrozenOpenCLIPEmbedder used to always permute [B, 77, D] -> [77, B, D].
    With B=1 that looks like seq_len=1, so PyTorch rejects CLIP's 77x77 mask.
    """
    attn = nn.MultiheadAttention(32, 4, batch_first=True)
    x = torch.randn(1, 77, 32)
    x_wrong = x.permute(1, 0, 2)
    mask = _causal_mask()
    with pytest.raises(RuntimeError, match=r"should be \(1, 1\)"):
        attn(x_wrong, x_wrong, x_wrong, need_weights=False, attn_mask=mask)
    out, _ = attn(x, x, x, need_weights=False, attn_mask=mask)
    assert out.shape == (1, 77, 32)


class _TinyBlock(nn.Module):
    def __init__(self, width: int, heads: int, batch_first: bool) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(width, heads, batch_first=batch_first)

    def forward(self, x, attn_mask=None):
        if attn_mask is not None and attn_mask.dtype != torch.bool:
            attn_mask = attn_mask.to(x.dtype)
        y, _ = self.attn(x, x, x, need_weights=False, attn_mask=attn_mask)
        return x + y


class _TinyTransformer(nn.Module):
    def __init__(self, width: int, heads: int, batch_first: bool) -> None:
        super().__init__()
        self.batch_first = batch_first
        self.grad_checkpointing = False
        self.resblocks = nn.ModuleList([_TinyBlock(width, heads, batch_first)])


class _TinyOpenCLIP(nn.Module):
    def __init__(self, batch_first: bool, width: int = 32, n_ctx: int = 77, vocab: int = 128) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(vocab, width)
        self.positional_embedding = nn.Parameter(torch.zeros(n_ctx, width))
        self.ln_final = nn.LayerNorm(width)
        self.attn_mask = _causal_mask(n_ctx)
        self.transformer = _TinyTransformer(width, heads=4, batch_first=batch_first)


def _stub_embedder(batch_first: bool):
    pytest.importorskip("open_clip")
    pytest.importorskip("transformers")
    pytest.importorskip("kornia")
    ensure_vendor_on_path()
    from lvdm.modules.encoders.condition import FrozenOpenCLIPEmbedder

    emb = FrozenOpenCLIPEmbedder.__new__(FrozenOpenCLIPEmbedder)
    nn.Module.__init__(emb)
    emb.model = _TinyOpenCLIP(batch_first=batch_first)
    emb.layer_idx = 0
    emb.device = torch.device("cpu")
    return emb


@pytest.mark.parametrize("batch_first", [True, False])
def test_frozen_openclip_embedder_text_path_accepts_77_mask(batch_first: bool) -> None:
    """Adapter load path: FrozenOpenCLIPEmbedder.encode_with_transformer, no CLIP file."""
    emb = _stub_embedder(batch_first)
    tokens = torch.randint(0, 128, (1, 77))
    out = emb.encode_with_transformer(tokens)
    assert out.shape == (1, 77, 32)
    assert torch.isfinite(out).all()


def test_openclip_layout_helper_matches_mha_batch_first() -> None:
    pytest.importorskip("open_clip")
    pytest.importorskip("transformers")
    pytest.importorskip("kornia")
    ensure_vendor_on_path()
    from lvdm.modules.encoders.condition import (
        apply_openclip_sequence_layout,
        openclip_transformer_is_batch_first,
    )

    bf = SimpleNamespace(batch_first=True)
    seq = SimpleNamespace(batch_first=False)
    x = torch.randn(1, 77, 8)
    assert openclip_transformer_is_batch_first(bf) is True
    assert openclip_transformer_is_batch_first(seq) is False
    assert apply_openclip_sequence_layout(x, bf).shape == (1, 77, 8)
    assert apply_openclip_sequence_layout(x, seq).shape == (77, 1, 8)
    attn_mod = SimpleNamespace(resblocks=[SimpleNamespace(attn=SimpleNamespace(batch_first=True))])
    assert openclip_transformer_is_batch_first(attn_mod) is True
