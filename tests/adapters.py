"""Attention-only adapters, extracted from the Apache-2.0 course scaffold.

The original online-softmax and fused-softmax-matmul adapters are omitted because
those separate exercises are outside the scope of this portfolio package.
"""
from __future__ import annotations

from typing import Type


def get_flashattention_autograd_function_pytorch() -> Type:
    from flash_attention.flash_attention import FlashAttentionPytorch
    return FlashAttentionPytorch


def get_flashattention_autograd_function_triton() -> Type:
    from flash_attention.flash_attention import FlashAttentionTriton
    return FlashAttentionTriton
