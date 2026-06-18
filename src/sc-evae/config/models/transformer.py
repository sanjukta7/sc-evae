"""Transformer backbone config (encoder / decoder / prior backbones)."""

from dataclasses import dataclass


@dataclass
class TransformerConfig:
    # "plain" | "cross_attention" | "dit" | "dit_cross_attention"
    type: str = "cross_attention"
    d_model: int = 256
    nhead: int = 8
    num_layers: int = 4
    dim_feedforward: int = 1024
    attn_dropout: float = 0.1
    layer_norm_eps: float = 1e-5
    activation: str = "gelu"
    causal_input_mask: bool = False
