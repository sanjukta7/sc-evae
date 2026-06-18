import torch
import torch.nn as nn
import torch.nn.functional as F
from sc_evae.config.models import TransformerConfig

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def get_activation_fn(name: str) -> nn.Module:
    if name == "gelu":
        return nn.GELU()
    elif name == "relu":
        return nn.ReLU()
    elif name == "silu":
        return nn.SiLU()
    raise ValueError(f"Unknown activation: {name!r}")


def _build_causal_mask(input_len: int) -> torch.Tensor:
    return torch.triu(torch.ones(input_len, input_len, dtype=torch.bool), diagonal=1)


def _attention(q, k, v, attn_mask, attn_dropout, training):
    return F.scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask=~attn_mask if attn_mask is not None else None,
        dropout_p=(attn_dropout if training else 0.0),
        is_causal=False,
    )


# ---------------------------------------------------------------------------
# Plain Transformer
# ---------------------------------------------------------------------------


class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.d_model = config.d_model
        self.nhead = config.nhead
        self.head_dim = config.d_model // config.nhead
        self.attn_dropout = config.attn_dropout

        self.norm1 = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.qkv_proj = nn.Linear(config.d_model, 3 * config.d_model)
        self.o_proj = nn.Linear(config.d_model, config.d_model)

        self.norm2 = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.ffn = nn.Sequential(
            nn.Linear(config.d_model, config.dim_feedforward),
            get_activation_fn(config.activation),
            nn.Dropout(config.attn_dropout),
            nn.Linear(config.dim_feedforward, config.d_model),
            nn.Dropout(config.attn_dropout),
        )

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        # --- Self-attention block ---
        residual = x
        x = self.norm1(x)

        qkv = self.qkv_proj(x).view(batch_size, seq_len, 3, self.nhead, self.head_dim)
        q = qkv[:, :, 0].transpose(1, 2)
        k = qkv[:, :, 1].transpose(1, 2)
        v = qkv[:, :, 2].transpose(1, 2)

        attn_out = _attention(q, k, v, attn_mask, self.attn_dropout, self.training)
        attn_out = (
            attn_out.transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_len, self.d_model)
        )
        x = residual + self.o_proj(attn_out)

        # --- FFN block ---
        residual = x
        x = residual + self.ffn(self.norm2(x))

        return x


class TransformerBackbone(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.causal_input_mask = config.causal_input_mask

        self.layers = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.num_layers)]
        )
        self.norm_f = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)

        self.register_buffer("attn_mask", None, persistent=False)

    def _get_mask(self, input_len: int, device: torch.device) -> torch.Tensor | None:
        if not self.causal_input_mask:
            return None
        if self.attn_mask is None or self.attn_mask.shape[0] != input_len:
            self.attn_mask = _build_causal_mask(input_len).to(device)
        return self.attn_mask

    def forward(self, input_embeds: torch.Tensor) -> torch.Tensor:
        _, input_len, _ = input_embeds.shape
        attn_mask = self._get_mask(input_len, input_embeds.device)

        x = input_embeds
        for layer in self.layers:
            x = layer(x, attn_mask)
        return self.norm_f(x)


# ---------------------------------------------------------------------------
# Plain Transformer + Cross-Attention
# ---------------------------------------------------------------------------


class TransformerCrossAttentionBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.d_model = config.d_model
        self.nhead = config.nhead
        self.head_dim = config.d_model // config.nhead
        self.attn_dropout = config.attn_dropout

        # Self-attention
        self.norm1 = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.qkv_proj = nn.Linear(config.d_model, 3 * config.d_model)
        self.o_proj = nn.Linear(config.d_model, config.d_model)

        # Cross-attention — Q from input, K/V from context
        self.norm_cross = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.q_cross_proj = nn.Linear(config.d_model, config.d_model)
        self.kv_cross_proj = nn.Linear(config.d_model, 2 * config.d_model)
        self.o_cross_proj = nn.Linear(config.d_model, config.d_model)

        # FFN
        self.norm2 = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.ffn = nn.Sequential(
            nn.Linear(config.d_model, config.dim_feedforward),
            get_activation_fn(config.activation),
            nn.Dropout(config.attn_dropout),
            nn.Linear(config.dim_feedforward, config.d_model),
            nn.Dropout(config.attn_dropout),
        )

    def forward(
        self, x: torch.Tensor, context: torch.Tensor, attn_mask: torch.Tensor | None
    ) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        ctx_len = context.shape[1]

        # --- Self-attention block ---
        residual = x
        x = self.norm1(x)

        qkv = self.qkv_proj(x).view(batch_size, seq_len, 3, self.nhead, self.head_dim)
        q = qkv[:, :, 0].transpose(1, 2)
        k = qkv[:, :, 1].transpose(1, 2)
        v = qkv[:, :, 2].transpose(1, 2)

        attn_out = _attention(q, k, v, attn_mask, self.attn_dropout, self.training)
        attn_out = (
            attn_out.transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_len, self.d_model)
        )
        x = residual + self.o_proj(attn_out)

        # --- Cross-attention block ---
        residual = x
        x = self.norm_cross(x)

        q = (
            self.q_cross_proj(x)
            .view(batch_size, seq_len, self.nhead, self.head_dim)
            .transpose(1, 2)
        )
        kv = self.kv_cross_proj(context).view(
            batch_size, ctx_len, 2, self.nhead, self.head_dim
        )
        k = kv[:, :, 0].transpose(1, 2)
        v = kv[:, :, 1].transpose(1, 2)

        # No mask — Q attends to all context positions
        attn_out = _attention(q, k, v, None, self.attn_dropout, self.training)
        attn_out = (
            attn_out.transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_len, self.d_model)
        )
        x = residual + self.o_cross_proj(attn_out)

        # --- FFN block ---
        residual = x
        x = residual + self.ffn(self.norm2(x))

        return x


class TransformerCrossAttentionBackbone(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.causal_input_mask = config.causal_input_mask

        self.layers = nn.ModuleList(
            [TransformerCrossAttentionBlock(config) for _ in range(config.num_layers)]
        )
        self.norm_f = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)

        self.register_buffer("attn_mask", None, persistent=False)

    def _get_mask(self, input_len: int, device: torch.device) -> torch.Tensor | None:
        if not self.causal_input_mask:
            return None
        if self.attn_mask is None or self.attn_mask.shape[0] != input_len:
            self.attn_mask = _build_causal_mask(input_len).to(device)
        return self.attn_mask

    def forward(
        self, input_embeds: torch.Tensor, context: torch.Tensor
    ) -> torch.Tensor:
        _, input_len, _ = input_embeds.shape
        attn_mask = self._get_mask(input_len, input_embeds.device)

        x = input_embeds
        for layer in self.layers:
            x = layer(x, context, attn_mask)
        return self.norm_f(x)


# ---------------------------------------------------------------------------
# DiT (adaLN-Zero)
# ---------------------------------------------------------------------------


class DITBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.d_model = config.d_model
        self.nhead = config.nhead
        self.head_dim = config.d_model // config.nhead
        self.attn_dropout = config.attn_dropout

        self.norm1 = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        # Single projection for all 6 adaLN params: [γ₁, β₁, α₁, γ₂, β₂, α₂]
        self.adaln_modulation = nn.Linear(config.d_model, 6 * config.d_model)
        self.qkv_proj = nn.Linear(config.d_model, 3 * config.d_model)
        self.o_proj = nn.Linear(config.d_model, config.d_model)

        self.norm2 = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.ffn = nn.Sequential(
            nn.Linear(config.d_model, config.dim_feedforward),
            get_activation_fn(config.activation),
            nn.Dropout(config.attn_dropout),
            nn.Linear(config.dim_feedforward, config.d_model),
            nn.Dropout(config.attn_dropout),
        )

        nn.init.zeros_(self.adaln_modulation.weight)
        nn.init.zeros_(self.adaln_modulation.bias)

    def forward(
        self, x: torch.Tensor, cond: torch.Tensor, attn_mask: torch.Tensor | None
    ) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        g1, b1, a1, g2, b2, a2 = self.adaln_modulation(cond).chunk(6, dim=-1)

        # --- Attention block ---
        residual = x
        x = (1 + g1.unsqueeze(1)) * self.norm1(x) + b1.unsqueeze(1)

        qkv = self.qkv_proj(x).view(batch_size, seq_len, 3, self.nhead, self.head_dim)
        q = qkv[:, :, 0].transpose(1, 2)
        k = qkv[:, :, 1].transpose(1, 2)
        v = qkv[:, :, 2].transpose(1, 2)

        attn_out = _attention(q, k, v, attn_mask, self.attn_dropout, self.training)
        attn_out = (
            attn_out.transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_len, self.d_model)
        )
        x = residual + a1.unsqueeze(1) * self.o_proj(attn_out)

        # --- FFN block ---
        residual = x
        x = residual + a2.unsqueeze(1) * self.ffn(
            (1 + g2.unsqueeze(1)) * self.norm2(x) + b2.unsqueeze(1)
        )

        return x


class DITBackbone(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.causal_input_mask = config.causal_input_mask

        self.layers = nn.ModuleList(
            [DITBlock(config) for _ in range(config.num_layers)]
        )
        self.norm_f = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)

        self.register_buffer("attn_mask", None, persistent=False)

    def _get_mask(self, input_len: int, device: torch.device) -> torch.Tensor | None:
        if not self.causal_input_mask:
            return None
        if self.attn_mask is None or self.attn_mask.shape[0] != input_len:
            self.attn_mask = _build_causal_mask(input_len).to(device)
        return self.attn_mask

    def forward(
        self, input_embeds: torch.Tensor, cond_embeds: torch.Tensor
    ) -> torch.Tensor:
        _, input_len, _ = input_embeds.shape
        attn_mask = self._get_mask(input_len, input_embeds.device)

        x = input_embeds
        for layer in self.layers:
            x = layer(x, cond_embeds, attn_mask)
        return self.norm_f(x)


# ---------------------------------------------------------------------------
# DiT + Cross-Attention (adaLN-Zero)
# ---------------------------------------------------------------------------


class DITCrossAttentionBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.d_model = config.d_model
        self.nhead = config.nhead
        self.head_dim = config.d_model // config.nhead
        self.attn_dropout = config.attn_dropout

        # Self-attention (adaLN modulated)
        self.norm1 = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        # Single projection for all 6 adaLN params: [γ₁, β₁, α₁, γ₂, β₂, α₂]
        self.adaln_modulation = nn.Linear(config.d_model, 6 * config.d_model)
        self.qkv_proj = nn.Linear(config.d_model, 3 * config.d_model)
        self.o_proj = nn.Linear(config.d_model, config.d_model)

        # Cross-attention — not modulated by adaLN, context is its own signal
        self.norm_cross = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.q_cross_proj = nn.Linear(config.d_model, config.d_model)
        self.kv_cross_proj = nn.Linear(config.d_model, 2 * config.d_model)
        self.o_cross_proj = nn.Linear(config.d_model, config.d_model)

        # FFN (adaLN modulated)
        self.norm2 = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.ffn = nn.Sequential(
            nn.Linear(config.d_model, config.dim_feedforward),
            get_activation_fn(config.activation),
            nn.Dropout(config.attn_dropout),
            nn.Linear(config.dim_feedforward, config.d_model),
            nn.Dropout(config.attn_dropout),
        )

        nn.init.zeros_(self.adaln_modulation.weight)
        nn.init.zeros_(self.adaln_modulation.bias)

    def forward(
        self,
        x: torch.Tensor,
        cond: torch.Tensor,
        context: torch.Tensor,
        attn_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        ctx_len = context.shape[1]
        g1, b1, a1, g2, b2, a2 = self.adaln_modulation(cond).chunk(6, dim=-1)

        # --- Self-attention block (adaLN modulated) ---
        residual = x
        x = (1 + g1.unsqueeze(1)) * self.norm1(x) + b1.unsqueeze(1)

        qkv = self.qkv_proj(x).view(batch_size, seq_len, 3, self.nhead, self.head_dim)
        q = qkv[:, :, 0].transpose(1, 2)
        k = qkv[:, :, 1].transpose(1, 2)
        v = qkv[:, :, 2].transpose(1, 2)

        attn_out = _attention(q, k, v, attn_mask, self.attn_dropout, self.training)
        attn_out = (
            attn_out.transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_len, self.d_model)
        )
        x = residual + a1.unsqueeze(1) * self.o_proj(attn_out)

        # --- Cross-attention block (plain, context is its own conditioning) ---
        residual = x
        x = self.norm_cross(x)

        q = (
            self.q_cross_proj(x)
            .view(batch_size, seq_len, self.nhead, self.head_dim)
            .transpose(1, 2)
        )
        kv = self.kv_cross_proj(context).view(
            batch_size, ctx_len, 2, self.nhead, self.head_dim
        )
        k = kv[:, :, 0].transpose(1, 2)
        v = kv[:, :, 1].transpose(1, 2)

        attn_out = _attention(q, k, v, None, self.attn_dropout, self.training)
        attn_out = (
            attn_out.transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_len, self.d_model)
        )
        x = residual + self.o_cross_proj(attn_out)

        # --- FFN block (adaLN modulated) ---
        residual = x
        x = residual + a2.unsqueeze(1) * self.ffn(
            (1 + g2.unsqueeze(1)) * self.norm2(x) + b2.unsqueeze(1)
        )

        return x


class DITCrossAttentionBackbone(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.causal_input_mask = config.causal_input_mask

        self.layers = nn.ModuleList(
            [DITCrossAttentionBlock(config) for _ in range(config.num_layers)]
        )
        self.norm_f = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)

        self.register_buffer("attn_mask", None, persistent=False)

    def _get_mask(self, input_len: int, device: torch.device) -> torch.Tensor | None:
        if not self.causal_input_mask:
            return None
        if self.attn_mask is None or self.attn_mask.shape[0] != input_len:
            self.attn_mask = _build_causal_mask(input_len).to(device)
        return self.attn_mask

    def forward(
        self,
        input_embeds: torch.Tensor,
        cond_embeds: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        _, input_len, _ = input_embeds.shape
        attn_mask = self._get_mask(input_len, input_embeds.device)

        x = input_embeds
        for layer in self.layers:
            x = layer(x, cond_embeds, context, attn_mask)
        return self.norm_f(x)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def build_backbone(config: TransformerConfig) -> nn.Module:
    if config.type == "plain":
        return TransformerBackbone(config)
    elif config.type == "cross_attention":
        return TransformerCrossAttentionBackbone(config)
    elif config.type == "dit":
        return DITBackbone(config)
    elif config.type == "dit_cross_attention":
        return DITCrossAttentionBackbone(config)
    raise ValueError(f"Unknown backbone type: {config.type!r}")
