"""
Adapted from https://github.com/facebookresearch/DiT/blob/main/models.py
"""

from typing import Tuple, Optional
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import SDPBackend
import math

from timm.models.vision_transformer import Mlp
from ..modules.embeddings import RotaryEmbeddingND

def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor):
    return x * (1 + scale) + shift

    """
    输入: [B, N, C]
    ↓ qkv线性层
    [B, N, 3×C]
    ↓ 重塑 + 置换
    [3, B, num_heads, N, head_dim]
    ↓ 分离
    q: [B, num_heads, N, head_dim]
    k: [B, num_heads, N, head_dim] 
    v: [B, num_heads, N, head_dim]
    q: [B, num_heads, N, head_dim]
    ↓ 转置k
    k^T: [B, num_heads, head_dim, N]
    ↓ 矩阵乘法
    attn: [B, num_heads, N, N]
    ↓ 与v相乘
    输出: [B, num_heads, N, head_dim]
    [B, num_heads, N, head_dim]
    ↓ 转置 + 重塑
    [B, N, num_heads × head_dim] = [B, N, C]
    ↓ 投影层
    [B, N, C]
    """
class Attention(nn.Module):
    """
    Adapted from timm.models.vision_transformer,
    to support the use of RoPE.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: nn.Module = nn.LayerNorm,
        rope: Optional[RotaryEmbeddingND] = None,
        fused_attn: bool = True,
        is_conditional: bool = False,
        conditioning_scale: float = 1.0,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, f"dim should be divisible by num_heads {dim} {num_heads}."
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.fused_attn = fused_attn

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.rope = rope
        self.is_conditional = is_conditional
        self.conditioning_scale = conditioning_scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape           #batch, sequence length, channels
        qkv = (
            self.qkv(x)                                         # -> (B, N, 3* C)
            .reshape(B, N, 3, self.num_heads, self.head_dim)    # (B, N, C*3) -> (B, N, 3, num_head, head_dim)
            .permute(2, 0, 3, 1, 4)                             # (B, N, 3, num_head, head_dim) -> (3, B, num_head, N, head_dim)
        )
        q, k, v = qkv.unbind(0)                 # 分离Q, K, V，每个形状为 [B, num_heads, N, head_dim]
        q, k = self.q_norm(q), self.k_norm(k)   # 可选QK归一化

        if self.rope is not None:
            if self.is_conditional:
                q_x, q_cond = q.chunk(2, dim=-2)
                k_x, k_cond = k.chunk(2, dim=-2)
                q = torch.cat((self.rope(q_x), self.rope(q_cond)), dim=-2)
                k = torch.cat((self.rope(k_x), self.rope(k_cond)), dim=-2)
            else:
                q = self.rope(q)
                k = self.rope(k)

        if self.is_conditional and self.conditioning_scale != 1.0:
            N_input = N // 2
            ## register attn bias
            if not hasattr(self, "attn_bias") or self.attn_bias.shape[-1] != N:
                self.attn_bias = torch.zeros(1, 1, N, N, device=x.device)
                self.attn_bias[:, :, :N_input, :N_input] = 0.0
                self.attn_bias[:, :, :N_input, N_input:] = math.log(self.conditioning_scale)
                self.attn_bias[:, :, N_input:, :N_input] = math.log(self.conditioning_scale)
                self.attn_bias[:, :, N_input:, N_input:] = 0.0
            self.attn_bias = self.attn_bias.to(x.dtype)
        else:
            self.attn_bias = None

        if self.fused_attn:
            # pylint: disable-next=not-callable, # 使用PyTorch的高效注意力实现
            x = F.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=self.attn_drop.p if self.training else 0.0,
                attn_mask=self.attn_bias,
            )
        else:
            # 手动实现注意力机制
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            if self.attn_bias is not None:
                attn = attn + self.attn_bias
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)  # 合并多头输出 [B, N, C]
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    
class CrossAttention(nn.Module):
    def __init__(
        self, 
        dim, 
        num_heads=12, 
        qkv_bias=False, 
        qk_norm: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: nn.Module = nn.LayerNorm,
        rope: Optional[RotaryEmbeddingND] = None,
        fused_attn: bool = True,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5
        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv_proj = nn.Linear(dim, int(dim * 2), bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.rope = rope
        self.fused_attn = fused_attn

    def forward(self, q, x):
        B, n, C = q.shape
        q = self.q_proj(q).reshape(B, n, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        q = self.q_norm(q) # 可选QK归一化

        B, N, C = x.shape
        kv = self.kv_proj(x).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]  # (batch_size, num_heads, seq_len, feature_dim_per_head)
        k = self.k_norm(k)   # 可选QK归一化

        if self.rope is not None:
            q = self.rope(q)
            k = self.rope(k)

        if self.fused_attn:
            if q.dtype == torch.bfloat16:
                with nn.attention.sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                    x =  F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p if self.training else 0.0,)
            else:
                with nn.attention.sdpa_kernel([SDPBackend.MATH, SDPBackend.EFFICIENT_ATTENTION]):
                    x =  F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p if self.training else 0.0,)

            # with torch.nn.attention.sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            #     q = F.scaled_dot_product_attention(q, k, v)
        else:
            xattn = (q @ k.transpose(-2, -1)) * self.scale
            xattn = xattn.softmax(dim=-1)  # (batch_size, num_heads, query_len, seq_len)
            q = xattn @ v

        q = q.transpose(1, 2).reshape(B, n, C)
        
        q = self.proj(q)
        q = self.proj_drop(q)

        return q


class CrossAttentionBlock(nn.Module):
    def __init__(self, 
        hidden_size, 
        num_heads, 
        mlp_ratio: Optional[float] = 4.0,
        rope: Optional[RotaryEmbeddingND] = None,
        **block_kwargs: dict,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size)
        self.xattn = CrossAttention(hidden_size, num_heads=num_heads, qkv_bias=True, rope=rope, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.use_mlp = mlp_ratio is not None
        if self.use_mlp:
            self.mlp = Mlp(
                in_features=hidden_size,
                hidden_features=int(hidden_size * mlp_ratio),
                act_layer=partial(nn.GELU, approximate="tanh"),
            )
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer linear layers:
        def _basic_init(module: nn.Module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        self.xattn.apply(_basic_init)
        if self.use_mlp:
            self.mlp.apply(_basic_init)


    def forward(self, q, x):
        y = self.xattn(q, self.norm1(x))
        q = q + y
        q = q + self.mlp(self.norm2(q))
        return q

class AdaLayerNorm(nn.Module):
    """
    Adaptive layer norm (AdaLN).
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        self.modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True),
        )
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.initialize_weights()

    def initialize_weights(self):
        # Zero-out:
        nn.init.zeros_(self.modulation[-1].weight)
        nn.init.zeros_(self.modulation[-1].bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the AdaLN layer.
        Args:
            x: Input tensor of shape (B, N, C).
            c: Conditioning tensor of shape (B, N, C).
        """
        shift, scale = self.modulation(c).chunk(2, dim=-1)
        return modulate(self.norm(x), shift, scale)


class AdaLayerNormZero(nn.Module):
    """
    Adaptive layer norm zero (AdaLN-Zero).
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        self.modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 3 * hidden_size, bias=True),
        )
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.initialize_weights()

    def initialize_weights(self):
        # Zero-out:
        nn.init.zeros_(self.modulation[-1].weight)
        nn.init.zeros_(self.modulation[-1].bias)

    def forward(
        self, x: torch.Tensor, c: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the AdaLN-Zero layer.
        Args:
            x: Input tensor of shape (B, N, C).
            c: Conditioning tensor of shape (B, N, C).
        """
        shift, scale, gate = self.modulation(c).chunk(3, dim=-1)
        return modulate(self.norm(x), shift, scale), gate


class DiTBlock(nn.Module):
    """
    A DiT transformer block with adaptive layer norm zero (AdaLN-Zero) conditioning.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: Optional[float] = 4.0,
        rope: Optional[RotaryEmbeddingND] = None,
        cross_attn: bool = False,
        **block_kwargs: Dict[str, Any],
    ):
        """
        Args:
            hidden_size: Number of features in the hidden layer.
            num_heads: Number of attention heads.
            mlp_ratio: Ratio of hidden layer size in the MLP. None to skip the MLP.
            block_kwargs: Additional arguments to pass to the Attention block.
        """
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6) 
        self.norm1 = AdaLayerNormZero(hidden_size)
        self.attn1 = Attention(
            hidden_size, num_heads=num_heads, qkv_bias=True, rope=rope, **block_kwargs
        )
        self.use_mlp = mlp_ratio is not None
        if self.use_mlp:
            self.norm2 = AdaLayerNormZero(hidden_size)
            self.mlp = Mlp(
                in_features=hidden_size,
                hidden_features=int(hidden_size * mlp_ratio),
                act_layer=partial(nn.GELU, approximate="tanh"),
            )
        self.initialize_weights()
        self.cross_attn=cross_attn
        if self.cross_attn:
            self.attn2 = CrossAttentionBlock(
                hidden_size, num_heads=num_heads, qkv_bias=True, mlp_ratio = mlp_ratio, rope=rope, **block_kwargs
            )

    def initialize_weights(self):
        # Initialize transformer linear layers:
        def _basic_init(module: nn.Module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        self.attn1.apply(_basic_init)
        if self.use_mlp:
            self.mlp.apply(_basic_init)

    def forward(self, x: torch.Tensor, c: torch.Tensor, external_cond: Optional[torch.Tensor] = None):
        """
        Forward pass of the DiT block.
        In original implementation, conditioning is uniform across all tokens in the sequence. Here, we extend it to support token-wise conditioning (e.g. noise level can be different for each token).
        Args:
            x: Input tensor of shape (B, N, C).
            c: Conditioning tensor of shape (B, N, C).
        """
        x, gate_msa = self.norm1(x, c)
        x = x + gate_msa * self.attn1(x)
        if self.use_mlp:
            x, gate_mlp = self.norm2(x, c)
            x = x + gate_mlp * self.mlp(x)
        if self.cross_attn:
            assert (
                external_cond is not None
            ), "External condition (camera pose) is required for cross attention."
            x = self.attn2(x, external_cond)

        return x


class DITFinalLayer(nn.Module):
    """
    The final layer of DiT.
    """

    def __init__(
        self,
        hidden_size: int,
        out_channels: int,
    ):
        super().__init__()
        self.norm_final = AdaLayerNorm(hidden_size)
        self.linear = nn.Linear(hidden_size, out_channels, bias=True)
        self.initialize_weights()

    def initialize_weights(self):
        # Zero-out:
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor):
        """
        Forward pass of the DiT final layer.
        Args:
            x: Input tensor of shape (B, N, C).
            c: Conditioning tensor of shape (B, N, C).
        """
        x = self.norm_final(x, c)
        x = self.linear(x)
        return x
