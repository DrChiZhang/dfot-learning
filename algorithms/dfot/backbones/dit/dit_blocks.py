"""
Adapted from https://github.com/facebookresearch/DiT/blob/main/models.py
"""

from typing import Tuple, Optional
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import SDPBackend

from timm.models.vision_transformer import Mlp
from ..modules.embeddings import RotaryEmbeddingND


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor):
    return x * (1 + scale) + shift


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
            q = self.rope(q)
            k = self.rope(k)

        if self.fused_attn:
            # pylint: disable-next=not-callable, # 使用PyTorch的高效注意力实现
            x = F.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=self.attn_drop.p if self.training else 0.0,
            )
        else:
            # 手动实现注意力机制
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
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
        mlp_ratio=4.0, 
        qkv_bias=False, 
        fused_attn=True,
        act_layer=nn.GELU, 
        norm_layer=nn.LayerNorm
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, f"dim should be divisible by num_heads {dim} {num_heads}."
        self.num_heads = num_heads
        self.fused_attn = fused_attn
        
        # 注意力相关参数
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5
        
        # 线性变换层
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, int(dim * 2), bias=qkv_bias)
        
        # 归一化层
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        
        # MLP层
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            act_layer(),
            nn.Linear(mlp_hidden_dim, dim)
        )

    def forward(self, q, x) -> torch.Tensor:
        """
        参数:
            q: 查询张量, shape (B, n, C)
            x: 上下文张量, shape (B, N, C)
        返回:
            q: 处理后的查询张量, shape (B, n, C)
        """
        # 保存原始查询用于残差连接
        residual = q
        
        # 归一化上下文
        x_norm = self.norm1(x)
        
        # 处理查询
        B, n, C = q.shape
        q_transformed = self.q(q).reshape(B, n, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        # 处理键值对
        B, N, C = x_norm.shape
        kv = self.kv(x_norm).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]  # (batch_size, num_heads, seq_len, feature_dim_per_head)

        # 注意力计算
        if self.fused_attn:
            with torch.nn.attention.sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                attended_q = F.scaled_dot_product_attention(q_transformed, k, v)
        else:
            xattn = (q_transformed @ k.transpose(-2, -1)) * self.scale
            xattn = xattn.softmax(dim=-1)  # (batch_size, num_heads, query_len, seq_len)
            attended_q = xattn @ v

        # 重塑输出
        attended_q = attended_q.transpose(1, 2).reshape(B, n, C)
        
        # 第一个残差连接
        q = residual + attended_q
        
        # 第二个残差连接 + MLP
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
        **block_kwargs: dict,
    ):
        """
        Args:
            hidden_size: Number of features in the hidden layer.
            num_heads: Number of attention heads.
            mlp_ratio: Ratio of hidden layer size in the MLP. None to skip the MLP.
            block_kwargs: Additional arguments to pass to the Attention block.
        """
        super().__init__()

        self.norm1 = AdaLayerNormZero(hidden_size)
        self.attn = Attention(
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

    def initialize_weights(self):
        # Initialize transformer linear layers:
        def _basic_init(module: nn.Module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        self.attn.apply(_basic_init)
        if self.use_mlp:
            self.mlp.apply(_basic_init)

    def forward(self, x: torch.Tensor, c: torch.Tensor):
        """
        Forward pass of the DiT block.
        In original implementation, conditioning is uniform across all tokens in the sequence. Here, we extend it to support token-wise conditioning (e.g. noise level can be different for each token).
        Args:
            x: Input tensor of shape (B, N, C).
            c: Conditioning tensor of shape (B, N, C).
        """
        x, gate_msa = self.norm1(x, c)
        x = x + gate_msa * self.attn(x)
        if self.use_mlp:
            x, gate_mlp = self.norm2(x, c)
            x = x + gate_mlp * self.mlp(x)
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
