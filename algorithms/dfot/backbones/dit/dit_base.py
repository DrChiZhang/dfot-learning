"""
Adapted from https://github.com/facebookresearch/DiT/blob/main/models.py
Extended to support:
- Temporal sequence modeling
- 1D input additionally to 2D spatial input
- Token-wise conditioning
"""

from typing import Literal, Optional, Tuple, Callable, Union
import numpy as np
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from einops import rearrange
from ..modules.embeddings import RotaryEmbedding3D
from .dit_blocks import (
    DiTBlock,
    DITFinalLayer,
)

Variant = Literal["full", "factorized_encoder", "factorized_attention"]
PosEmb = Literal[
    "learned_1d", "sinusoidal_1d", "sinusoidal_3d", "sinusoidal_factorized", "rope_3d"
]


def rearrange_contiguous_many(
    tensors: Tuple[torch.Tensor, ...], *args, **kwargs
) -> Tuple[torch.Tensor, ...]:
    return tuple(rearrange(t, *args, **kwargs).contiguous() for t in tensors)


class DiTBase(nn.Module):
    """
    A DiT base model.
    """

    def __init__(
        self,
        num_patches: Optional[int] = None,
        max_temporal_length: int = 16,
        out_channels: int = 4,
        variant: Variant = "full",
        pos_emb_type: PosEmb = "learned_1d",
        hidden_size: int = 1152,
        depth: int = 28,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        learn_sigma: bool = True,
        use_gradient_checkpointing: bool = False,
    ):
        """
        Args:
            num_patches: Number of patches in the image, None for 1D inputs.
            max_temporal_length: Maximum length of the temporal sequence.
            variant: Variant of the DiT model to use.
                - "full": process all tokens at once
                - "factorized_encoder": alternate between spatial transformer blocks and temporal transformer blocks
                - "factorized_attention": decompose the multi-head attention in the transformer block, compute spatial self-attention and then temporal self-attention
            pos_emb_type: Type of positional embedding to use.
                - "learned_1d": learned 1D positional embeddings
                - "sinusoidal_1d": sinusoidal 1D positional embeddings
                - "sinusoidal_3d": sinusoidal 3D positional embeddings
                - "sinusoidal_factorized": sinusoidal 2D positional embeddings for spatial and 1D for temporal
                - "rope_3d": rope 3D positional embeddings
        """
        super().__init__()
        self._check_args(num_patches, variant, pos_emb_type)
        self.learn_sigma = learn_sigma
        self.out_channels = out_channels * (2 if learn_sigma else 1)
        self.num_patches = num_patches
        self.max_temporal_length = max_temporal_length
        self.max_tokens = self.max_temporal_length * (num_patches or 1)
        self.hidden_size = hidden_size
        self.depth = depth
        self.mlp_ratio = mlp_ratio
        self.variant = variant
        self.pos_emb_type = pos_emb_type
        self.use_gradient_checkpointing = use_gradient_checkpointing

        match self.pos_emb_type:
            case "learned_1d":
                self.pos_emb = SinusoidalPositionalEmbedding(
                    embed_dim=self.hidden_size,
                    shape=(self.max_tokens,),
                    learnable=True,
                )
            case "sinusoidal_1d":
                self.pos_emb = SinusoidalPositionalEmbedding(
                    embed_dim=self.hidden_size,
                    shape=(self.max_tokens,),
                )

            case "sinusoidal_3d":
                self.pos_emb = SinusoidalPositionalEmbedding(
                    embed_dim=self.hidden_size,
                    shape=(
                        self.max_temporal_length,
                        self.spatial_grid_size,
                        self.spatial_grid_size,
                    ),
                )
            case "sinusoidal_factorized":
                self.spatial_pos_emb = SinusoidalPositionalEmbedding(
                    embed_dim=self.hidden_size,
                    shape=(self.spatial_grid_size, self.spatial_grid_size),
                )
                self.temporal_pos_emb = SinusoidalPositionalEmbedding(
                    embed_dim=self.hidden_size,
                    shape=(self.max_temporal_length,),
                )
            case "rope_3d":
                rope = RotaryEmbedding3D(
                    dim=self.hidden_size // num_heads,
                    sizes=(
                        self.max_temporal_length,
                        self.spatial_grid_size,
                        self.spatial_grid_size,
                    ),
                )

        self.blocks = nn.ModuleList(
            [
                DiTBlock(
                    hidden_size=hidden_size,
                    num_heads=num_heads,
                    mlp_ratio=(
                        mlp_ratio if self.variant != "factorized_attention" else None
                    ),
                    rope=rope if self.pos_emb_type == "rope_3d" else None,
                )
                for _ in range(depth)
            ]
        )
        self.temporal_blocks = (
            nn.ModuleList(
                [
                    DiTBlock(
                        hidden_size=hidden_size,
                        num_heads=num_heads,
                        mlp_ratio=mlp_ratio,
                    )
                    for _ in range(depth)
                ]
            )
            if self.is_factorized
            else None
        )

        self.final_layer = DITFinalLayer(hidden_size, self.out_channels)
    """
    @propery 装饰器
    - 将方法转换为只读属性
    - 允许像访问属性一样调用方法：obj.is_factorized 而不是 obj.is_factorized()
    """
    @property
    def is_factorized(self) -> bool:
        return self.variant in {"factorized_encoder", "factorized_attention"}

    @property
    def is_pos_emb_absolute_once(self) -> bool:
        return self.pos_emb_type in {"learned_1d", "sinusoidal_1d", "sinusoidal_3d"}

    @property
    def is_pos_emb_absolute_factorized(self) -> bool:
        return self.pos_emb_type == "sinusoidal_factorized"

    @property
    def spatial_grid_size(self) -> Optional[int]:
        if self.num_patches is None:
            return None
        grid_size = int(self.num_patches**0.5)
        assert (
            grid_size * grid_size == self.num_patches
        ), "num_patches must be a square number"
        return grid_size

    @staticmethod
    def _check_args(num_patches: Optional[int], variant: Variant, pos_emb_type: PosEmb):
        if variant not in {"full", "factorized_encoder", "factorized_attention"}:
            raise ValueError(f"Unknown variant {variant}")
        if pos_emb_type not in {
            "learned_1d",
            "sinusoidal_1d",
            "sinusoidal_3d",
            "sinusoidal_factorized",
            "rope_3d",
        }:
            raise ValueError(f"Unknown positional embedding type {pos_emb_type}")
        if num_patches is None:
            assert (
                variant == "full"
            ), "For 1D inputs, factorized variants are not supported"
            assert pos_emb_type in {
                "learned_1d",
                "sinusoidal_1d",
            }, "For 1D inputs, only 1D positional embeddings are supported"

        if pos_emb_type == "rope_3d":
            assert variant == "full", "Rope3D is only supported with full variant"

    def checkpoint(self, module: nn.Module, *args):
        if self.use_gradient_checkpointing:
            return checkpoint(module, *args, use_reentrant=False)
        return module(*args)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the DiTBase model.
        Args:
            x: Input tensor of shape (B, N, C). 输入张量，形状为(批次大小, 序列长度, 通道数)
            c: Conditioning tensor of shape (B, N, C). 条件张量，形状同输入
        Returns:
            Output tensor of shape (B, N, OC). 输出张量，形状为(批次大小, 序列长度, 输出通道数)
        """
        # 初始化图像数据为None，用于后续的条件处理
        x_img = None
        
        # 图像-视频联合训练的分割逻辑
        if x.size(1) > self.max_tokens:
            # 非训练模式或未设置num_patches时抛出错误
            if not self.training or self.num_patches is None:
                raise ValueError(
                    f"Input sequence length {x.size(1)} exceeds the maximum length {self.max_tokens}"
                )
            else:  
                # 图像-视频联合训练模式：分割输入数据
                # 计算视频数据的结束位置（时间长度×每帧的patch数）
                video_end = self.max_temporal_length * self.num_patches
                
                # 分割输入和条件张量：前部分为视频，后部分为图像
                x, x_img, c, c_img = (
                    x[:, :video_end],  # 视频序列数据   （B, (T P), C)
                    x[:, video_end:],  # 图像数据       (B, P, C)
                    c[:, :video_end],  # 视频条件数据
                    c[:, video_end:],  # 图像条件数据
                )
                
                # 对图像数据进行维度重排：将批次和时间维度合并
                # 从 (批次, 时间×patch, 通道) 转换为 (批次×时间, patch, 通道)
                x_img, c_img = rearrange_contiguous_many(
                    (x_img, c_img), "b (t p) c -> (b t) p c", p=self.num_patches
                )  # 重排后，图像数据被视为长度为1的序列

        # 获取批次大小信息
        seq_batch_size = x.size(0)  # 序列数据的批次大小
        img_batch_size = x_img.size(0) if x_img is not None else None  # 图像数据的批次大小（可能为None）

        # 创建状态字典，统一管理序列和图像数据的处理状态
        seq_states = {"x": x, "c": c, "batch_size": seq_batch_size}  # 序列状态
        img_states = (
            {"x": x_img, "c": c_img, "batch_size": img_batch_size}
            if x_img is not None  # 只有当图像数据存在时才创建图像状态
            else None
        )

        def execute_in_parallel(
            fn: Callable[
                [torch.Tensor, torch.Tensor, int],
                Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor],
            ]
        ):
            """
            在序列和图像张量上并行执行函数
            对seq_states和img_states（如果存在）应用相同的函数操作
            """
            # 处理序列数据
            seq_result = fn(seq_states["x"], seq_states["c"], seq_states["batch_size"])
            # 根据函数返回类型更新状态：可能是元组（更新x和c）或单个张量（只更新x）
            if isinstance(seq_result, tuple):
                seq_states["x"], seq_states["c"] = seq_result
            else:
                seq_states["x"] = seq_result
                
            # 如果图像状态存在，同样处理图像数据
            if img_states is not None:
                img_result = fn(
                    img_states["x"], img_states["c"], img_states["batch_size"]
                )
                if isinstance(img_result, tuple):
                    img_states["x"], img_states["c"] = img_result
                else:
                    img_states["x"] = img_result

        # ==================== 位置编码处理阶段 ====================

        # 一次性绝对位置编码：直接应用位置编码到输入
        if self.is_pos_emb_absolute_once:
            execute_in_parallel(lambda x, c, batch_size: self.pos_emb(x))
            
        # 因子化绝对位置编码：分别应用空间和时间位置编码
        if self.is_pos_emb_absolute_factorized and not self.is_factorized:

            def add_pos_emb(
                x: torch.Tensor, _: torch.Tensor, batch_size: int
            ) -> torch.Tensor:
                """应用因子化位置编码：先空间编码，再时间编码"""
                # 重排：从 (批次, 时间×空间patch, 通道) 到 (批次×时间, 空间patch, 通道)
                x = rearrange(x, "b (t p) c -> (b t) p c", p=self.num_patches)
                # 应用空间位置编码
                x = self.spatial_pos_emb(x)
                # 重排：从 (批次×时间, 空间patch, 通道) 到 (批次×空间patch, 时间, 通道)
                x = rearrange(x, "(b t) p c -> (b p) t c", b=batch_size)
                # 应用时间位置编码
                x = self.temporal_pos_emb(x)
                # 重排：恢复原始形状 (批次, 时间×空间patch, 通道)
                x = rearrange(x, "(b p) t c -> b (t p) c", b=batch_size)
                return x

            execute_in_parallel(add_pos_emb)

        # ==================== 因子化预处理阶段 ====================

        # 如果启用因子化处理，进行初始维度重排
        if self.is_factorized:
            # 将输入重排为空间块格式：从序列格式转换为空间块格式
            execute_in_parallel(
                lambda x, c, batch_size: rearrange_contiguous_many(
                    (x, c), "b (t p) c -> (b t) p c", p=self.num_patches
                )
            )
            # 如果使用因子化绝对位置编码，应用空间位置编码
            if self.is_pos_emb_absolute_factorized:
                execute_in_parallel(lambda x, c, batch_size: self.spatial_pos_emb(x))

        # ==================== 主要块处理阶段 ====================

        # 遍历所有块（空间块和时间块配对）
        for i, (block, temporal_block) in enumerate(
            zip(self.blocks, self.temporal_blocks or [None for _ in range(self.depth)])
        ):
            # 应用空间Transformer块（使用梯度检查点以节省内存）
            execute_in_parallel(lambda x, c, batch_size: self.checkpoint(block, x, c))

            # 如果启用因子化处理，进行时间维度处理
            if self.is_factorized:
                # 重排：从空间维度到时间维度
                execute_in_parallel(
                    lambda x, c, batch_size: rearrange_contiguous_many(
                        (x, c), "(b t) p c -> (b p) t c", b=batch_size
                    )
                )
                
                # 仅在第一个块且使用正弦因子化位置编码时应用时间位置编码
                if i == 0 and self.pos_emb_type == "sinusoidal_factorized":
                    execute_in_parallel(
                        lambda x, c, batch_size: self.temporal_pos_emb(x)
                    )
                    
                # 应用时间Transformer块
                execute_in_parallel(
                    lambda x, c, batch_size: self.checkpoint(temporal_block, x, c)
                )
                
                # 重排：从时间维度回到空间维度
                execute_in_parallel(
                    lambda x, c, batch_size: rearrange_contiguous_many(
                        (x, c), "(b p) t c -> (b t) p c", b=batch_size
                    )
                )
                
        # ==================== 后处理阶段 ====================

        # 如果启用因子化处理，进行最终维度重排
        if self.is_factorized:
            # 从空间块格式恢复为序列格式
            execute_in_parallel(
                lambda x, c, batch_size: rearrange_contiguous_many(
                    (x, c), "(b t) p c -> b (t p) c", b=batch_size
                )
            )

        # 应用最终输出层
        execute_in_parallel(lambda x, c, batch_size: self.final_layer(x, c))

        # ==================== 输出合并阶段 ====================

        # 提取处理后的序列数据
        x = seq_states["x"]
        # 提取处理后的图像数据（如果存在）
        x_img = img_states["x"] if img_states is not None else None
        
        # 如果存在图像数据，进行维度重排并合并到输出中
        if x_img is not None:
            # 重排图像数据：从 (批次×时间, patch, 通道) 回到 (批次, 时间×patch, 通道)
            x_img = rearrange(x_img, "(b t) p c -> b (t p) c", b=seq_batch_size)
            # 将图像数据连接到序列数据后面
            x = torch.cat([x, x_img], dim=1)
            
        # 返回最终的输出张量
        return x


class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, embed_dim: int, shape: Tuple[int, ...], learnable: bool = False):
        super().__init__()
        if learnable:
            max_tokens = np.prod(shape)
            self.pos_emb = nn.Parameter(
                torch.zeros(1, max_tokens, embed_dim).normal_(std=0.02),
                requires_grad=True,
            )

        else:
            self.register_buffer(
                "pos_emb",
                torch.from_numpy(get_nd_sincos_pos_embed(embed_dim, shape))
                .float()
                .unsqueeze(0),
                persistent=False,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.shape[1]
        return x + self.pos_emb[:, :seq_len]


def get_nd_sincos_pos_embed(
    embed_dim: int,
    shape: Tuple[int, ...],
) -> np.ndarray:
    """
    Get n-dimensional sinusoidal positional embeddings.
    Args:
        embed_dim: Embedding dimension.
        shape: Shape of the input tensor.
    Returns:
        Positional embeddings with shape (shape_flattened, embed_dim).
    """
    assert embed_dim % (2 * len(shape)) == 0
    grid = np.meshgrid(*[np.arange(s, dtype=np.float32) for s in shape])
    grid = np.stack(grid, axis=0)  # (ndim, *shape)
    return np.concatenate(
        [
            get_1d_sincos_pos_embed_from_grid(embed_dim // len(shape), grid[i])
            for i in range(len(shape))
        ],
        axis=1,
    )


def get_1d_sincos_pos_embed_from_grid(embed_dim: int, pos: np.ndarray) -> np.ndarray:
    """
    Args:
        embed_dim: Embedding dimension.
        pos: Position tensor of shape (...).
    Returns:
        Positional embeddings with shape (-1, embed_dim).
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb
