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


class MMDiTBase(nn.Module):
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
        conditioning_scale: float = 1.0,
    ):
        """
        Args:
            num_patches: Number of patches in the image, None for 1D inputs.
            max_temporal_length: Maximum length of the temporal sequence.
            variant: Variant of the DiT model to use.
                - "full": process all tokens at once
            pos_emb_type: Type of positional embedding to use.
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
        self.conditioning_scale = conditioning_scale

        match self.pos_emb_type:
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
                    is_conditional=True,
                    conditioning_scale=self.conditioning_scale,
                )
                for _ in range(depth)
            ]
        )
        self.final_layer = DITFinalLayer(hidden_size, self.out_channels)

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
        assert variant == 'full' and pos_emb_type == 'rope_3d', "Only full variant with rope_3d is supported in this implementation."

    def checkpoint(self, module: nn.Module, *args):
        if self.use_gradient_checkpointing:
            return checkpoint(module, *args, use_reentrant=False)
        return module(*args)

    def forward(self, x: torch.Tensor, c: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the DiTBase model.
        Args:
            x: Input tensor of shape (B, N, C).
            c: Conditioning tensor of shape (B, N, C).
            cond: pose or control conditioning tensor of shape (B, N, C).
        Returns:
            Output tensor of shape (B, N, OC).
        """
        x_img = None
        if x.size(1) > self.max_tokens:
            if not self.training or self.num_patches is None:
                raise ValueError(
                    f"Input sequence length {x.size(1)} exceeds the maximum length {self.max_tokens}"
                )

            else:  # image-video joint training
                video_end = self.max_temporal_length * self.num_patches
                x, x_img, c, c_img, cond, cond_img = (
                    x[:, :video_end],
                    x[:, video_end:],
                    c[:, :video_end],
                    c[:, video_end:],
                    cond[:, :video_end],
                    cond[:, video_end:],
                )
                x_img, c_img, cond_img = rearrange_contiguous_many(
                    (x_img, c_img, cond_img), "b (t p) c -> (b t) p c", p=self.num_patches
                )  # as if they are sequences of length 1
        seq_batch_size = x.size(0)
        img_batch_size = x_img.size(0) if x_img is not None else None

        x = torch.cat([x, cond], dim=1)
        c = torch.cat([c, c], dim=1)
        x_img = torch.cat([x_img, cond_img], dim=1) if x_img is not None else None
        c_img = torch.cat([c_img, c_img], dim=1) if x_img is not None else None
        
        seq_states = {"x": x, "c": c, "batch_size": seq_batch_size}
        img_states = (
            {"x": x_img, "c": c_img, "batch_size": img_batch_size}
            if x_img is not None
            else None
        )

        def execute_in_parallel(
            fn: Callable[
                [torch.Tensor, torch.Tensor, int],
                Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor],
            ]
        ):
            """execute a function in parallel on the sequence and image tensors"""
            seq_result = fn(seq_states["x"], seq_states["c"], seq_states["batch_size"])
            if isinstance(seq_result, tuple):
                seq_states["x"], seq_states["c"] = seq_result
            else:
                seq_states["x"] = seq_result
            if img_states is not None:
                img_result = fn(
                    img_states["x"], img_states["c"], img_states["batch_size"]
                )
                if isinstance(img_result, tuple):
                    img_states["x"], img_states["c"] = img_result
                else:
                    img_states["x"] = img_result

        for i, block in enumerate(self.blocks):
            execute_in_parallel(lambda x, c, batch_size: self.checkpoint(block, x, c))

        execute_in_parallel(lambda x, c, batch_size: self.final_layer(x, c))

        x = seq_states["x"].chunk(2, dim=1)[0]  # remove conditioning part
        x_img = img_states["x"].chunk(2, dim=1)[0] if img_states is not None else None
        if x_img is not None:
            x_img = rearrange(x_img, "(b t) p c -> b (t p) c", b=seq_batch_size)
            x = torch.cat([x, x_img], dim=1)
        return x



