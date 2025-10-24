# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin  # used for model hub

from einops import rearrange
from external.vggt.models.aggregator import Aggregator
from external.vggt.heads.camera_head import CameraHead
from external.vggt.heads.dpt_head import DPTHead
from external.vggt.heads.track_head import TrackHead
import torch.nn.functional as F

class VGGT(nn.Module, PyTorchModelHubMixin):
    def __init__(self, img_size=518, patch_size=14, embed_dim=1024):
        super().__init__()

        self.aggregator = Aggregator(img_size=img_size, patch_size=patch_size, embed_dim=embed_dim)
        self.camera_head = CameraHead(dim_in=2 * embed_dim)
        self.point_head = DPTHead(dim_in=2 * embed_dim, output_dim=4, activation="inv_log", conf_activation="expp1")
        self.depth_head = DPTHead(dim_in=2 * embed_dim, output_dim=2, activation="exp", conf_activation="expp1")
        self.track_head = TrackHead(dim_in=2 * embed_dim, patch_size=patch_size)

    def forward(
        self,
        images: torch.Tensor,
        query_points: torch.Tensor = None,
    ):
        """
        Forward pass of the VGGT model.

        Args:
            images (torch.Tensor): Input images with shape [S, 3, H, W] or [B, S, 3, H, W], in range [0, 1].
                B: batch size, S: sequence length, 3: RGB channels, H: height, W: width
            query_points (torch.Tensor, optional): Query points for tracking, in pixel coordinates.
                Shape: [N, 2] or [B, N, 2], where N is the number of query points.
                Default: None

        Returns:
            dict: A dictionary containing the following predictions:
                - pose_enc (torch.Tensor): Camera pose encoding with shape [B, S, 9] (from the last iteration)
                - depth (torch.Tensor): Predicted depth maps with shape [B, S, H, W, 1]
                - depth_conf (torch.Tensor): Confidence scores for depth predictions with shape [B, S, H, W]
                - world_points (torch.Tensor): 3D world coordinates for each pixel with shape [B, S, H, W, 3]
                - world_points_conf (torch.Tensor): Confidence scores for world points with shape [B, S, H, W]
                - images (torch.Tensor): Original input images, preserved for visualization

                If query_points is provided, also includes:
                - track (torch.Tensor): Point tracks with shape [B, S, N, 2] (from the last iteration), in pixel coordinates
                - vis (torch.Tensor): Visibility scores for tracked points with shape [B, S, N]
                - conf (torch.Tensor): Confidence scores for tracked points with shape [B, S, N]
        """

        # If without batch dimension, add it
        if len(images.shape) == 4:
            images = images.unsqueeze(0)
        if query_points is not None and len(query_points.shape) == 2:
            query_points = query_points.unsqueeze(0)

        aggregated_tokens_list, patch_start_idx = self.aggregator(images)
        predictions = {}

        with torch.amp.autocast('cuda',enabled=False):
            if self.camera_head is not None:
                pose_enc_list = self.camera_head(aggregated_tokens_list)
                predictions["pose_enc"] = pose_enc_list[-1]  # pose encoding of the last iteration

            if self.depth_head is not None:
                # print(f"VGGT depth head input aggregated_tokens_list={aggregated_tokens_list[0].shape}, patch_start_idx={patch_start_idx}")
                depth, depth_conf = self.depth_head(
                    aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx
                )
                predictions["depth"] = depth
                predictions["depth_conf"] = depth_conf

            if self.point_head is not None:
                pts3d, pts3d_conf = self.point_head(
                    aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx
                )
                predictions["world_points"] = pts3d
                predictions["world_points_conf"] = pts3d_conf

        if self.track_head is not None and query_points is not None:
            track_list, vis, conf = self.track_head(
                aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx, query_points=query_points
            )
            predictions["track"] = track_list[-1]  # track of the last iteration
            predictions["vis"] = vis
            predictions["conf"] = conf

        predictions["images"] = images

        return predictions
    
    def token_list_to_predictions(self, aggregated_tokens_list, images: torch.Tensor, query_points: torch.Tensor = None,patch_start_idx=None):
        predictions = {}
        with torch.cuda.amp.autocast(enabled=False):
            if self.camera_head is not None:
                pose_enc_list = self.camera_head(aggregated_tokens_list)
                predictions["pose_enc"] = pose_enc_list[-1]  # pose encoding of the last iteration

            if self.depth_head is not None:
                depth, depth_conf = self.depth_head(
                    aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx
                )
                predictions["depth"] = depth
                predictions["depth_conf"] = depth_conf

            if self.point_head is not None:
                pts3d, pts3d_conf = self.point_head(
                    aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx
                )
                predictions["world_points"] = pts3d
                predictions["world_points_conf"] = pts3d_conf

        if self.track_head is not None and query_points is not None:
            track_list, vis, conf = self.track_head(
                aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx, query_points=query_points
            )
            predictions["track"] = track_list[-1]  # track of the last iteration
            predictions["vis"] = vis
            predictions["conf"] = conf

        predictions["images"] = images
        return predictions 

    def shortcut_forward(self, images: torch.Tensor):
        """
        Forward pass of the VGGT model with a shortcut for faster inference.
        """
        # If without batch dimension, add it
        if len(images.shape) == 4:
            images = images.unsqueeze(0)
            
        aggregated_tokens_list, patch_start_idx = self.aggregator(images)
        return aggregated_tokens_list, patch_start_idx
    
def forward_image_features(self, images: torch.Tensor):
    """
    Forward pass of the VGGT model to extract features. for image level alignment 
    for imagenet training
    """
    logger = logging.getLogger(__name__)
    
    # 开始计时（用于性能分析）
    start_time = time.time() if logger.isEnabledFor(logging.DEBUG) else None
    
    try:
        # 记录输入图像信息
        logger.debug(f"[START] Input images shape: {images.shape}, dtype: {images.dtype}, device: {images.device}")
        
        # If without batch dimension, add it
        original_shape = images.shape
        if len(images.shape) == 4:
            logger.info(f"Adding batch dimension. {original_shape} -> {(*images.shape[:-3], 1, *images.shape[-3:])}")
            images = images.unsqueeze(0)
            logger.debug(f"After unsqueeze: {images.shape}")

        # 聚合器前向传播
        logger.debug(f"Calling aggregator with shape: {images.shape}")
        aggregated_tokens_list, patch_start_idx = self.aggregator(images)
        
        # 详细记录聚合器输出
        logger.debug(f"Aggregator returned {len(aggregated_tokens_list)} token lists")
        for i, token in enumerate(aggregated_tokens_list):
            logger.debug(f"  Token list {i}: shape {token.shape}")
        
        aggregated_tokens = torch.stack(aggregated_tokens_list)  # [B, S, H*W, D]
        logger.debug(f"Stacked tokens: {aggregated_tokens.shape}")
        
        # 维度重排
        logger.debug(f"Rearranging: 'c t b h w -> b t c h w'")
        target_feats = rearrange(aggregated_tokens,'c t b h w -> b t c h w')
        logger.debug(f"After rearrange: {target_feats.shape}")
        
        # 验证形状
        seq_dim = target_feats.shape[1]
        logger.debug(f"Sequence dimension check: {seq_dim} == 1")
        if seq_dim != 1:
            logger.warning(f"Unexpected sequence dimension: {seq_dim}, expected 1")
        
        assert target_feats.shape[1]==1,f"target_feats {target_feats.shape} should be [B, S, 1, H, W]"
        
        # 压缩维度
        target_feats = target_feats.squeeze(1)  # [B, S, H, W]
        logger.debug(f"After squeeze(1): {target_feats.shape}")
        
        # 插值操作
        target_size = [256, 768]
        logger.info(f"Interpolating from {target_feats.shape[-2:]} to {target_size}")
        
        # 记录插值前统计
        if logger.isEnabledFor(logging.DEBUG):
            stats_msg = (
                f"Pre-interpolation stats - min: {target_feats.min().item():.4f}, "
                f"max: {target_feats.max().item():.4f}, mean: {target_feats.mean().item():.4f}"
            )
            logger.debug(stats_msg)
        
        target_feats = nn.functional.interpolate(
            target_feats, 
            size=target_size, 
            mode='bilinear', 
            align_corners=False
        )
        
        # 记录最终结果
        logger.info(f"[COMPLETE] Final features shape: {target_feats.shape}")
        
        # 检查输出质量
        if torch.isnan(target_feats).any():
            nan_count = torch.isnan(target_feats).sum().item()
            logger.error(f"Output contains {nan_count} NaN values!")
        
        if torch.isinf(target_feats).any():
            inf_count = torch.isinf(target_feats).sum().item()
            logger.error(f"Output contains {inf_count} Inf values!")
        
        # 记录性能信息
        if start_time:
            duration = time.time() - start_time
            logger.debug(f"forward_image_features completed in {duration:.4f} seconds")
        
        return target_feats
        
    except Exception as e:
        logger.error(f"Error in forward_image_features: {str(e)}", exc_info=True)
        raise