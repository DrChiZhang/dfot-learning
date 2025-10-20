
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
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer)

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

class DualCrossAttention(nn.Module):
    """
    双注意力模块：先自注意力，后交叉注意力
    自注意力：x -> x
    交叉注意力：condition -> x (condition作为q,k，x作为v)
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
        rope: Optional[nn.Module] = None,
        fused_attn: bool = True,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, f"dim should be divisible by num_heads {dim} {num_heads}."
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.fused_attn = fused_attn

        # 自注意力的QKV投影
        self.self_qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        
        # 交叉注意力的QKV投影
        # 注意：交叉注意力中，condition作为q,k，x作为v
        self.cross_q = nn.Linear(dim, dim, bias=qkv_bias)  # condition -> q
        self.cross_k = nn.Linear(dim, dim, bias=qkv_bias)  # condition -> k  
        self.cross_v = nn.Linear(dim, dim, bias=qkv_bias)  # x -> v

        # 归一化层
        self.self_q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.self_k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.cross_q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.cross_k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        
        # Dropout层
        self.attn_drop = nn.Dropout(attn_drop)
        
        # 输出投影
        self.self_proj = nn.Linear(dim, dim)
        self.cross_proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        # RoPE位置编码
        self.rope = rope

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量 [B, N, C]
            condition: 条件张量 [B, M, C]
            
        Returns:
            输出张量 [B, N, C]
        """
        B, N, C = x.shape
        M = condition.shape[1]  # 条件序列长度
        
        # ========== 第一阶段：自注意力 (x -> x) ==========
        
        # QKV投影和重塑
        self_qkv = (
            self.self_qkv(x)
            .reshape(B, N, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        self_q, self_k, self_v = self_qkv.unbind(0)
        self_q, self_k = self.self_q_norm(self_q), self.self_k_norm(self_k)
        
        # 应用RoPE
        if self.rope is not None:
            self_q = self.rope(self_q)
            self_k = self.rope(self_k)
        
        # 自注意力计算
        if self.fused_attn:
            self_attn_out = F.scaled_dot_product_attention(
                self_q, self_k, self_v,
                dropout_p=self.attn_drop.p if self.training else 0.0,
            )
        else:
            self_q = self_q * self.scale
            self_attn = self_q @ self_k.transpose(-2, -1)
            self_attn = self_attn.softmax(dim=-1)
            self_attn = self.attn_drop(self_attn)
            self_attn_out = self_attn @ self_v
        
        # 自注意力输出处理
        self_attn_out = self_attn_out.transpose(1, 2).reshape(B, N, C)
        self_attn_out = self.self_proj(self_attn_out)
        self_attn_out = self.proj_drop(self_attn_out)
        
        # ========== 第二阶段：交叉注意力 (condition -> x) ==========
        
        # 交叉注意力QKV投影
        # q, k 来自 condition; v 来自自注意力输出
        cross_q = (
            self.cross_q(condition)
            .reshape(B, M, self.num_heads, self.head_dim)
            .permute(0, 2, 1, 3)
        )  # [B, num_heads, M, head_dim]
        
        cross_k = (
            self.cross_k(condition) 
            .reshape(B, M, self.num_heads, self.head_dim)
            .permute(0, 2, 1, 3)
        )  # [B, num_heads, M, head_dim]
        
        cross_v = (
            self.cross_v(self_attn_out)
            .reshape(B, N, self.num_heads, self.head_dim)
            .permute(0, 2, 1, 3)
        )  # [B, num_heads, N, head_dim]
        
        # 归一化
        cross_q, cross_k = self.cross_q_norm(cross_q), self.cross_k_norm(cross_k)
        
        # 应用RoPE
        if self.rope is not None:
            cross_q = self.rope(cross_q)
            cross_k = self.rope(cross_k)
        
        # 交叉注意力计算
        if self.fused_attn:
            cross_attn_out = F.scaled_dot_product_attention(
                cross_q, cross_k, cross_v,
                dropout_p=self.attn_drop.p if self.training else 0.0,
            )
        else:
            cross_q = cross_q * self.scale
            cross_attn = cross_q @ cross_k.transpose(-2, -1)
            cross_attn = cross_attn.softmax(dim=-1)
            cross_attn = self.attn_drop(cross_attn)
            cross_attn_out = cross_attn @ cross_v
        
        # 交叉注意力输出处理
        cross_attn_out = cross_attn_out.transpose(1, 2).reshape(B, N, C)
        cross_attn_out = self.cross_proj(cross_attn_out)
        cross_attn_out = self.proj_drop(cross_attn_out)
        
        return cross_attn_out

class DualCrossAttention(nn.Module):
    """
    双注意力模块：先自注意力，后交叉注意力
    自注意力：x -> x
    交叉注意力：x -> condition (x作为q，condition作为k,v)
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
        rope: Optional[nn.Module] = None,
        fused_attn: bool = True,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, f"dim should be divisible by num_heads {dim} {num_heads}."
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.fused_attn = fused_attn

        # 自注意力的QKV投影
        self.self_qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        
        # 交叉注意力的QKV投影
        # 注意：交叉注意力中，x作为q，condition作为k,v
        self.cross_q = nn.Linear(dim, dim, bias=qkv_bias)  # x -> q
        self.cross_k = nn.Linear(dim, dim, bias=qkv_bias)  # condition -> k  
        self.cross_v = nn.Linear(dim, dim, bias=qkv_bias)  # condition -> v

        # 归一化层
        self.self_q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.self_k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.cross_q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.cross_k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        
        # Dropout层
        self.attn_drop = nn.Dropout(attn_drop)
        
        # 输出投影
        self.self_proj = nn.Linear(dim, dim)
        self.cross_proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        # RoPE位置编码
        self.rope = rope

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量 [B, N, C]
            condition: 条件张量 [B, M, C]
            
        Returns:
            输出张量 [B, N, C]
        """
        B, N, C = x.shape
        M = condition.shape[1]  # 条件序列长度
        
        # ========== 第一阶段：自注意力 (x -> x) ==========
        
        # QKV投影和重塑
        self_qkv = (
            self.self_qkv(x)
            .reshape(B, N, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        self_q, self_k, self_v = self_qkv.unbind(0)
        self_q, self_k = self.self_q_norm(self_q), self.self_k_norm(self_k)
        
        # 应用RoPE
        if self.rope is not None:
            self_q = self.rope(self_q)
            self_k = self.rope(self_k)
        
        # 自注意力计算
        if self.fused_attn:
            self_attn_out = F.scaled_dot_product_attention(
                self_q, self_k, self_v,
                dropout_p=self.attn_drop.p if self.training else 0.0,
            )
        else:
            self_q = self_q * self.scale
            self_attn = self_q @ self_k.transpose(-2, -1)
            self_attn = self_attn.softmax(dim=-1)
            self_attn = self.attn_drop(self_attn)
            self_attn_out = self_attn @ self_v
        
        # 自注意力输出处理
        self_attn_out = self_attn_out.transpose(1, 2).reshape(B, N, C)
        self_attn_out = self.self_proj(self_attn_out)
        self_attn_out = self.proj_drop(self_attn_out)
        
        # ========== 第二阶段：交叉注意力 (x -> condition) ==========
        
        # 交叉注意力QKV投影
        # q 来自自注意力输出; k, v 来自 condition
        cross_q = (
            self.cross_q(self_attn_out)
            .reshape(B, N, self.num_heads, self.head_dim)
            .permute(0, 2, 1, 3)
        )  # [B, num_heads, N, head_dim]
        
        cross_k = (
            self.cross_k(condition) 
            .reshape(B, M, self.num_heads, self.head_dim)
            .permute(0, 2, 1, 3)
        )  # [B, num_heads, M, head_dim]
        
        cross_v = (
            self.cross_v(condition)
            .reshape(B, M, self.num_heads, self.head_dim)
            .permute(0, 2, 1, 3)
        )  # [B, num_heads, M, head_dim]
        
        # 归一化
        cross_q, cross_k = self.cross_q_norm(cross_q), self.cross_k_norm(cross_k)
        
        # 应用RoPE
        if self.rope is not None:
            cross_q = self.rope(cross_q)
            cross_k = self.rope(cross_k)
        
        # 交叉注意力计算
        if self.fused_attn:
            cross_attn_out = F.scaled_dot_product_attention(
                cross_q, cross_k, cross_v,
                dropout_p=self.attn_drop.p if self.training else 0.0,
            )
        else:
            cross_q = cross_q * self.scale
            cross_attn = cross_q @ cross_k.transpose(-2, -1)
            cross_attn = cross_attn.softmax(dim=-1)
            cross_attn = self.attn_drop(cross_attn)
            cross_attn_out = cross_attn @ cross_v
        
        # 交叉注意力输出处理
        cross_attn_out = cross_attn_out.transpose(1, 2).reshape(B, N, C)
        cross_attn_out = self.cross_proj(cross_attn_out)
        cross_attn_out = self.proj_drop(cross_attn_out)
        
        return cross_attn_out
    
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class PluckerRayConditionedAttention(nn.Module):
    """
    专门为图像特征+Plucker射线相机条件设计的注意力模块
    Plucker射线: [B, H, W, 6] 表示相机光线
    """
    
    def __init__(
        self,
        dim: int,                    # 图像特征维度
        num_heads: int = 8,
        qkv_bias: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        # Plucker射线特定参数
        plucker_dim: int = 6,        # Plucker射线维度
        plucker_proj_dim: int = 64,  # Plucker投影维度
        use_plucker_proj: bool = True,
        plucker_embed_type: str = "linear",  # "linear" 或 "mlp"
    ) -> None:
        super().__init__()
        
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        
        # 自注意力
        self.self_qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.self_proj = nn.Linear(dim, dim)
        
        # 交叉注意力 - 图像特征作为q，Plucker射线作为k,v
        self.cross_q = nn.Linear(dim, dim, bias=qkv_bias)
        
        # Plucker射线处理
        self.plucker_dim = plucker_dim
        self.use_plucker_proj = use_plucker_proj
        
        if use_plucker_proj:
            if plucker_embed_type == "linear":
                # 简单线性投影
                self.plucker_proj = nn.Linear(plucker_dim, plucker_proj_dim)
            else:
                # MLP投影，更好地捕捉Plucker射线的几何特性
                self.plucker_proj = nn.Sequential(
                    nn.Linear(plucker_dim, plucker_proj_dim),
                    nn.ReLU(),
                    nn.Linear(plucker_proj_dim, plucker_proj_dim),
                    nn.ReLU(),
                    nn.Linear(plucker_proj_dim, plucker_proj_dim)
                )
            
            # 将投影后的Plucker特征映射到注意力维度
            self.plucker_k_proj = nn.Linear(plucker_proj_dim, dim, bias=qkv_bias)
            self.plucker_v_proj = nn.Linear(plucker_proj_dim, dim, bias=qkv_bias)
        else:
            # 直接映射
            self.plucker_k_proj = nn.Linear(plucker_dim, dim, bias=qkv_bias)
            self.plucker_v_proj = nn.Linear(plucker_dim, dim, bias=qkv_bias)
        
        self.cross_k = nn.Linear(dim, dim, bias=qkv_bias)
        self.cross_v = nn.Linear(dim, dim, bias=qkv_bias)
        self.cross_proj = nn.Linear(dim, dim)
        
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)
        
        # 位置编码相关（可选）
        self.use_ray_direction = True

    def forward(self, x: torch.Tensor, plucker_rays: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 图像特征 [B, N, C] 或 [B, H, W, C]
            plucker_rays: Plucker射线 [B, H, W, 6]
            
        Returns:
            输出张量 [B, N, C]
        """
        # 处理输入形状
        if x.dim() == 4:  # [B, H, W, C]
            B, H, W, C = x.shape
            x = x.reshape(B, H*W, C)
            N = H * W
        else:  # [B, N, C]
            B, N, C = x.shape
            H = W = int(N ** 0.5)  # 假设是正方形特征图
        
        # ===== 自注意力 =====
        self_qkv = self.self_qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        self_q, self_k, self_v = self_qkv.unbind(0)
        
        # 自注意力计算
        self_attn_out = F.scaled_dot_product_attention(
            self_q, self_k, self_v,
            dropout_p=self.attn_drop.p if self.training else 0.0,
        )
        self_attn_out = self_attn_out.transpose(1, 2).reshape(B, N, C)
        self_attn_out = self.proj_drop(self.self_proj(self_attn_out))
        
        # ===== 处理Plucker射线 =====
        # 确保Plucker射线与图像特征空间对齐
        if plucker_rays.shape[1:3] != (H, W):
            # 如果空间尺寸不匹配，进行插值
            plucker_rays = F.interpolate(
                plucker_rays.permute(0, 3, 1, 2),  # [B, 6, H_orig, W_orig]
                size=(H, W),
                mode='bilinear',
                align_corners=False
            ).permute(0, 2, 3, 1)  # [B, H, W, 6]
        
        # 展平Plucker射线
        plucker_flat = plucker_rays.reshape(B, H*W, self.plucker_dim)  # [B, N, 6]
        
        # 投影Plucker射线
        if self.use_plucker_proj:
            plucker_projected = self.plucker_proj(plucker_flat)  # [B, N, plucker_proj_dim]
        else:
            plucker_projected = plucker_flat
        
        # ===== 交叉注意力 =====
        # Q来自自注意力输出
        cross_q = self.cross_q(self_attn_out).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        
        # K,V来自投影后的Plucker射线
        plucker_k = self.plucker_k_proj(plucker_projected).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        plucker_v = self.plucker_v_proj(plucker_projected).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        
        # 交叉注意力计算
        cross_attn_out = F.scaled_dot_product_attention(
            cross_q, plucker_k, plucker_v,
            dropout_p=self.attn_drop.p if self.training else 0.0,
        )
        cross_attn_out = cross_attn_out.transpose(1, 2).reshape(B, N, C)
        cross_attn_out = self.proj_drop(self.cross_proj(cross_attn_out))
        
        return cross_attn_out

class MultiScalePluckerAttention(nn.Module):
    """
    多尺度Plucker射线注意力，处理不同分辨率的特征
    """
    
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        plucker_dim: int = 6,
        num_scales: int = 3,
        **kwargs
    ):
        super().__init__()
        
        self.num_scales = num_scales
        self.attentions = nn.ModuleList([
            PluckerRayConditionedAttention(
                dim=dim,
                num_heads=num_heads,
                plucker_dim=plucker_dim,
                **kwargs
            ) for _ in range(num_scales)
        ])
        
        # 多尺度融合
        self.fusion = nn.Sequential(
            nn.Linear(dim * num_scales, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )
    
    def forward(self, multi_scale_features, plucker_rays):
        """
        Args:
            multi_scale_features: 多尺度特征列表 [feature1, feature2, ...]
            plucker_rays: Plucker射线 [B, H, W, 6]
        """
        outputs = []
        
        for i, feature in enumerate(multi_scale_features):
            # 对每个尺度应用注意力
            output = self.attentions[i](feature, plucker_rays)
            outputs.append(output)
        
        # 融合多尺度输出
        if self.num_scales > 1:
            # 上采样所有输出到最大分辨率
            target_size = outputs[0].shape[1]  # 最大分辨率
            upsampled_outputs = []
            
            for output in outputs:
                if output.shape[1] != target_size:
                    # 需要上采样
                    H = W = int(output.shape[1] ** 0.5)
                    output_2d = output.reshape(output.shape[0], H, W, -1).permute(0, 3, 1, 2)
                    output_upsampled = F.interpolate(
                        output_2d, 
                        size=(int(target_size ** 0.5), int(target_size ** 0.5)),
                        mode='bilinear',
                        align_corners=False
                    ).permute(0, 2, 3, 1).reshape(output.shape[0], target_size, -1)
                    upsampled_outputs.append(output_upsampled)
                else:
                    upsampled_outputs.append(output)
            
            # 拼接并融合
            fused = torch.cat(upsampled_outputs, dim=-1)
            final_output = self.fusion(fused)
        else:
            final_output = outputs[0]
        
        return final_output
    
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, List

class GeometryAwarePluckerAttention(nn.Module):
    """
    完整版的几何感知Plucker射线注意力模块
    专门为神经渲染、新视角合成等任务设计
    """
    
    def __init__(
        self,
        dim: int,                    # 图像特征维度
        num_heads: int = 8,
        qkv_bias: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        # Plucker射线特定参数
        plucker_dim: int = 6,        # Plucker射线维度
        plucker_proj_dim: int = 128, # Plucker投影维度
        plucker_embed_type: str = "mlp",  # "linear", "mlp", "fourier"
        # 几何感知参数
        use_ray_direction_bias: bool = True,
        use_distance_aware_scale: bool = True,
        use_geometric_attention_bias: bool = True,
        use_relative_position: bool = True,
        max_relative_distance: int = 16,
        # 位置编码
        use_rope: bool = True,
        rope_theta: float = 10000.0,
        **kwargs
    ) -> None:
        super().__init__()
        
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.plucker_dim = plucker_dim
        
        # ===== 自注意力部分 =====
        self.self_qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.self_proj = nn.Linear(dim, dim)
        self.self_attn_drop = nn.Dropout(attn_drop)
        
        # ===== Plucker射线处理 =====
        self.plucker_proj_dim = plucker_proj_dim
        self.plucker_embed_type = plucker_embed_type
        
        if plucker_embed_type == "linear":
            self.plucker_proj = nn.Linear(plucker_dim, plucker_proj_dim)
        elif plucker_embed_type == "mlp":
            # 深度MLP更好地捕捉Plucker几何
            self.plucker_proj = nn.Sequential(
                nn.Linear(plucker_dim, plucker_proj_dim * 2),
                nn.GELU(),
                nn.Linear(plucker_proj_dim * 2, plucker_proj_dim),
                nn.GELU(),
                nn.Linear(plucker_proj_dim, plucker_proj_dim)
            )
        elif plucker_embed_type == "fourier":
            # 傅里叶特征编码，适合表示高频几何信息
            self.num_fourier_features = 64
            self.plucker_proj = FourierFeatureProjection(
                plucker_dim, 
                self.num_fourier_features, 
                plucker_proj_dim
            )
        
        # ===== 交叉注意力部分 =====
        self.cross_q = nn.Linear(dim, dim, bias=qkv_bias)
        self.plucker_k_proj = nn.Linear(plucker_proj_dim, dim, bias=qkv_bias)
        self.plucker_v_proj = nn.Linear(plucker_proj_dim, dim, bias=qkv_bias)
        self.cross_proj = nn.Linear(dim, dim)
        self.cross_attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)
        
        # ===== 几何感知机制 =====
        self.use_ray_direction_bias = use_ray_direction_bias
        self.use_distance_aware_scale = use_distance_aware_scale
        self.use_geometric_attention_bias = use_geometric_attention_bias
        self.use_relative_position = use_relative_position
        
        # 射线方向偏置
        if use_ray_direction_bias:
            self.ray_direction_bias = nn.Parameter(torch.zeros(num_heads))
            
        # 距离感知缩放
        if use_distance_aware_scale:
            self.distance_aware_scale = nn.Sequential(
                nn.Linear(plucker_dim, num_heads * 4),
                nn.GELU(),
                nn.Linear(num_heads * 4, num_heads),
                nn.Tanh()  # 限制在[-1, 1]范围内
            )
            
        # 几何注意力偏置
        if use_geometric_attention_bias:
            self.geometric_bias_net = nn.Sequential(
                nn.Linear(plucker_proj_dim * 2, dim // 2),
                nn.GELU(),
                nn.Linear(dim // 2, num_heads),
                nn.Tanh()
            )
            
        # 相对位置编码
        if use_relative_position:
            self.max_relative_distance = max_relative_distance
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros((2 * max_relative_distance - 1) * (2 * max_relative_distance - 1), num_heads)
            )
            self.register_buffer("relative_index", self._create_relative_index())
            
        # ===== RoPE位置编码 =====
        self.use_rope = use_rope
        if use_rope:
            self.rope_theta = rope_theta
            
        # ===== 几何特征增强 =====
        self.geometric_feature_enhance = nn.Sequential(
            nn.Linear(dim + plucker_proj_dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim)
        )
        
        # 初始化
        self._init_weights()
        
    def _init_weights(self):
        # 初始化偏置参数
        if self.use_ray_direction_bias:
            nn.init.normal_(self.ray_direction_bias, mean=0.0, std=0.02)
            
        if self.use_relative_position:
            nn.init.normal_(self.relative_position_bias_table, mean=0.0, std=0.02)
            
    def _create_relative_index(self):
        """创建相对位置索引"""
        coords = torch.arange(self.max_relative_distance)
        coords = torch.stack(torch.meshgrid(coords, coords, indexing='ij'))
        coords_flat = coords.flatten(1)
        relative_coords = coords_flat[:, :, None] - coords_flat[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.max_relative_distance - 1
        relative_coords[:, :, 1] += self.max_relative_distance - 1
        relative_coords[:, :, 0] *= 2 * self.max_relative_distance - 1
        return relative_coords.sum(-1).flatten()
    
    def _apply_rope(self, x: torch.Tensor, seq_dim: int) -> torch.Tensor:
        """应用RoPE位置编码"""
        batch_size, num_heads, seq_len, head_dim = x.shape
        
        # 简化版RoPE实现
        inv_freq = 1.0 / (self.rope_theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
        inv_freq = inv_freq.to(x.device)
        
        # 生成位置索引
        pos = torch.arange(seq_len, device=x.device).type_as(inv_freq)
        sinusoid_inp = torch.einsum("i,j->ij", pos, inv_freq)
        
        # 计算正弦余弦
        sin = torch.sin(sinusoid_inp)
        cos = torch.cos(sinusoid_inp)
        
        # 应用旋转
        x_rotated = torch.stack([
            x[..., 0::2] * cos[..., None] - x[..., 1::2] * sin[..., None],
            x[..., 1::2] * cos[..., None] + x[..., 0::2] * sin[..., None]
        ], dim=-1)
        
        return x_rotated.flatten(-2)
    
    def _extract_plucker_components(self, plucker_rays: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """提取Plucker射线的方向、矩和距离信息"""
        # Plucker射线: [..., 6] = [direction(3), moment(3)]
        direction = plucker_rays[..., :3]  # 射线方向
        moment = plucker_rays[..., 3:]     # Plucker矩
        
        # 计算射线距离（原点到射线的距离）
        # 距离 = ||moment|| / ||direction||，但方向通常已归一化
        ray_distance = torch.norm(moment, dim=-1, keepdim=True)
        
        # 方向归一化
        direction_norm = F.normalize(direction, p=2, dim=-1)
        
        return direction_norm, moment, ray_distance
    
    def _compute_geometric_bias(self, plucker_features: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """计算几何注意力偏置"""
        B, N, _ = plucker_features.shape
        
        # 重塑为2D格式以计算空间关系
        plucker_2d = plucker_features.reshape(B, H, W, -1)
        
        # 计算相邻位置的特征差异
        horizontal_diff = torch.abs(plucker_2d[:, :, 1:] - plucker_2d[:, :, :-1])
        vertical_diff = torch.abs(plucker_2d[:, 1:, :] - plucker_2d[:, :-1, :])
        
        # 填充以保持原始尺寸
        horizontal_diff = F.pad(horizontal_diff, (0, 0, 0, 1, 0, 0))
        vertical_diff = F.pad(vertical_diff, (0, 0, 0, 0, 0, 1))
        
        # 合并差异特征
        spatial_diff = horizontal_diff + vertical_diff
        spatial_diff_flat = spatial_diff.reshape(B, N, -1)
        
        # 计算几何偏置
        geometric_bias = self.geometric_bias_net(spatial_diff_flat)  # [B, N, num_heads]
        geometric_bias = geometric_bias.permute(0, 2, 1).unsqueeze(2)  # [B, num_heads, 1, N]
        
        return geometric_bias
    
    def _compute_relative_position_bias(self, H: int, W: int) -> torch.Tensor:
        """计算相对位置偏置"""
        relative_bias = self.relative_position_bias_table[self.relative_index]
        relative_bias = relative_bias.reshape(
            H * W, H * W, -1
        ).permute(2, 0, 1).unsqueeze(0)  # [1, num_heads, N, N]
        return relative_bias
    
    def forward(self, x: torch.Tensor, plucker_rays: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 图像特征 [B, N, C] 或 [B, H, W, C]
            plucker_rays: Plucker射线 [B, H, W, 6]
            
        Returns:
            输出张量 [B, N, C]
        """
        # ===== 输入形状处理 =====
        if x.dim() == 4:  # [B, H, W, C]
            B, H, W, C = x.shape
            x = x.reshape(B, H*W, C)
            N = H * W
        else:  # [B, N, C]
            B, N, C = x.shape
            H = W = int(N ** 0.5)  # 假设是正方形特征图
        
        original_x = x  # 保存原始输入用于残差连接
        
        # ===== 自注意力阶段 =====
        self_qkv = self.self_qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        self_q, self_k, self_v = self_qkv.unbind(0)
        
        # 应用RoPE
        if self.use_rope:
            self_q = self._apply_rope(self_q, N)
            self_k = self._apply_rope(self_k, N)
        
        # 自注意力计算
        self_attn = (self_q @ self_k.transpose(-2, -1)) * self.scale
        
        # 添加相对位置偏置
        if self.use_relative_position:
            relative_bias = self._compute_relative_position_bias(H, W)
            self_attn = self_attn + relative_bias
        
        self_attn = self_attn.softmax(dim=-1)
        self_attn = self.self_attn_drop(self_attn)
        
        self_attn_out = (self_attn @ self_v).transpose(1, 2).reshape(B, N, C)
        self_attn_out = self.self_proj(self_attn_out)
        
        # ===== Plucker射线处理 =====
        # 确保空间尺寸匹配
        if plucker_rays.shape[1:3] != (H, W):
            plucker_rays = F.interpolate(
                plucker_rays.permute(0, 3, 1, 2),
                size=(H, W),
                mode='bilinear',
                align_corners=False
            ).permute(0, 2, 3, 1)
        
        plucker_flat = plucker_rays.reshape(B, H*W, self.plucker_dim)
        
        # 提取几何组件
        direction, moment, ray_distance = self._extract_plucker_components(plucker_flat)
        
        # 投影Plucker射线
        plucker_projected = self.plucker_proj(plucker_flat)
        
        # ===== 交叉注意力阶段 =====
        cross_q = self.cross_q(self_attn_out).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        cross_k = self.plucker_k_proj(plucker_projected).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        cross_v = self.plucker_v_proj(plucker_projected).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        
        # 应用RoPE
        if self.use_rope:
            cross_q = self._apply_rope(cross_q, N)
            cross_k = self._apply_rope(cross_k, N)
        
        # 交叉注意力计算
        cross_attn = (cross_q @ cross_k.transpose(-2, -1)) * self.scale
        
        # ===== 应用几何感知机制 =====
        
        # 1. 射线方向偏置
        if self.use_ray_direction_bias:
            # 计算方向相似性矩阵
            direction_similarity = torch.matmul(
                direction, direction.transpose(-2, -1)
            )  # [B, N, N]
            
            # 应用可学习的偏置
            direction_bias = direction_similarity.unsqueeze(1) * self.ray_direction_bias.view(1, -1, 1, 1)
            cross_attn = cross_attn + direction_bias
        
        # 2. 距离感知缩放
        if self.use_distance_aware_scale:
            distance_scale = self.distance_aware_scale(plucker_flat)  # [B, N, num_heads]
            distance_scale = distance_scale.permute(0, 2, 1).unsqueeze(2)  # [B, num_heads, 1, N]
            cross_attn = cross_attn * (1.0 + distance_scale * 0.1)  # 轻微调整
        
        # 3. 几何注意力偏置
        if self.use_geometric_attention_bias:
            geometric_bias = self._compute_geometric_bias(plucker_projected, H, W)
            cross_attn = cross_attn + geometric_bias
        
        # 4. 相对位置偏置
        if self.use_relative_position:
            relative_bias = self._compute_relative_position_bias(H, W)
            cross_attn = cross_attn + relative_bias
        
        # 完成注意力计算
        cross_attn = cross_attn.softmax(dim=-1)
        cross_attn = self.cross_attn_drop(cross_attn)
        
        cross_attn_out = (cross_attn @ cross_v).transpose(1, 2).reshape(B, N, C)
        
        # ===== 几何特征增强 =====
        # 将几何信息与注意力输出融合
        enhanced_features = torch.cat([cross_attn_out, plucker_projected], dim=-1)
        enhanced_output = self.geometric_feature_enhance(enhanced_features)
        
        # 残差连接 + 最终投影
        final_output = self.cross_proj(enhanced_output)
        final_output = self.proj_drop(final_output)
        
        # 残差连接回原始输入
        final_output = final_output + original_x
        
        return final_output


class FourierFeatureProjection(nn.Module):
    """傅里叶特征投影，用于高频几何信息编码"""
    
    def __init__(self, input_dim: int, num_fourier_features: int, output_dim: int):
        super().__init__()
        self.num_fourier_features = num_fourier_features
        self.output_dim = output_dim
        
        # 傅里叶投影矩阵
        self.fourier_proj = nn.Linear(input_dim, num_fourier_features * 2, bias=False)
        
        # 频率参数（可学习）
        self.frequencies = nn.Parameter(torch.randn(num_fourier_features) * 10.0)
        
        # 最终投影
        self.output_proj = nn.Linear(num_fourier_features * 2, output_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 应用频率缩放
        x_scaled = x.unsqueeze(-1) * self.frequencies.view(1, 1, 1, -1)
        
        # 傅里叶特征：sin和cos
        fourier_features = torch.cat([
            torch.sin(x_scaled * 2 * math.pi),
            torch.cos(x_scaled * 2 * math.pi)
        ], dim=-1)
        
        # 投影到目标维度
        output = self.output_proj(fourier_features)
        return output


class MultiResolutionGeometryAttention(nn.Module):
    """多分辨率几何感知注意力"""
    
    def __init__(self, dim: int, num_resolutions: int = 3, **kwargs):
        super().__init__()
        self.num_resolutions = num_resolutions
        self.attentions = nn.ModuleList([
            GeometryAwarePluckerAttention(dim=dim, **kwargs)
            for _ in range(num_resolutions)
        ])
        
        # 分辨率融合
        self.fusion_net = nn.Sequential(
            nn.Linear(dim * num_resolutions, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim)
        )
        
    def forward(self, multi_scale_features: List[torch.Tensor], 
                plucker_rays: torch.Tensor) -> torch.Tensor:
        """
        Args:
            multi_scale_features: 多尺度特征列表，从低分辨率到高分辨率
            plucker_rays: Plucker射线 [B, H, W, 6]
        """
        assert len(multi_scale_features) == self.num_resolutions
        
        outputs = []
        original_H, original_W = plucker_rays.shape[1:3]
        
        for i, features in enumerate(multi_scale_features):
            B, N_i, C = features.shape
            H_i = W_i = int(N_i ** 0.5)
            
            # 调整Plucker射线到当前分辨率
            if (H_i, W_i) != (original_H, original_W):
                rays_resized = F.interpolate(
                    plucker_rays.permute(0, 3, 1, 2),
                    size=(H_i, W_i),
                    mode='bilinear',
                    align_corners=False
                ).permute(0, 2, 3, 1)
            else:
                rays_resized = plucker_rays
            
            # 应用几何注意力
            output = self.attentions[i](features, rays_resized)
            outputs.append(output)
        
        # 上采样所有输出到最高分辨率
        target_H, target_W = multi_scale_features[-1].shape[1:3]
        target_H = int(target_H ** 0.5)
        upsampled_outputs = []
        
        for output in outputs:
            B, N, C = output.shape
            H = W = int(N ** 0.5)
            
            if (H, W) != (target_H, target_W):
                output_2d = output.reshape(B, H, W, C).permute(0, 3, 1, 2)
                output_upsampled = F.interpolate(
                    output_2d, 
                    size=(target_H, target_W),
                    mode='bilinear',
                    align_corners=False
                ).permute(0, 2, 3, 1).reshape(B, target_H * target_W, C)
                upsampled_outputs.append(output_upsampled)
            else:
                upsampled_outputs.append(output)
        
        # 融合多分辨率特征
        fused = torch.cat(upsampled_outputs, dim=-1)
        final_output = self.fusion_net(fused)
        
        return final_output
    
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, List

class GeometryAwarePluckerAttention(nn.Module):
    """
    完整版的几何感知Plucker射线注意力模块
    输入 x 维度: (B, N, C)
    输入 plucker_rays 维度: (B, H, W, 6) 或 (B, N, 6) 如果已经展平
    """
    
    def __init__(
        self,
        dim: int,                    # 输入特征维度 C
        num_heads: int = 8,
        qkv_bias: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        # Plucker射线特定参数
        plucker_dim: int = 6,        # Plucker射线维度
        plucker_proj_dim: int = 128, # Plucker投影维度
        plucker_embed_type: str = "mlp",  # "linear", "mlp", "fourier"
        # 几何感知参数
        use_ray_direction_bias: bool = True,
        use_distance_aware_scale: bool = True,
        use_geometric_attention_bias: bool = True,
        use_relative_position: bool = True,
        max_relative_distance: int = 16,
        # 位置编码
        use_rope: bool = True,
        rope_theta: float = 10000.0,
        # 残差连接
        use_residual: bool = True,
        **kwargs
    ) -> None:
        super().__init__()
        
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.plucker_dim = plucker_dim
        self.use_residual = use_residual
        
        # 验证维度可除性
        assert dim % num_heads == 0, f"dim {dim} must be divisible by num_heads {num_heads}"
        
        # ===== 自注意力部分 =====
        self.self_qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.self_proj = nn.Linear(dim, dim)
        self.self_attn_drop = nn.Dropout(attn_drop)
        
        # ===== Plucker射线处理 =====
        self.plucker_proj_dim = plucker_proj_dim
        self.plucker_embed_type = plucker_embed_type
        
        if plucker_embed_type == "linear":
            self.plucker_proj = nn.Linear(plucker_dim, plucker_proj_dim)
        elif plucker_embed_type == "mlp":
            # 深度MLP更好地捕捉Plucker几何
            self.plucker_proj = nn.Sequential(
                nn.Linear(plucker_dim, plucker_proj_dim * 2),
                nn.GELU(),
                nn.Linear(plucker_proj_dim * 2, plucker_proj_dim),
                nn.GELU(),
                nn.Linear(plucker_proj_dim, plucker_proj_dim)
            )
        elif plucker_embed_type == "fourier":
            # 傅里叶特征编码
            self.num_fourier_features = 64
            self.plucker_proj = FourierFeatureProjection(
                plucker_dim, 
                self.num_fourier_features, 
                plucker_proj_dim
            )
        
        # ===== 交叉注意力部分 =====
        self.cross_q = nn.Linear(dim, dim, bias=qkv_bias)
        self.plucker_k_proj = nn.Linear(plucker_proj_dim, dim, bias=qkv_bias)
        self.plucker_v_proj = nn.Linear(plucker_proj_dim, dim, bias=qkv_bias)
        self.cross_proj = nn.Linear(dim, dim)
        self.cross_attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)
        
        # ===== 几何感知机制 =====
        self.use_ray_direction_bias = use_ray_direction_bias
        self.use_distance_aware_scale = use_distance_aware_scale
        self.use_geometric_attention_bias = use_geometric_attention_bias
        self.use_relative_position = use_relative_position
        
        # 射线方向偏置
        if use_ray_direction_bias:
            self.ray_direction_bias = nn.Parameter(torch.zeros(num_heads))
            
        # 距离感知缩放
        if use_distance_aware_scale:
            self.distance_aware_scale = nn.Sequential(
                nn.Linear(plucker_dim, num_heads * 4),
                nn.GELU(),
                nn.Linear(num_heads * 4, num_heads),
                nn.Tanh()  # 限制在[-1, 1]范围内
            )
            
        # 几何注意力偏置
        if use_geometric_attention_bias:
            self.geometric_bias_net = nn.Sequential(
                nn.Linear(plucker_proj_dim * 2, dim // 2),
                nn.GELU(),
                nn.Linear(dim // 2, num_heads),
                nn.Tanh()
            )
            
        # 相对位置编码
        if use_relative_position:
            self.max_relative_distance = max_relative_distance
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros((2 * max_relative_distance - 1) * (2 * max_relative_distance - 1), num_heads)
            )
            
        # ===== RoPE位置编码 =====
        self.use_rope = use_rope
        if use_rope:
            self.rope_theta = rope_theta
            
        # ===== 几何特征增强 =====
        self.geometric_feature_enhance = nn.Sequential(
            nn.Linear(dim + plucker_proj_dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim)
        )
        
        # 初始化
        self._init_weights()
        
    def _init_weights(self):
        # 初始化偏置参数
        if self.use_ray_direction_bias:
            nn.init.normal_(self.ray_direction_bias, mean=0.0, std=0.02)
            
        if self.use_relative_position:
            nn.init.normal_(self.relative_position_bias_table, mean=0.0, std=0.02)
    
    def _compute_relative_position_bias(self, H: int, W: int) -> torch.Tensor:
        """计算相对位置偏置"""
        # 创建相对位置索引
        coords = torch.arange(max(H, W))
        coords = torch.stack(torch.meshgrid(coords, coords, indexing='ij'))
        coords_flat = coords.flatten(1)
        relative_coords = coords_flat[:, :, None] - coords_flat[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.max_relative_distance - 1
        relative_coords[:, :, 1] += self.max_relative_distance - 1
        relative_coords[:, :, 0] *= 2 * self.max_relative_distance - 1
        relative_index = relative_coords.sum(-1).flatten()
        
        # 从表中获取偏置
        relative_bias = self.relative_position_bias_table[relative_index]
        relative_bias = relative_bias.reshape(
            H * W, H * W, -1
        ).permute(2, 0, 1).unsqueeze(0)  # [1, num_heads, N, N]
        return relative_bias
    
    def _apply_rope(self, x: torch.Tensor, spatial_dim: int) -> torch.Tensor:
        """应用RoPE位置编码"""
        batch_size, num_heads, seq_len, head_dim = x.shape
        
        # 简化版RoPE实现
        inv_freq = 1.0 / (self.rope_theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
        inv_freq = inv_freq.to(x.device)
        
        # 生成位置索引
        pos = torch.arange(seq_len, device=x.device).type_as(inv_freq)
        sinusoid_inp = torch.einsum("i,j->ij", pos, inv_freq)
        
        # 计算正弦余弦
        sin = torch.sin(sinusoid_inp)
        cos = torch.cos(sinusoid_inp)
        
        # 应用旋转
        x_rotated = torch.stack([
            x[..., 0::2] * cos[..., None] - x[..., 1::2] * sin[..., None],
            x[..., 1::2] * cos[..., None] + x[..., 0::2] * sin[..., None]
        ], dim=-1)
        
        return x_rotated.flatten(-2)
    
    def _extract_plucker_components(self, plucker_rays: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """提取Plucker射线的方向、矩和距离信息"""
        # Plucker射线: [..., 6] = [direction(3), moment(3)]
        direction = plucker_rays[..., :3]  # 射线方向
        moment = plucker_rays[..., 3:]     # Plucker矩
        
        # 计算射线距离（原点到射线的距离）
        ray_distance = torch.norm(moment, dim=-1, keepdim=True)
        
        # 方向归一化
        direction_norm = F.normalize(direction, p=2, dim=-1)
        
        return direction_norm, moment, ray_distance
    
    def _compute_geometric_bias(self, plucker_features: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """计算几何注意力偏置"""
        B, N, _ = plucker_features.shape
        
        # 重塑为2D格式以计算空间关系
        plucker_2d = plucker_features.reshape(B, H, W, -1)
        
        # 计算相邻位置的特征差异
        horizontal_diff = torch.abs(plucker_2d[:, :, 1:] - plucker_2d[:, :, :-1])
        vertical_diff = torch.abs(plucker_2d[:, 1:, :] - plucker_2d[:, :-1, :])
        
        # 填充以保持原始尺寸
        horizontal_diff = F.pad(horizontal_diff, (0, 0, 0, 1, 0, 0))
        vertical_diff = F.pad(vertical_diff, (0, 0, 0, 0, 0, 1))
        
        # 合并差异特征
        spatial_diff = horizontal_diff + vertical_diff
        spatial_diff_flat = spatial_diff.reshape(B, N, -1)
        
        # 计算几何偏置
        geometric_bias = self.geometric_bias_net(spatial_diff_flat)  # [B, N, num_heads]
        geometric_bias = geometric_bias.permute(0, 2, 1).unsqueeze(2)  # [B, num_heads, 1, N]
        
        return geometric_bias
    
    def forward(self, x: torch.Tensor, plucker_rays: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入特征 [B, N, C]
            plucker_rays: Plucker射线 [B, H, W, 6] 或 [B, N, 6]
            
        Returns:
            输出张量 [B, N, C]
        """
        B, N, C = x.shape
        
        # 保存原始输入用于残差连接
        original_x = x
        
        # ===== 处理Plucker射线输入 =====
        if plucker_rays.dim() == 4:  # [B, H, W, 6]
            B_rays, H, W, D = plucker_rays.shape
            assert B == B_rays and D == self.plucker_dim, f"Plucker rays shape mismatch: {plucker_rays.shape}"
            
            # 展平Plucker射线
            plucker_flat = plucker_rays.reshape(B, H*W, self.plucker_dim)
            
            # 如果特征数量不匹配，需要插值
            if H*W != N:
                # 将Plucker射线重塑为2D并进行插值
                target_H = target_W = int(math.sqrt(N))
                plucker_2d = plucker_rays.permute(0, 3, 1, 2)  # [B, 6, H, W]
                plucker_resized = F.interpolate(
                    plucker_2d, 
                    size=(target_H, target_W), 
                    mode='bilinear', 
                    align_corners=False
                )
                plucker_flat = plucker_resized.permute(0, 2, 3, 1).reshape(B, N, self.plucker_dim)
                H, W = target_H, target_W
            else:
                H, W = int(math.sqrt(N)), int(math.sqrt(N))
        else:  # [B, N, 6]
            plucker_flat = plucker_rays
            H = W = int(math.sqrt(N))
        
        # ===== 自注意力阶段 =====
        # QKV投影和重塑
        self_qkv = self.self_qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        self_q, self_k, self_v = self_qkv.unbind(0)  # 每个 [B, num_heads, N, head_dim]
        
        # 应用RoPE
        if self.use_rope:
            self_q = self._apply_rope(self_q, N)
            self_k = self._apply_rope(self_k, N)
        
        # 自注意力计算
        self_attn = (self_q @ self_k.transpose(-2, -1)) * self.scale  # [B, num_heads, N, N]
        
        # 添加相对位置偏置
        if self.use_relative_position:
            relative_bias = self._compute_relative_position_bias(H, W)
            self_attn = self_attn + relative_bias
        
        self_attn = self_attn.softmax(dim=-1)
        self_attn = self.self_attn_drop(self_attn)
        
        # 应用注意力到值
        self_attn_out = (self_attn @ self_v).transpose(1, 2).reshape(B, N, C)  # [B, N, C]
        self_attn_out = self.self_proj(self_attn_out)
        
        # ===== Plucker射线投影 =====
        plucker_projected = self.plucker_proj(plucker_flat)  # [B, N, plucker_proj_dim]
        
        # ===== 交叉注意力阶段 =====
        # Q来自自注意力输出，K,V来自投影后的Plucker射线
        cross_q = self.cross_q(self_attn_out).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        cross_k = self.plucker_k_proj(plucker_projected).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        cross_v = self.plucker_v_proj(plucker_projected).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        
        # 应用RoPE
        if self.use_rope:
            cross_q = self._apply_rope(cross_q, N)
            cross_k = self._apply_rope(cross_k, N)
        
        # 交叉注意力计算
        cross_attn = (cross_q @ cross_k.transpose(-2, -1)) * self.scale  # [B, num_heads, N, N]
        
        # ===== 应用几何感知机制 =====
        
        # 1. 射线方向偏置
        if self.use_ray_direction_bias:
            # 提取方向分量
            direction, _, _ = self._extract_plucker_components(plucker_flat)
            
            # 计算方向相似性矩阵
            direction_similarity = torch.matmul(
                direction, direction.transpose(-2, -1)
            )  # [B, N, N]
            
            # 应用可学习的偏置
            direction_bias = direction_similarity.unsqueeze(1) * self.ray_direction_bias.view(1, -1, 1, 1)
            cross_attn = cross_attn + direction_bias
        
        # 2. 距离感知缩放
        if self.use_distance_aware_scale:
            distance_scale = self.distance_aware_scale(plucker_flat)  # [B, N, num_heads]
            distance_scale = distance_scale.permute(0, 2, 1).unsqueeze(2)  # [B, num_heads, 1, N]
            cross_attn = cross_attn * (1.0 + distance_scale * 0.1)  # 轻微调整
        
        # 3. 几何注意力偏置
        if self.use_geometric_attention_bias:
            geometric_bias = self._compute_geometric_bias(plucker_projected, H, W)
            cross_attn = cross_attn + geometric_bias
        
        # 4. 相对位置偏置
        if self.use_relative_position:
            relative_bias = self._compute_relative_position_bias(H, W)
            cross_attn = cross_attn + relative_bias
        
        # 完成注意力计算
        cross_attn = cross_attn.softmax(dim=-1)
        cross_attn = self.cross_attn_drop(cross_attn)
        
        cross_attn_out = (cross_attn @ cross_v).transpose(1, 2).reshape(B, N, C)  # [B, N, C]
        
        # ===== 几何特征增强 =====
        # 将几何信息与注意力输出融合
        enhanced_features = torch.cat([cross_attn_out, plucker_projected], dim=-1)
        enhanced_output = self.geometric_feature_enhance(enhanced_features)
        
        # 最终投影
        final_output = self.cross_proj(enhanced_output)
        final_output = self.proj_drop(final_output)
        
        # 残差连接
        if self.use_residual:
            final_output = final_output + original_x
        
        return final_output


class FourierFeatureProjection(nn.Module):
    """傅里叶特征投影，用于高频几何信息编码"""
    
    def __init__(self, input_dim: int, num_fourier_features: int, output_dim: int):
        super().__init__()
        self.num_fourier_features = num_fourier_features
        self.output_dim = output_dim
        
        # 傅里叶投影矩阵
        self.fourier_proj = nn.Linear(input_dim, num_fourier_features * 2, bias=False)
        
        # 频率参数（可学习）
        self.frequencies = nn.Parameter(torch.randn(num_fourier_features) * 10.0)
        
        # 最终投影
        self.output_proj = nn.Linear(num_fourier_features * 2, output_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        
        # 应用频率缩放
        x_scaled = x.unsqueeze(-1) * self.frequencies.view(1, 1, 1, -1)  # [B, N, D, num_fourier_features]
        
        # 傅里叶特征：sin和cos
        fourier_features = torch.cat([
            torch.sin(x_scaled * 2 * math.pi),
            torch.cos(x_scaled * 2 * math.pi)
        ], dim=-1)  # [B, N, D, num_fourier_features * 2]
        
        # 展平并投影
        fourier_flat = fourier_features.reshape(B, N, -1)
        output = self.output_proj(fourier_flat)  # [B, N, output_dim]
        
        return output


# 使用示例
def test_geometry_aware_attention():
    """测试函数"""
    B, N, C = 2, 64, 256  # batch_size, sequence_length, feature_dim
    H, W = 8, 8  # 假设空间尺寸
    
    # 创建输入
    x = torch.randn(B, N, C)
    plucker_rays = torch.randn(B, H, W, 6)
    
    # 创建注意力模块
    attention = GeometryAwarePluckerAttention(
        dim=C,
        num_heads=8,
        plucker_dim=6,
        plucker_proj_dim=128,
        use_ray_direction_bias=True,
        use_distance_aware_scale=True,
        use_geometric_attention_bias=True
    )
    
    # 前向传播
    output = attention(x, plucker_rays)
    
    print(f"输入形状: {x.shape}")
    print(f"Plucker射线形状: {plucker_rays.shape}")
    print(f"输出形状: {output.shape}")
    
    # 验证输出形状与输入相同
    assert output.shape == (B, N, C), f"输出形状 {output.shape} 与输入形状 {(B, N, C)} 不匹配"
    print("测试通过!")

if __name__ == "__main__":
    test_geometry_aware_attention()