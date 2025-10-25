import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ConvProjector(nn.Module):
    def __init__(self, in_channels=576, in_h=32, in_w=32, out_h=1374, out_w=2048, mid_channels=512,out_channels=24):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.pool = nn.AdaptiveAvgPool2d((out_h, out_w))
        # 用1x1卷积降维通道到1，方便输出二维张量
        self.channel_reducer = nn.Conv2d(mid_channels, out_channels, kernel_size=1)
        print(f"importing conv projector from {__file__}")
    def forward(self, x):
        """
        x: (batch, 576, 32, 32)
        output: (batch,24, 1374, 2048) 
        """
        x = self.conv(x)             # (batch, mid_channels, 32, 32)
        x = self.pool(x)             # (batch, mid_channels, 1374, 2048)
        x = self.channel_reducer(x)  # (batch, 1, 1374, 2048)
        x = x.squeeze(1)             # (batch,24, 1374, 2048)
        return x

# ==================== 方法3: UNet风格编码器-解码器 ====================
class UNetStyleCompression(nn.Module):
    def __init__(self, in_channels=24):
        super().__init__()
        
        # 编码器 - 逐步减少通道但保持空间分辨率
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        
        self.enc2 = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        
        self.enc3 = nn.Sequential(
            nn.Conv2d(32, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        
        # 中间层
        self.middle = nn.Sequential(
            nn.Conv2d(16, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        
        # 解码器 - 逐步重建空间细节
        self.dec3 = nn.Sequential(
            nn.Conv2d(32, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        
        self.dec2 = nn.Sequential(
            nn.Conv2d(48, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        
        self.dec1 = nn.Sequential(
            nn.Conv2d(96, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        
        # 最终输出单通道
        self.final_conv = nn.Conv2d(32, 1, 1)
        
    def forward(self, x):
        # 编码
        e1 = self.enc1(x)  # 24->64
        e2 = self.enc2(e1) # 64->32
        e3 = self.enc3(e2) # 32->16
        
        # 中间层
        middle = self.middle(e3)
        
        # 解码（带跳跃连接）
        d3 = self.dec3(torch.cat([middle, e3], dim=1))  # 16+16=32->16
        d2 = self.dec2(torch.cat([d3, e2], dim=1))      # 16+32=48->32
        d1 = self.dec1(torch.cat([d2, e1], dim=1))      # 32+64=96->32
        
        # 最终输出
        output = self.final_conv(d1)
        
        return output

# ==================== 方法5: 完整的空间信息保留流程 ====================
class SpatialPreservingProjection(nn.Module):
    def __init__(self, in_channels=24, embed_dim=768, target_patches=196):
        super().__init__()
        self.target_patches = target_patches
        self.target_size = int(np.sqrt(target_patches))  # 14 for 196 patches
        
        # 步骤1: 使用UNet风格压缩通道但保留空间信息
        self.channel_compression = UNetStyleCompression(in_channels)
        
        # 步骤2: 空间信息增强
        self.spatial_enhance = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 1, 3, padding=1),
            nn.Tanh()  # 增强对比度
        )
        
        # 步骤3: 投影到目标维度
        self.projection = nn.Sequential(
            nn.Conv2d(1, embed_dim, kernel_size=1),
            nn.AdaptiveAvgPool2d((self.target_size, self.target_size))
        )
        
        # 初始化权重
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
    def forward(self, x):
        # 输入: (B, 24, H, W)
        original_size = x.shape[2:]
        
        # 1. 通道压缩，保留空间信息
        compressed = self.channel_compression(x)  # (B, 1, H, W)
        
        # 2. 空间信息增强
        enhanced = self.spatial_enhance(compressed)  # (B, 1, H, W)
        
        # 3. 投影到目标形状
        projected = self.projection(enhanced)  # (B, 768, 14, 14)
        
        # 4. 重塑为最终输出
        output = projected.permute(0, 2, 3, 1)  # (B, 14, 14, 768)
        output = output.reshape(-1, 1, self.target_patches, 768)  # (B, 1, 196, 768)
        
        return output, compressed, enhanced
    
    # batch_size = 4
    # in_channels = 24
    # height, width = 1374, 2048
    # embed_dim = 768
    # target_patches = 196
    
    # # 创建模型
    # model = SpatialPreservingProjection(
    #     in_channels=in_channels,
    #     embed_dim=embed_dim,
    #     target_patches=target_patches
    # )
    
    # # 创建测试输入
    # x = torch.randn(batch_size, in_channels, height, width)
    # print(f"输入形状: {x.shape}")
    
    # # 前向传播
    # model.eval()
    # with torch.no_grad():
    #     output, compressed, enhanced = model(x)