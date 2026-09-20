# ==========================================
# models/efficientvit.py — EfficientViT Backbone
# Based on official implementation: https://github.com/mit-han-lab/efficientvit
# ==========================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict


class Conv2d_BN(nn.Module):
    def __init__(self, in_c, out_c, kernel_size, stride=1, padding=0, groups=1):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, kernel_size, stride, padding, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_c)
    
    def forward(self, x):
        return self.bn(self.conv(x))


class PDConv2d_BN(nn.Module):
    """Partial Depthwise Convolution"""
    def __init__(self, in_c, out_c, kernel_size, stride=1, padding=0, groups=1):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, kernel_size, stride, padding, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_c)
        self.act = nn.ReLU6(inplace=True)
    
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class MBConv(nn.Module):
    """Mobile Inverted Bottleneck Conv"""
    def __init__(self, in_c, out_c, expand_ratio, stride=1):
        super().__init__()
        hidden_c = int(in_c * expand_ratio)
        
        self.conv1 = Conv2d_BN(in_c, hidden_c, 1)
        self.act1 = nn.ReLU6(inplace=True)
        
        self.conv2 = PDConv2d_BN(hidden_c, hidden_c, 3, stride, 1, groups=hidden_c)
        
        self.conv3 = Conv2d_BN(hidden_c, out_c, 1)
        
        self.use_residual = (stride == 1) and (in_c == out_c)
    
    def forward(self, x):
        if self.use_residual:
            return x + self.conv3(self.act1(self.conv1(x)))
        else:
            return self.conv3(self.act1(self.conv1(x)))


class EfficientViTBlock(nn.Module):
    """
    Simplified EfficientViT block with multi-scale linear attention.
    This is a minimal implementation for thesis purposes.
    For production, use the official implementation.
    """
    def __init__(self, in_c, out_c, num_heads=4, expand_ratio=4):
        super().__init__()
        
        # MBConv for local features
        self.mbconv = MBConv(in_c, out_c, expand_ratio)
        
        # Multi-scale attention (simplified)
        self.attention = MultiScaleLinearAttention(out_c, num_heads)
        
        # Skip connection
        self.use_residual = (in_c == out_c)
    
    def forward(self, x):
        x = self.mbconv(x)
        if self.use_residual:
            x = x + self.attention(x)
        else:
            x = self.attention(x)
        return x


class MultiScaleLinearAttention(nn.Module):
    """
    Lightweight multi-scale linear attention module.
    Replaces expensive self-attention with ReLU-based linear attention.
    """
    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        
        self.qkv = nn.Conv2d(dim, dim * 3, 1, bias=False)
        self.proj = nn.Conv2d(dim, dim, 1, bias=False)
        self.bn = nn.BatchNorm2d(dim)
        self.act = nn.ReLU6(inplace=True)
    
    def forward(self, x):
        B, C, H, W = x.shape
        
        # Generate Q, K, V
        qkv = self.qkv(x).reshape(B, 3, self.num_heads, self.head_dim, H * W)
        q, k, v = qkv.unbind(1)
        
        # ReLU-based linear attention
        q = F.relu(q)
        k = F.relu(k)
        
        # Attention computation (simplified)
        kv = torch.einsum('bhdn,bhen->bhde', k, v)
        out = torch.einsum('bhdn,bhde->bhen', q, kv)
        
        out = out.reshape(B, C, H, W)
        out = self.proj(out)
        out = self.bn(out)
        out = self.act(out)
        
        return out


class EfficientViTEncoder(nn.Module):
    """
    EfficientViT Encoder for semantic segmentation.
    Returns multi-scale features for decoder.
    """
    
    def __init__(self, variant: str = "b0"):
        super().__init__()
        
        # Configuration for different variants
        configs = {
            "b0": {"channels": [32, 64, 128, 256], "blocks": [1, 2, 3, 4]},
            "b1": {"channels": [48, 96, 192, 384], "blocks": [2, 3, 4, 6]},
            "b2": {"channels": [64, 128, 256, 512], "blocks": [3, 4, 6, 8]},
            "b3": {"channels": [96, 192, 384, 768], "blocks": [4, 6, 8, 12]},
        }
        
        cfg = configs[variant]
        channels = cfg["channels"]
        blocks = cfg["blocks"]
        
        # Input stem
        self.stem = nn.Sequential(
            Conv2d_BN(3, channels[0], 3, 2, 1),
            nn.ReLU6(inplace=True)
        )
        
        # Stage 1
        self.stage1 = nn.Sequential(
            MBConv(channels[0], channels[0], 1, stride=1),
            *[EfficientViTBlock(channels[0], channels[0]) for _ in range(blocks[0])]
        )
        
        # Stage 2
        self.stage2 = nn.Sequential(
            MBConv(channels[0], channels[1], 4, stride=2),
            *[EfficientViTBlock(channels[1], channels[1]) for _ in range(blocks[1])]
        )
        
        # Stage 3
        self.stage3 = nn.Sequential(
            MBConv(channels[1], channels[2], 4, stride=2),
            *[EfficientViTBlock(channels[2], channels[2]) for _ in range(blocks[2])]
        )
        
        # Stage 4
        self.stage4 = nn.Sequential(
            MBConv(channels[2], channels[3], 4, stride=2),
            *[EfficientViTBlock(channels[3], channels[3]) for _ in range(blocks[3])]
        )
        
        self._out_channels = channels
    
    @property
    def out_channels(self):
        return self._out_channels
    
    def forward(self, x):
        features = []
        
        x = self.stem(x)
        x = self.stage1(x)
        features.append(x)  # F1: 1/2 resolution
        
        x = self.stage2(x)
        features.append(x)  # F2: 1/4 resolution
        
        x = self.stage3(x)
        features.append(x)  # F3: 1/8 resolution
        
        x = self.stage4(x)
        features.append(x)  # F4: 1/16 resolution
        
        return features