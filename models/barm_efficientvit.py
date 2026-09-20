# ==========================================
# models/barm_efficientvit.py — Boundary-Aware EfficientViT
# ==========================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from .efficientvit import EfficientViTEncoder


class SobelEdgeExtractor(nn.Module):
    """
    Differentiable Sobel edge extraction.
    Applied to projected feature maps for better edge cues.
    """
    
    def __init__(self, channels: int = 1):
        super().__init__()
        
        # Sobel kernels
        sx = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]).view(1, 1, 3, 3)
        sy = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]).view(1, 1, 3, 3)
        
        self.register_buffer('kernel_x', sx.repeat(channels, 1, 1, 1))
        self.register_buffer('kernel_y', sy.repeat(channels, 1, 1, 1))
        self.channels = channels
    
    def forward(self, x):
        # x: (B, C, H, W)
        # Project to single channel if needed
        if x.shape[1] > 1:
            x = x.mean(dim=1, keepdim=True)
        
        gx = F.conv2d(x, self.kernel_x, padding=1, groups=self.channels)
        gy = F.conv2d(x, self.kernel_y, padding=1, groups=self.channels)
        
        magnitude = torch.sqrt(gx**2 + gy**2 + 1e-6)
        return magnitude


class DepthwiseSeparableConv(nn.Module):
    """Lightweight depthwise-separable convolution"""
    
    def __init__(self, in_c, out_c, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.dw = nn.Conv2d(in_c, in_c, kernel_size, stride, padding, groups=in_c, bias=False)
        self.pw = nn.Conv2d(in_c, out_c, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_c)
        self.act = nn.ReLU6(inplace=True)
    
    def forward(self, x):
        return self.act(self.bn(self.pw(self.dw(x))))


class BoundaryAwareRefinementModule(nn.Module):
    """
    BARM: Boundary-Aware Refinement Module
    Fuses shallow edge features with deep semantic features.
    """
    
    def __init__(self, low_c=64, deep_c=256, out_c=128):
        super().__init__()
        
        # Edge extraction from shallow features
        self.edge_project = nn.Conv2d(low_c, 1, 1, bias=False)
        self.sobel = SobelEdgeExtractor(channels=1)
        
        # Process shallow features
        self.low_conv = nn.Sequential(
            DepthwiseSeparableConv(low_c, out_c // 2),
            DepthwiseSeparableConv(out_c // 2, out_c // 2)
        )
        
        # Process deep features
        self.deep_conv = nn.Sequential(
            DepthwiseSeparableConv(deep_c, out_c // 2),
            DepthwiseSeparableConv(out_c // 2, out_c // 2)
        )
        
        # Fusion
        self.fusion = nn.Sequential(
            DepthwiseSeparableConv(out_c, out_c),
            DepthwiseSeparableConv(out_c, out_c)
        )
        
        # Edge prediction head
        self.edge_head = nn.Conv2d(out_c, 1, 1, bias=False)
    
    def forward(self, low_feat, deep_feat):
        # Extract edges from shallow features
        edge_input = self.edge_project(low_feat)
        edges = self.sobel(edge_input)
        
        # Enhance shallow features with edge cues
        low_enhanced = low_feat + edges
        low_processed = self.low_conv(low_enhanced)
        
        # Upsample deep features
        deep_upsampled = F.interpolate(
            deep_feat,
            size=low_feat.shape[2:],
            mode='bilinear',
            align_corners=False
        )
        deep_processed = self.deep_conv(deep_upsampled)
        
        # Fuse features
        fused = torch.cat([low_processed, deep_processed], dim=1)
        refined = self.fusion(fused)
        
        # Predict edge map
        pred_edge = self.edge_head(refined)
        
        return refined, pred_edge


class BoundaryAwareEfficientViT(nn.Module):
    """
    Complete Boundary-Aware EfficientViT for semantic segmentation.
    """
    
    def __init__(
        self,
        variant: str = "b0",
        num_classes: int = 1,
        use_barm: bool = True,
        low_level_idx: int = 0,  # Which stage to use for low-level features
        high_level_idx: int = 3  # Which stage to use for high-level features
    ):
        super().__init__()
        
        self.use_barm = use_barm
        self.low_level_idx = low_level_idx
        self.high_level_idx = high_level_idx
        
        # EfficientViT encoder
        self.encoder = EfficientViTEncoder(variant)
        encoder_channels = self.encoder.out_channels
        
        # BARM module
        if use_barm:
            self.barm = BoundaryAwareRefinementModule(
                low_c=encoder_channels[low_level_idx],
                deep_c=encoder_channels[high_level_idx],
                out_c=128
            )
            decoder_c = 128
        else:
            decoder_c = encoder_channels[high_level_idx]
        
        # Segmentation head
        self.seg_head = nn.Sequential(
            DepthwiseSeparableConv(decoder_c, decoder_c // 2),
            nn.Conv2d(decoder_c // 2, num_classes, 1, bias=False)
        )
    
    def forward(self, x):
        # Get multi-scale features
        features = self.encoder(x)
        
        low_feat = features[self.low_level_idx]
        high_feat = features[self.high_level_idx]
        
        if self.use_barm:
            # Use BARM for feature refinement
            refined_feat, pred_edge = self.barm(low_feat, high_feat)
            seg_logits = self.seg_head(refined_feat)
            
            # Upsample to input resolution
            seg_logits = F.interpolate(
                seg_logits,
                size=x.shape[2:],
                mode='bilinear',
                align_corners=False
            )
            pred_edge = F.interpolate(
                pred_edge,
                size=x.shape[2:],
                mode='bilinear',
                align_corners=False
            )
            
            return seg_logits, pred_edge
        else:
            # Standard decoder
            seg_logits = self.seg_head(high_feat)
            seg_logits = F.interpolate(
                seg_logits,
                size=x.shape[2:],
                mode='bilinear',
                align_corners=False
            )
            return seg_logits, None
    
    def count_parameters(self):
        """Count trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)