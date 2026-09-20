# ==========================================
# models/barm_efficientvit.py — LIGHTWEIGHT VERSION
# ==========================================

import torch
import torch.nn as nn
import torch.nn.functional as F


class SobelEdgeExtractor(nn.Module):
    def __init__(self, channels=1):
        super().__init__()
        sx = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]).view(1, 1, 3, 3)
        sy = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]).view(1, 1, 3, 3)
        self.register_buffer('kernel_x', sx.repeat(channels, 1, 1, 1))
        self.register_buffer('kernel_y', sy.repeat(channels, 1, 1, 1))
        self.channels = channels
    
    def forward(self, x):
        if x.shape[1] > 1:
            x = x.mean(dim=1, keepdim=True)
        gx = F.conv2d(x, self.kernel_x, padding=1, groups=self.channels)
        gy = F.conv2d(x, self.kernel_y, padding=1, groups=self.channels)
        return torch.sqrt(gx**2 + gy**2 + 1e-6)


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_c, out_c, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.dw = nn.Conv2d(in_c, in_c, kernel_size, stride, padding, groups=in_c, bias=False)
        self.pw = nn.Conv2d(in_c, out_c, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_c)
        self.act = nn.ReLU6(inplace=True)
    
    def forward(self, x):
        return self.act(self.bn(self.pw(self.dw(x))))


class BoundaryAwareRefinementModule(nn.Module):
    def __init__(self, low_c=64, deep_c=128, out_c=64):
        super().__init__()
        self.edge_project = nn.Conv2d(low_c, 1, 1, bias=False)
        self.sobel = SobelEdgeExtractor(channels=1)
        self.low_conv = DepthwiseSeparableConv(low_c, out_c // 2)
        self.deep_conv = DepthwiseSeparableConv(deep_c, out_c // 2)
        self.fusion = DepthwiseSeparableConv(out_c, out_c)
        self.edge_head = nn.Conv2d(out_c, 1, 1, bias=False)
    
    def forward(self, low_feat, deep_feat):
        edge_input = self.edge_project(low_feat)
        edges = self.sobel(edge_input)
        low_enhanced = low_feat + edges
        low_processed = self.low_conv(low_enhanced)
        deep_upsampled = F.interpolate(deep_feat, size=low_feat.shape[2:], mode='bilinear', align_corners=False)
        deep_processed = self.deep_conv(deep_upsampled)
        fused = torch.cat([low_processed, deep_processed], dim=1)
        refined = self.fusion(fused)
        pred_edge = self.edge_head(refined)
        return refined, pred_edge


class LightweightEncoder(nn.Module):
    """Simple lightweight encoder (replaces EfficientViT)"""
    def __init__(self):
        super().__init__()
        self.stage1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU6(),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU6()
        )
        self.stage2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU6(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU6()
        )
        self.stage3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.BatchNorm2d(128), nn.ReLU6(),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU6()
        )
    
    def forward(self, x):
        f1 = self.stage1(x)  # 1/2 resolution
        f2 = self.stage2(f1)  # 1/4 resolution
        f3 = self.stage3(f2)  # 1/8 resolution
        return [f1, f2, f3]


class BoundaryAwareEfficientViT(nn.Module):
    def __init__(self, variant="b0", num_classes=1, use_barm=True):
        super().__init__()
        self.use_barm = use_barm
        self.encoder = LightweightEncoder()
        
        if use_barm:
            self.barm = BoundaryAwareRefinementModule(low_c=32, deep_c=128, out_c=64)
            self.seg_head = nn.Sequential(
                DepthwiseSeparableConv(64, 32),
                nn.Conv2d(32, num_classes, 1)
            )
        else:
            self.seg_head = nn.Sequential(
                DepthwiseSeparableConv(128, 64),
                nn.Conv2d(64, num_classes, 1)
            )
    
    def forward(self, x):
        features = self.encoder(x)
        low_feat = features[0]  # 1/2 resolution
        high_feat = features[2]  # 1/8 resolution
        
        if self.use_barm:
            refined_feat, pred_edge = self.barm(low_feat, high_feat)
            seg_logits = self.seg_head(refined_feat)
            seg_logits = F.interpolate(seg_logits, size=x.shape[2:], mode='bilinear', align_corners=False)
            pred_edge = F.interpolate(pred_edge, size=x.shape[2:], mode='bilinear', align_corners=False)
            return seg_logits, pred_edge
        else:
            seg_logits = self.seg_head(high_feat)
            seg_logits = F.interpolate(seg_logits, size=x.shape[2:], mode='bilinear', align_corners=False)
            return seg_logits, None
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
