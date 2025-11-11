# src/models.py
import torch
import torch.nn as nn
from timm import create_model
from einops import rearrange

class VideoKeywordModel(nn.Module):
    def __init__(self, vit_name='vit_base_patch16_224', num_labels=300, temporal_dim=512, pretrained=True):
        super().__init__()
        # Create ViT as feature extractor (remove classifier)
        self.vit = create_model(vit_name, pretrained=pretrained, num_classes=0)  # returns features
        # timm models have .num_features
        feat_dim = getattr(self.vit, 'num_features', None)
        if feat_dim is None:
            # fallback assumption
            feat_dim = 768
        self.feat_dim = feat_dim
        self.temporal_proj = nn.Linear(feat_dim, temporal_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=temporal_dim, nhead=8, dim_feedforward=2048, dropout=0.1, activation='gelu')
        self.temporal_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.classifier = nn.Sequential(
            nn.LayerNorm(temporal_dim),
            nn.Linear(temporal_dim, num_labels)
        )

    def forward(self, video_frames):
        """
        video_frames: (B, T, C, H, W)
        returns: logits (B, num_labels)
        """
        B, T, C, H, W = video_frames.shape
        # merge batch & time for frame-wise encoding
        frames = rearrange(video_frames, 'b t c h w -> (b t) c h w')
        feats = self.vit.forward_features(frames)  # (B*T, feat_dim), depends on timm model implementation
        # ensure shape:
        feats = feats.view(B, T, -1)  # (B, T, feat_dim)
        x = self.temporal_proj(feats)  # (B, T, temporal_dim)
        # transformer expects (seq_len, batch, dim)
        x = x.permute(1, 0, 2)
        x = self.temporal_encoder(x)  # (T, B, dim)
        # pool across time
        x = x.mean(dim=0)  # (B, dim)
        logits = self.classifier(x)  # (B, num_labels)
        return logits
