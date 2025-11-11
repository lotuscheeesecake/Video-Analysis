# src/models_enhanced.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm import create_model
from einops import rearrange
import torchvision.models as tv_models
from typing import Dict, List, Optional, Tuple

class SpatialFeatureExtractor(nn.Module):
    """Extract spatial features using CNNs or ViT for object detection and scene understanding"""
    def __init__(self, backbone='resnet50', pretrained=True, feature_dim=2048):
        super().__init__()
        if 'vit' in backbone:
            self.backbone = create_model(backbone, pretrained=pretrained, num_classes=0)
            self.feature_dim = self.backbone.num_features
        else:
            # Use ResNet/EfficientNet
            if backbone == 'resnet50':
                model = tv_models.resnet50(pretrained=pretrained)
                self.backbone = nn.Sequential(*list(model.children())[:-2])
                self.feature_dim = 2048
            elif backbone == 'efficientnet_b0':
                self.backbone = create_model('efficientnet_b0', pretrained=pretrained, num_classes=0)
                self.feature_dim = self.backbone.num_features
        
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
    def forward(self, x):
        """
        Args:
            x: (B, C, H, W) or (B*T, C, H, W)
        Returns:
            features: (B, feature_dim) or (B*T, feature_dim)
        """
        if hasattr(self.backbone, 'forward_features'):
            feat = self.backbone.forward_features(x)
        else:
            feat = self.backbone(x)
        
        if len(feat.shape) == 4:  # Conv features (B, C, H, W)
            feat = self.pool(feat).flatten(1)
        return feat


class TemporalTransformer(nn.Module):
    """Transformer encoder for temporal modeling (similar to TimeSformer approach)"""
    def __init__(self, d_model=768, nhead=8, num_layers=6, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pos_embedding = nn.Parameter(torch.randn(1, 128, d_model))  # max 128 frames
        
    def forward(self, x):
        """
        Args:
            x: (B, T, D)
        Returns:
            (B, T, D)
        """
        B, T, D = x.shape
        x = x + self.pos_embedding[:, :T, :]
        x = self.transformer(x)
        return x


class SlowFastBackbone(nn.Module):
    """Simplified SlowFast dual-pathway network for motion capture"""
    def __init__(self, slow_frames=8, fast_frames=32, feature_dim=512):
        super().__init__()
        self.slow_frames = slow_frames
        self.fast_frames = fast_frames
        
        # Slow pathway: spatial semantics (low frame rate)
        self.slow_conv = nn.Sequential(
            nn.Conv3d(3, 64, kernel_size=(1, 7, 7), stride=(1, 2, 2), padding=(0, 3, 3)),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        )
        
        # Fast pathway: temporal dynamics (high frame rate)
        self.fast_conv = nn.Sequential(
            nn.Conv3d(3, 16, kernel_size=(5, 7, 7), stride=(1, 2, 2), padding=(2, 3, 3)),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        )
        
        # Lateral connections (fast -> slow)
        self.lateral = nn.Conv3d(16, 64, kernel_size=5, stride=1, padding=2)
        
        self.global_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(64 + 16, feature_dim)
        
    def forward(self, slow_frames, fast_frames):
        """
        Args:
            slow_frames: (B, 3, T_slow, H, W)
            fast_frames: (B, 3, T_fast, H, W)
        Returns:
            features: (B, feature_dim)
        """
        slow_feat = self.slow_conv(slow_frames)  # (B, 64, T, H', W')
        fast_feat = self.fast_conv(fast_frames)  # (B, 16, T, H', W')
        
        # Lateral connection
        lateral_feat = self.lateral(fast_feat)
        slow_feat = slow_feat + lateral_feat
        
        # Pool and concatenate
        slow_pooled = self.global_pool(slow_feat).flatten(1)
        fast_pooled = self.global_pool(fast_feat).flatten(1)
        
        combined = torch.cat([slow_pooled, fast_pooled], dim=1)
        out = self.fc(combined)
        return out


class AudioFeatureExtractor(nn.Module):
    """Extract audio features from spectrograms (simulating VGGish/YAMNet)"""
    def __init__(self, feature_dim=512):
        super().__init__()
        # Simple CNN for mel-spectrograms
        self.conv = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, feature_dim)
        
    def forward(self, spectrogram):
        """
        Args:
            spectrogram: (B, 1, freq_bins, time_steps)
        Returns:
            features: (B, feature_dim)
        """
        x = self.conv(spectrogram)
        x = self.pool(x).flatten(1)
        x = self.fc(x)
        return x


class MultiModalFusionTransformer(nn.Module):
    """Cross-modal fusion (inspired by VideoBERT/CLIP)"""
    def __init__(self, visual_dim=768, audio_dim=512, text_dim=512, fusion_dim=768, num_layers=4):
        super().__init__()
        self.visual_proj = nn.Linear(visual_dim, fusion_dim)
        self.audio_proj = nn.Linear(audio_dim, fusion_dim)
        self.text_proj = nn.Linear(text_dim, fusion_dim)
        
        # Modality type embeddings
        self.modality_embed = nn.Embedding(3, fusion_dim)  # 0=visual, 1=audio, 2=text
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=fusion_dim,
            nhead=8,
            dim_feedforward=2048,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        self.fusion_transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
    def forward(self, visual_feat, audio_feat=None, text_feat=None):
        """
        Args:
            visual_feat: (B, T_v, D_v) or (B, D_v)
            audio_feat: (B, T_a, D_a) or (B, D_a) or None
            text_feat: (B, T_t, D_t) or (B, D_t) or None
        Returns:
            fused_features: (B, fusion_dim)
        """
        batch_size = visual_feat.size(0)
        
        # Ensure sequence dimension
        if len(visual_feat.shape) == 2:
            visual_feat = visual_feat.unsqueeze(1)
        
        visual_proj = self.visual_proj(visual_feat)
        visual_proj = visual_proj + self.modality_embed(torch.zeros(1, dtype=torch.long, device=visual_feat.device))
        
        features = [visual_proj]
        
        if audio_feat is not None:
            if len(audio_feat.shape) == 2:
                audio_feat = audio_feat.unsqueeze(1)
            audio_proj = self.audio_proj(audio_feat)
            audio_proj = audio_proj + self.modality_embed(torch.ones(1, dtype=torch.long, device=audio_feat.device))
            features.append(audio_proj)
        
        if text_feat is not None:
            if len(text_feat.shape) == 2:
                text_feat = text_feat.unsqueeze(1)
            text_proj = self.text_proj(text_feat)
            text_proj = text_proj + self.modality_embed(torch.full((1,), 2, dtype=torch.long, device=text_feat.device))
            features.append(text_proj)
        
        # Concatenate all modalities
        multimodal_seq = torch.cat(features, dim=1)  # (B, total_seq, fusion_dim)
        
        # Cross-modal fusion
        fused = self.fusion_transformer(multimodal_seq)
        
        # Global pooling
        fused_global = fused.mean(dim=1)  # (B, fusion_dim)
        return fused_global


class ComprehensiveVideoAnalysisModel(nn.Module):
    """
    Complete SOTA video analysis model with:
    - Spatial feature extraction (objects, scenes)
    - Temporal modeling (actions, motion)
    - Audio processing
    - Multimodal fusion
    - Multiple prediction heads
    """
    def __init__(
        self,
        num_object_classes=80,      # COCO objects
        num_action_classes=400,     # Kinetics actions
        num_scene_classes=365,      # Places365 scenes
        num_keywords=300,           # Your custom keywords
        visual_backbone='vit_base_patch16_224',
        use_slowfast=False,
        fusion_dim=768
    ):
        super().__init__()
        
        # Spatial features
        self.spatial_extractor = SpatialFeatureExtractor(visual_backbone)
        spatial_dim = self.spatial_extractor.feature_dim
        
        # Temporal modeling
        self.use_slowfast = use_slowfast
        if use_slowfast:
            self.slowfast = SlowFastBackbone(feature_dim=512)
            temporal_dim = 512
        else:
            self.temporal_transformer = TemporalTransformer(d_model=spatial_dim, num_layers=6)
            temporal_dim = spatial_dim
        
        # Audio features
        self.audio_extractor = AudioFeatureExtractor(feature_dim=512)
        
        # Multimodal fusion
        self.fusion = MultiModalFusionTransformer(
            visual_dim=temporal_dim,
            audio_dim=512,
            text_dim=512,  # For ASR transcripts
            fusion_dim=fusion_dim
        )
        
        # Multiple prediction heads
        self.object_classifier = nn.Linear(fusion_dim, num_object_classes)
        self.action_classifier = nn.Linear(fusion_dim, num_action_classes)
        self.scene_classifier = nn.Linear(fusion_dim, num_scene_classes)
        self.keyword_classifier = nn.Linear(fusion_dim, num_keywords)
        
        # Embedding for semantic retrieval
        self.semantic_projection = nn.Sequential(
            nn.Linear(fusion_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512)
        )
        
    def forward(
        self, 
        frames,           # (B, T, C, H, W)
        audio_spec=None,  # (B, 1, freq, time) or None
        text_embed=None   # (B, text_dim) or None
    ):
        """
        Returns dict with:
            - object_logits
            - action_logits
            - scene_logits
            - keyword_logits
            - semantic_embedding
        """
        B, T, C, H, W = frames.shape
        
        # Extract spatial features per frame
        frames_flat = rearrange(frames, 'b t c h w -> (b t) c h w')
        spatial_feats = self.spatial_extractor(frames_flat)  # (B*T, spatial_dim)
        spatial_feats = rearrange(spatial_feats, '(b t) d -> b t d', b=B, t=T)
        
        # Temporal modeling
        if self.use_slowfast:
            # Sample slow and fast frames
            slow_indices = torch.linspace(0, T-1, 8).long()
            fast_indices = torch.linspace(0, T-1, 32).long()
            slow_frames = frames[:, slow_indices]
            fast_frames = frames[:, fast_indices]
            slow_frames = rearrange(slow_frames, 'b t c h w -> b c t h w')
            fast_frames = rearrange(fast_frames, 'b t c h w -> b c t h w')
            temporal_feat = self.slowfast(slow_frames, fast_frames)  # (B, 512)
        else:
            temporal_feat = self.temporal_transformer(spatial_feats)  # (B, T, D)
            temporal_feat = temporal_feat.mean(dim=1)  # (B, D)
        
        # Audio features
        audio_feat = None
        if audio_spec is not None:
            audio_feat = self.audio_extractor(audio_spec)  # (B, 512)
        
        # Multimodal fusion
        fused_feat = self.fusion(
            temporal_feat.unsqueeze(1),  # Add sequence dim
            audio_feat.unsqueeze(1) if audio_feat is not None else None,
            text_embed.unsqueeze(1) if text_embed is not None else None
        )  # (B, fusion_dim)
        
        # Multiple outputs
        outputs = {
            'object_logits': self.object_classifier(fused_feat),
            'action_logits': self.action_classifier(fused_feat),
            'scene_logits': self.scene_classifier(fused_feat),
            'keyword_logits': self.keyword_classifier(fused_feat),
            'semantic_embedding': F.normalize(self.semantic_projection(fused_feat), dim=-1)
        }
        
        return outputs


# Simplified version for initial testing
class EnhancedVideoKeywordModel(nn.Module):
    """Enhanced version of your current model with better temporal modeling"""
    def __init__(self, vit_name='vit_base_patch16_224', num_labels=300, 
                 temporal_dim=768, num_temporal_layers=6, pretrained=True):
        super().__init__()
        self.vit = create_model(vit_name, pretrained=pretrained, num_classes=0)
        feat_dim = self.vit.num_features
        
        # Better temporal modeling with proper transformer
        self.temporal_transformer = TemporalTransformer(
            d_model=feat_dim,
            nhead=12,
            num_layers=num_temporal_layers,
            dim_feedforward=feat_dim * 4
        )
        
        # Attention pooling instead of mean
        self.attention_pool = nn.Sequential(
            nn.Linear(feat_dim, 1),
            nn.Softmax(dim=1)
        )
        
        self.classifier = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Dropout(0.1),
            nn.Linear(feat_dim, num_labels)
        )

    def forward(self, video_frames):
        B, T, C, H, W = video_frames.shape
        frames = rearrange(video_frames, 'b t c h w -> (b t) c h w')
        feats = self.vit.forward_features(frames)
        feats = feats.view(B, T, -1)
        
        # Temporal modeling
        temporal_feats = self.temporal_transformer(feats)
        
        # Attention pooling
        attn_weights = self.attention_pool(temporal_feats)  # (B, T, 1)
        pooled = (temporal_feats * attn_weights).sum(dim=1)  # (B, D)
        
        logits = self.classifier(pooled)
        return logits