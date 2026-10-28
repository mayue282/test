from __future__ import annotations

import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvNormAct(nn.Sequential):
    def __init__(self, cin: int, cout: int, k: int = 3, stride: int = 1, groups: int = 1):
        super().__init__(
            nn.Conv2d(cin, cout, k, stride, k // 2, groups=groups, bias=False),
            nn.GroupNorm(min(8, cout), cout),
            nn.GELU(),
        )


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.net = nn.Sequential(ConvNormAct(channels, channels), nn.Conv2d(channels, channels, 3, 1, 1, bias=False), nn.GroupNorm(min(8, channels), channels))

    def forward(self, x):
        return F.gelu(x + self.net(x))


class Encoder(nn.Module):
    def __init__(self, widths=(32, 64, 128, 256)):
        super().__init__()
        stages = []
        cin = 3
        for width in widths:
            stages.append(nn.Sequential(ConvNormAct(cin, width, 3, 2), ResidualBlock(width)))
            cin = width
        self.stages = nn.ModuleList(stages)
        self.widths = widths

    def forward(self, x):
        feats = []
        for stage in self.stages:
            x = stage(x)
            feats.append(x)
        return feats


class Decoder(nn.Module):
    def __init__(self, widths=(32, 64, 128, 256), num_classes=8):
        super().__init__()
        self.blocks = nn.ModuleList([
            ConvNormAct(widths[3] + widths[2], widths[2]),
            ConvNormAct(widths[2] + widths[1], widths[1]),
            ConvNormAct(widths[1] + widths[0], widths[0]),
        ])
        self.head = nn.Sequential(ConvNormAct(widths[0], widths[0]), nn.Conv2d(widths[0], num_classes, 1))

    def forward(self, feats, output_size):
        x = feats[-1]
        for skip, block in zip(reversed(feats[:-1]), self.blocks):
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = block(torch.cat([x, skip], dim=1))
        x = self.head(x)
        return F.interpolate(x, size=output_size, mode="bilinear", align_corners=False)


class TimmUNet(nn.Module):
    """U-Net decoder over an ImageNet-pretrained ResNet-50 feature pyramid."""
    def __init__(self, num_classes=8, pretrained=True):
        super().__init__()
        import timm

        self.encoder = timm.create_model("resnet50", pretrained=pretrained, features_only=True)
        channels = self.encoder.feature_info.channels()
        decoder_channels = [256, 128, 64, 32]
        blocks = []
        current = channels[-1]
        for skip_channels, out_channels in zip(reversed(channels[:-1]), decoder_channels):
            blocks.append(nn.Sequential(ConvNormAct(current + skip_channels, out_channels), ResidualBlock(out_channels)))
            current = out_channels
        self.blocks = nn.ModuleList(blocks)
        self.head = nn.Conv2d(current, num_classes, 1)

    def forward(self, x, return_features=False):
        feats = self.encoder(x)
        decoded = feats[-1]
        for skip, block in zip(reversed(feats[:-1]), self.blocks):
            decoded = F.interpolate(decoded, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            decoded = block(torch.cat([decoded, skip], 1))
        logits = F.interpolate(self.head(decoded), size=x.shape[-2:], mode="bilinear", align_corners=False)
        return (logits, feats) if return_features else logits


class HFSegFormer(nn.Module):
    """Official SegFormer-B2 architecture and ADE20K-pretrained checkpoint."""
    def __init__(self, num_classes=8, pretrained=True):
        super().__init__()
        from transformers import SegformerConfig, SegformerForSemanticSegmentation

        if pretrained:
            self.model = SegformerForSemanticSegmentation.from_pretrained(
                "nvidia/segformer-b2-finetuned-ade-512-512",
                num_labels=num_classes,
                ignore_mismatched_sizes=True,
            )
        else:
            config = SegformerConfig(
                num_labels=num_classes,
                depths=[3, 4, 6, 3],
                hidden_sizes=[64, 128, 320, 512],
                num_attention_heads=[1, 2, 5, 8],
                sr_ratios=[8, 4, 2, 1],
                decoder_hidden_size=768,
            )
            self.model = SegformerForSemanticSegmentation(config)

    def forward(self, x, return_features=False):
        output = self.model(pixel_values=x, output_hidden_states=return_features)
        logits = F.interpolate(output.logits, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return (logits, list(output.hidden_states)) if return_features else logits


class SpatialTransformer(nn.Module):
    def __init__(self, channels=256, depth=3, heads=8):
        super().__init__()
        layer = nn.TransformerEncoderLayer(channels, heads, channels * 4, batch_first=True, norm_first=True, activation="gelu")
        self.blocks = nn.TransformerEncoder(layer, depth)

    def forward(self, x):
        b, c, h, w = x.shape
        tokens = x.flatten(2).transpose(1, 2)
        tokens = self.blocks(tokens)
        return tokens.transpose(1, 2).reshape(b, c, h, w)


class ScanMambaBlock(nn.Module):
    """Pure PyTorch four-direction selective-scan approximation for Apple MPS."""
    def __init__(self, channels=256):
        super().__init__()
        self.in_proj = nn.Conv2d(channels, channels * 2, 1)
        self.local = ConvNormAct(channels, channels, 3, 1, channels)
        self.out = nn.Conv2d(channels, channels, 1)
        self.decay = nn.Parameter(torch.full((1, channels, 1, 1), -2.0))

    def forward(self, x):
        value, gate = self.in_proj(x).chunk(2, 1)
        value = self.local(value)
        a = torch.sigmoid(self.decay)
        lr = torch.cumsum(value, -1) * a
        rl = torch.flip(torch.cumsum(torch.flip(value, [-1]), -1), [-1]) * a
        tb = torch.cumsum(value, -2) * a
        bt = torch.flip(torch.cumsum(torch.flip(value, [-2]), -2), [-2]) * a
        scale = 1.0 / math.sqrt(max(value.shape[-1], value.shape[-2]))
        return x + self.out((lr + rl + tb + bt) * scale * torch.sigmoid(gate))


class ReinsAdapter(nn.Module):
    """Token adapter following the released Rein simple_reins equations."""
    def __init__(self, channels=256, token_length=32, scale_init=1e-3):
        super().__init__()
        self.tokens = nn.Parameter(torch.empty(token_length, channels))
        self.token_to_feat = nn.Linear(channels, channels)
        self.delta = nn.Linear(channels, channels)
        self.scale = nn.Parameter(torch.tensor(scale_init))
        nn.init.normal_(self.tokens, std=0.02)

    def forward(self, x):
        b, c, h, w = x.shape
        feats = x.flatten(2).transpose(1, 2)
        attn = torch.softmax(torch.einsum("bnc,mc->bnm", feats, self.tokens) * (c ** -0.5), -1)
        delta = torch.einsum("bnm,mc->bnc", attn, self.token_to_feat(self.tokens))
        feats = feats + self.scale * self.delta(delta + feats)
        return feats.transpose(1, 2).reshape(b, c, h, w)


class CompactSegmenter(nn.Module):
    def __init__(self, variant="unet", num_classes=8, widths=(32, 64, 128, 256)):
        super().__init__()
        self.variant = variant
        self.encoder = Encoder(widths)
        if variant in {"segformer_b2", "transformer"}:
            self.context = SpatialTransformer(widths[-1], depth=3, heads=8)
        elif variant in {"rs3mamba", "mamba"}:
            self.context = nn.Sequential(ScanMambaBlock(widths[-1]), ScanMambaBlock(widths[-1]), ScanMambaBlock(widths[-1]))
        elif variant in {"rein"}:
            self.context = nn.Sequential(ReinsAdapter(widths[-1]), ReinsAdapter(widths[-1]))
        else:
            self.context = nn.Sequential(ResidualBlock(widths[-1]), ResidualBlock(widths[-1]))
        self.decoder = Decoder(widths, num_classes)

    def forward_features(self, x):
        feats = self.encoder(x)
        feats[-1] = self.context(feats[-1])
        return feats

    def decode(self, feats, output_size):
        return self.decoder(feats, output_size)

    def forward(self, x, return_features=False):
        feats = self.forward_features(x)
        logits = self.decode(feats, x.shape[-2:])
        return (logits, feats) if return_features else logits


class DINOv2Segmenter(nn.Module):
    def __init__(self, num_classes=8, pretrained=True, model_name="vit_base_patch14_dinov2", image_size=256):
        super().__init__()
        import timm

        # timm 0.9 adapts the pretrained positional embedding to this fixed grid.
        self.encoder = timm.create_model(model_name, pretrained=pretrained, num_classes=0, img_size=image_size)
        channels = self.encoder.num_features
        for parameter in self.encoder.parameters():
            parameter.requires_grad = False
        self.decoder = nn.Sequential(
            ConvNormAct(channels, 256, 3),
            ConvNormAct(256, 128, 3),
            nn.Conv2d(128, num_classes, 1),
        )

    def train(self, mode=True):
        super().train(mode)
        self.encoder.eval()
        return self

    def forward(self, x, return_features=False):
        with torch.no_grad():
            tokens = self.encoder.forward_features(x)
            if tokens.ndim == 3:
                # DINOv2 emits five register/prefix tokens in timm.
                prefix = getattr(self.encoder, "num_prefix_tokens", 1)
                tokens = tokens[:, prefix:]
                n = tokens.shape[1]
                h = int(round(math.sqrt(n)))
                feat = tokens[:, : h * h].transpose(1, 2).reshape(x.shape[0], -1, h, h)
            else:
                feat = tokens.permute(0, 3, 1, 2) if tokens.shape[-1] == self.encoder.num_features else tokens
        logits = self.decoder(feat)
        logits = F.interpolate(logits, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return (logits, [feat]) if return_features else logits


class DepthConditionedSegmenter(CompactSegmenter):
    """DepthForge-compatible structural conditioning without CUDA-only DAv2 ops."""
    def __init__(self, num_classes=8):
        super().__init__("rein", num_classes)
        c = self.encoder.widths[-1]
        self.depth_encoder = nn.Sequential(ConvNormAct(1, 32, 5, 2), ConvNormAct(32, 64, 3, 2), ConvNormAct(64, 128, 3, 2), ConvNormAct(128, c, 3, 2))
        self.depth_gate = nn.Sequential(nn.Conv2d(c, c, 1), nn.Sigmoid())

    def forward_features(self, x):
        feats = self.encoder(x)
        gray = (0.299 * x[:, :1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3])
        depth = self.depth_encoder(gray)
        depth = F.interpolate(depth, feats[-1].shape[-2:], mode="bilinear", align_corners=False)
        feats[-1] = self.context(feats[-1]) * (1.0 + self.depth_gate(depth))
        return feats


def make_model(name: str, pretrained=True):
    if name == "unet_resnet50":
        return TimmUNet(pretrained=pretrained)
    if name == "segformer_b2":
        return HFSegFormer(pretrained=pretrained)
    if name == "rs3mamba":
        return CompactSegmenter("rs3mamba")
    if name == "dinov2_b14_mask2former":
        return DINOv2Segmenter(pretrained=pretrained)
    if name in {"daformer", "mic", "mrm", "crossearth"}:
        return CompactSegmenter("transformer")
    if name == "rein":
        return CompactSegmenter("rein")
    if name == "depthforge":
        return DepthConditionedSegmenter()
    raise ValueError(name)


def ema_copy(model: nn.Module) -> nn.Module:
    teacher = copy.deepcopy(model).eval()
    for p in teacher.parameters():
        p.requires_grad = False
    return teacher


@torch.no_grad()
def update_ema(teacher: nn.Module, student: nn.Module, momentum=0.999):
    for tp, sp in zip(teacher.parameters(), student.parameters()):
        tp.mul_(momentum).add_(sp, alpha=1.0 - momentum)
