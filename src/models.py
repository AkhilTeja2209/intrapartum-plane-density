"""Models.

The comparison only isolates what we claim it isolates if the encoder is
byte-for-byte the same object in both arms. So there is exactly one encoder
definition here, and both heads wrap it.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models


def make_encoder(arch: str = "resnet18", pretrained: bool = True) -> tuple[nn.Module, int]:
    """ResNet-18 with the classifier removed. Returns (encoder, feat_dim)."""
    if arch == "resnet18":
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        net = models.resnet18(weights=weights)
        dim = net.fc.in_features          # 512
        net.fc = nn.Identity()
    elif arch == "resnet50":
        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        net = models.resnet50(weights=weights)
        dim = net.fc.in_features
        net.fc = nn.Identity()
    elif arch == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        net = models.efficientnet_b0(weights=weights)
        dim = net.classifier[1].in_features
        net.classifier = nn.Identity()
    else:
        raise ValueError(f"unsupported arch {arch!r}")
    return net, dim


class FrameClassifier(nn.Module):
    """Arm 1: one frame in, one prediction out. No memory of any kind."""

    def __init__(self, arch="resnet18", pretrained=True, n_classes=2, dropout=0.2):
        super().__init__()
        self.encoder, dim = make_encoder(arch, pretrained)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(dim, n_classes))
        self.feat_dim = dim

    def forward(self, x):                      # (B,3,H,W)
        return self.head(self.encoder(x))      # (B,C)


class TemporalClassifier(nn.Module):
    """Arm 2: T frames in, T predictions out.

    Per-frame output rather than per-clip, for three reasons:
      * the label is defined per frame, so a per-clip label would throw
        supervision away;
      * it keeps the output space identical to the frame-wise arm, so the same
        metrics apply without any re-derivation;
      * clinically, the deliverable is "which frames in this video are usable",
        which is a per-frame question.

    `bidirectional` is worth reporting both ways. Bi-LSTM is the stronger
    offline model and the fairer test of "does temporal context help at all".
    Unidirectional is the only variant deployable during live scanning, since
    it cannot see the future. A gap between the two is itself a finding.
    """

    def __init__(self, arch="resnet18", pretrained=True, n_classes=2,
                 rnn="lstm", hidden=256, layers=1, bidirectional=True,
                 dropout=0.2, freeze_encoder=False):
        super().__init__()
        self.encoder, dim = make_encoder(arch, pretrained)
        self.feat_dim = dim
        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False

        rnn_cls = {"lstm": nn.LSTM, "gru": nn.GRU}[rnn.lower()]
        self.rnn = rnn_cls(dim, hidden, num_layers=layers, batch_first=True,
                           bidirectional=bidirectional,
                           dropout=dropout if layers > 1 else 0.0)
        out_dim = hidden * (2 if bidirectional else 1)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(out_dim, n_classes))

    def forward(self, x):                      # (B,T,3,H,W)
        B, T = x.shape[:2]
        f = self.encoder(x.flatten(0, 1))      # (B*T, dim)
        f = f.view(B, T, -1)
        h, _ = self.rnn(f)                     # (B,T,out_dim)
        return self.head(h)                    # (B,T,C)


def load_encoder_from(model: nn.Module, ckpt_path: str, strict: bool = False):
    """Warm-start the temporal arm's encoder from the trained frame-wise arm.

    Report BOTH initialisations. From-ImageNet is the clean comparison; from
    the frame-wise checkpoint is the practical one and usually converges in
    far fewer epochs, which matters when GPU time is the binding constraint.
    """
    sd = torch.load(ckpt_path, map_location="cpu")
    sd = sd.get("model", sd)
    enc = {k[len("encoder."):]: v for k, v in sd.items() if k.startswith("encoder.")}
    missing, unexpected = model.encoder.load_state_dict(enc, strict=strict)
    return missing, unexpected


def build(cfg: dict) -> nn.Module:
    m = cfg.get("model", {})
    if m.get("type", "frame") == "frame":
        return FrameClassifier(arch=m.get("arch", "resnet18"),
                               pretrained=m.get("pretrained", True),
                               dropout=m.get("dropout", 0.2))
    return TemporalClassifier(arch=m.get("arch", "resnet18"),
                              pretrained=m.get("pretrained", True),
                              rnn=m.get("rnn", "lstm"),
                              hidden=m.get("hidden", 256),
                              layers=m.get("layers", 1),
                              bidirectional=m.get("bidirectional", True),
                              dropout=m.get("dropout", 0.2),
                              freeze_encoder=m.get("freeze_encoder", False))
