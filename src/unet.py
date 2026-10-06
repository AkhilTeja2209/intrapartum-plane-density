"""U-Net for pubic-symphysis / fetal-head segmentation.

Why a U-Net is the right second model here, and what it can and cannot do:

The IUGC 2024 challenge this corpus comes from is a *multi-task* problem --
classify standard planes, segment PS and FH, then measure the angle of
progression. Most challenge entries solved it as a **cascade**: a classifier
picks standard planes, and a segmenter runs only on those frames. Classic 2D
U-Net was the organisers' baseline and the backbone for several teams, with
reported Dice in the 0.89-0.93 range. So the frame classifier already in this
repo is stage one, and this module is stage two.

What it cannot be, on this data, is a drop-in replacement for the classifier.
Masks exist for 2,915 frames and **every one of them is a standard plane** --
there is no "anatomy absent" supervision anywhere in the corpus. A segmenter
trained on it has never been shown a negative and cannot be assumed to produce
empty output on one.

That makes "can a U-Net classify?" an empirical question rather than an
architectural one, and `src/train_unet.py` answers it: run the segmenter over
the whole test set, derive a plane score from how much PS and FH it finds, and
score that against the same labels the ResNet is scored on. The answer is worth
having either way -- a segmentation-derived decision is interpretable by
construction, because the evidence for it is the mask it drew.

Encoder is the same ImageNet ResNet-18 the classifier uses, so the comparison
is between decoders and objectives rather than between backbones.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


def _block(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False),
        nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False),
        nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
    )


class Up(nn.Module):
    """Upsample, concatenate the skip connection, then two convs."""

    def __init__(self, cin: int, skip: int, cout: int):
        super().__init__()
        self.conv = _block(cin + skip, cout)

    def forward(self, x, skip=None):
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        if skip is not None:
            # guard against odd input sizes drifting the two tensors apart
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear",
                                  align_corners=False)
            x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class ResUNet(nn.Module):
    """U-Net with a ResNet-18 encoder.

    The encoder is the same architecture and the same ImageNet initialisation
    the frame classifier uses, so any difference between the two models is the
    decoder and the objective, not the backbone. `in_ch=1` because the frames
    are greyscale; the pretrained stem's three input channels are summed, which
    is equivalent to feeding a replicated grey image and keeps the pretrained
    filters intact.
    """

    def __init__(self, n_classes: int = 3, pretrained: bool = True,
                 in_ch: int = 1):
        super().__init__()
        w = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        net = models.resnet18(weights=w)

        if in_ch != 3:
            old = net.conv1
            new = nn.Conv2d(in_ch, 64, 7, 2, 3, bias=False)
            with torch.no_grad():
                new.weight.copy_(old.weight.sum(dim=1, keepdim=True)
                                 if in_ch == 1 else old.weight[:, :in_ch])
            net.conv1 = new

        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu)   # /2   64
        self.pool = net.maxpool                                   # /4
        self.enc1, self.enc2 = net.layer1, net.layer2             # /4 64, /8 128
        self.enc3, self.enc4 = net.layer3, net.layer4             # /16 256, /32 512

        self.up4 = Up(512, 256, 256)
        self.up3 = Up(256, 128, 128)
        self.up2 = Up(128, 64, 64)
        self.up1 = Up(64, 64, 32)
        self.up0 = Up(32, 0, 16)
        self.head = nn.Conv2d(16, n_classes, 1)

    def forward(self, x):
        s0 = self.stem(x)            # /2
        s1 = self.enc1(self.pool(s0))
        s2 = self.enc2(s1)
        s3 = self.enc3(s2)
        s4 = self.enc4(s3)

        d = self.up4(s4, s3)
        d = self.up3(d, s2)
        d = self.up2(d, s1)
        d = self.up1(d, s0)
        d = self.up0(d)
        return self.head(d)


# ---------------------------------------------------------------- losses ---

def dice_loss(logits, target, n_classes: int = 3, eps: float = 1.0):
    """Soft Dice over the foreground classes.

    Background is excluded: it is ~85% of every frame, so including it lets a
    model score well by predicting background everywhere, which is exactly the
    degenerate solution to avoid when the pubic symphysis is 1.5% of the image.
    """
    prob = F.softmax(logits, dim=1)
    oh = F.one_hot(target, n_classes).permute(0, 3, 1, 2).float()
    dims = (0, 2, 3)
    inter = (prob * oh).sum(dims)
    card = prob.sum(dims) + oh.sum(dims)
    dice = (2 * inter + eps) / (card + eps)
    return 1.0 - dice[1:].mean()


def combined_loss(logits, target, ce_weight=None, n_classes: int = 3):
    """Cross-entropy + Dice, the standard pairing for small structures."""
    ce = F.cross_entropy(logits, target, weight=ce_weight)
    return ce + dice_loss(logits, target, n_classes)


@torch.no_grad()
def dice_per_class(logits, target, n_classes: int = 3, eps: float = 1e-6):
    """Hard Dice per class, returned as a (n_classes,) tensor."""
    pred = logits.argmax(1)
    out = []
    for c in range(n_classes):
        p, t = pred == c, target == c
        inter = (p & t).sum().float()
        out.append((2 * inter + eps) / (p.sum() + t.sum() + eps))
    return torch.stack(out)
