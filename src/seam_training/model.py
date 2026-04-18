import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    @staticmethod
    def make_group_norm(num_channels, max_groups=8):
        num_groups = min(max_groups, num_channels)
        while num_channels % num_groups != 0 and num_groups > 1:
            num_groups -= 1
        return nn.GroupNorm(num_groups, num_channels)

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            self.make_group_norm(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            self.make_group_norm(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.layers(x)


class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = DoubleConv(in_channels, out_channels)
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        features = self.conv(x)
        return features, self.pool(features)


class AttentionGate(nn.Module):
    def __init__(self, x_channels, g_channels, inter_channels):
        super().__init__()
        self.wx = nn.Sequential(
            nn.Conv2d(x_channels, inter_channels, kernel_size=1, bias=False),
            DoubleConv.make_group_norm(inter_channels),
        )
        self.wg = nn.Sequential(
            nn.Conv2d(g_channels, inter_channels, kernel_size=1, bias=False),
            DoubleConv.make_group_norm(inter_channels),
        )
        self.psi = nn.Sequential(
            nn.Conv2d(inter_channels, 1, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, g):
        if x.shape[-2:] != g.shape[-2:]:
            g = F.interpolate(g, size=x.shape[-2:], mode="bilinear", align_corners=False)
        attention = self.relu(self.wx(x) + self.wg(g))
        alpha = self.psi(attention)
        return alpha * x


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.attention = AttentionGate(
            x_channels=skip_channels,
            g_channels=out_channels,
            inter_channels=max(out_channels // 2, 1),
        )
        self.conv = DoubleConv(out_channels + skip_channels, out_channels)

    def forward(self, x, skip):
        x = self.up(x)
        x = UNet._align_to(x, skip)
        gated_skip = self.attention(skip, x)
        return self.conv(torch.cat([x, gated_skip], dim=1))


class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, base_channels=32):
        super().__init__()
        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4
        c4 = base_channels * 8
        c5 = base_channels * 16

        self.enc1 = DoubleConv(in_channels, c1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.enc2 = DoubleConv(c1, c2)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.enc3 = DoubleConv(c2, c3)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.enc4 = DoubleConv(c3, c4)
        self.pool4 = nn.MaxPool2d(2, 2)
        self.bottleneck = DoubleConv(c4, c5)
        self.up4 = nn.ConvTranspose2d(c5, c4, 2, 2)
        self.dec4 = DoubleConv(c4 + c4, c4)
        self.up3 = nn.ConvTranspose2d(c4, c3, 2, 2)
        self.dec3 = DoubleConv(c3 + c3, c3)
        self.up2 = nn.ConvTranspose2d(c3, c2, 2, 2)
        self.dec2 = DoubleConv(c2 + c2, c2)
        self.up1 = nn.ConvTranspose2d(c2, c1, 2, 2)
        self.dec1 = DoubleConv(c1 + c1, c1)
        self.out_conv = nn.Conv2d(c1, out_channels, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))
        b = self.bottleneck(self.pool4(e4))
        d4 = self.dec4(torch.cat([self._align_to(self.up4(b), e4), e4], dim=1))
        d3 = self.dec3(torch.cat([self._align_to(self.up3(d4), e3), e3], dim=1))
        d2 = self.dec2(torch.cat([self._align_to(self.up2(d3), e2), e2], dim=1))
        d1 = self.dec1(torch.cat([self._align_to(self.up1(d2), e1), e1], dim=1))
        return self.out_conv(d1)

    @staticmethod
    def _align_to(source, target):
        if source.shape[-2:] == target.shape[-2:]:
            return source
        return F.interpolate(source, size=target.shape[-2:], mode="bilinear", align_corners=False)


class AttentionUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, base_channels=64):
        super().__init__()
        c1 = base_channels
        c2 = 128 if base_channels == 64 else base_channels * 2
        c3 = 256 if base_channels == 64 else base_channels * 4
        c4 = 512 if base_channels == 64 else base_channels * 8
        c5 = 1024 if base_channels == 64 else base_channels * 16

        self.stem = DoubleConv(in_channels, c1)
        self.enc2 = EncoderBlock(c1, c2)
        self.enc3 = EncoderBlock(c2, c3)
        self.enc4 = EncoderBlock(c3, c4)
        self.pool4 = nn.MaxPool2d(2, 2)
        self.bottleneck = DoubleConv(c4, c5)

        self.dec4 = DecoderBlock(c5, c4, c4)
        self.dec3 = DecoderBlock(c4, c3, c3)
        self.dec2 = DecoderBlock(c3, c2, c2)
        self.dec1 = DecoderBlock(c2, c1, c1)
        self.out_conv = nn.Conv2d(c1, out_channels, kernel_size=1)

    def forward(self, x):
        e1 = self.stem(x)
        e2, p2 = self.enc2(F.max_pool2d(e1, kernel_size=2, stride=2))
        e3, p3 = self.enc3(p2)
        e4, p4 = self.enc4(p3)
        b = self.bottleneck(self.pool4(e4))
        d4 = self.dec4(b, e4)
        d3 = self.dec3(d4, e3)
        d2 = self.dec2(d3, e2)
        d1 = self.dec1(d2, e1)
        return self.out_conv(d1)


def build_model(model_name="unet", in_channels=1, out_channels=1, base_channels=32):
    normalized = str(model_name).strip().lower()
    if normalized == "unet":
        return UNet(in_channels=in_channels, out_channels=out_channels, base_channels=base_channels)
    if normalized == "attention_unet":
        return AttentionUNet(in_channels=in_channels, out_channels=out_channels, base_channels=base_channels)
    raise ValueError(f"Unsupported model_name: {model_name!r}")


class SegmentationCriterion(nn.Module):
    def __init__(self, pos_weight=3.0, bce_weight=1.0, dice_weight=1.0):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.register_buffer("pos_weight", torch.tensor(float(pos_weight), dtype=torch.float32))

    def forward(self, pred, targets, valid_mask=None, smooth=1e-6):
        if valid_mask is None:
            valid_mask = torch.ones_like(targets)
        valid_mask = valid_mask.float()
        bce_map = nn.functional.binary_cross_entropy_with_logits(
            pred,
            targets,
            pos_weight=self.pos_weight,
            reduction="none",
        )
        bce = (bce_map * valid_mask).sum() / valid_mask.sum().clamp_min(1.0)
        probs = torch.sigmoid(pred)
        probs = probs.contiguous().view(probs.size(0), -1)
        targets = targets.contiguous().view(targets.size(0), -1)
        valid_mask = valid_mask.contiguous().view(valid_mask.size(0), -1)
        intersection = (probs * targets * valid_mask).sum(dim=1)
        union = (probs * valid_mask).sum(dim=1) + (targets * valid_mask).sum(dim=1)
        dice = 1 - (2.0 * intersection + smooth) / (union + smooth)
        dice = dice.mean()
        return self.bce_weight * bce + self.dice_weight * dice
