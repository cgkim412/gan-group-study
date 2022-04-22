import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from timm import create_model
from adversarial_nets import Dropout


def upsample_like(x1, x2, mode="bilinear", align_corners=False):
    b, c, h, w = x2.shape
    return nn.Upsample(size=(h, w), mode=mode, align_corners=align_corners)(x1)

def upconcat(x1, x2):
    return torch.cat([upsample_like(x1, x2), x2], dim=1)

  
class Conv3x3(nn.Module):
    def __init__(self, cin, cout, droprate=0.0, activation=nn.GELU) -> None:
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, 3, padding=1, bias=False)
        self.norm = nn.GroupNorm(1, cout)
        self.dropout = Dropout(droprate)
        self.act = activation()

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.dropout(x)
        x = self.act(x)
        return x


class DoubleConv3x3(nn.Module):
    def __init__(self, cin, cout, droprate=0.0) -> None:
        super().__init__()
        self.conv1 = Conv3x3(cin, cout, droprate)
        self.conv2 = Conv3x3(cout, cout, droprate, activation=nn.Identity)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.conv1(x)
        x = x + self.conv2(x)
        return self.act(x)


class Encoder(nn.Module):
    def __init__(self, cin, pretrained=True) -> None:
        super().__init__()
        self.inconv = Conv3x3(cin, 16)
        self.backbone = create_model("efficientnet_b4", pretrained, in_chans=cin, features_only=True)

    def forward(self, x):
        """
        B2: 16-24-48-120-352
        B3: 24-32-48-136-384
        B4: 24-32-56-160-448
        """
        return self.inconv(x), *self.backbone(x)


class Decoder(nn.Module):
    def __init__(self, cout, encoder_config=(16, 24, 32, 56, 160, 448)) -> None:
        super().__init__()
        dims = encoder_config
        self.convs = nn.ModuleList(
            [
                DoubleConv3x3(dims[5] + dims[4], 256, droprate=0.5), # 1/16
                DoubleConv3x3(256 + dims[3], 192), # 1/8
                DoubleConv3x3(192 + dims[2], 128), # 1/4
                DoubleConv3x3(128 + dims[1], 96), # 1/2
                Conv3x3(96 + dims[0], 64) # 1/1
            ]
        )
        self.outconv = nn.Conv2d(64, cout, 3, padding=1)

    def forward(self, features):
        *features, x = features  # x = bottleneck
        for x2, conv in zip(reversed(features), self.convs):
            x = upconcat(x, x2)
            x = conv(x)
        return self.outconv(x)


class Generator(nn.Module):
    def __init__(self, cin, cout, activation=nn.Tanh) -> None:
        super().__init__()
        self.encoder = Encoder(cin)
        self.decoder = Decoder(cout)
        self.act = activation()

    def forward(self, x):
        x = self.encoder(x)
        logits = self.decoder(x)
        return self.act(logits)


class DownConv(nn.Module):
    def __init__(self, cin, cout) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 4, 2, padding=1)
        self.conv2 = nn.Conv2d(cout, cout, 3, padding=1)
        self.norm1 = nn.GroupNorm(1, cout)
        self.norm2 = nn.GroupNorm(1, cout)
        self.act = nn.LeakyReLU(0.2)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act(x)
        shortcut = x
        x = self.conv2(x)
        x = self.norm2(x)
        return self.act(shortcut + x)


class Discriminator(nn.Module):
    def __init__(self, cin) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            DownConv(cin, 64),
            DownConv(64, 128),
            DownConv(128, 256),
            DownConv(256, 512),
            nn.Conv2d(512, 1, 4, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.layers(x)
