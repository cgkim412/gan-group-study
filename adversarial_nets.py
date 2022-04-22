import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial


"""
1. Conv - BN - ReLU 가 기본이다 (실제론 IN)
2. Conv - BN - Dropout(0.5) - ReLU 도 있다
3. 모든 conv는 kernel_size=4, stride=2

[G] 64-128-256-512-512-512-512-512 (다운샘플 8번...)
4. 인코더의 첫 번째 conv에는 BN이 없다
5. 인코더는 LeakyReLU(0.2), 디코더는 일반 ReLU를 사용한다.

[D] 64-128-256-512
6. 첫 번째 conv에는 BN이 없다
7. 전부 LeakyReLU(0.2)
"""

"""
Enc     Dec
3       3
64      128
128     256
256     512
512     1024
512     1024
512     1024
512     1024
    512
"""

class Dropout(nn.Module):
    def __init__(self, droprate=0.0) -> None:
        super().__init__()
        self.dropout = partial(F.dropout, p=droprate) if droprate > 0 else nn.Identity()
    
    def forward(self, x):
        return self.dropout(x)

class _Conv4x4(nn.Module):
    def __init__(
        self, cin, cout, leaky, normalize=True, droprate=0.0, stride=2, conv_layer=nn.Conv2d, norm_layer=nn.BatchNorm2d
    ) -> None:
        super().__init__()
        self.conv = conv_layer(cin, cout, 4, stride, 1)
        self.norm = norm_layer(cout, track_running_stats=False, affine=False) if normalize else nn.Identity()
        self.act = nn.LeakyReLU(0.2) if leaky else nn.ReLU()
        self.dropout = Dropout(droprate)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.dropout(x)
        x = self.act(x)
        return x

class DownConv4x4(_Conv4x4):
    def __init__(self, cin, cout, leaky=True, normalize=True, droprate=0.0, stride=2) -> None:
        super().__init__(cin, cout, leaky, normalize, droprate, stride)


class UpConv4x4(_Conv4x4):
    def __init__(self, cin, cout, leaky=False, normalize=True, droprate=0.0, stride=2) -> None:
        super().__init__(cin, cout, leaky, normalize, droprate, stride, nn.ConvTranspose2d)


class Encoder(nn.Module):
    """
    C64-C128-C256-C512-C512-C512-C512-C512
    """
    def __init__(self, cin, n_blocks=8, min_dim=64, max_dim=512) -> None:
        super().__init__()
        self.convs = nn.ModuleList([DownConv4x4(cin, min_dim, normalize=False)])
        for i in range(n_blocks - 1):
            din = min(2 ** i * min_dim, max_dim)
            dout = min(2 * din, max_dim)
            normalize = i < 6
            self.convs.append(DownConv4x4(din, dout, normalize=normalize))

    def forward(self, x):
        features = []
        for conv in self.convs:
            x = conv(x)
            features.append(x)
        return features  # 고해상도가 앞


class Decoder(nn.Module):
    """
    CD512-CD1024-CD1024-C1024-C1024-C512-C256-C128
    """
    def __init__(self, cout) -> None:
        super().__init__()
        self.convs = nn.ModuleList(
            [
                UpConv4x4(512, 512, droprate=0.5),
                UpConv4x4(1024, 512, droprate=0.5),
                UpConv4x4(1024, 512, droprate=0.5),
                UpConv4x4(1024, 512),
                UpConv4x4(1024, 256),
                UpConv4x4(512, 128),
                UpConv4x4(256, 64),
            ]
        )
        self.outconv = nn.ConvTranspose2d(128, cout, 4, 2, 1)

    def forward(self, features):
        *features, x = features  # x = bottleneck
        for feat, conv in zip(reversed(features), self.convs):
            x = conv(x)
            x = torch.cat((feat, x), dim=1)
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


class Discriminator(nn.Module):
    """
    The 70x70 version: C64-C128-C256-C512
    """
    def __init__(self, cin) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            DownConv4x4(cin, 64, normalize=False),
            DownConv4x4(64, 128),
            DownConv4x4(128, 256),
            DownConv4x4(256, 512, stride=1),
            nn.Conv2d(512, 1, 4, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.layers(x)

