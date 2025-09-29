import torch
import torch.nn as nn

class ConvBNReLU(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, k, s, p, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.block(x)

class UNetEncoder(nn.Module):
    def __init__(self, in_ch=3, base=64, depth=4):
        super().__init__()
        self.depth = depth
        ch = base
        self.enc = nn.ModuleList()
        self.pools = nn.ModuleList()
        for d in range(depth):
            self.enc.append(ConvBNReLU(in_ch if d==0 else ch, ch))
            self.pools.append(nn.MaxPool2d(2))
            ch *= 2
        self.bottleneck = ConvBNReLU(ch//2, ch)
        self.out_channels = [base*(2**i) for i in range(depth)]

    def forward(self, x):
        feats = []
        for block, pool in zip(self.enc, self.pools):
            x = block(x)
            feats.append(x)
            x = pool(x)
        x = self.bottleneck(x)
        return feats, x

class UNetDecoder(nn.Module):
    def __init__(self, out_ch_per_stage, base=64, depth=4, dropout=0.0):
        super().__init__()
        ch = base*(2**depth)
        self.up = nn.ModuleList()
        self.dec = nn.ModuleList()
        for d in reversed(range(depth)):
            self.up.append(nn.ConvTranspose2d(ch, ch//2, 2, 2))
            self.dec.append(ConvBNReLU(ch//2 + out_ch_per_stage[d], ch//2))
            ch//=2
        self.out_ch = ch
        self.drop = nn.Dropout2d(dropout) if dropout>0 else nn.Identity()

    def forward(self, x, feats):
        for up, dec, skip in zip(self.up, self.dec, reversed(feats)):
            x = up(x)
            x = torch.cat([x, skip], dim=1)
            x = dec(x)
        return self.drop(x)
