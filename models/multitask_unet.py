import torch.nn as nn
from .unet_backbone import UNetEncoder, UNetDecoder
from .decoders import SegHead, ClsPresenceHead

class MultiTaskUNet(nn.Module):
    def __init__(self, in_ch=3, base=64, depth=4, dropout=0.1, n_seg_classes=1, n_cls_labels=0):
        super().__init__()
        self.encoder = UNetEncoder(in_ch=in_ch, base=base, depth=depth)
        self.decoder = UNetDecoder(out_ch_per_stage=self.encoder.out_channels, base=base, depth=depth, dropout=dropout)
        self.seg_head = SegHead(base, n_seg_classes) if n_seg_classes>0 else None
        self.cls_head = ClsPresenceHead(base, n_cls_labels) if n_cls_labels>0 else None

        # project to 'base' channels if needed
        self.project = nn.Conv2d(self.decoder.out_ch, base, 1) if self.decoder.out_ch != base else nn.Identity()

    def forward(self, x):
        feats, bott = self.encoder(x)
        x = self.decoder(bott, feats)
        x = self.project(x)
        out = {}
        if self.seg_head:
            out['segmentation'] = self.seg_head(x)
        if self.cls_head:
            out['classification'] = self.cls_head(x)
        return out
