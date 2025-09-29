import torch.nn as nn

class SegHead(nn.Module):
    def __init__(self, in_ch, n_classes):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, n_classes, 1)
        )
    def forward(self, x):
        return self.head(x)

class ClsPresenceHead(nn.Module):
    def __init__(self, in_ch, n_labels):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(in_ch, n_labels)
    def forward(self, x):
        x = self.pool(x).flatten(1)
        return self.fc(x)
