import torch.nn as nn


class ResidualBlock(nn.Module):

    def __init__(self, in_filters, out_filters):
        super().__init__()
        self.conv1 = nn.Conv2d(in_filters, out_filters, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_filters)
        self.conv2 = nn.Conv2d(out_filters, out_filters, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_filters)
        self.relu = nn.ReLU()

        self.skip = nn.Sequential()
        if in_filters != out_filters:
            self.skip = nn.Sequential(
                nn.Conv2d(in_filters, out_filters, 1, bias=False),
                nn.BatchNorm2d(out_filters),
            )

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = out + self.skip(x)
        out = self.relu(out)
        return out


class LightweightResNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU()
        self.block1 = ResidualBlock(32, 32)
        self.block2 = ResidualBlock(32, 64)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(64, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x
