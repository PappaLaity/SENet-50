import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Bloc SE ---
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.fc1 = nn.Linear(channels, channels // reduction, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(channels // reduction, channels, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        y = F.adaptive_avg_pool2d(x, 1).view(b, c)
        y = self.fc1(y)
        y = self.relu(y)
        y = self.fc2(y)
        y = self.sigmoid(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


# --- Bloc Bottleneck + SE ---
class SEBottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, mid_channels, stride=1, downsample=None, reduction=16):
        super(SEBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(mid_channels)

        self.conv2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=stride,
                               padding=1)
        self.bn2 = nn.BatchNorm2d(mid_channels)

        self.conv3 = nn.Conv2d(mid_channels, mid_channels * self.expansion, kernel_size=1)
        self.bn3 = nn.BatchNorm2d(mid_channels * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.se = SEBlock(mid_channels * self.expansion, reduction)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out = self.se(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


# --- SENet50 (binaire ou multi-classes) ---
class SENet50(nn.Module):
    def __init__(self, num_classes=1, reduction=16):
        """
        num_classes=1 -> classification binaire (sigmoid)
        num_classes>1 -> classification multi-classes (softmax/logits)
        """
        super(SENet50, self).__init__()
        self.inplanes = 64
        self.num_classes = num_classes

        # Étape initiale
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Stages
        self.layer1 = self._make_layer(SEBottleneck, 64, 3, reduction=reduction)
        self.layer2 = self._make_layer(SEBottleneck, 128, 4, stride=2, reduction=reduction)
        self.layer3 = self._make_layer(SEBottleneck, 256, 6, stride=2, reduction=reduction)
        self.layer4 = self._make_layer(SEBottleneck, 512, 3, stride=2, reduction=reduction)

        # Pooling + FC
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * SEBottleneck.expansion, num_classes)

    def _make_layer(self, block, mid_channels, blocks, stride=1, reduction=16):
        downsample = None
        if stride != 1 or self.inplanes != mid_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, mid_channels * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(mid_channels * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, mid_channels, stride, downsample, reduction))
        self.inplanes = mid_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, mid_channels, reduction=reduction))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        # --- Différence ici ---
        if self.num_classes == 1:  # Binaire
            return torch.sigmoid(x)
        else:  # Multi-classes
            return x  # logits (on applique CrossEntropyLoss)
        



