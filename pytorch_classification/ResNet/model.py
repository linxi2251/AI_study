from torchinfo import summary
from torch import nn
import torch


# 该block是针对与18layer 34layer的网络
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channel, out_channel, stride=1, downsample=None, **kwargs) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channel, out_channel, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channel, out_channel,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        # 捷径分支是否有block模块
        if self.downsample is not None:
            identity = self.downsample(x)
        # 主分支
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # 主干分支加上捷径分支和送入激活函数
        out += identity
        out = self.relu(out)

        return out

# 该block是针对于非18层与34层网络的


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channel, out_channel, stride=1, downsample=None, groups=1, width_per_group=64) -> None:
        super().__init__()
        width = int(out_channel * (width_per_group / 64.)) * groups
        self.conv1 = nn.Conv2d(
            in_channels=in_channel, out_channels=width, kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width)

        self.conv2 = nn.Conv2d(in_channels=width, out_channels=width,
                               kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width)

        self.conv3 = nn.Conv2d(in_channels=width, out_channels=out_channel *
                               self.expansion, kernel_size=1, stride=1, bias=False)  # unsqueeze channels

        self.bn3 = nn.BatchNorm2d(out_channel*self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, block_num_ls, num_classes=1000, include_top=True, groups=1, width_per_group=64) -> None:
        super().__init__()
        self.include_top = include_top
        self.in_channel = 64

        self.groups = groups
        self.width_per_group = width_per_group
        self.conv1 = nn.Conv2d(
            3, self.in_channel, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channel)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, block_num_ls[0])
        self.layer2 = self._make_layer(block, 128, block_num_ls[1], stride=2)
        self.layer3 = self._make_layer(block, 256, block_num_ls[2], stride=2)
        self.layer4 = self._make_layer(block, 512, block_num_ls[3], stride=2)

        if self.include_top:
            self.avg_pool = nn.AdaptiveAvgPool2d(
                (1, 1))  # output size = (1, 1)
            self.fc = nn.Linear(512*block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if self.include_top:
            x = self.avg_pool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)
        return x

    def _make_layer(self, block, out_channel, block_num, stride=1):
        downsample = None
        if stride != 1 or self.in_channel != out_channel * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel, out_channel * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channel * block.expansion)
            )

        layers = []
        # 残差块入口
        layers.append(block(self.in_channel,
                            out_channel,
                            downsample=downsample,
                            stride=stride,
                            groups=self.groups,
                            width_per_group=self.width_per_group
                            ))
        # 堆叠残差块
        self.in_channel = out_channel * block.expansion
        for _ in range(1, block_num):
            layers.append(block(
                self.in_channel,
                out_channel,
                groups=self.groups,
                width_per_group=self.width_per_group
            ))
        return nn.Sequential(*layers)


def resnet34(num_classes=1000, include_top=True):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top)


def resnet50(num_classes=1000, include_top=True):
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top)


def resnet101(num_classes=1000, include_top=True):
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes, include_top=include_top)


def resnet50_32x4d(num_classes=1000, include_top=True):
    groups = 32
    width_per_group = 4
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top, groups=groups, width_per_group=width_per_group)


if __name__ == '__main__':
    model = resnet34()
    summary(model)
