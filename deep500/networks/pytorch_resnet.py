'''ResNet in PyTorch.
copied and extended from: https://github.com/kuangliu/pytorch-cifar
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, in_channels=3):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


class Cifar10ResNet(nn.Module):
    def __init__(self, block, n, num_classes=10, in_channels=3):
        super(Cifar10ResNet, self).__init__()
        self.in_planes = 16
        num_blocks = n

        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks, stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks, stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks, stride=2)
        self.linear = nn.Linear(64 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet18(num_classes=10, in_channels=3):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, in_channels=in_channels)


def ResNet20(num_classes=10, in_channels=3):
    return Cifar10ResNet(BasicBlock, n=3, num_classes=num_classes, in_channels=in_channels)


def ResNet32(num_classes=10, in_channels=3):
    return Cifar10ResNet(BasicBlock, n=5, num_classes=num_classes, in_channels=in_channels)


def ResNet34(num_classes=10, in_channels=3):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, in_channels=in_channels)


def ResNet44(num_classes=10, in_channels=3):
    return Cifar10ResNet(BasicBlock, n=7, num_classes=num_classes, in_channels=in_channels)


def ResNet50(num_classes=10, in_channels=3):
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, in_channels=in_channels)


def ResNet56(num_classes=10, in_channels=3):
    return Cifar10ResNet(BasicBlock, n=9, num_classes=num_classes, in_channels=in_channels)


def ResNet101(num_classes=10, in_channels=3):
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes, in_channels=in_channels)


def ResNet110(num_classes=10, in_channels=3):
    return Cifar10ResNet(BasicBlock, n=18, num_classes=num_classes, in_channels=in_channels)


def ResNet152(num_classes=10, in_channels=3):
    return ResNet(Bottleneck, [3, 8, 36, 3], num_classes=num_classes, in_channels=in_channels)

_DEPTH_TO_FUNCTION = { 18: ResNet18, 20: ResNet20, 32: ResNet32, 34: ResNet34,
                       44: ResNet44, 50: ResNet50, 56: ResNet56, 101: ResNet101,
                       110: ResNet110, 152: ResNet152 }

def export_resnet(batch_size, depth=50, classes=10, file_path='resnet.onnx',
                  shape=(3, 32, 32)):
    if depth not in _DEPTH_TO_FUNCTION:
        raise ValueError('ResNet depth %d not defined' % depth)
        
    net = _DEPTH_TO_FUNCTION[depth](classes, shape[0])
    dummy_input = Variable(torch.randn(batch_size, *shape))

    torch.onnx.export(net, dummy_input, file_path, verbose=True)
    return file_path
