import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride):
        super(ResBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(outchannel)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outchannel)
            )

    def forward(self, x):
        out = self.left(x)
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.ResBlock1 = ResBlock(64, 64, 1)
        self.ResBlock2 = ResBlock(64, 64, 1)
        self.ResBlock3 = ResBlock(64, 128, 2)
        self.ResBlock4 = ResBlock(128, 128, 1)
        self.ResBlock5 = ResBlock(128, 256, 2)
        self.ResBlock6 = ResBlock(256, 256, 1)
        self.ResBlock7 = ResBlock(256, 512, 2)
        self.ResBlock8 = ResBlock(512, 512, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.ResBlock1(x)
        x = self.ResBlock2(x)
        x = self.ResBlock3(x)
        x = self.ResBlock4(x)
        x = self.ResBlock5(x)
        x = self.ResBlock6(x)
        x = self.ResBlock7(x)
        x = self.ResBlock8(x)
        return x

def HandsResNet():
    return ResNet()