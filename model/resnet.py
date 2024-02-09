import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F

class Resnet50(nn.Module):
    def __init__(self):
        super(Resnet50, self).__init__()
        self.model = models.resnet50(pretrained=True)
        del self.model.avgpool
        del self.model.fc

        
    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)

        return x


def conv3x3(in_planes, out_planes, stride=1, dilation=1):
    """3x3 convolution with padding"""

    # Fractional strides correspond to transpose convolution
    if stride < 1:
        stride = int(round(1 / stride))
        kernel_size = stride + 2
        padding = int((dilation * (kernel_size - 1) - stride + 1) / 2)
        return nn.ConvTranspose2d(
            in_planes, out_planes, kernel_size, stride, padding, 
            output_padding=0, dilation=dilation, bias=False)
    
    # Otherwise return normal convolution
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=int(stride),
                     dilation=dilation, padding=dilation, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""

    # Fractional strides correspond to transpose convolution
    if int(1 / stride) > 1:
        stride = int(1 / stride)
        return nn.ConvTranspose2d(
            in_planes, out_planes, kernel_size=stride, stride=stride,bias=False)
    
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=1, stride=int(stride), bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, dilation=1):
        super(BasicBlock, self).__init__()

        self.conv1 = conv3x3(inplanes, planes, stride, dilation)
        self.bn1 = nn.GroupNorm(16, planes)

        self.conv2 = conv3x3(planes, planes, 1, dilation)
        self.bn2 = nn.GroupNorm(16, planes)

        if stride != 1 or inplanes != planes:
            self.downsample = nn.Sequential(
                conv1x1(inplanes, planes, stride), nn.GroupNorm(16, planes))
        else:
            self.downsample = None


    def forward(self, x):
        identity = x

        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = F.relu(out, inplace=True)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.GroupNorm(16, planes)
        self.conv2 = conv3x3(planes, planes, stride, dilation)
        self.bn2 = nn.GroupNorm(16, planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.GroupNorm(16, planes * self.expansion)

        if stride != 1 or inplanes != planes * self.expansion:
            self.downsample = nn.Sequential(
                conv1x1(inplanes, planes * self.expansion, stride), 
                nn.GroupNorm(16, planes * self.expansion))
        else:
            self.downsample = None

    def forward(self, x):
        identity = x

        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = F.relu(self.bn2(self.conv2(out)), inplace=True)
        out = self.bn3(self.conv3(out))
 
        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = F.relu(out)

        return out


class ResNetLayer(nn.Sequential):

    def __init__(self, in_channels, channels, num_blocks, stride=1, 
                 dilation=1, blocktype='bottleneck'):
        
        # Get block type
        if blocktype == 'basic':
            block = BasicBlock
        elif blocktype == 'bottleneck':
            block = Bottleneck
        else:
            raise Exception("Unknown residual block type: " + str(blocktype))

        if blocktype == 'basic':
            expansion = 1
        elif blocktype == 'bottleneck':
            expansion = 4
        
        # Construct layers
        layers = [block(in_channels, channels, stride, dilation)]
        for _ in range(1, num_blocks):
            layers.append(block(channels * expansion, channels, 1, dilation))

        self.in_channels = in_channels
        self.out_channels = channels * expansion

        super(ResNetLayer, self).__init__(*layers)

if __name__ == '__main__':
    x = torch.rand(1,3,64,128)
    net = Resnet50()
    y = net(x)
    print(y.shape) #1*3*2*4

    '''
    for name,i in net.named_modules():
        print(name)
    '''