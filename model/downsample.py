import torch
import torch.nn as nn

class Conv(nn.Module):
    def __init__(self, C_in, C_out):
        super(Conv, self).__init__()
        self.layer = nn.Sequential(

            nn.Conv2d(C_in, C_out, 1, 1),
            nn.BatchNorm2d(C_out),
            nn.LeakyReLU(),

            nn.Conv2d(C_out, C_out, 3, 1, 1),
            nn.BatchNorm2d(C_out),
            nn.LeakyReLU(),
        )

    def forward(self, x):
        return self.layer(x)

class DownChannel(nn.Module):
    def __init__(self, C_in, C_out):
        super(DownChannel, self).__init__()
        layers = []
        for i,j in zip(C_in,C_out):
            layers.append(Conv(i,j))
        
        self.layers = nn.ModuleList(layers)

    def forward(self,x):
        y = x
        for l in self.layers:
            y = l(y)
        return y

class DownSampling(nn.Module):
    def __init__(self, C):
        super(DownSampling, self).__init__()
        self.Down = nn.Sequential(
            nn.Conv2d(C, C, 3, 2, 1),
            nn.BatchNorm2d(C),
            nn.LeakyReLU()
        )

    def forward(self, x):
        return self.Down(x)

class Down(nn.Module):
    def __init__(self, C_in, C_out):
        super(Down, self).__init__()
        layers = []
        for i,j in zip(C_in,C_out):
            layers.append(Conv(i,j))
            layers.append(DownSampling(j))
        
        self.layers = nn.ModuleList(layers)

    def forward(self,x):
        y = x
        for l in self.layers:
            y = l(y)
        return y

if __name__ == '__main__':
    x = torch.rand(1,256,98,100)
    net1 = Down([256,512,512],[512,512,512])
    y = net1(x)
    print(y.shape) #1*512*13*13

    net2 = Down([256,512],[512,512])
    x = torch.rand(1,256,47,156)
    y = net2(x)
    print(y.shape) #1*512*12*39
