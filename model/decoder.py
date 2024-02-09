import torch
import torch.nn as nn

class decoder_16to121_bev(nn.Module):
    def __init__(self,args,last_channel=1):
        super(decoder_16to121_bev, self).__init__()
        self.deconv1 = nn.ConvTranspose2d(256,128,kernel_size=(3,3),stride=(2,2),padding=(1,1))
        self.deconv2 = nn.ConvTranspose2d(128,64,kernel_size=(3,3),stride=(2,2),padding=(1,1))
        self.deconv3 = nn.ConvTranspose2d(64,last_channel,kernel_size=(3,3),stride=(2,2),padding=(1,1))
        self.lrelu = nn.LeakyReLU()
        self.sig = nn.Sigmoid()

    def forward(self,x):
        x = self.lrelu(self.deconv1(x))
        x = self.lrelu(self.deconv2(x))
        return self.sig(self.deconv3(x))

class decoder_16to121_bid(nn.Module):
    def __init__(self,args,last_channel=1):
        super(decoder_16to121_bid, self).__init__()
        self.deconv1 = nn.ConvTranspose2d(2*args.d_model,512,kernel_size=(3,3),stride=(2,2),padding=(1,1))
        self.deconv2 = nn.ConvTranspose2d(512,256,kernel_size=(3,3),stride=(2,2),padding=(1,1))
        self.deconv3 = nn.ConvTranspose2d(256,last_channel,kernel_size=(3,3),stride=(2,2),padding=(1,1))
        self.lrelu = nn.LeakyReLU()
        self.sig = nn.Sigmoid()

    def forward(self,x):
        x = self.lrelu(self.deconv1(x))
        x = self.lrelu(self.deconv2(x))
        return self.sig(self.deconv3(x))

class decoder_16to121(nn.Module):
    def __init__(self,args,last_channel=1):
        super(decoder_16to121, self).__init__()
        self.deconv1 = nn.ConvTranspose2d(args.d_model,256,kernel_size=(3,3),stride=(2,2),padding=(1,1))
        self.deconv2 = nn.ConvTranspose2d(256,128,kernel_size=(3,3),stride=(2,2),padding=(1,1))
        self.deconv3 = nn.ConvTranspose2d(128,last_channel,kernel_size=(3,3),stride=(2,2),padding=(1,1))
        self.lrelu = nn.LeakyReLU()
        self.sig = nn.Sigmoid()

    def forward(self,x):
        x = self.lrelu(self.deconv1(x))
        x = self.lrelu(self.deconv2(x))
        return self.sig(self.deconv3(x))


class decoder_16to121_vis(nn.Module):
    def __init__(self,args):
        super(decoder_16to121_vis, self).__init__()
        self.deconv1 = nn.ConvTranspose2d(args.d_model,256,kernel_size=(3,3),stride=(2,2),padding=(1,1))
        self.deconv2 = nn.ConvTranspose2d(256,128,kernel_size=(3,3),stride=(2,2),padding=(1,1))
        self.deconv3 = nn.ConvTranspose2d(128,2,kernel_size=(3,3),stride=(2,2),padding=(1,1))
        self.lrelu = nn.LeakyReLU()
        self.sig = nn.Sigmoid()

    def forward(self,x):
        x = self.lrelu(self.deconv1(x))
        x = self.lrelu(self.deconv2(x))
        return self.sig(self.deconv3(x))

class decoder_16to121_vis_bidir_concate(nn.Module):
    def __init__(self,args):
        super(decoder_16to121_vis_bidir_concate, self).__init__()
        self.deconv1 = nn.ConvTranspose2d(2*args.d_model,512,kernel_size=(3,3),stride=(2,2),padding=(1,1))
        self.deconv2 = nn.ConvTranspose2d(512,128,kernel_size=(3,3),stride=(2,2),padding=(1,1))
        self.deconv3 = nn.ConvTranspose2d(128,2,kernel_size=(3,3),stride=(2,2),padding=(1,1))
        self.lrelu = nn.LeakyReLU()
        self.sig = nn.Sigmoid()

    def forward(self,x):
        x = self.lrelu(self.deconv1(x))
        x = self.lrelu(self.deconv2(x))
        return self.sig(self.deconv3(x))

if __name__ == '__main__':
    class A():
        def __init__(self):
            self.d_model = 512

    args = A()
    net = decoder_16to121_bid(args)
    x = torch.rand(1,args.d_model,16,16)
    print(net(x).shape)