# -*- coding: utf-8 -*-
import numpy as np

from torch import nn
from torch.nn import functional as F
import torch
from torchvision import models
import torchvision
from torchsummary import summary 
from tensorboardX import SummaryWriter

writer = SummaryWriter()

class VGGBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, act_func=nn.ReLU(inplace=True)):
        super(VGGBlock, self).__init__()
        self.act_func = act_func
        self.conv1 = nn.Conv2d(in_channels, middle_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(middle_channels)
        self.conv2 = nn.Conv2d(middle_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act_func(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.act_func(out)

        return out


# class Unet(nn.Module):
#     def __init__(self, num_channels = 3):
#         super().__init__()

#         self.pool = nn.MaxPool2d(2, 2)
#         self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
#         self.relu = nn.ReLU()
#         self.dropout = nn.Dropout(0.5)

#         self.conv11 = nn.Conv2d( num_channels, 64, kernel_size = 3, padding=1)
#         self.conv12 = nn.Conv2d( 64, 64, kernel_size = 3, padding=1)

#         self.conv21 = nn.Conv2d( 64, 128 ,kernel_size = 3, padding=1 )
#         self.conv22 = nn.Conv2d( 128, 128 ,kernel_size = 3, padding=1 )

#         self.conv31 = nn.Conv2d( 128, 256, kernel_size= 3, padding= 1)
#         self.conv32 = nn.Conv2d( 256, 256 ,kernel_size = 3, padding=1 )

#         self.conv41 = nn.Conv2d( 256, 512, kernel_size = 3, padding = 1)
#         self.conv42 = nn.Conv2d( 512, 512, kernel_size = 3, padding = 1)

#         self.conv51 = nn.Conv2d( 512, 1024 ,kernel_size =3, padding=1)
#         self.conv52 = nn.Conv2d( 1024, 1024 ,kernel_size =3, padding=1)
        
#         self.upconv1 = nn.Conv2d( 1024, 512, kernel_size =1 )

#         self.conv61 = nn.Conv2d( 512*2, 512, kernel_size = 3, padding=1)
#         self.conv62 = nn.Conv2d( 512, 512, kernel_size = 3, padding=1)
        
#         self.upconv2 = nn.Conv2d( 512, 256 , kernel_size = 1)

#         self.conv71 = nn.Conv2d( 256*2, 256, kernel_size = 3, padding=1)
#         self.conv72 = nn.Conv2d( 256, 256, kernel_size = 3, padding=1)        

#         self.upconv3 = nn.Conv2d( 256, 128 , kernel_size = 1)

#         self.conv81 = nn.Conv2d( 128*2, 128, kernel_size = 3, padding=1)
#         self.conv82 = nn.Conv2d( 128, 128, kernel_size = 3, padding=1)
        
    
#         self.upconv4 = nn.Conv2d( 128, 64 , kernel_size = 1)

#         self.conv91 = nn.Conv2d( 64*2, 64, kernel_size = 3, padding=1)
#         self.conv92 = nn.Conv2d( 64, 64, kernel_size = 3, padding=1)

#         self.conv101 = nn.Conv2d( 64,2 , kernel_size = 3, padding=1 )

#         self.conv102 = nn.Conv2d(2, 1,  kernel_size = 1 )
 

#     def forward(self, x):
        
#         x11 = self.conv11(x)
#         x12 = self.conv12(x11)
#         x13 = self.pool(x12)

#         x21 = self.conv21(x13)
#         x22 = self.conv22(x21)
#         x23 = self.pool(x22)

#         x31 = self.conv31(x23)
#         x32 = self.conv32(x31)
#         x33 = self.pool(x32)

#         x41 = self.conv41(x33)
#         x42 = self.conv42(x41)

#         x43 = self.dropout(x42)
#         x44 = self.pool(x43)

#         x51 = self.conv51(x44)
#         x52 = self.conv52(x51)

#         x53 = self.dropout(x52)
       
#         x54 = self.up(x53)
#         x55 = self.upconv1(x54)
#         x56 = torch.cat((x55,x43),dim=1)

#         x61 = self.conv61(x56)
#         x62 = self.conv62(x61)

#         x63 = self.up(x62)
#         x64 = self.upconv2(x63)
#         x65 = torch.cat((x64,x32),dim=1)

#         x71 = self.conv71(x65)
#         x72 = self.conv72(x71)

#         x73 = self.up(x72)
#         x74 = self.upconv3(x73)
#         x75 = torch.cat((x74,x22),dim = 1)

#         x81 = self.conv81(x75)
#         x82 = self.conv82(x81)

#         x83 = self.up(x82)
#         x84 = self.upconv4(x83)
#         x85 = torch.cat((x84,x12),dim = 1)

#         x91 = self.conv91(x85)
#         x92 = self.conv92(x91)

#         x101 = self.conv101(x92)
#         x102 = self.conv102(x101)

#         out = F.sigmoid(x102)

#         return out 


class conv_block(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(conv_block,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )


    def forward(self,x):
        x = self.conv(x)
        return x

class up_conv(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(up_conv,self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in,ch_out,kernel_size=3,stride=1,padding=1,bias=True),
		    nn.BatchNorm2d(ch_out),
			nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.up(x)
        return x


class Unet(nn.Module):
    def __init__(self,img_ch=3,output_ch=1):
        super(Unet,self).__init__()
        
        self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)

        self.Conv1 = conv_block(ch_in=img_ch,ch_out=64)
        self.Conv2 = conv_block(ch_in=64,ch_out=128)
        self.Conv3 = conv_block(ch_in=128,ch_out=256)
        self.Conv4 = conv_block(ch_in=256,ch_out=512)
        self.Conv5 = conv_block(ch_in=512,ch_out=1024)

        self.Up5 = up_conv(ch_in=1024,ch_out=512)
        self.Up_conv5 = conv_block(ch_in=1024, ch_out=512)

        self.Up4 = up_conv(ch_in=512,ch_out=256)
        self.Up_conv4 = conv_block(ch_in=512, ch_out=256)
        
        self.Up3 = up_conv(ch_in=256,ch_out=128)
        self.Up_conv3 = conv_block(ch_in=256, ch_out=128)
        
        self.Up2 = up_conv(ch_in=128,ch_out=64)
        self.Up_conv2 = conv_block(ch_in=128, ch_out=64)

        self.Conv_1x1 = nn.Conv2d(64,output_ch,kernel_size=1,stride=1,padding=0)


    def forward(self,x):
        # encoding path
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)
        
        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        d5 = torch.cat((x4,d5),dim=1)
        
        d5 = self.Up_conv5(d5)
        
        d4 = self.Up4(d5)
        d4 = torch.cat((x3,d4),dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2,d3),dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1,d2),dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)

        d1 = F.sigmoid(d1)

        return d1




class Unetpp(nn.Module):
    def __init__(self):
        super().__init__()

        nb_filter = [32, 64, 128, 256, 512]

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = VGGBlock( 3, nb_filter[0], nb_filter[0])
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0 = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3])
        self.conv4_0 = VGGBlock(nb_filter[3], nb_filter[4], nb_filter[4])

        self.conv0_1 = VGGBlock(nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_1 = VGGBlock(nb_filter[1]+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_1 = VGGBlock(nb_filter[2]+nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv3_1 = VGGBlock(nb_filter[3]+nb_filter[4], nb_filter[3], nb_filter[3])

        self.conv0_2 = VGGBlock(nb_filter[0]*2+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_2 = VGGBlock(nb_filter[1]*2+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_2 = VGGBlock(nb_filter[2]*2+nb_filter[3], nb_filter[2], nb_filter[2])

        self.conv0_3 = VGGBlock(nb_filter[0]*3+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_3 = VGGBlock(nb_filter[1]*3+nb_filter[2], nb_filter[1], nb_filter[1])

        self.conv0_4 = VGGBlock(nb_filter[0]*4+nb_filter[1], nb_filter[0], nb_filter[0])

        self.final = nn.Conv2d(nb_filter[0], 1, kernel_size=1)


    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))

        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))

        
        output = self.final(x0_4)
        out = F.sigmoid(output)

        return out


device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
model = Unetpp().to(device)

summary( model, (3, 512, 512))