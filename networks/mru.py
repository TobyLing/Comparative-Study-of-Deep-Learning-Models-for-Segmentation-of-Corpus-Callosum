
import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F

from torchsummary import summary 

from functools import partial
from tensorboardX import SummaryWriter
import numpy as np


writer = SummaryWriter()

# nonlinearity = partial(F.relu, inplace=True)


class multiresblock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(multiresblock,self).__init__()

        self.alpha = 1.67 
        self.W = self.alpha * out_channels

        self.channels = int(self.W*0.167) + int(self.W*0.333) + int(self.W*0.5)

        self.conv1 = nn.Conv2d(in_channels,self.channels, kernel_size= 1)
        self.batch_norm1 = nn.BatchNorm2d(self.channels)

        self.conv2 = nn.Conv2d(in_channels, int(self.W*0.167), kernel_size= 3, padding=1)
        self.batch_norm2 = nn.BatchNorm2d(int(self.W*0.167))

        self.conv3 = nn.Conv2d( int(self.W*0.167), int(self.W*0.333), kernel_size =3, padding=1 )
        self.batch_norm3 = nn.BatchNorm2d(int(self.W*0.333))
        
        self.conv4 = nn.Conv2d( int(self.W*0.333), int(self.W*0.5), kernel_size =3, padding=1 )
        self.batch_norm4 = nn.BatchNorm2d(int(self.W*0.5))

        self.relu = nn.ReLU()


    def forward(self,x): 
        y = self.conv1(x)
        y = self.batch_norm1(y) 

        a = self.conv2(x)
        a = self.batch_norm2(a)
        a = self.relu(a)

        b = self.conv3(a)
        b = self.batch_norm3(b)
        b = self.relu(b)

        c = self.conv4(b)
        c = self.batch_norm4(c)
        c = self.relu(c)
        
        out = torch.cat((a,b,c), dim=1)
        out = self.batch_norm1(out)

        out = torch.add(y,out)

        out = self.relu(out)

        out = self.batch_norm1(out)

        return out

class respath(nn.Module):

    def __init__(self, in_channels,out_channels):
        super(respath,self).__init__()

        self.conv5 = nn.Conv2d(in_channels,out_channels,kernel_size=1)
        self.batch_norm = nn.BatchNorm2d(out_channels)

        self.conv6 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        self.relu2 = nn.ReLU()
        

    def forward(self,x): 

        y = self.conv5(x)
        y = self.batch_norm(y)

        out = self.conv6(x)
        out = self.batch_norm(out)

        out = torch.add(y,out)

        out = self.relu2(out)

        out = self.batch_norm(out)

        return out

# x = 0 
def channels(x): 
    y  = 1.67 * x
    return int(y*0.167) + int(y*0.333) + int(y*0.5)



class MRU(nn.Module):      
    def __init__(self, num_classes= 1, num_channels=3):
        super(MRU, self).__init__()

        self.multiresblock1 = multiresblock(num_channels,32)
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.respath11 = respath( channels(32) , 32)
        self.respath12 = respath( 32 , 32 ) # 4

        self.multiresblock2 = multiresblock( channels(32),32*2)
        self.respath21 = respath( channels(32*2), 32*2)
        self.respath22 = respath( 32 *2  , 32 * 2 ) # 3 

        
        self.multiresblock3 = multiresblock(channels(32*2),32* 4 )
        self.respath31 = respath( channels(32*4) , 32*4)
        self.respath32 = respath( 32 *4  , 32 * 4 ) # 2

        self.multiresblock4 = multiresblock(channels(32*4),32* 8 )
        self.respath41 = respath( channels(32*8) , 32*8)
        self.respath42 = respath( 32 *8  , 32 * 8 ) # 1

        self.multiresblock5 = multiresblock(channels(32*8), 32 * 16 )

        self.up1 = nn.ConvTranspose2d( channels(32*16),  32 *8 , kernel_size =2 ,stride=2, padding= 0)
        self.multiresblock6 = multiresblock( 2 *32 *8 ,32 *8)

        self.up2 = nn.ConvTranspose2d( channels(32*8) ,  32 *4 , kernel_size =2 ,stride=2, padding= 0)
        self.multiresblock7 = multiresblock( 2 *32 *4 ,32 *4)

        self.up3 = nn.ConvTranspose2d( channels(32*4) ,  32 *2 , kernel_size =2 ,stride=2, padding= 0)  
        self.multiresblock8 = multiresblock( 2 *32 *2 ,32 *2)

        self.up4 = nn.ConvTranspose2d( channels(32*2) ,  32  , kernel_size =2 ,stride=2, padding= 0)
        self.multiresblock9 = multiresblock( 2 *32 ,32)

        self.convl = nn.Conv2d(channels(32),1, kernel_size = 1)
        self.batch_norml = nn.BatchNorm2d(1)




    def forward(self, x):

        x11 = self.multiresblock1(x)
        x111 = self.pool(x11)
        x12 = self.respath11(x11)
        x12 = self.respath12(x12)
        x12 = self.respath12(x12)
        x12 = self.respath12(x12)
        x12 = self.respath12(x12)

        x21 = self.multiresblock2(x111)
        x211 = self.pool(x21)
        x22 = self.respath21(x21)
        x22 = self.respath22(x22)
        x22 = self.respath22(x22)
        x22 = self.respath22(x22)

        x31 = self.multiresblock3(x211)
        x311 = self.pool(x31)
        x32 = self.respath31(x31)
        x32 = self.respath32(x32)
        x32 = self.respath32(x32)

        x41 = self.multiresblock4(x311)
        x411 = self.pool(x41)
        x42 = self.respath41(x41)
        x42 = self.respath42(x42)

        x51 = self.multiresblock5(x411)

        up1 = self.up1(x51)
        up1 = torch.cat((up1,x42),dim=1)
        x61 = self.multiresblock6(up1)

        up2 = self.up2(x61)
        up2 = torch.cat((up2,x32),dim=1)
        x71 = self.multiresblock7(up2)

        up3 = self.up3(x71)
        up3 = torch.cat((up3,x22),dim=1)
        x81 = self.multiresblock8(up3)

        up4 = self.up4(x81)
        up4 = torch.cat((up4,x12),dim=1)
        x91 = self.multiresblock9(up4)

        x10 = self.convl(x91)
        x10 = self.batch_norml(x10)

        out = F.sigmoid(x10)

        return out
    
    
    




device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
model = MRU().to(device)

summary( model, (3, 512, 512))