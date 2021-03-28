""" Full assembly of the parts to form the complete network """
import torch
import torch.nn as nn
import torch.nn.functional as F

from .unet_parts import *

InputNorm = SwitchNorm2d

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

class InUNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(InUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inputnorm = InputNorm(n_channels)
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x = self.inputnorm(x)
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

class AttU_Net(nn.Module):
    """
    Attention Unet implementation
    Paper: https://arxiv.org/abs/1804.03999
    """
    def __init__(self, n_channels=3, n_classes=1, bilinear=True):
        super(AttU_Net, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        n1 = 64
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]
        
        self.inc = DoubleConv(n_channels, filters[0])
        self.down1 = Down(filters[0], filters[1])
        self.down2 = Down(filters[1], filters[2])
        self.down3 = Down(filters[2], filters[3])
        factor = 1 if bilinear else 1
        self.down4 = Down(filters[3], filters[4] // factor)
        self.Up4 = up_conv(filters[4], filters[3], bilinear)
        self.Att4 = Attention_block(F_g=filters[3], F_l=filters[3], F_int=filters[2])
        self.Up_conv4 = DoubleConv(filters[4], filters[3])

        self.Up3 = up_conv(filters[3], filters[2], bilinear)
        self.Att3 = Attention_block(F_g=filters[2], F_l=filters[2], F_int=filters[1])
        self.Up_conv3 = DoubleConv(filters[3], filters[2])

        self.Up2 = up_conv(filters[2], filters[1], bilinear)
        self.Att2 = Attention_block(F_g=filters[1], F_l=filters[1], F_int=filters[0])
        self.Up_conv2 = DoubleConv(filters[2], filters[1])

        self.Up1 = up_conv(filters[1], filters[0], bilinear)
        self.Att1 = Attention_block(F_g=filters[0], F_l=filters[0], F_int=32)
        self.Up_conv1 = DoubleConv(filters[1], filters[0])

        self.Conv = nn.Conv2d(filters[0], n_classes, kernel_size=1, stride=1, padding=0)

        #self.active = torch.nn.Sigmoid()


    def forward(self, x):
        e1 = self.inc(x)
        e2 = self.down1(e1)
        e3 = self.down2(e2)
        e4 = self.down3(e3)
        e5 = self.down4(e4)

        d4 = self.Up4(e5)
        x4 = self.Att4(g=d4, x=e4)
        d4 = torch.cat((x4, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        x3 = self.Att3(g=d3, x=e3)
        d3 = torch.cat((x3, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        x2 = self.Att2(g=d2, x=e2)
        d2 = torch.cat((x2, d2), dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Up1(d2)
        x1 = self.Att1(g=d1, x=e1)
        d1 = torch.cat((x1, d1), dim=1)
        d1 = self.Up_conv1(d1)

        out = self.Conv(d1)

        return out

class InAttU_Net(nn.Module):
    """
    Attention Unet with Input Normalization Layer implementation
    Paper: https://arxiv.org/abs/1804.03999
    """
    def __init__(self, n_channels=3, n_classes=1, bilinear=True):
        super(InAttU_Net, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        n1 = 64
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]
        
        self.inputnorm = InputNorm(n_channels)
        self.inc = DoubleConv(n_channels, filters[0])
        self.down1 = Down(filters[0], filters[1])
        self.down2 = Down(filters[1], filters[2])
        self.down3 = Down(filters[2], filters[3])
        factor = 1 if bilinear else 1
        self.down4 = Down(filters[3], filters[4] // factor)
        self.Up4 = up_conv(filters[4], filters[3], bilinear)
        self.Att4 = Attention_block(F_g=filters[3], F_l=filters[3], F_int=filters[2])
        self.Up_conv4 = DoubleConv(filters[4], filters[3])

        self.Up3 = up_conv(filters[3], filters[2], bilinear)
        self.Att3 = Attention_block(F_g=filters[2], F_l=filters[2], F_int=filters[1])
        self.Up_conv3 = DoubleConv(filters[3], filters[2])

        self.Up2 = up_conv(filters[2], filters[1], bilinear)
        self.Att2 = Attention_block(F_g=filters[1], F_l=filters[1], F_int=filters[0])
        self.Up_conv2 = DoubleConv(filters[2], filters[1])

        self.Up1 = up_conv(filters[1], filters[0], bilinear)
        self.Att1 = Attention_block(F_g=filters[0], F_l=filters[0], F_int=32)
        self.Up_conv1 = DoubleConv(filters[1], filters[0])

        self.Conv = nn.Conv2d(filters[0], n_classes, kernel_size=1, stride=1, padding=0)

        #self.active = torch.nn.Sigmoid()


    def forward(self, x):
        x = self.inputnorm(x)
        e1 = self.inc(x)
        e2 = self.down1(e1)
        e3 = self.down2(e2)
        e4 = self.down3(e3)
        e5 = self.down4(e4)

        d4 = self.Up4(e5)
        x4 = self.Att4(g=d4, x=e4)
        d4 = torch.cat((x4, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        x3 = self.Att3(g=d3, x=e3)
        d3 = torch.cat((x3, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        x2 = self.Att2(g=d2, x=e2)
        d2 = torch.cat((x2, d2), dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Up1(d2)
        x1 = self.Att1(g=d1, x=e1)
        d1 = torch.cat((x1, d1), dim=1)
        d1 = self.Up_conv1(d1)

        out = self.Conv(d1)

        return out
class ECAU_Net(nn.Module):
    """
    Double Attention Unet implementation with Attention gate and ECA.
    Paper: https://arxiv.org/abs/1804.03999
           https://arxiv.org/abs/1910.03151
    """
    def __init__(self, n_channels=3, n_classes=1, bilinear=True):
        super(ECAU_Net, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        n1 = 64
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]
        
        self.inc = DoubleConv(n_channels, filters[0])
        self.down1 = Down(filters[0], filters[1])
        self.down2 = Down(filters[1], filters[2])
        self.down3 = Down(filters[2], filters[3])
        factor = 1 if bilinear else 1
        self.down4 = Down(filters[3], filters[4] // factor)
        self.Up4 = up_conv(filters[4], filters[3], bilinear)
        self.Eca4 = eca_layer(filters[3])
        self.Up_conv4 = DoubleConv(filters[4], filters[3])

        self.Up3 = up_conv(filters[3], filters[2], bilinear)
        self.Eca3 = eca_layer(filters[2])
        self.Up_conv3 = DoubleConv(filters[3], filters[2])

        self.Up2 = up_conv(filters[2], filters[1], bilinear)
        self.Eca2 = eca_layer(filters[1])
        self.Up_conv2 = DoubleConv(filters[2], filters[1])

        self.Up1 = up_conv(filters[1], filters[0], bilinear)
        self.Eca1 = eca_layer(filters[0])
        self.Up_conv1 = DoubleConv(filters[1], filters[0])

        self.Conv = nn.Conv2d(filters[0], n_classes, kernel_size=1, stride=1, padding=0)

        #self.active = torch.nn.Sigmoid()


    def forward(self, x):
        e1 = self.inc(x)
        e2 = self.down1(e1)
        e3 = self.down2(e2)
        e4 = self.down3(e3)
        e5 = self.down4(e4)

        d4 = self.Up4(e5)
        e4 = self.Eca4(e4)
        d4 = torch.cat((e4, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        e3 = self.Eca3(e3)
        d3 = torch.cat((e3, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        e2 = self.Eca2(e2)
        d2 = torch.cat((e2, d2), dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Up1(d2)
        e1 = self.Eca1(e1)
        d1 = torch.cat((e1, d1), dim=1)
        d1 = self.Up_conv1(d1)

        out = self.Conv(d1)

        return out

class Att2U_NetA(nn.Module):
    """
    Double Attention Unet implementation with Attention gate and ECA.
    Paper: https://arxiv.org/abs/1804.03999
           https://arxiv.org/abs/1910.03151
    """
    def __init__(self, n_channels=3, n_classes=1, bilinear=True):
        super(Att2U_NetA, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        n1 = 64
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]
        
        self.inc = DoubleConv(n_channels, filters[0])
        self.down1 = Down(filters[0], filters[1])
        self.down2 = Down(filters[1], filters[2])
        self.down3 = Down(filters[2], filters[3])
        factor = 1 if bilinear else 1
        self.down4 = Down(filters[3], filters[4] // factor)
        self.Up4 = up_conv(filters[4], filters[3], bilinear)
        self.Eca4 = eca_layer(filters[3])
        self.Att4 = Attention_block(F_g=filters[3], F_l=filters[3], F_int=filters[2])
        self.Up_conv4 = DoubleConv(filters[4], filters[3])

        self.Up3 = up_conv(filters[3], filters[2], bilinear)
        self.Eca3 = eca_layer(filters[2])
        self.Att3 = Attention_block(F_g=filters[2], F_l=filters[2], F_int=filters[1])
        self.Up_conv3 = DoubleConv(filters[3], filters[2])

        self.Up2 = up_conv(filters[2], filters[1], bilinear)
        self.Eca2 = eca_layer(filters[1])
        self.Att2 = Attention_block(F_g=filters[1], F_l=filters[1], F_int=filters[0])
        self.Up_conv2 = DoubleConv(filters[2], filters[1])

        self.Up1 = up_conv(filters[1], filters[0], bilinear)
        self.Eca1 = eca_layer(filters[0])
        self.Att1 = Attention_block(F_g=filters[0], F_l=filters[0], F_int=32)
        self.Up_conv1 = DoubleConv(filters[1], filters[0])

        self.Conv = nn.Conv2d(filters[0], n_classes, kernel_size=1, stride=1, padding=0)

        #self.active = torch.nn.Sigmoid()


    def forward(self, x):
        e1 = self.inc(x)
        e2 = self.down1(e1)
        e3 = self.down2(e2)
        e4 = self.down3(e3)
        e5 = self.down4(e4)

        d4 = self.Up4(e5)
        e4 = self.Eca4(e4)
        x4 = self.Att4(g=d4, x=e4)
        d4 = torch.cat((x4, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        e3 = self.Eca3(e3)
        x3 = self.Att3(g=d3, x=e3)
        d3 = torch.cat((x3, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        e2 = self.Eca2(e2)
        x2 = self.Att2(g=d2, x=e2)
        d2 = torch.cat((x2, d2), dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Up1(d2)
        e1 = self.Eca1(e1)
        x1 = self.Att1(g=d1, x=e1)
        d1 = torch.cat((x1, d1), dim=1)
        d1 = self.Up_conv1(d1)

        out = self.Conv(d1)

        return out

class Att2U_NetB(nn.Module):
    """
    Double Attention Unet implementation with Attention gate and ECA.
    Paper: https://arxiv.org/abs/1804.03999
           https://arxiv.org/abs/1910.03151
    """
    def __init__(self, n_channels=3, n_classes=1, bilinear=True):
        super(Att2U_NetB, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        n1 = 64
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]
        
        self.inc = DoubleConv(n_channels, filters[0])
        self.down1 = Down(filters[0], filters[1])
        self.down2 = Down(filters[1], filters[2])
        self.down3 = Down(filters[2], filters[3])
        factor = 1 if bilinear else 1
        self.down4 = Down(filters[3], filters[4] // factor)
        self.Up4 = up_conv(filters[4], filters[3], bilinear)
        self.Att4 = Attention_block(F_g=filters[3], F_l=filters[3], F_int=filters[2])
        self.Eca4 = eca_layer(filters[3])
        self.Up_conv4 = DoubleConv(filters[4], filters[3])

        self.Up3 = up_conv(filters[3], filters[2], bilinear)
        self.Att3 = Attention_block(F_g=filters[2], F_l=filters[2], F_int=filters[1])
        self.Eca3 = eca_layer(filters[2])
        self.Up_conv3 = DoubleConv(filters[3], filters[2])

        self.Up2 = up_conv(filters[2], filters[1], bilinear)
        self.Att2 = Attention_block(F_g=filters[1], F_l=filters[1], F_int=filters[0])
        self.Eca2 = eca_layer(filters[1])
        self.Up_conv2 = DoubleConv(filters[2], filters[1])

        self.Up1 = up_conv(filters[1], filters[0], bilinear)
        self.Att1 = Attention_block(F_g=filters[0], F_l=filters[0], F_int=32)
        self.Eca1 = eca_layer(filters[0])
        self.Up_conv1 = DoubleConv(filters[1], filters[0])

        self.Conv = nn.Conv2d(filters[0], n_classes, kernel_size=1, stride=1, padding=0)

        #self.active = torch.nn.Sigmoid()


    def forward(self, x):
        e1 = self.inc(x)
        e2 = self.down1(e1)
        e3 = self.down2(e2)
        e4 = self.down3(e3)
        e5 = self.down4(e4)

        d4 = self.Up4(e5)
        x4 = self.Att4(g=d4, x=e4)
        d4 = torch.cat((x4, d4), dim=1)
        d4 = self.Eca4(d4)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        x3 = self.Att3(g=d3, x=e3)
        d3 = torch.cat((x3, d3), dim=1)
        d3 = self.Eca3(d3)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        x2 = self.Att2(g=d2, x=e2)
        d2 = torch.cat((x2, d2), dim=1)
        d2 = self.Eca2(d2)
        d2 = self.Up_conv2(d2)

        d1 = self.Up1(d2)
        x1 = self.Att1(g=d1, x=e1)
        d1 = torch.cat((x1, d1), dim=1)
        d1 = self.Eca1(d1)
        d1 = self.Up_conv1(d1)

        out = self.Conv(d1)

        return out

class Att2U_NetC(nn.Module):
    """
    Double Attention Unet implementation with Attention gate and ECA.
    Paper: https://arxiv.org/abs/1804.03999
           https://arxiv.org/abs/1910.03151
    """
    def __init__(self, n_channels=3, n_classes=1, bilinear=True):
        super(Att2U_NetC, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        n1 = 64
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]
        
        self.inc = DoubleConv(n_channels, filters[0])
        self.down1 = Down(filters[0], filters[1])
        self.down2 = Down(filters[1], filters[2])
        self.down3 = Down(filters[2], filters[3])
        factor = 1 if bilinear else 1
        self.down4 = Down(filters[3], filters[4] // factor)
        self.Up4 = up_conv(filters[4], filters[3], bilinear)
        self.Eca4_0 = eca_layer(filters[3])
        self.Att4 = Attention_block(F_g=filters[3], F_l=filters[3], F_int=filters[2])
        self.Eca4_1 = eca_layer(filters[3])
        self.Up_conv4 = DoubleConv(filters[4], filters[3])

        self.Up3 = up_conv(filters[3], filters[2], bilinear)
        self.Eca3_0 = eca_layer(filters[2])
        self.Att3 = Attention_block(F_g=filters[2], F_l=filters[2], F_int=filters[1])
        self.Eca3_1 = eca_layer(filters[2])
        self.Up_conv3 = DoubleConv(filters[3], filters[2])

        self.Up2 = up_conv(filters[2], filters[1], bilinear)
        self.Eca2_0 = eca_layer(filters[2])
        self.Att2 = Attention_block(F_g=filters[1], F_l=filters[1], F_int=filters[0])
        self.Eca2_1 = eca_layer(filters[1])
        self.Up_conv2 = DoubleConv(filters[2], filters[1])

        self.Up1 = up_conv(filters[1], filters[0], bilinear)
        self.Eca1_0 = eca_layer(filters[0])
        self.Att1 = Attention_block(F_g=filters[0], F_l=filters[0], F_int=32)
        self.Eca1_1 = eca_layer(filters[0])
        self.Up_conv1 = DoubleConv(filters[1], filters[0])

        self.Conv = nn.Conv2d(filters[0], n_classes, kernel_size=1, stride=1, padding=0)

        #self.active = torch.nn.Sigmoid()


    def forward(self, x):
        e1 = self.inc(x)
        e2 = self.down1(e1)
        e3 = self.down2(e2)
        e4 = self.down3(e3)
        e5 = self.down4(e4)

        d4 = self.Up4(e5)
        e4 = self.Eca4_0(e4)
        x4 = self.Att4(g=d4, x=e4)
        d4 = torch.cat((x4, d4), dim=1)
        d4 = self.Eca4_1(d4)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        e3 = self.Eca3_0(e3)
        x3 = self.Att3(g=d3, x=e3)
        d3 = torch.cat((x3, d3), dim=1)
        d3 = self.Eca3_1(d3)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        e2 = self.Eca2_0(e2)
        x2 = self.Att2(g=d2, x=e2)
        d2 = torch.cat((x2, d2), dim=1)
        d2 = self.Eca2_1(d2)
        d2 = self.Up_conv2(d2)

        d1 = self.Up1(d2)
        e1 = self.Eca1_0(e1)
        x1 = self.Att1(g=d1, x=e1)
        d1 = torch.cat((x1, d1), dim=1)
        d1 = self.Eca1_1(d1)
        d1 = self.Up_conv1(d1)

        out = self.Conv(d1)

        return out

def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0)
    # 也可以判断是否为conv2d，使用相应的初始化方式 
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
     # 是否为批归一化层
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)