import torch
import torch.nn as nn
import torch.nn.functional as F
# from backbone.mobilenetv2 import mobilenet_v2
# from toolbox.models.bbbmodels.mobilenetv2 import mobilenet_v2
# from backbone.VGG import VGG16
# from backbone.resnet import Backbone_ResNet34_in3
from torchvision import transforms
import cv2
from backbone.convnext import convnext_small

###------------------------------DFM1--加入d=18的膨胀卷积--将d4（DAM输出）加入到末尾concat---------------------

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        rgb = convnext_small(True)
        t = convnext_small(True)
        self.layer0_r = nn.Sequential(rgb.downsample_layers[0], rgb.stages[0])
        self.layer1_r = nn.Sequential(rgb.downsample_layers[1], rgb.stages[1])
        self.layer2_r = nn.Sequential(rgb.downsample_layers[2], rgb.stages[2])
        self.layer3_r = nn.Sequential(rgb.downsample_layers[3], rgb.stages[3])

        self.layer0_t = nn.Sequential(t.downsample_layers[0], t.stages[0])
        self.layer1_t = nn.Sequential(t.downsample_layers[1], t.stages[1])
        self.layer2_t = nn.Sequential(t.downsample_layers[2], t.stages[2])
        self.layer3_t = nn.Sequential(t.downsample_layers[3], t.stages[3])



class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7, ):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2 , 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)    #1*h*w

class ChannelAttention(nn.Module):
    # def __init__(self, in_planes, ratio=16):
    def __init__(self, in_planes, ratio=8):
        super(ChannelAttention, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # self.fc1 = nn.Conv2d(in_planes, in_planes / 16, 1, bias=False)
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.LeakyReLU()
        # self.fc2 = nn.Conv2d(in_planes / 16, in_planes, 1, bias=False)
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self,x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)   #输出格式为 1*1*c

class ChannelShuffle(nn.Module):
    def __init__(self, groups):
        super(ChannelShuffle, self).__init__()
        self.groups = groups

    def forward(self, x):
        batch_size, num_channels, height, width = x.data.size()
        channels_per_group = num_channels // self.groups

        # reshape
        x = x.view(batch_size, self.groups, channels_per_group, height, width)

        # transpose
        x = torch.transpose(x, 1, 2).contiguous()

        # reshape back
        x = x.view(batch_size, -1, height, width)

        return x

class CFI(nn.Module):
    def __init__(self, in_ch):
        super(CFI, self).__init__()

        self.conv_r = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=in_ch, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_ch),
            nn.LeakyReLU(),
        )
        self.conv_d = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=in_ch, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_ch),
            nn.LeakyReLU(),
        )
        self.conv_cat1 = nn.Sequential(
            nn.Conv2d(in_channels=2 * in_ch, out_channels=in_ch, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(in_channels=in_ch, out_channels=in_ch, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_ch),
            nn.LeakyReLU(),
        )
        self.conv_cat2 = nn.Sequential(
            nn.Conv2d(in_channels=3 * in_ch, out_channels=in_ch, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(in_channels=in_ch, out_channels=in_ch, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_ch),
            nn.LeakyReLU(),
        )
        self.cs = ChannelShuffle(groups=4)

    def forward(self, r, d, fus):
        r = self.conv_r(r)
        d = self.conv_d(d)
        rd1 = self.conv_cat1(torch.cat((r, d), dim=1))
        b, c, h, w = r.shape

        num = 5
        for _ in range(num):
            # Randomly select a position for cutting
            x = torch.randint(0, h - h // 4, (1,))
            y = torch.randint(0, w - w // 4, (1,))

            # Cut out a piece from feature_map1
            piece = r[:, :, x:x + h // 4, y:y + w // 4]

            # Replace the corresponding position in feature_map2 with the pi       print(rd1.shape, d.shape, fus.shape)ece
            d[:, :, x:x + h // 4, y:y + w // 4] = piece

        rd2 = self.conv_cat2(self.cs(torch.cat((rd1, d, fus), dim=1)))
        return rd2

class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, dilation=1, groups=1, bn=True, relu=True):
        padding = ((kernel_size - 1) * dilation + 1) // 2
        # padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, dilation=dilation, groups=groups,
                              bias=False if bn else True)
        self.bn = bn
        if bn:
            self.bnop = nn.BatchNorm2d(out_planes)
        self.relu = relu
        if relu:
            # self.reluop = nn.ReLU6(inplace=True)
            self.reluop = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if self.bn:
            x = self.bnop(x)
        if self.relu:
            x = self.reluop(x)
        return x


class MDA(nn.Module):
    def __init__(self):
        super(MDA,self).__init__()
        self.up2x = nn.UpsamplingBilinear2d(scale_factor=2)
        # self.conv_rt = nn.Conv2d(in_channels=1536, out_channels=128, kernel_size=1)
        # self.conv_r1 = nn.Conv2d(in_channels=512, out_channels=128, kernel_size=1)
        # self.conv_t1 = nn.Conv2d(in_channels=512, out_channels=128, kernel_size=1)
        self.d3_r = ConvBNReLU(in_planes=768, out_planes=128, kernel_size=3, stride=1, dilation=3)
        self.d6_r = ConvBNReLU(in_planes=896, out_planes=128, kernel_size=3, stride=1, dilation=6)
        self.d12_r = ConvBNReLU(in_planes=1024, out_planes=128, kernel_size=3, stride=1, dilation=12)
        # self.d18_r = ConvBNReLU(in_planes=512, out_planes=64, kernel_size=3, stride=1, dilation=18)
        self.d3_t = ConvBNReLU(in_planes=768, out_planes=128, kernel_size=3, stride=1, dilation=3)
        self.d6_t = ConvBNReLU(in_planes=896, out_planes=128, kernel_size=3, stride=1, dilation=6)
        self.d12_t = ConvBNReLU(in_planes=1024, out_planes=128, kernel_size=3, stride=1, dilation=12)
        # self.d18_t = ConvBNReLU(in_planes=512, out_planes=64, kernel_size=3, stride=1, dilation=18)
        self.avp = nn.AdaptiveAvgPool2d(1)
        self.max = nn.AdaptiveMaxPool2d(1)
        self.conv_cat_r = nn.Conv2d(in_channels=1152, out_channels=128, kernel_size=1, stride=1, padding=0)
        self.conv_cat_rr = ConvBNReLU(in_planes=128, out_planes=128, kernel_size=3, stride=1, dilation=1)

        self.conv_cat_t = nn.Conv2d(in_channels=1152, out_channels=128, kernel_size=1, stride=1, padding=0)
        self.conv_cat_tt = ConvBNReLU(in_planes=128, out_planes=128, kernel_size=3, stride=1, dilation=1)
        self.conv_fus = nn.Conv2d(in_channels=384, out_channels=128, kernel_size=1, stride=1, padding=0)
        self.conv_fus_1 = ConvBNReLU(in_planes=128, out_planes=128, kernel_size=3, stride=1, dilation=1)

        self.conv_d4 = nn.Conv2d(in_channels=384, out_channels=128, kernel_size=1, stride=2, padding=0)

        self.conv_w1 = nn.Conv2d(128, 128, 1, 1)
        self.conv_w2 = nn.Conv2d(128, 128, 1, 1)
        # self.resize = nn.UpsamplingBilinear2d((13, 13))

    def forward(self, r, t, d4):
        r3 = self.d3_r(r)  # 128
        r_13 = torch.cat([r, r3], dim=1)  # 640
        r6 = self.d6_r(r_13)  # 128
        r_136 = torch.cat([r_13, r6], dim=1)  # 768
        r12 = self.d12_r(r_136)  # 128
        cat_r = torch.cat([r, r3, r6, r12], dim=1)  # 896

        cat_r = self.conv_cat_r(cat_r)  # 128
        cat_r = self.conv_cat_rr(cat_r)
        weight_r = self.conv_w1(self.avp(cat_r) + self.max(cat_r))

        t3 = self.d3_t(t)  # 128
        t_13 = torch.cat([t, t3], dim=1)  # 640
        t6 = self.d6_t(t_13)  # 128
        t_136 = torch.cat([t_13, t6], dim=1)  # 768
        t12 = self.d12_t(t_136)  # 128
        cat_t = torch.cat([t, t3, t6, t12], dim=1)  # 896

        cat_t = self.conv_cat_t(cat_t)  # 128
        cat_t = self.conv_cat_tt(cat_t)

        weight_t = self.conv_w2(self.avp(cat_t) + self.max(cat_t))

        we_r = torch.mul(cat_r, weight_t)
        we_t = torch.mul(cat_t, weight_r)
        we_r = we_r + cat_r
        we_t = we_t + cat_t
        conv_d4 = self.conv_d4(d4)

        fus = torch.cat((we_r, we_t, conv_d4), dim=1)  # 256
        out = self.conv_fus(fus)
        out = self.conv_fus_1(out)

        return out

class NAE(nn.Module):
    def __init__(self, in_ch1, in_ch2, in_ch3):
        super(NAE, self).__init__()
        self.conv_x1 = nn.Sequential(nn.Conv2d(in_channels=in_ch1, out_channels=128, kernel_size=1, stride=1, padding=0),
                                     nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
                                     nn.BatchNorm2d(128),
                                     nn.LeakyReLU()
                                     )
        self.conv_x2 = nn.Sequential(nn.Conv2d(in_channels=in_ch2, out_channels=128, kernel_size=1, stride=1, padding=0),
                                     nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
                                     nn.BatchNorm2d(128),
                                     nn.LeakyReLU()
                                     )
        self.conv_x3 = nn.Sequential(nn.Conv2d(in_channels=in_ch3, out_channels=128, kernel_size=1, stride=1, padding=0),
                                     nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
                                     nn.BatchNorm2d(128),
                                     nn.LeakyReLU()
                                     )

        self.up2x = nn.UpsamplingBilinear2d(scale_factor=2)
        self.up4x = nn.UpsamplingBilinear2d(scale_factor=4)
        self.up8x = nn.UpsamplingBilinear2d(scale_factor=8)
        # self.mp = nn.MaxPool2d(1)
        # self.avp = nn.AdaptiveAvgPool2d(1)
        self.conv_att_x1 = nn.Conv2d(2, 1, 1, 1)
        self.conv_att_x3 = nn.Conv2d(2, 1, 1, 1)
        self.conv_fus = nn.Sequential(nn.Conv2d(in_channels=384, out_channels=128, kernel_size=1, stride=1, padding=0),
                                      nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
                                      nn.BatchNorm2d(128),
                                      nn.LeakyReLU()
                                      )

    def forward(self, x1, x2, x3):
        conv_x1 = self.conv_x1(x1)
        conv_x2 = self.conv_x2(x2)
        up_x2 = self.up2x(conv_x2)
        conv_x3 = self.conv_x3(x3)
        x1_mean = torch.mean(conv_x1, dim=1, keepdim=True)
        x1_max,_ = torch.max(conv_x1, dim=1, keepdim=True)
        cat_x1 = torch.cat([x1_max, x1_mean], dim=1)
        cat_x1 = self.up4x(cat_x1)
        att_x1 = torch.sigmoid(self.conv_att_x1(cat_x1))
        # we_3avp = self.avp(x3)
        # we_3mp = self.mp(x3)
        x3_mean = torch.mean(conv_x3, dim=1, keepdim=True)
        x3_max,_ = torch.max(conv_x3, dim=1, keepdim=True)
        cat_x3 = torch.cat([x3_max, x3_mean], dim=1)
        att_x3 = torch.sigmoid(self.conv_att_x3(cat_x3))
        # print(att_x1.shape, up_x2.shape)
        att_12 = torch.mul(att_x1, up_x2)
        att_32 = torch.mul(att_x3, up_x2)
        fus = torch.cat([up_x2, att_12, att_32], dim=1)  #384*60*80
        out = self.conv_fus(fus)

        return out

class CCF(nn.Module):
    def __init__(self):
        super(CCF, self).__init__()
        self.up2x = nn.UpsamplingBilinear2d(scale_factor=2)
        self.up4x = nn.UpsamplingBilinear2d(scale_factor=4)
        self.up8x = nn.UpsamplingBilinear2d(scale_factor=8)
        self.conv_x4 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1)
        self.conv_x5 = nn.Conv2d(in_channels=512, out_channels=128, kernel_size=1)
        # self.conv_cat = nn.Conv2d(in_channels=384, out_channels=128, kernel_size=1)
        self.conv_11 = nn.Sequential(nn.Conv2d(in_channels=128, out_channels=128, kernel_size=1, stride=1),
                                     nn.BatchNorm2d(128),
                                     nn.LeakyReLU())
        self.conv_12 = ConvBNReLU(in_planes=128, out_planes=128, kernel_size=3, stride=1, dilation=1)
        self.conv_21 = nn.Sequential(nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
                                     nn.BatchNorm2d(128),
                                     nn.LeakyReLU())
        self.conv_22 = ConvBNReLU(in_planes=128, out_planes=128, kernel_size=3, stride=1, dilation=3)
        self.conv_31 = nn.Sequential(nn.Conv2d(in_channels=128, out_channels=128, kernel_size=5, padding=2),
                                     nn.BatchNorm2d(128),
                                     nn.LeakyReLU())
        self.conv_32 = ConvBNReLU(in_planes=128, out_planes=128, kernel_size=3, stride=1, dilation=5)
        self.conv_41 = nn.Sequential(nn.Conv2d(in_channels=128, out_channels=128, kernel_size=7, padding=3),
                                     nn.BatchNorm2d(128),
                                     nn.LeakyReLU())
        self.conv_42 = ConvBNReLU(in_planes=128, out_planes=128, kernel_size=3, stride=1, dilation=7)

        self.conv_down1 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1, stride=1, padding=0)
        self.conv_down2 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1, stride=1, padding=0)
        self.conv1x1_1 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv1x1_2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv_cat = nn.Sequential(nn.Conv2d(in_channels=512, out_channels=128, kernel_size=3, stride=1, padding=1),
                                      nn.BatchNorm2d(128),
                                      nn.LeakyReLU()
                                      )
        self.conv_out = nn.Sequential(nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
                                      nn.BatchNorm2d(128),
                                      nn.LeakyReLU(),
                                      nn.Conv2d(in_channels=128, out_channels=1, kernel_size=3, stride=1, padding=1)
                                      )

    def forward(self, x1, x2, x3):

        up_x1 = self.up2x(x1)
        up_x2 = self.up2x(x2)
        # print(up_x1.shape, x2.shape)
        cat_12 = torch.cat([up_x1, x2], dim=1)
        cat_12_down = self.conv_down1(cat_12)
        att1 = F.avg_pool2d(cat_12_down, cat_12_down.size()[2:])
        att1 = torch.sigmoid(self.conv1x1_1(att1))
        add_12 = up_x1 + x2
        f12 = torch.mul(att1, add_12)

        up_f12 = self.up2x(f12)
        cat_123 = torch.cat([up_f12, x3], dim=1)
        cat_123_down = self.conv_down2(cat_123)
        att2 = F.avg_pool2d(cat_123_down, cat_123_down.size()[2:])
        att2 = torch.sigmoid(self.conv1x1_2(att2))
        add_123 = up_f12 + x3
        f123 = torch.mul(att2, add_123)
        cat = torch.cat([f123, self.up4x(x1), up_x2, x3], dim=1)    #512*240*320
        cat = self.conv_cat(cat)               #128*240*320
        conv_11_cat = self.conv_11(cat)
        conv_12_cat = self.conv_12(conv_11_cat)
        conv_21_cat = self.conv_21(cat)
        conv_22_cat = self.conv_22(conv_21_cat)
        conv_31_cat = self.conv_31(cat)
        conv_32_cat = self.conv_32(conv_31_cat)
        conv_41_cat = self.conv_41(cat)
        conv_42_cat = self.conv_42(conv_41_cat)
        tfm_out = conv_12_cat + conv_22_cat + conv_32_cat + conv_42_cat
        tfm_out = self.conv_out(tfm_out)

        return tfm_out

class HAFNet(nn.Module):
    def __init__(self):
        super(HAFNet,self).__init__()

        # self.mobile_rgb = mobilenet_v2(pretrained=True)
        # self.mobile_dep = mobilenet_v2(pretrained=True)

        self.reshape1 = nn.Conv2d(96, 96, 3, stride=2, padding=1)
        self.reshape2 = nn.Conv2d(96, 192, 3, stride=2, padding=1)
        self.reshape3 = nn.Conv2d(192, 384, 3, stride=2, padding=1)
        self.reshape4 = nn.Conv2d(384, 768, 3, stride=2, padding=1)

        self.CFI1 = CFI(96)
        self.CFI2 = CFI(96)
        self.CFI3 = CFI(192)
        self.CFI4 = CFI(384)

        self.up2x = nn.UpsamplingBilinear2d(scale_factor=2)
        self.up4x = nn.UpsamplingBilinear2d(scale_factor=4)
        self.up8x = nn.UpsamplingBilinear2d(scale_factor=8)
        self.up16x = nn.UpsamplingBilinear2d(scale_factor=16)
        self.up32x = nn.UpsamplingBilinear2d(scale_factor=32)

        self.conv1 = nn.Conv2d(in_channels=512, out_channels=64, kernel_size=1)
        # self.dfm = DFM()
        self.MDA = MDA()

        self.NAE1 = NAE(in_ch1=128, in_ch2=384, in_ch3=192)
        self.NAE2 = NAE(in_ch1=384, in_ch2=192, in_ch3=96)
        self.NAE3 = NAE(in_ch1=192, in_ch2=96, in_ch3=96)
        self.CCF = CCF()

        self.conv_out1 = nn.Conv2d(in_channels=128, out_channels=1, kernel_size=1)

        self.conv_out2 = nn.Conv2d(in_channels=128, out_channels=1, kernel_size=1)

        self.conv_out3 = nn.Conv2d(in_channels=128, out_channels=1, kernel_size=1)

        self.conv_out4 = nn.Conv2d(in_channels=128, out_channels=1, kernel_size=1)

        rgb = convnext_small(True)
        t = convnext_small(True)
        self.layer0_r = nn.Sequential(rgb.downsample_layers[0])
        self.layer1_r = nn.Sequential(rgb.stages[0])
        self.layer2_r = nn.Sequential(rgb.downsample_layers[1], rgb.stages[1])
        self.layer3_r = nn.Sequential(rgb.downsample_layers[2], rgb.stages[2])
        self.layer4_r = nn.Sequential(rgb.downsample_layers[3], rgb.stages[3])

        self.layer0_t = nn.Sequential(t.downsample_layers[0])
        self.layer1_t = nn.Sequential(t.stages[0])
        self.layer2_t = nn.Sequential(t.downsample_layers[1], t.stages[1])
        self.layer3_t = nn.Sequential(t.downsample_layers[2], t.stages[2])
        self.layer4_t = nn.Sequential(t.downsample_layers[3], t.stages[3])


    def forward(self, img, depth):
        conv1_convnext_r = self.layer0_r(img)
        conv1_convnext_t = self.layer0_t(depth)

        fus1 = self.CFI1(conv1_convnext_r, conv1_convnext_t, conv1_convnext_t + conv1_convnext_r)
        cfus1_r = fus1 + conv1_convnext_r
        cfus1_t = fus1 + conv1_convnext_t
        fus1 = self.up2x(fus1)
        fus1_reshape = self.reshape1(fus1)

        conv2_convnext_r = self.layer1_r(conv1_convnext_r)
        conv2_convnext_t = self.layer1_t(cfus1_t)

        fus2 = self.CFI2(conv2_convnext_r, conv2_convnext_t, fus1_reshape)

        cfus2_r = fus2 + conv2_convnext_r
        cfus2_t = fus2 + conv2_convnext_t

        fus2_reshape = self.reshape2(fus2)

        conv3_convnext_r = self.layer2_r(conv2_convnext_r)
        conv3_convnext_t = self.layer2_t(cfus2_t)
        # print(conv3_convnext_r.shape, conv3_convnext_t.shape)
        # conv3_convnext_r = self.layer2_r(cfus2_r)
        # conv3_convnext_t = self.layer2_t(conv2_convnext_t)
        fus3 = self.CFI3(conv3_convnext_r, conv3_convnext_t, fus2_reshape)
        # print(fus3.shape)
        cfus3_r = fus3 + conv3_convnext_r
        cfus3_t = fus3 + conv3_convnext_t

        fus3_reshape = self.reshape3(fus3)

        conv4_convnext_r = self.layer3_r(cfus3_r)
        conv4_convnext_t = self.layer3_t(conv3_convnext_t)
        # print(conv4_convnext_r.shape, conv4_convnext_t.shape)
        # conv4_convnext_r = self.layer3_r(conv3_convnext_r)
        # conv4_convnext_t = self.layer3_t(cfus3_t)
        fus4 = self.CFI4(conv4_convnext_r, conv4_convnext_t, fus3_reshape)
        # print(fus4.shape)
        cfus4_r = fus4 + conv4_convnext_r
        cfus4_t = fus4 + conv4_convnext_t

        # fus4_reshape = self.reshape4(fus4)
        #
        # conv5_mobile_r = self.mobile_rgb.features[14:18](cfus4_r)
        # conv5_mobile_t = self.mobile_dep.features[14:18](conv4_mobile_t)
        conv5_convnext_r = self.layer4_r(cfus4_r)
        conv5_convnext_t = self.layer4_t(conv4_convnext_t)
        # print(conv5_convnext_r.shape, conv5_convnext_t.shape, fus4.shape)
        # conv5_convnext_r = self.layer4_r(conv4_convnext_r)
        # conv5_convnext_t = self.layer4_t(cfus4_t)
        out1 = self.MDA(conv5_convnext_r, conv5_convnext_t, fus4)
        out2 = self.NAE1(out1, fus4, fus3)
        out3 = self.NAE2(fus4, fus3, fus2)
        out4 = self.NAE3(fus3, fus2, fus1)

        out = self.CCF(out2, out3, out4)
        out = self.up2x(out)
        out4 = self.up2x(self.conv_out4(out4))
        out3 = self.up4x(self.conv_out3(out3))
        out2 = self.up8x(self.conv_out2(out2))
        out1 = self.up32x(self.conv_out1(out1))

        return out, out4, out3, out2, out1


if __name__ == "__main__":
    rgb = torch.randn(2, 3, 416, 416)
    t = torch.randn(2, 3, 416, 416)
    model = HAFNet()
    out = model(rgb, t)
    print(out[0].shape, out[1].shape, out[2].shape, out[3].shape, out[4].shape)
