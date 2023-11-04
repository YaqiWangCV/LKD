from torch import nn
from torch.nn import functional as F
import torch
from torchvision import models
import torchvision


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, input):
        return self.conv(input)


# class VGGBlock(nn.Module):
#     def __init__(self, in_channels, middle_channels, out_channels, act_func=nn.ReLU(inplace=True)):
#         super(VGGBlock, self).__init__()
#         self.act_func = act_func
#         self.conv1 = nn.Conv2d(in_channels, middle_channels, 3, padding=1)
#         self.bn1 = nn.BatchNorm2d(middle_channels)
#         self.conv2 = nn.Conv2d(middle_channels, out_channels, 3, padding=1)
#         self.bn2 = nn.BatchNorm2d(out_channels)
#
#     def forward(self, x):
#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.act_func(out)
#
#         out = self.conv2(out)
#         out = self.bn2(out)
#         out = self.act_func(out)
#         return out

class NestedUNet(nn.Module):
    def __init__(self, in_channel, out_channel, deepsupervision=False, feature=False):
        # def __init__(self, args, in_channel, out_channel):
        super().__init__()
        self.deepsupervision = deepsupervision
        # nb_filter = [24, 32, 64, 128, 256]
        nb_filter = [32, 64, 128, 256, 512]
        self.feature = feature
        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = DoubleConv(in_channel, nb_filter[0])
        self.conv1_0 = DoubleConv(nb_filter[0], nb_filter[1])
        self.conv2_0 = DoubleConv(nb_filter[1], nb_filter[2])
        self.conv3_0 = DoubleConv(nb_filter[2], nb_filter[3])
        self.conv4_0 = DoubleConv(nb_filter[3], nb_filter[4])

        self.conv0_1 = DoubleConv(nb_filter[0] + nb_filter[1], nb_filter[0])
        self.conv1_1 = DoubleConv(nb_filter[1] + nb_filter[2], nb_filter[1])
        self.conv2_1 = DoubleConv(nb_filter[2] + nb_filter[3], nb_filter[2])
        self.conv3_1 = DoubleConv(nb_filter[3] + nb_filter[4], nb_filter[3])

        self.conv0_2 = DoubleConv(nb_filter[0] * 2 + nb_filter[1], nb_filter[0])
        self.conv1_2 = DoubleConv(nb_filter[1] * 2 + nb_filter[2], nb_filter[1])
        self.conv2_2 = DoubleConv(nb_filter[2] * 2 + nb_filter[3], nb_filter[2])

        self.conv0_3 = DoubleConv(nb_filter[0] * 3 + nb_filter[1], nb_filter[0])
        self.conv1_3 = DoubleConv(nb_filter[1] * 3 + nb_filter[2], nb_filter[1])

        self.conv0_4 = DoubleConv(nb_filter[0] * 4 + nb_filter[1], nb_filter[0])
        self.sigmoid = nn.Sigmoid()
        if self.deepsupervision:
            # self.final_tool1 = nn.Conv2d(nb_filter[0], 24, kernel_size=1)
            # self.final_tool2 = nn.Conv2d(nb_filter[0], 24, kernel_size=1)
            # self.final_tool3 = nn.Conv2d(nb_filter[0], 24, kernel_size=1)
            # self.final_tool4 = nn.Conv2d(nb_filter[0], 24, kernel_size=1)
            self.final1 = nn.Conv2d(nb_filter[0], out_channel, kernel_size=1)
            self.final2 = nn.Conv2d(nb_filter[0], out_channel, kernel_size=1)
            self.final3 = nn.Conv2d(nb_filter[0], out_channel, kernel_size=1)
            self.final4 = nn.Conv2d(nb_filter[0], out_channel, kernel_size=1)
            # self.final1 = nn.Conv2d(24, out_channel, kernel_size=1)
            # self.final2 = nn.Conv2d(24, out_channel, kernel_size=1)
            # self.final3 = nn.Conv2d(24, out_channel, kernel_size=1)
            # self.final4 = nn.Conv2d(24, out_channel, kernel_size=1)

        else:
            self.final = nn.Conv2d(nb_filter[0], out_channel, kernel_size=1)

    def get_last_feat_modules(self):
        return self.final4

    def forward(self, input, all_features=False):
        x0_0 = self.conv0_0(input)
        f0 = x0_0
        x1_0 = self.conv1_0(self.pool(x0_0))
        f1 = x1_0
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))

        x2_0 = self.conv2_0(self.pool(x1_0))
        f2 = x2_0
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))

        x3_0 = self.conv3_0(self.pool(x2_0))
        f3 = x3_0
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))

        x4_0 = self.conv4_0(self.pool(x3_0))
        f4 = x4_0
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))
        f5 = x3_1
        f6 = x2_2
        f7 = x1_3
        f8 = x0_4

        if self.deepsupervision:
            x0_1 = self.final1(x0_1)
            output1 = self.sigmoid(x0_1).permute(0, 2, 3, 1)
            x0_2 = self.final2(x0_2)
            output2 = self.sigmoid(x0_2).permute(0, 2, 3, 1)
            x0_3 = self.final3(x0_3)
            output3 = self.sigmoid(x0_3).permute(0, 2, 3, 1)
            x0_4 = self.final4(x0_4)          
            output4 = self.sigmoid(x0_4).permute(0, 2, 3, 1)

            if all_features:
                return [f0, f1, f2, f3, f4, f5, f6, f7, f8], output4

            return output1, output2, output3, output4

        else:
            output = self.final(x0_4)
            output = self.sigmoid(output).permute(0, 2, 3, 1)
            return output


if __name__ == '__main__':
    images = torch.rand(1, 3, 640, 640)
    model = NestedUNet(3, 5, deepsupervision=True)
    feature, output = model(images, all_features=True)
    print('output:', output.shape)
    for i, o in enumerate(feature):
        print('d{}:{}'.format(i, o.shape))
