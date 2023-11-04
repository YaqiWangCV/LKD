import os

import torch
from torch import nn
from .layers import SN_CS_Parallel_Attention_block, SN_PostRes2d
from torch import optim
from torch.backends import cudnn
from torch.nn.functional import interpolate
from .layers import SwitchNorm2d


class CAUNet(nn.Module):

    def __init__(self, input_channel=3, output_channel=1, bn_momentum=0.2, feature=False):
        super(CAUNet, self).__init__()
        # super(SN_MSF_UpDec_ResUNet, self).__init__()
        # The first few layers consumes the most memory, so use simple convolution to save memory.
        # Call these layers preBlock, i.e., before the residual blocks of later layers.
        self.original_size = 640
        self.feature = feature
        self.nb_filter = [32, 64, 128, 256, 512]
        self.preBlock = nn.Sequential(
            nn.Conv2d(input_channel, self.nb_filter[0], kernel_size=3, padding=1),
            SwitchNorm2d(self.nb_filter[0], momentum=bn_momentum),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.nb_filter[0], self.nb_filter[0], kernel_size=3, padding=1),
            SwitchNorm2d(self.nb_filter[0], momentum=bn_momentum),
            nn.ReLU(inplace=True),
            # nn.Conv2d(24, 24, kernel_size=3, padding=1),
            # SwitchNorm2d(24, momentum=bn_momentum),
            # nn.ReLU(inplace=True)
        )

        self.outBlock = nn.Sequential(

            nn.Conv2d(48, 24, kernel_size=3, padding=1),
            SwitchNorm2d(24, momentum=bn_momentum),
            nn.ReLU(inplace=True),
            nn.Conv2d(24, 24, kernel_size=3, padding=1),
            SwitchNorm2d(24, momentum=bn_momentum),
            nn.ReLU(inplace=True)
        )

        self.pred = nn.Sequential(
            # nn.Conv2d(24, 24, kernel_size=3, padding=1),
            # SwitchNorm2d(24, momentum=bn_momentum),
            # nn.ReLU(inplace=True),
            nn.Conv2d(self.nb_filter[0], 12, kernel_size=1),
            SwitchNorm2d(12, momentum=bn_momentum),
            nn.ReLU(inplace=True),
            nn.Conv2d(12, output_channel, kernel_size=1),
            nn.Sigmoid()
        )

        num_blocks_forw = [2, 2, 3, 4]
        # self.featureNum_forw = [24, 32, 64, 128, 256]
        self.featureNum_forw = self.nb_filter
        # [32, 64, 128, 256, 512]
        for i in range(len(num_blocks_forw)):
            blocks = []
            for j in range(num_blocks_forw[i]):
                if j == 0:
                    blocks.append(SN_PostRes2d(self.featureNum_forw[i], self.featureNum_forw[i + 1]))
                else:
                    blocks.append(SN_PostRes2d(self.featureNum_forw[i + 1], self.featureNum_forw[i + 1]))
            setattr(self, 'forw' + str(i + 1), nn.Sequential(*blocks))

        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

        # self.nb_filter = [32, 64, 128, 256, 512]
        self.back1 = nn.Sequential(
            SN_PostRes2d(992, self.nb_filter[3]),
            # SN_PostRes2d(256, 128),
            SN_PostRes2d(self.nb_filter[3], self.nb_filter[3]),
        )
        self.back2 = nn.Sequential(
            SN_PostRes2d(608, self.nb_filter[2]),
            # PostRes2d(128, 64),
            SN_PostRes2d(self.nb_filter[2], self.nb_filter[2]),
        )
        self.back3 = nn.Sequential(
            SN_PostRes2d(416, self.nb_filter[1]),
            # PostRes2d(96, 32),
            SN_PostRes2d(self.nb_filter[1], self.nb_filter[1]),
        )

        self.back4 = nn.Sequential(
            SN_PostRes2d(320, self.nb_filter[0]),
            # SN_PostRes2d(48, 24),
            SN_PostRes2d(self.nb_filter[0], self.nb_filter[0]),
        )

        self.deconv4 = nn.Sequential(
            nn.ConvTranspose2d(self.nb_filter[4], self.nb_filter[3], kernel_size=2, stride=2),
            SwitchNorm2d(self.nb_filter[3], momentum=bn_momentum),
            nn.ReLU(inplace=True)
        )
        self.deconv3 = nn.Sequential(
            nn.ConvTranspose2d(self.nb_filter[3], self.nb_filter[2], kernel_size=2, stride=2),
            SwitchNorm2d(self.nb_filter[2], momentum=bn_momentum),
            nn.ReLU(inplace=True)
        )
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(self.nb_filter[2], self.nb_filter[1], kernel_size=2, stride=2),
            SwitchNorm2d(self.nb_filter[1], momentum=bn_momentum),
            nn.ReLU(inplace=True)
        )
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(self.nb_filter[1], self.nb_filter[0], kernel_size=2, stride=2),
            SwitchNorm2d(self.nb_filter[0], momentum=bn_momentum),
            nn.ReLU(inplace=True)
        )

        self.up4 = nn.Sequential(
            nn.Conv2d(self.nb_filter[4], self.nb_filter[3], kernel_size=3),
            SwitchNorm2d(self.nb_filter[3], momentum=bn_momentum),
            nn.ReLU(inplace=True),
            # nn.UpsamplingBilinear2d(size=(64, 64))
        )
        self.up3 = nn.Sequential(
            nn.Conv2d(self.nb_filter[3], self.nb_filter[2], kernel_size=3),
            SwitchNorm2d(self.nb_filter[2], momentum=bn_momentum),
            nn.ReLU(inplace=True),
            # nn.UpsamplingBilinear2d(size=(128, 128))

        )
        self.up2 = nn.Sequential(
            nn.Conv2d(self.nb_filter[2], self.nb_filter[1], kernel_size=3),
            SwitchNorm2d(self.nb_filter[1], momentum=bn_momentum),
            nn.ReLU(inplace=True),
            # nn.UpsamplingBilinear2d(size=(256, 256))
        )
        self.up1 = nn.Sequential(
            nn.Conv2d(self.nb_filter[1], self.nb_filter[0], kernel_size=3),
            SwitchNorm2d(self.nb_filter[0], momentum=bn_momentum),
            nn.ReLU(inplace=True),
            # nn.UpsamplingBilinear2d(size=(512, 512)),
        )

        # output layers
        self.out_conv1 = nn.Sequential(
            nn.ConvTranspose2d(self.nb_filter[3], output_channel, kernel_size=8, stride=8),
            # nn.Conv2d(in_channels=128,out_channels=1,kernel_size=1),
            nn.Sigmoid()
        )
        self.out_conv2 = nn.Sequential(
            nn.ConvTranspose2d(self.nb_filter[2], output_channel, kernel_size=4, stride=4),
            # nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1),
            nn.Sigmoid()
        )
        self.out_conv3 = nn.Sequential(
            nn.ConvTranspose2d(self.nb_filter[1], output_channel, kernel_size=2, stride=2),
            # nn.Conv2d(in_channels=32,out_channels=1,kernel_size=1),
            nn.Sigmoid()
        )

        # ================================================================================================
        # Fully Connected Layers
        # ================================================================================================
        self.pool_preblock_gate1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool_preblock_gate2 = nn.MaxPool2d(kernel_size=4, stride=4)
        self.pool_preblock_gate3 = nn.MaxPool2d(kernel_size=8, stride=8)
        self.pool_out1_gate2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool_out1_gate3 = nn.MaxPool2d(kernel_size=4, stride=4)
        self.pool_out2_gate3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.up_out1_gate0 = nn.Sequential(
            nn.ConvTranspose2d(self.nb_filter[1], self.nb_filter[0], kernel_size=2, stride=2),
            SwitchNorm2d(self.nb_filter[0], momentum=bn_momentum),
            nn.ReLU(inplace=True))
        self.up_out2_gate0 = nn.Sequential(
            nn.ConvTranspose2d(self.nb_filter[2], self.nb_filter[1], kernel_size=4, stride=4),
            SwitchNorm2d(self.nb_filter[1], momentum=bn_momentum),
            nn.ReLU(inplace=True))
        self.up_out2_gate1 = nn.Sequential(
            nn.ConvTranspose2d(self.nb_filter[2], self.nb_filter[1], kernel_size=2, stride=2),
            SwitchNorm2d(self.nb_filter[1], momentum=bn_momentum),
            nn.ReLU(inplace=True))
        self.up_out3_gate0 = nn.Sequential(
            nn.ConvTranspose2d(self.nb_filter[3], self.nb_filter[2], kernel_size=8, stride=8),
            SwitchNorm2d(self.nb_filter[2], momentum=bn_momentum),
            nn.ReLU(inplace=True))
        self.up_out3_gate1 = nn.Sequential(
            nn.ConvTranspose2d(self.nb_filter[3], self.nb_filter[2], kernel_size=4, stride=4),
            SwitchNorm2d(self.nb_filter[2], momentum=bn_momentum),
            nn.ReLU(inplace=True))
        self.up_out3_gate2 = nn.Sequential(
            nn.ConvTranspose2d(self.nb_filter[3], self.nb_filter[2], kernel_size=2, stride=2),
            SwitchNorm2d(self.nb_filter[2], momentum=bn_momentum),
            nn.ReLU(inplace=True))

        # self.Fint_num = [128, 64, 32, 24]
        self.Fint_num = [256, 128, 64, 32]
        # self.nb_filter = [32, 64, 128, 256, 512]
        self.att3_1 = SN_CS_Parallel_Attention_block(F_g=self.Fint_num[0], F_x=self.Fint_num[3], F_int=self.Fint_num[0])
        self.att3_2 = SN_CS_Parallel_Attention_block(F_g=self.Fint_num[0], F_x=self.Fint_num[2], F_int=self.Fint_num[0])
        self.att3_3 = SN_CS_Parallel_Attention_block(F_g=self.Fint_num[0], F_x=self.Fint_num[1], F_int=self.Fint_num[0])
        self.att3_4 = SN_CS_Parallel_Attention_block(F_g=self.Fint_num[0], F_x=self.Fint_num[0], F_int=self.Fint_num[0])

        self.att2_1 = SN_CS_Parallel_Attention_block(F_g=self.Fint_num[1], F_x=self.Fint_num[3], F_int=self.Fint_num[1])
        self.att2_2 = SN_CS_Parallel_Attention_block(F_g=self.Fint_num[1], F_x=self.Fint_num[2], F_int=self.Fint_num[1])
        self.att2_3 = SN_CS_Parallel_Attention_block(F_g=self.Fint_num[1], F_x=self.Fint_num[1], F_int=self.Fint_num[1])
        self.att2_4 = SN_CS_Parallel_Attention_block(F_g=self.Fint_num[1], F_x=self.Fint_num[1], F_int=self.Fint_num[1])

        self.att1_1 = SN_CS_Parallel_Attention_block(F_g=self.Fint_num[2], F_x=self.Fint_num[3], F_int=self.Fint_num[2])
        self.att1_2 = SN_CS_Parallel_Attention_block(F_g=self.Fint_num[2], F_x=self.Fint_num[2], F_int=self.Fint_num[2])
        self.att1_3 = SN_CS_Parallel_Attention_block(F_g=self.Fint_num[2], F_x=self.Fint_num[2], F_int=self.Fint_num[2])
        self.att1_4 = SN_CS_Parallel_Attention_block(F_g=self.Fint_num[2], F_x=self.Fint_num[1], F_int=self.Fint_num[2])

        self.att0_1 = SN_CS_Parallel_Attention_block(F_g=self.Fint_num[3], F_x=self.Fint_num[3], F_int=self.Fint_num[3])
        self.att0_2 = SN_CS_Parallel_Attention_block(F_g=self.Fint_num[3], F_x=self.Fint_num[3], F_int=self.Fint_num[3])
        self.att0_3 = SN_CS_Parallel_Attention_block(F_g=self.Fint_num[3], F_x=self.Fint_num[2], F_int=self.Fint_num[3])
        self.att0_4 = SN_CS_Parallel_Attention_block(F_g=self.Fint_num[3], F_x=self.Fint_num[1], F_int=self.Fint_num[3])

    def get_last_feat_modules(self):
        return self.pred

    def forward(self, x, all_features=False, same=False):

        preblock = self.preBlock(x)  # 24 -> 32
        f0 = preblock
        out_pool, indices0 = self.maxpool1(preblock)

        out1 = self.forw1(out_pool)  # 32 -> 64
        f1 = out1
        out1_pool, indices1 = self.maxpool2(out1)

        out2 = self.forw2(out1_pool)  # 64 -> 128
        f2 = out2
        out2_pool, indices2 = self.maxpool3(out2)

        out3 = self.forw3(out2_pool)  # 128 -> 256
        f3 = out3
        out3_pool, indices3 = self.maxpool4(out3)

        out4 = self.forw4(out3_pool)  # 256 -> 512
        f4 = out4
        # ============================================================================
        # Fully Connected Modules
        # ============================================================================

        # ---------Gate0--------------
        out1_gate0 = self.up_out1_gate0(out1)  # 24 -> 64-32
        out2_gate0 = self.up_out2_gate0(out2)  # 32 -> 128-64
        out3_gate0 = self.up_out3_gate0(out3)  # 64 -> 256-128
        # out4_gate0 = self.up_out4_gate0(out4)

        # ---------Gate1--------------
        preblock_gate1 = self.pool_preblock_gate1(preblock)  # 24 -> 32
        out2_gate1 = self.up_out2_gate1(out2)  # 32 -> 128-64
        out3_gate1 = self.up_out3_gate1(out3)  # 64 -> 256-128

        # ---------Gate2--------------
        preblock_gate2 = self.pool_preblock_gate2(preblock)  # 24 -> 32
        out1_gate2 = self.pool_out1_gate2(out1)  # 32 -> 64
        out3_gate2 = self.up_out3_gate2(out3)  # 64 -> 256
        # out4_gate2 = self.up_out4_gate2(out4)

        # ---------Gate3--------------
        preblock_gate3 = self.pool_preblock_gate3(preblock)  # 24
        out1_gate3 = self.pool_out1_gate3(out1)  # 32
        out2_gate3 = self.pool_out2_gate3(out2)  # 64
        # out4_gate3 = self.up_out4_gate3(out4)

        # print('out1', out1.shape)  # [1, 64, 320, 320]
        # print('out2', out2.shape)  # [1, 128, 160, 160]
        # print('out3', out3.shape)  # [1, 256, 80, 80]
        # print('out4', out4.shape)  # [1, 512, 40, 40]

        up4 = self.up4(out4)  # 128
        # up4 = self.up4(out4)
        up4 = nn.functional.interpolate(up4, size=(int(self.original_size / 8), int(self.original_size / 8)), mode='nearest')
        # up4 = nn.functional.interpolate(up4, size=(80, 80), mode='nearest')

        deconv4 = self.deconv4(out4)
        up4_sig = up4 + deconv4
        up4 = torch.cat((up4_sig, deconv4), dim=1)

        # 原来 [24, 32, 64, 128, 256]  现在
        preblock_gate3 = self.att3_1(g=up4_sig, x=preblock_gate3)   # g[1, 256, 80, 80]  x[1, 32, 80, 80]
        out1_gate3 = self.att3_2(g=up4_sig, x=out1_gate3)
        out2_gate3 = self.att3_3(g=up4_sig, x=out2_gate3)
        out3 = self.att3_4(g=up4_sig, x=out3)
        # gate3_fusion = torch.cat((preblock_gate3, out1_gate3, out2_gate3, out3), dim=1)  # 24+32+64+128=248
        # gate3_back = self.gate3_back(gate3_fusion)  # 128
        gate3 = torch.cat((up4, preblock_gate3, out1_gate3, out2_gate3, out3), dim=1)  # 128 + 128 = 256
        # gate3 = self.em_att_unit3(comb3)
        comb3 = self.back1(gate3)
        f5 = comb3

        up3 = self.up3(comb3)
        up3 = nn.functional.interpolate(up3, size=(int(self.original_size / 4), int(self.original_size / 4)), mode='nearest')
        # up3 = nn.functional.interpolate(up3, size=(160, 160), mode='nearest')
        deconv3 = self.deconv3(comb3)
        up3_sig = up3 + deconv3
        up3 = torch.cat((up3, deconv3), dim=1)
        preblock_gate2 = self.att2_1(g=up3_sig, x=preblock_gate2)
        out1_gate2 = self.att2_2(g=up3_sig, x=out1_gate2)
        out2 = self.att2_3(g=up3_sig, x=out2)
        out3_gate2 = self.att2_4(g=up3_sig, x=out3_gate2)
        # gate2_fusion = torch.cat((preblock_gate2, out1_gate2, out2, out3_gate2), dim=1)  # 24+32+64+64=184
        # gate2_back = self.gate2_back(gate2_fusion)  # 64
        gate2 = torch.cat((up3, preblock_gate2, out1_gate2, out2, out3_gate2), dim=1)  # 64 + 64 = 128
        # gate2 = self.em_att_unit2(comb2)
        comb2 = self.back2(gate2)
        f6 = comb2

        up2 = self.up2(comb2)  # 32
        up2 = nn.functional.interpolate(up2, size=(int(self.original_size / 2), int(self.original_size / 2)), mode='nearest')
        # up2 = nn.functional.interpolate(up2, size=(320, 320), mode='nearest')
        deconv2 = self.deconv2(comb2)
        up2_sig = up2 + deconv2
        up2 = torch.cat((up2, deconv2), dim=1)
        preblock_gate1 = self.att1_1(g=up2_sig, x=preblock_gate1)
        out1 = self.att1_2(g=up2_sig, x=out1)
        out2_gate1 = self.att1_3(g=up2_sig, x=out2_gate1)
        out3_gate1 = self.att1_4(g=up2_sig, x=out3_gate1)
        # gate1_fusion = torch.cat((preblock_gate1, out1, out2_gate1, out3_gate1), dim=1)  # 24+32+32+64=152
        # gate1_back = self.gate1_back(gate1_fusion)  # 32
        gate1 = torch.cat((up2, preblock_gate1, out1, out2_gate1, out3_gate1), dim=1)  # 32 + 32 = 64
        # gate1 = self.em_att_unit1(comb1)
        comb1 = self.back3(gate1)  # 32
        f7 = comb1
        up1 = self.up1(comb1)  # 24
        up1 = nn.functional.interpolate(up1, size=(self.original_size, self.original_size), mode='nearest')
        # up1 = nn.functional.interpolate(up1, size=(640, 640), mode='nearest')

        deconv1 = self.deconv1(comb1)
        up1_sig = up1 + deconv1
        up1 = torch.cat((up1, deconv1), dim=1)
        preblock = self.att0_1(g=up1_sig, x=preblock)
        out1_gate0 = self.att0_2(g=up1_sig, x=out1_gate0)
        out2_gate0 = self.att0_3(g=up1_sig, x=out2_gate0)
        out3_gate0 = self.att0_4(g=up1_sig, x=out3_gate0)
        gate0 = torch.cat((up1, preblock, out1_gate0, out2_gate0, out3_gate0), dim=1)  # 24 + 24 = 48
        # gate0 = self.em_att_unit0(out)
        out = self.back4(gate0)  # 48
        feature = out
        f8 = out

        # out = self.outBlock(out)  # 24
        out = self.pred(out)
        out3 = self.out_conv1(comb3)
        out2 = self.out_conv2(comb2)
        out1 = self.out_conv3(comb1)

        if same:
            return out.permute(0, 2, 3, 1), out3.permute(0, 2, 3, 1), \
                   out2.permute(0, 2, 3, 1), out1.permute(0, 2, 3, 1), [f0, f1, f2, f3, f4, f5, f6, f7, f8]

        if all_features:
            # return [f0, f1, f2, f3, f4],  out.permute(0, 2, 3, 1)

            # return [f5, f6, f7, f8],  out.permute(0, 2, 3, 1)
            return [f0, f1, f2, f3, f4, f5, f6, f7, f8], out.permute(0, 2, 3, 1)

        # if not self.feature: print('in') return out.permute(0, 2, 3, 1), out3.permute(0, 2, 3, 1), out2.permute(0,
        # 2, 3, 1), out1.permute(0, 2, 3, 1)  # 640, 80, 160, 320
        # return out.permute(0, 2, 3, 1), out3.permute(0, 2, 3, 1), \
        #        out2.permute(0, 2, 3, 1), out1.permute(0, 2, 3, 1), feature
        return out.permute(0, 2, 3, 1), out3.permute(0, 2, 3, 1), \
               out2.permute(0, 2, 3, 1), out1.permute(0, 2, 3, 1)


if __name__ == '__main__':
    images = torch.rand(1, 3, 640, 640)
    model = CAUNet(3, 5)
    feature, output = model(images, all_features=True)
    print('output:', output.shape)
    for i, o in enumerate(feature):
        print('d{}:{}'.format(i, o.shape))
