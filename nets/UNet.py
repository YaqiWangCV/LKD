import torch.nn as nn
import torch


def get_activation(activation_type):
    activation_type = activation_type.lower()
    if hasattr(nn, activation_type):
        return getattr(nn, activation_type)()
    else:
        return nn.ReLU()


def _make_nConv(in_channels, out_channels, nb_Conv, activation='ReLU'):
    layers = []
    layers.append(ConvBatchNorm(in_channels, out_channels, activation))

    for _ in range(nb_Conv - 1):
        layers.append(ConvBatchNorm(out_channels, out_channels, activation))
    return nn.Sequential(*layers)


class ConvBatchNorm(nn.Module):
    """(convolution => [BN] => ReLU)"""

    def __init__(self, in_channels, out_channels, activation='ReLU'):
        super(ConvBatchNorm, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=3, padding=1)
        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = get_activation(activation)

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        return self.activation(out)


class DownBlock(nn.Module):
    """Downscaling with maxpool convolution"""

    def __init__(self, in_channels, out_channels, nb_Conv, activation='ReLU'):
        super(DownBlock, self).__init__()
        self.maxpool = nn.MaxPool2d(2)
        self.nConvs = _make_nConv(in_channels, out_channels, nb_Conv, activation)

    def forward(self, x):
        out = self.maxpool(x)
        return self.nConvs(out)


class UpBlock(nn.Module):
    """Upscaling then conv"""

    def __init__(self, in_channels, out_channels, nb_Conv, activation='ReLU'):
        super(UpBlock, self).__init__()

        self.up = nn.Upsample(scale_factor=2)
        self.nConvs = _make_nConv(in_channels, out_channels, nb_Conv, activation)

    def forward(self, x, skip_x):
        out = self.up(x)
        x = torch.cat([out, skip_x], dim=1)  # dim 1 is the channel dimension
        return self.nConvs(x)


class UNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=5, feature=False):
        '''
    n_channels : number of channels of the input.
                    By default 3, because we have RGB images
    n_labels : number of channels of the ouput.
                  By default 3 (2 labels + 1 for the background)
    '''
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.feature = feature
        nb_filter = [32, 64, 128, 256, 512]
        # Question here
        # self.inc = ConvBatchNorm(n_channels, 64)
        # self.down1 = DownBlock(64, 128, nb_Conv=2)
        # self.down2 = DownBlock(128, 256, nb_Conv=2)
        # self.down3 = DownBlock(256, 512, nb_Conv=2)
        # self.down4 = DownBlock(512, 512, nb_Conv=2)
        # self.up1 = UpBlock(1024, 256, nb_Conv=2)
        # self.up2 = UpBlock(512, 128, nb_Conv=2)
        # self.up3 = UpBlock(256, 64, nb_Conv=2)
        # self.up4 = UpBlock(128, 64, nb_Conv=2)
        self.inc = ConvBatchNorm(n_channels, nb_filter[0])
        self.down1 = DownBlock(nb_filter[0], nb_filter[1], nb_Conv=2)
        self.down2 = DownBlock(nb_filter[1], nb_filter[2], nb_Conv=2)
        self.down3 = DownBlock(nb_filter[2], nb_filter[3], nb_Conv=2)
        self.down4 = DownBlock(nb_filter[3], nb_filter[4], nb_Conv=2)
        self.up1 = UpBlock(nb_filter[3]+nb_filter[4], nb_filter[3], nb_Conv=2)
        self.up2 = UpBlock(nb_filter[2]+nb_filter[3], nb_filter[2], nb_Conv=2)
        self.up3 = UpBlock(nb_filter[1]+nb_filter[2], nb_filter[1], nb_Conv=2)
        self.up4 = UpBlock(nb_filter[0]+nb_filter[1], nb_filter[0], nb_Conv=2)
        # self.outc1 = nn.Conv2d(64, 24, kernel_size=3, stride=1, padding=1)
        self.outc2 = nn.Conv2d(nb_filter[0], n_classes, kernel_size=3, stride=1, padding=1)
        # self.last_activation = get_activation('Softmax')
        self.last_activation = nn.Sigmoid()  # if using BCELoss

    def forward(self, x, all_features=False):
        # Question here
        x = x.float()
        x1 = self.inc(x)
        f0 = x1
        x2 = self.down1(x1)
        f1 = x2
        x3 = self.down2(x2)
        f2 = x3
        x4 = self.down3(x3)
        f3 = x4
        x5 = self.down4(x4)
        f4 = x5
        x = self.up1(x5, x4)
        f5 = x
        x = self.up2(x, x3)
        f6 = x
        x = self.up3(x, x2)
        f7 = x
        x = self.up4(x, x1)
        # feature = self.outc1(x)
        f8 = x

        logits = self.outc2(x)  # if using BCEWithLogitsLoss
        logits = self.last_activation(logits)

        if all_features:
            #return [f0, f1, f2, f3, f4, f5],  logits.permute(0, 2, 3, 1)
            # return [f5, f6, f7, f8],  logits.permute(0, 2, 3, 1)
            return [f0, f1, f2, f3, f4, f5, f6, f7, f8],  logits.permute(0, 2, 3, 1)
        if not self.feature:
            return logits.permute(0, 2, 3, 1)
        return logits.permute(0, 2, 3, 1), f8


if __name__ == '__main__':
    images = torch.rand(1, 3, 640, 640)
    model = UNet(3, 5)
    feature, output = model(images, all_features=True)
    print('output:', output.shape)
    for i, o in enumerate(feature):
        print('d{}:{}'.format(i, o.shape))
