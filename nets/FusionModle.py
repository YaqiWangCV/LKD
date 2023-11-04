import math
import torch
from torch import nn
import torch.nn.functional as F
import warnings
warnings.filterwarnings("ignore")


class FusionModule(nn.Module):
    def __init__(self, channel, output_channel):
        super(FusionModule, self).__init__()
        self.conv1 = nn.Conv2d(channel * 2, channel * 2, kernel_size=3, stride=1, padding=1, groups=channel * 2,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(channel * 2)
        self.conv1_1 = nn.Conv2d(channel * 2, channel, kernel_size=1, groups=1, bias=False)
        self.bn1_1 = nn.BatchNorm2d(channel)

        self.final = nn.Conv2d(channel, output_channel, kernel_size=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x, y):
        input = torch.cat((x, y), 1)
        x = F.relu(self.bn1((self.conv1(input))))
        x = F.relu(self.bn1_1(self.conv1_1(x)))
        x = F.sigmoid(self.final(x))
        return x.permute(0, 2, 3, 1)


if __name__ == '__main__':
    data1 = torch.rand((1, 24, 640, 640))
    data2 = torch.rand((1, 24, 640, 640))

    module = FusionModule(24, 5)
    print(module(data1, data2).shape)
