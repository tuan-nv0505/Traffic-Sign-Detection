import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary


class AveragePoolingChannel(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.avg_pool2d(x, kernel_size=2, stride=2)

class MaxPoolingChannel(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.max_pool2d(x, kernel_size=2, stride=2)

class SE(nn.Module):
    def __init__(self, in_channels, reduction):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_features=in_channels, out_features=in_channels // reduction, bias=False),
            nn.SiLU(),
            nn.Linear(in_features=in_channels // reduction, out_features=in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.SiLU()
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=1),
            nn.BatchNorm2d(32),
            nn.SiLU(),
            nn.Conv2d(in_channels=32, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.SiLU()
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.SiLU(),
            nn.Conv2d(in_channels=64, out_channels=3, kernel_size=3, padding=1)
        )

        self.se = SE(in_channels=3, reduction=3)

        self.max_pool = MaxPoolingChannel()
        self.avg_pool = AveragePoolingChannel()

        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=1),
            nn.BatchNorm2d(num_features=64),
            nn.SiLU(),
            nn.Conv2d(in_channels=64, out_channels=3, kernel_size=3, padding=1)
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1),
            nn.SiLU(),
        )

        self.conv6 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=1),
            nn.BatchNorm2d(num_features=64),
            nn.SiLU(),
            nn.Conv2d(in_channels=64, out_channels=3, kernel_size=5, padding=2)
        )

        self.alpha = nn.Parameter(torch.tensor(0.2))
        self.beta = nn.Parameter(torch.tensor(0.6))
        self.gamma = nn.Parameter(torch.tensor(0.2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        channel_1_max_pool = self.max_pool(x)
        desired_size = (x.size(2), x.size(3))
        channel_1_max_pool_out = F.interpolate(channel_1_max_pool, size=desired_size, mode='bilinear',
                                               align_corners=False)

        channel_2_1 = self.conv4(x)
        channel_2_2 = self.conv5(x)
        channel_2_3 = self.conv6(x)

        channel_2_sum = 0.2 * channel_2_1 + 0.6 * channel_2_2 + 0.2 * channel_2_3

        channel_2_x_1 = self.conv1(channel_2_sum)
        channel_2_x_2 = self.conv2(channel_2_x_1)
        channel_2_x_3 = self.conv3(channel_2_x_2)
        channel_2_x_4 = self.se(channel_2_x_3)

        channel_2_total = channel_2_x_4 * 0.6 + channel_2_x_3 * 0.4
        channel_2_total_avg_pool = self.avg_pool(channel_2_total)
        desired_size = (x.size(2), x.size(3))
        channel_2_avg_pool_out = F.interpolate(
            channel_2_total_avg_pool,
            size=desired_size, mode='bilinear',
            align_corners=False
        )

        channel_3 = x

        conv_net_out = self.alpha * channel_1_max_pool_out + self.beta * channel_2_avg_pool_out + self.gamma * channel_3

        return conv_net_out

