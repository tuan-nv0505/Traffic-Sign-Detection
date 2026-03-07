import torch
from torch import nn as nn
from torch.nn import functional as F
from torchinfo import summary
from pprint import pprint

from models.backbone.backbone import BackBone

class FeaturePyramidNetwork(nn.Module):
    def __init__(self, list_in_channels, out_channels=256):
        super().__init__()
        self.list_in_channels = list_in_channels
        self.list_feature_map_1 = nn.ModuleList()
        for in_channels in list_in_channels:
            self.list_feature_map_1.append(nn.Conv2d(in_channels, out_channels, kernel_size=1))

        if len(list_in_channels) > 1:
            self.list_feature_map_2 = nn.ModuleList()
            for _ in range(len(list_in_channels) - 1):
                self.list_feature_map_2.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1))

    def forward(self, list_feature_map: list[torch.Tensor]) -> list[torch.Tensor]:
        list_feature_map = [conv(x) for conv, x in zip(self.list_feature_map_1, list_feature_map)]

        for i in range(len(list_feature_map) - 1, 0, -1):
            up_shape = list_feature_map[i - 1].shape[2:]
            list_feature_map[i - 1] = list_feature_map[i - 1] + F.interpolate(
                list_feature_map[i], size=up_shape, mode='nearest'
            )

        p_outputs = []
        if len(self.list_in_channels) > 1:
            for i in range(len(list_feature_map) - 1):
                p_outputs.append(self.list_feature_map_2[i](list_feature_map[i]))

        p_outputs.append(list_feature_map[-1])

        return p_outputs


class FeatureMapExtractor(BackBone):
    def __init__(self, in_channels=3, features='last', out_channels=128, depth=4, ssm_d_state=16, weight=None):
        super().__init__(dims=in_channels, depth=depth, ssm_d_state=ssm_d_state)
        self.features = features
        self.out_channels = out_channels
        if weight:
            result = self.load_state_dict(weight, strict=False)
            print("Missing keys:", len(result.missing_keys))
            print("Unexpected keys:", len(result.unexpected_keys))
            print("Missing keys list:")
            pprint(result.missing_keys)
            print("Unexpected keys list:")
            pprint(result.unexpected_keys)

        if features == 'last':
            dims = self.dims[-1:]
        elif features == 'all':
            dims = self.dims[1:]
        else:
            raise NotImplementedError

        self.fpn = FeaturePyramidNetwork(list_in_channels=dims, out_channels=out_channels)

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        outputs = []
        x = self.pre_embd(x)
        for i in range(0, len(self.layers), 2):
            downsample = self.layers[i]
            vss_block = self.layers[i + 1]

            x = downsample(x)
            x = vss_block(x)

            outputs.append(x)

        if self.features == 'last':
            outputs = outputs[-1:]

        return self.fpn(outputs)

if __name__ == '__main__':
    checkpoint = torch.load('../../best_checkpoint.pth', map_location='cpu', weights_only=True)
    new_checkpoint = {}
    for k, v in checkpoint['state_dict'].items():
        if k.startswith("backbone."):
            new_k = k[len("backbone."):]
        else:
            new_k = k
        new_checkpoint[new_k] = v
    net = FeatureMapExtractor(in_channels=3, features='last', out_channels=64, depth=3, ssm_d_state=8, weight=new_checkpoint)
    x = torch.rand(1, 3, 608, 1024)
    # summary(net, (1, 3, 608, 1024))
    out = net(x)
    # for x in out:
    #     print(x.shape)