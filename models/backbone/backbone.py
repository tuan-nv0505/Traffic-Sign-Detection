from collections import OrderedDict

import torch
import torch.nn as nn
from models.backbone.conv import ConvNet
from models.backbone.vmamba import VSSBlock, PatchMerging2D
from models.backbone.permute import Permute


class BackBone(nn.Module):
    def __init__(self, dims=3, depth=4):
        super().__init__()
        self.depth = depth
        self.pre_embd = ConvNet()
        self.dims = [dims * 2 ** i_layer for i_layer in range(self.depth + 1)]
        self.num_features = self.dims[-1]
        self.layers = nn.ModuleList()
        for i_layer in range(self.depth):
            downsample = PatchMerging2D(
                dim=self.dims[i_layer],
                out_dim=self.dims[i_layer + 1],
                norm_layer=nn.LayerNorm,
            )
            vss_block = VSSBlock(hidden_dim=self.dims[i_layer + 1], drop_path=0.0, ssm_d_state=16)
            self.layers.append(downsample)
            self.layers.append(vss_block)

        self.out_norm = nn.Sequential(OrderedDict(
            permute_in=Permute(0, 2, 3, 1),
            norm=nn.LayerNorm(self.num_features),
            permute_out=Permute(0, 3, 1, 2),
        ))

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x: torch.Tensor):
        x = self.pre_embd(x)
        for layer in self.layers:
            x = layer(x)
        x = self.out_norm(x)
        return x