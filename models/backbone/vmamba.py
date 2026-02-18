import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat
from timm.layers import DropPath

from models.backbone.conv import SE

def cross_scan(x: torch.Tensor):
    B, C, H, W = x.shape
    L = H * W

    scanned = torch.stack([
        x.view(B, C, L),
        x.view(B, C, L).flip(dims=[-1]),
        x.permute(0, 1, 3, 2).contiguous().view(B, C, L),
        x.permute(0, 1, 3, 2).contiguous().view(B, C, L).flip(dims=[-1])
    ], dim=1)

    return scanned

def cross_merge(y: torch.Tensor, H, W):
    B, _, C, L = y.shape
    y1, y2, y3, y4 = y.chunk(4, 1)

    y1 = y1.squeeze(dim=1)
    y2 = y2.squeeze(dim=1).flip(dims=[-1])
    y3 = y3.squeeze(dim=1).view(B, C, W, H).permute(0, 1, 3, 2).contiguous().view(B, C, L)
    y4 = y4.squeeze(dim=1).flip(dims=[-1]).view(B, C, W, H).permute(0, 1, 3, 2).contiguous().view(B, C, L)

    return (y1 + y2 + y3 + y4).view(B, C, H, W).contiguous()

def selective_scan(
        x: torch.Tensor,
        x_projection_weight: torch.Tensor, # (K, 2 * d_state + delta_rank, d_inner)
        delta_projection_weight: torch.Tensor, # (K, d_inner, d_rank)
        delta_projection_bias: torch.Tensor, # (K, d_inner)
        A_log: torch.Tensor, # (K, d_inner, d_state)
        Ds: torch.Tensor, # (K, d_inner)
        delta_softplus = True,
):
    _, _, H, W = x.shape
    x_scanned = cross_scan(x)
    _, _, R = delta_projection_weight.shape
    d_state = A_log.shape[-1]

    delta_b_c = x_scanned.permute(0, 1, 3, 2) @ x_projection_weight.unsqueeze(0).permute(0, 1, 3, 2)
    delta_b_c = delta_b_c.permute(0, 1, 3, 2)
    delta, b, c = delta_b_c.split((R, d_state, d_state), dim=2)

    delta = delta.permute(0, 1, 3, 2) @ delta_projection_weight.unsqueeze(0).permute(0, 1, 3, 2)
    delta = delta.permute(0, 1, 3, 2) + delta_projection_bias[None, :, :, None]
    if delta_softplus:
        delta = F.softplus(delta)

    delta_A = (delta[:, :, :, :, None] * A_log[None, :, :, None, :]).to(torch.float32)
    phi = torch.exp(-torch.cumsum(delta_A, dim=3))
    delta_b_x_scanned = (
            delta[:, :, :, None, :] *
            b[:, :, None, :, :] *
            x_scanned[:, :, :, None, :]
    ).permute(0, 1, 2, 4, 3).contiguous()

    h = phi * torch.cumsum(delta_b_x_scanned / (phi + 1e-12), dim=3)
    y = (h.permute(0, 1, 3, 2, 4) @ c.permute(0, 1, 3, 2).unsqueeze(-1)).squeeze(-1).permute(0, 1, 3, 2)

    return cross_merge(y + Ds[None, :, :, None], H, W)


class SS2D(nn.Module):
    def __init__(self,
                 # basic dims ===========
                 d_model,
                 d_state=16,
                 ssm_ratio=2,
                 ssm_rank_ratio=2,
                 delta_rank="auto",
                 # conv_dw ===============
                 d_conv=3,
                 conv_bias=True,
                 # ======================
                 dropout=0.0,
                 bias=False,
                 # delta init ==============
                 delta_min=0.001,
                 delta_max=0.1,
                 delta_init="random",
                 delta_scale=1.0,
                 delta_init_floor=1e-4,
    ):
        super().__init__()
        d_expand = d_model * ssm_ratio
        d_inner = int(min(ssm_ratio, ssm_rank_ratio) * d_model) if ssm_rank_ratio > 0 else d_expand
        self.d_state = math.ceil(d_model / 6) if d_state == "auto" else d_state
        self.delta_rank = math.ceil(d_model / 16) if delta_rank == "auto" else delta_rank
        self.d_conv = d_conv

        self.in_projection = nn.Linear(in_features=d_model, out_features=d_expand * 2, bias=bias)

        self.conv2d = nn.Conv2d(
            in_channels=d_expand,
            out_channels=d_expand,
            kernel_size=self.d_conv,
            padding=(self.d_conv - 1) // 2,
            groups=d_expand
        )

        self.ssm_low_rank = False
        if d_inner < d_expand:
            self.ssm_low_rank = True
            self.in_rank = nn.Conv2d(in_channels=d_expand, out_channels=d_inner, kernel_size=1, bias=False)
            self.out_rank = nn.Linear(in_features=d_inner, out_features=d_expand, bias=False)

        self.x_projection = [
            nn.Linear(in_features=d_inner, out_features=self.delta_rank + self.d_state * 2, bias=False)
            for _ in range(4)
        ]
        # x_projection_weight (K, 2 * d_state + delta_rank, d_inner)
        self.x_projection_weight = nn.Parameter(torch.stack([k.weight for k in self.x_projection], dim=0))
        # print(f"shape x_projection_weight: (K, 2 * d_state, d_inner) {self.x_projection_weight.shape}")
        del self.x_projection

        self.delta_projection = [
            self.delta_init(
                delta_rank=self.delta_rank,
                d_inner=d_inner,
                delta_min=delta_min,
                delta_max=delta_max,
                delta_init=delta_init,
                delta_scale=delta_scale,
                delta_init_floor=delta_init_floor
            )
            for _ in range(4)
        ]
        # delta_projection_weight (K, d_inner, delta_rank)
        self.delta_projection_weight = nn.Parameter(torch.stack([k.weight for k in self.delta_projection], dim=0))
        # print(f"shape delta_projection_weight: (K, d_inner, delta_rank) {self.delta_projection_weight.shape}")
        # delta_projection_bias (K, d_inner)
        self.delta_projection_bias = nn.Parameter(torch.stack([k.bias for k in self.delta_projection], dim=0))
        # print(f"shape delta_projection_bias: (K, d_inner) {self.delta_projection_bias.shape}")
        del self.delta_projection

        # A_log (K, d_inner, d_state)
        self.A_log = self.A_log_init(d_state, d_inner)
        # print(f"shape A_log: (K, d_inner, d_state) {self.A_log.shape}")
        # Ds (K, d_inner)
        self.Ds = self.D_init(d_inner)
        # print(f"shape Ds: (K, d_inner) {self.Ds.shape}")

        self.out_projection = nn.Linear(d_expand, d_model, bias=bias)
        self.norm = nn.LayerNorm(d_expand)
        self.effn = EFFN(d_expand)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()


    @staticmethod
    def delta_init(delta_rank,
                   d_inner,
                   delta_scale=1.0,
                   delta_init="random",
                   delta_min=0.001,
                   delta_max=0.1,
                   delta_init_floor=1e-4,
                   **factory_kwargs
    ):
        delta_projection = nn.Linear(delta_rank, d_inner, bias=True)

        delta_init_std = delta_rank ** -0.5 * delta_scale
        if delta_init == "constant":
            nn.init.constant_(delta_projection.weight, delta_init_std)
        elif delta_init == "random":
            nn.init.uniform_(delta_projection.weight, -delta_init_std, delta_init_std)
        else:
            raise NotImplementedError

        delta = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(delta_max) - math.log(delta_min))
            + math.log(delta_min)
        ).clamp(min=delta_init_floor)
        inv_delta = delta + torch.log(-torch.expm1(-delta))
        with torch.no_grad():
            delta_projection.bias.copy_(inv_delta)

        return delta_projection

    @staticmethod
    def A_log_init(d_state, d_inner):
        A = repeat(torch.arange(1, d_state + 1, dtype=torch.float32),"n -> d n", d=d_inner).contiguous()
        A_log = torch.log(A)
        A_log = repeat(A_log, "d n -> k d n", k=4).clone().contiguous()
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner):
        D = torch.ones(d_inner)
        D = repeat(D, "n1 -> k n1", k=4).clone().contiguous()
        D = nn.Parameter(D)
        D._no_weight_decay = True
        return D

    def forward_ss2d(self, x: torch.Tensor):
        if self.ssm_low_rank:
            x = self.in_rank(x)

        x = selective_scan(
            x=x,
            x_projection_weight=self.x_projection_weight,
            delta_projection_weight=self.delta_projection_weight,
            delta_projection_bias=self.delta_projection_bias,
            A_log=self.A_log,
            Ds=self.Ds
        )

        if self.ssm_low_rank:
            x = self.out_rank(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        return x

    def forward(self, x: torch.Tensor):
        xz = x.permute(0, 2, 3, 1)
        xz = self.in_projection(xz)
        x, z = xz.chunk(2, dim=-1)
        x, z = x.permute(0, 3, 1, 2), z.permute(0, 3, 1, 2)
        x = F.silu(self.conv2d(x))
        x = self.forward_ss2d(x)
        x = self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        return self.effn(x * z)


class PatchMerging2D(nn.Module):
    def __init__(self, dim, out_dim=-1, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, (2 * dim) if out_dim < 0 else out_dim, bias=False)
        self.norm = norm_layer(4 * dim)

    @staticmethod
    def _patch_merging_pad(x: torch.Tensor):
        H, W, _ = x.shape[-3:]
        if (W % 2 != 0) or (H % 2 != 0):
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))
        x0 = x[..., 0::2, 0::2, :]  # ... H/2 W/2 C
        x1 = x[..., 1::2, 0::2, :]  # ... H/2 W/2 C
        x2 = x[..., 0::2, 1::2, :]  # ... H/2 W/2 C
        x3 = x[..., 1::2, 1::2, :]  # ... H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # ... H/2 W/2 4*C
        return x

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = self._patch_merging_pad(x)
        x = self.norm(x)
        x = self.reduction(x)
        return x.permute(0, 3, 1, 2)

class EFFN(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels * 2, kernel_size=1),
            nn.BatchNorm2d(num_features=in_channels * 2),
            nn.ReLU()
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels * 2, out_channels=in_channels * 2, kernel_size=3, padding=1, groups=in_channels * 2),
            nn.BatchNorm2d(num_features=in_channels * 2),
            nn.ReLU()
        )

        self.conv3 = nn.Conv2d(in_channels=in_channels * 2, out_channels=in_channels // 2,kernel_size=1)

        self.se = SE(in_channels=in_channels // 2, reduction=2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x + self.se(x)
        return x

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0, channels_first=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        Linear = partial(nn.Conv2d, kernel_size=1, padding=0) if channels_first else nn.Linear
        self.fc1 = Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        x = x.permute(0, 3, 1, 2)
        return x


class VSSBlock(nn.Module):
    def __init__(
            self,
            hidden_dim,
            drop_path=0.0,
            norm_layer = partial(nn.LayerNorm, eps=1e-6),
            # =============================
            ssm_d_state= 16,
            ssm_ratio=2,
            ssm_rank_ratio=2,
            ssm_delta_rank="auto",
            ssm_conv=3,
            ssm_conv_bias=True,
            ssm_drop_rate=0,
            # =============================
            mlp_ratio=4.0,
            mlp_act_layer=nn.GELU,
            mlp_drop_rate: float = 0.0,
            **kwargs,
    ):
        super().__init__()
        self.norm = norm_layer(hidden_dim)
        self.ss2d = SS2D(
            d_model=hidden_dim,
            d_state=ssm_d_state,
            ssm_ratio=ssm_ratio,
            ssm_rank_ratio=ssm_rank_ratio,
            delta_rank=ssm_delta_rank,
            # ==========================
            d_conv=ssm_conv,
            conv_bias=ssm_conv_bias,
            # ==========================
            dropout=ssm_drop_rate,
            # bias=False,
            # ==========================
            # dt_min=0.001,
            # dt_max=0.1,
            # dt_init="random",
            # dt_scale="random",
            # dt_init_floor=1e-4,
        )
        self.drop_path = DropPath(drop_path)

        self.mlp_branch = mlp_ratio > 0
        if self.mlp_branch:
            self.norm2 = norm_layer(hidden_dim)
            mlp_hidden_dim = int(hidden_dim * mlp_ratio)
            self.mlp = Mlp(
                in_features=hidden_dim,
                hidden_features=mlp_hidden_dim,
                act_layer=mlp_act_layer,
                drop=mlp_drop_rate,
                channels_first=False
            )


    def forward(self, input: torch.Tensor):
        x = self.norm(input.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x = input + self.drop_path(self.ss2d(x))
        if self.mlp_branch:
            x = self.norm2(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
            x = x + self.drop_path(self.mlp(x))
        return x
