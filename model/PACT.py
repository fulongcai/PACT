# 主题：守恒平流 + 轨迹连续性
# 核心思想：把“看哪里/记多久”的选择性交给物理量（平流残差与速度模长），
# 在不破坏守恒与稀疏索引的前提下，自适应调节时间步 Δt、核宽 σ 与多假设权重，

import os
import math
import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
import spconv.pytorch as spconv
from spconv.pytorch import SparseModule
from spconv.pytorch import functional as Fsp



class ResidualShortcut(nn.Module):
    def __init__(self, in_channels, out_channels, norm_fn, stride=1):
        super(ResidualShortcut, self).__init__()
        self.conv = spconv.SparseSequential(
            spconv.SubMConv3d(in_channels, out_channels, kernel_size=1, bias=False),
            norm_fn(out_channels)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class SparseAvgPool(nn.Module):
    def forward(self, x: spconv.SparseConvTensor):
        feats = x.features              # [N, C]
        bidx = x.indices[:, 0].long()   # [N]
        B = int(bidx.max().item()) + 1
        C = feats.size(1)
        out = feats.new_zeros(B, C)
        cnt = feats.new_zeros(B, 1)
        out.index_add_(0, bidx, feats)
        cnt.index_add_(0, bidx, feats.new_ones(feats.size(0), 1))
        return out / cnt.clamp_min(1.0)


class MLPBlock(nn.Module):
    def __init__(self, channel, reduction=4):
        super().__init__()
        self.avg_pool = SparseAvgPool()
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: spconv.SparseConvTensor):
        se = self.avg_pool(x)                   # [B, C]
        bidx = x.indices[:, 0].long()           # [N]
        scale = self.fc(se)[bidx]               # [N, C]
        return x.replace_feature(scale * x.features)


class DilatedConv(SparseModule):
    def __init__(self, out_channels, norm_fn, stride=1, dilations=[1, 2, 3, 4], indice_key=None):
        super().__init__()
        num_splits = len(dilations)
        assert (out_channels % num_splits == 0)
        temp = out_channels // num_splits
        convs = []
        for d in dilations:
            convs.append(
                spconv.SubMConv3d(
                    temp, temp,
                    kernel_size=3,
                    padding=d,
                    dilation=d,
                    stride=stride
                )
            )
        self.convs = nn.ModuleList(convs)
        self.num_splits = num_splits
        self.temp = temp

    def forward(self, x):
        res = []
        for i in range(self.num_splits):
            features_split = x.features[:, i * self.temp:(i + 1) * self.temp]
            x_split = spconv.SparseConvTensor(features_split, x.indices, x.spatial_shape, x.batch_size)
            res.append(self.convs[i](x_split).features)
        x_features = torch.cat(res, dim=1)
        x = x.replace_feature(x_features)
        return x


# ContextAggregationBlock
class SCABlock(SparseModule):
    def __init__(self, in_channels, out_channels, kernel_size, stride, norm_fn,
                 dilations=[1, 2, 3, 4], indice_key=None, bias=False, ad_channels=16):
        super().__init__()

        self.ResidualShortcut = ResidualShortcut(in_channels, out_channels, norm_fn)
        add_channel = ad_channels
        self.pwconv = spconv.SparseSequential(
            spconv.SubMConv3d(in_channels, out_channels + add_channel, kernel_size=1, padding=1, bias=bias),
            norm_fn(out_channels + add_channel),
            nn.ReLU(),
        )

        # 主体空间上下文聚合
        self.DilatedConv_spa = spconv.SparseSequential(
            DilatedConv(out_channels + add_channel, norm_fn, stride, dilations=dilations, indice_key=indice_key),
            norm_fn(out_channels + add_channel),
            nn.ReLU(),
        )

        self.mlp = MLPBlock(out_channels + add_channel, reduction=2)

        self.channel_att = spconv.SparseSequential(
            spconv.SubMConv3d(out_channels + add_channel, out_channels, kernel_size=1, bias=False),
            norm_fn(out_channels),
        )

        self.act = spconv.SparseSequential(nn.ReLU())

    def forward(self, input):
        identity = spconv.SparseConvTensor(input.features, input.indices, input.spatial_shape, input.batch_size)
        identity = self.ResidualShortcut(identity)
        x = self.pwconv(input)
        x = self.DilatedConv_spa(x)
        x = self.mlp(x)
        x = self.channel_att(x)

        x = x.replace_feature(x.features + identity.features)
        x = self.act(x)
        return x


class VelocityHeadSparse(nn.Module):
    def __init__(self, in_ch, v_max=2.5):
        super().__init__()
        hid = max(16, in_ch // 2)
        self.pred = nn.Sequential(
            nn.Linear(in_ch, hid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hid, 2, bias=True)
        )
        self.v_max = float(v_max)

    def forward(self, sp: 'spconv.SparseConvTensor'):
        v = self.pred(sp.features)           # [N,2]
        v = torch.tanh(v) * self.v_max       # 限幅，防发散
        return v


class SoftAdvectSparseConservative(nn.Module):
    def __init__(self, max_shift_xy=2.5, base_sigma=0.6):
        super().__init__()
        self.max_shift_xy = float(max_shift_xy)
        self.base_sigma = float(base_sigma)
        # Δt 由一致性先验与速度模长共同决定：可信走大步，存疑走小步
        self.dt_alpha = 0.6  # 一致性占比
        self.dt_beta = 0.4  # 速度抑制占比
        self.dt_min = 0.25  # 最小有效时间步，避免冻结

    @staticmethod
    def _hash4(b, x, y, t):
        return (((b & 0xFFFF) << 48) | ((x & 0xFFFF) << 32) | ((y & 0xFFFF) << 16) | (t & 0xFFFF)).long()

    def _gather_hits(self, key_src_sorted, perm, key_tgt):
        pos = torch.bucketize(key_tgt, key_src_sorted, right=False)
        valid = (pos > 0) & (pos <= key_src_sorted.numel())
        posc = (pos - 1).clamp(0, key_src_sorted.numel() - 1)
        hit = torch.zeros_like(valid)
        hit[valid] = (key_src_sorted[posc[valid]] == key_tgt[valid])
        dst = torch.full_like(posc, -1)
        dst[hit] = perm[posc[hit]]
        return hit, dst

    def _gauss4(self, x_tgt, y_tgt, x0, x1, y0, y1):
        dx0 = (x_tgt - x0).abs()
        dx1 = (x_tgt - x1).abs()
        dy0 = (y_tgt - y0).abs()
        dy1 = (y_tgt - y1).abs()
        w00 = torch.exp(-(dx0 ** 2 + dy0 ** 2))
        w10 = torch.exp(-(dx1 ** 2 + dy0 ** 2))
        w01 = torch.exp(-(dx0 ** 2 + dy1 ** 2))
        w11 = torch.exp(-(dx1 ** 2 + dy1 ** 2))
        return [(x0, y0, w00), (x1, y0, w10), (x0, y1, w01), (x1, y1, w11)]

    def _advect_once(self, coords, feats, vx, vy, gain):
        device = feats.device
        b = coords[:, 0]
        xi = coords[:, 1].float()
        yi = coords[:, 2].float()
        ti = coords[:, 3].float()

        vx = vx.clamp(-self.max_shift_xy, self.max_shift_xy)
        vy = vy.clamp(-self.max_shift_xy, self.max_shift_xy)
        x_tgt = xi + vx
        y_tgt = yi + vy
        t1 = (ti + 1.0).long()

        x0 = torch.floor(x_tgt)
        x1 = x0 + 1
        y0 = torch.floor(y_tgt)
        y1 = y0 + 1
        x0 = x0.long()
        x1 = x1.long()
        y0 = y0.long()
        y1 = y1.long()

        key_src = self._hash4(b, coords[:, 1], coords[:, 2], coords[:, 3])
        key_src_sorted, perm = torch.sort(key_src)

        neigh = self._gauss4(x_tgt, y_tgt, x0, x1, y0, y1)
        accum = torch.zeros_like(feats)
        weight_sum_dst = torch.zeros(feats.size(0), device=device)

        dst_lists, w_lists, src_mask_lists = [], [], []
        for nx, ny, w in neigh:
            key_tgt = self._hash4(b, nx, ny, t1)
            hit, dst = self._gather_hits(key_src_sorted, perm, key_tgt)
            if hit.any():
                dst_h = dst[hit]
                w_h = w[hit]
                dst_lists.append(dst_h)
                w_lists.append(w_h)
                src_mask_lists.append(hit.nonzero(as_tuple=False).squeeze(1))
                weight_sum_dst.index_add_(0, dst_h, w_h)

        weight_sum_dst = weight_sum_dst.clamp_min(1e-6)
        for dst_h, w_h, src_mask in zip(dst_lists, w_lists, src_mask_lists):
            norm_w = (w_h / weight_sum_dst[dst_h])
            accum.index_add_(0, dst_h, (norm_w.unsqueeze(1) * feats[src_mask]) * gain)

        return accum

    def forward(self, x: 'spconv.SparseConvTensor',
                vel_xy: torch.Tensor,
                gain: float = 1.0,
                return_prior: bool = False):

        coords = x.indices.long()
        feats = x.features

        vx = vel_xy[:, 0]
        vy = vel_xy[:, 1]

        accum = self._advect_once(coords, feats, vx, vy, gain)

        # accum = accum.cpu().numpy()

        # 速度模长，后面做温控和反混叠都要用
        speed = (vx.abs() + vy.abs())  # [N]

        # =============== 一致性能量 E 和温度 T ===============
        # 特征差异（相对误差）
        diff = (accum - feats).norm(p=1, dim=1) / (feats.norm(p=1, dim=1).clamp_min(1e-6))
        cons_mask = torch.exp(-diff).unsqueeze(1)

        # 反混叠：速度越大，抑制越强
        anti_alias = (1.0 / (1.0 + 0.25 * speed)).unsqueeze(1)

        # 守恒融合门
        gate_adv = cons_mask * anti_alias

        mixed = gate_adv * accum + (1.0 - gate_adv) * feats

        if return_prior:
            return (x.replace_feature(mixed),
                    cons_mask.squeeze(1).detach(),   # cons（高=一致）
                    cons_mask.squeeze(1).detach(),   # prior(暂沿用)
                    anti_alias.squeeze(1).detach(),  # anti-alias
                    gate_adv.squeeze(1).detach())    # gate
        else:
            return x.replace_feature(mixed)

class TrajectoryPriorHead(nn.Module):
    def __init__(self, max_shift_xy=2.5, base_sigma=0.6, gamma=0.8):
        super().__init__()
        self.adv = SoftAdvectSparseConservative(max_shift_xy=max_shift_xy, base_sigma=base_sigma)
        self.gamma = float(gamma)

    def forward(self, x: 'spconv.SparseConvTensor', vel_xy: torch.Tensor):
        # 1) 利用守恒平流得到“轨迹一致性”先验 cons \in [0,1]
        x_adv, cons, cons_mask, anti_alias, gate_adv  = self.adv(x, vel_xy, return_prior=True)  # x_adv 只用于计算，不直接写回

        feats = x.features                              # [N,C]

        # 2) 用通道能量近似 token 的“信息量”
        energy = (feats ** 2).sum(dim=1)                # [N]
        energy = energy / (energy.mean().detach() + 1e-6)
        energy = energy.clamp(0.0, 4.0) / 4.0           # 归一化到 [0,1]

        # 3) 前景得分：轨迹一致 + 能量足够
        traj_score = cons * energy                      # [N]

        # 4) 前景增强
        score = traj_score.unsqueeze(1)                 # [N,1]
        feats_center = feats.mean(dim=0, keepdim=True)  # 全局背景中心
        feats_out = feats + self.gamma * score * (feats - feats_center)

        x_out = x.replace_feature(feats_out)
        return x_out, traj_score.detach()

@torch.no_grad()
def build_c12l_map(idx_src: torch.Tensor, idx_lvl: torch.Tensor, stride: tuple):
    sx, sy, st = map(int, stride)
    b_s, x_s, y_s, t_s = idx_src.long().unbind(1)
    x_c = torch.div(x_s, sx, rounding_mode='floor')
    y_c = torch.div(y_s, sy, rounding_mode='floor')
    t_c = torch.div(t_s, st, rounding_mode='floor')

    def h4(b, x, y, t):
        return (((b & 0xFFFF) << 48) |
                ((x & 0xFFFF) << 32) |
                ((y & 0xFFFF) << 16) |
                (t & 0xFFFF)).long()

    key_src = h4(b_s, x_c, y_c, t_c)
    b_l, x_l, y_l, t_l = idx_lvl.long().unbind(1)
    key_lvl = h4(b_l, x_l, y_l, t_l)

    key_lvl_sorted, perm = torch.sort(key_lvl)
    pos = torch.searchsorted(key_lvl_sorted, key_src, right=False)
    posc = pos.clamp(max=key_lvl_sorted.numel() - 1)
    hit = key_lvl_sorted[posc] == key_src
    dst = torch.full_like(posc, -1)
    dst[hit] = perm[posc[hit]]
    return dst  # [N_src] -> [0..Nv-1] or -1


def aggregate_velocity_to_level(v1: torch.Tensor, map12l: torch.Tensor, Nv: int):
    device = v1.device
    out = torch.zeros(Nv, 2, device=device)
    cnt = torch.zeros(Nv, device=device)
    m = map12l >= 0
    if m.any():
        dst = map12l[m]
        out.index_add_(0, dst, v1[m])
        cnt.index_add_(0, dst, torch.ones_like(dst, dtype=cnt.dtype))
        cnt = cnt.clamp_min(1.0)
        out = out / cnt.unsqueeze(1)
    return out

def make_small_deltas(v: torch.Tensor, K: int = 5, radius: float = 0.6):
    if K == 1:
        return [torch.zeros_like(v)]
    base = [
        (0.0, 0.0),
        (1.0, 0.0),
        (-1.0, 0.0),
        (0.0, 1.0),
        (0.0, -1.0),
    ]
    base = base[:K]
    deltas = []
    for dx, dy in base:
        d = torch.zeros_like(v)
        d[:, 0] = dx * radius
        d[:, 1] = dy * radius
        deltas.append(d)
    return deltas


class VelocityAligner(nn.Module):
    def __init__(self, in_ch, use_mhv=True, K=5, delta_radius=0.6, prior_gain=0.75):
        super().__init__()
        self.use_mhv = use_mhv
        self.K = K
        self.delta_radius = delta_radius
        self.prior_gain = float(prior_gain)

        hid = max(16, in_ch // 2)
        self.score = nn.Sequential(
            nn.Linear(in_ch, hid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hid, 1, bias=True)
        )
        self.hint = nn.Sequential(
            nn.Linear(in_ch, hid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hid, in_ch, bias=True)
        )
        self.gate = nn.Sequential(
            nn.Linear(in_ch * 2, hid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hid, 1, bias=True),
            nn.Sigmoid()
        )

        self.soft_advect = SoftAdvectSparseConservative(max_shift_xy=2.5, base_sigma=0.6)
        self.pre_bn = nn.BatchNorm1d(in_ch, eps=1e-5, momentum=0.2)

    def forward(self, F_l_sp: 'spconv.SparseConvTensor', v_l: torch.Tensor):
        x = F_l_sp
        feats = x.features
        feats_n = self.pre_bn(feats)

        if not self.use_mhv:
            x_adv, prior, cons_mask, anti_alias, gate_adv = self.soft_advect(x.replace_feature(feats_n), v_l, return_prior=True)
            H = self.hint(x_adv.features)
            G = torch.cat([feats_n, H], dim=1)
            alpha = self.gate(G)
            out = feats + alpha * H
            return x.replace_feature(out)

        deltas = make_small_deltas(v_l, K=self.K, radius=self.delta_radius)
        A_stack, S_stack, P_stack = [], [], []

        for d in deltas:
            x_adv, prior, cons_mask, anti_alias, gate_adv = self.soft_advect(x.replace_feature(feats_n), v_l + d, return_prior=True)
            A_stack.append(x_adv.features.unsqueeze(0))              # [1,N,C]
            S_stack.append(self.score(x_adv.features).unsqueeze(0))  # [1,N,1]
            P_stack.append(prior.unsqueeze(0).unsqueeze(-1))         # [1,N,1]

        A_stack = torch.cat(A_stack, 0)  # [K,N,C]
        S_stack = torch.cat(S_stack, 0)  # [K,N,1]
        P_stack = torch.cat(P_stack, 0)  # [K,N,1]

        # 信息优先偏置：prior 越大，S 越小（更易被 softmax 选中）
        S_stack = S_stack - self.prior_gain * P_stack

        pi = torch.softmax(-S_stack.squeeze(-1), dim=0).unsqueeze(-1)  # [K,N,1]
        A_bar = (pi * A_stack).sum(0)  # [N,C]

        H = self.hint(A_bar)
        G = torch.cat([feats_n, H], dim=1)
        alpha = self.gate(G)
        out = feats + alpha * H
        return x.replace_feature(out)

def post_act_block(in_channels, out_channels, kernel_size, indice_key=None, stride=1, padding=0,
                   conv_type='subm', norm_fn=None, algo=None, dilations=[1, 2, 3, 4], ad_channels=16):
    if conv_type == 'subm':
        conv = spconv.SubMConv3d(in_channels, out_channels, kernel_size, bias=False, indice_key=indice_key, algo=algo)
    elif conv_type == 'spconv':
        conv = spconv.SparseConv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                                   bias=False, indice_key=indice_key, algo=algo)
    elif conv_type == 'inverseconv':
        conv = spconv.SparseInverseConv3d(in_channels, out_channels, kernel_size, indice_key=indice_key,
                                          bias=False, algo=algo)
    elif conv_type == 'SCA':
        conv = SCABlock(in_channels, out_channels, kernel_size, stride, norm_fn, dilations=dilations,
                        indice_key=indice_key, bias=False, ad_channels=ad_channels)
    else:
        raise NotImplementedError

    m = spconv.SparseSequential(
        conv,
        norm_fn(out_channels),
        nn.ReLU(),
    )
    return m


class BottleneckBlock(spconv.SparseModule):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, indice_key=None, norm_fn=None):
        super().__init__()
        self.conv1 = spconv.SubMConv3d(inplanes, planes, kernel_size=3, stride=stride, padding=1,
                                       bias=False, indice_key=indice_key)
        self.bn1 = norm_fn(planes)
        self.relu = nn.ReLU()
        self.conv2 = spconv.SubMConv3d(planes, planes, kernel_size=3, stride=1, padding=1,
                                       bias=False, indice_key=indice_key)
        self.bn2 = norm_fn(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x.features

        out = self.conv1(x)
        out = out.replace_feature(self.bn1(out.features))
        out = out.replace_feature(self.relu(out.features))

        out = self.conv2(out)
        out = out.replace_feature(self.bn2(out.features))

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out.replace_feature(out.features + identity)
        out = out.replace_feature(self.relu(out.features))
        return out

class PACT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        input_channels = cfg.input_channel
        width = cfg.width

        # ===== 可调超参 =====
        norm_fn = functools.partial(nn.BatchNorm1d, eps=1e-5, momentum=0.1)
        v1_vmax = 5.0  # level-1 速度限幅

        # 输入
        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(input_channels, width, 3, padding=1, bias=False, indice_key='subm1'),
            norm_fn(width),
            nn.ReLU(),
        )
        block = post_act_block

        self.SCA1 = spconv.SparseSequential(
            block(width, width, 3, norm_fn=norm_fn, padding=1, indice_key='m1', conv_type='SCA'),
        )
        self.SCA2 = spconv.SparseSequential(
            block(width, 2 * width, 3, norm_fn=norm_fn, stride=[2, 2, 4], padding=1,
                  indice_key='spc2', conv_type='spconv'),
            block(2 * width, 2 * width, 3, norm_fn=norm_fn, padding=1, indice_key='m2',
                  conv_type='SCA', ad_channels=16),
        )
        self.SCA3 = spconv.SparseSequential(
            block(2 * width, 4 * width, 3, norm_fn=norm_fn, stride=[2, 2, 4], padding=1,
                  indice_key='spc3', conv_type='spconv'),
            block(4 * width, 4 * width, 3, norm_fn=norm_fn, padding=1, indice_key='m3',
                  conv_type='SCA', ad_channels=8),
        )
        self.SCA4 = spconv.SparseSequential(
            block(4 * width, 4 * width, 3, norm_fn=norm_fn, stride=[2, 2, 4], padding=1,
                  indice_key='spc4', conv_type='spconv'),
            block(4 * width, 4 * width, 3, norm_fn=norm_fn, padding=1, indice_key='m4',
                  conv_type='SCA', ad_channels=0),
        )

        self.vel1 = VelocityHeadSparse(in_ch=width, v_max=v1_vmax)

        # 轨迹前景先验头：越深层 gamma 越小，避免在粗尺度过度拉扯
        self.traj_prior1 = TrajectoryPriorHead(
            max_shift_xy=v1_vmax,
            base_sigma=0.6,
            gamma=0.8
        )
        self.traj_prior2 = TrajectoryPriorHead(
            max_shift_xy=v1_vmax / 2.0,
            base_sigma=0.6,
            gamma=0.6
        )
        self.traj_prior3 = TrajectoryPriorHead(
            max_shift_xy=v1_vmax / 4.0,
            base_sigma=0.6,
            gamma=0.4
        )
        self.traj_prior4 = TrajectoryPriorHead(
            max_shift_xy=v1_vmax / 4.0,
            base_sigma=0.6,
            gamma=0.2
        )

        # 解码器
        self.mlp4 = BottleneckBlock(4 * width, 4 * width, indice_key='m4', norm_fn=norm_fn)
        self.channel_att4 = block(8 * width, 4 * width, 3, norm_fn=norm_fn,
                                  padding=1, indice_key='m4', conv_type='SCA')
        self.invconv4 = block(4 * width, 4 * width, 3, norm_fn=norm_fn,
                              indice_key='spc4', conv_type='inverseconv')

        self.mlp3 = BottleneckBlock(4 * width, 4 * width, indice_key='m3', norm_fn=norm_fn)
        self.channel_att3 = block(8 * width, 4 * width, 3, norm_fn=norm_fn,
                                  padding=1, indice_key='m3', conv_type='SCA')
        self.invconv3 = block(4 * width, 2 * width, 3, norm_fn=norm_fn,
                              indice_key='spc3', conv_type='inverseconv')

        self.mlp2 = BottleneckBlock(2 * width, 2 * width, indice_key='m2', norm_fn=norm_fn)
        self.channel_att2 = block(4 * width, 2 * width, 3, norm_fn=norm_fn,
                                  padding=1, indice_key='m2', conv_type='SCA')
        self.invconv2 = block(2 * width, width, 3, norm_fn=norm_fn,
                              indice_key='spc2', conv_type='inverseconv')

        self.mlp1 = BottleneckBlock(width, width, indice_key='m1', norm_fn=norm_fn)
        self.channel_att1 = block(2 * width, width, 3, norm_fn=norm_fn,
                                  indice_key='m1', conv_type='SCA')

        self.conv5 = spconv.SparseSequential(
            block(width, width, 3, norm_fn=norm_fn, padding=1, indice_key='m1', conv_type='SCA')
        )

        self.semantic_linear = nn.Sequential(
            nn.Linear(width, 1),
            nn.Sigmoid()
        )

        # 外置对齐器（每个解码阶段一个），多假设 + 一致性先验
        self.align4 = VelocityAligner(in_ch=4 * width, use_mhv=True, K=5, delta_radius=0.6)
        self.align3 = VelocityAligner(in_ch=2 * width, use_mhv=True, K=5, delta_radius=0.6)
        self.align2 = VelocityAligner(in_ch=1 * width, use_mhv=True, K=5, delta_radius=0.6)

    @staticmethod
    def attention(x, out_channels):
        features = x.features
        n, in_channels = features.shape
        x = x.replace_feature(features.view(n, out_channels, -1).sum(dim=2))
        return x

    def ACRBlock(self, x_lateral, x_bottom, mlp, channel_att, conv_inv):
        x_trans = mlp(x_lateral)
        x = x_trans
        x = x.replace_feature(torch.cat((x_bottom.features, x.features), dim=1))
        x_m = channel_att(x)
        x = self.attention(x, x_m.features.shape[1])
        x = x.replace_feature(x_m.features + x.features)
        x = conv_inv(x)
        return x

    def forward(self, input):
        # ===== 编码器 =====
        x = self.conv_input(input)
        x_1 = self.SCA1(x)           # level-1

        v1 = self.vel1(x_1)          # [N1, 2]
        idx_v1 = x_1.indices

        x_1, traj_score1 = self.traj_prior1(x_1, v1)

        # level-2
        x_2 = self.SCA2(x_1)         # level-2 特征
        # 将 v1 聚合到 level-2，保证世界线连续
        map1to2 = build_c12l_map(idx_v1, x_2.indices, stride=(2, 2, 4))
        v2 = aggregate_velocity_to_level(v1, map1to2, Nv=x_2.features.size(0))
        x_2, traj_score2 = self.traj_prior2(x_2, v2)

        # level-3
        x_3 = self.SCA3(x_2)         # level-3
        map1to3 = build_c12l_map(idx_v1, x_3.indices, stride=(4, 4, 16))
        v3 = aggregate_velocity_to_level(v1, map1to3, Nv=x_3.features.size(0))
        x_3, traj_score3 = self.traj_prior3(x_3, v3)

        # level-4（最粗）
        x_4 = self.SCA4(x_3)
        map1to4 = build_c12l_map(idx_v1, x_4.indices, stride=(8, 8, 64))
        v4 = aggregate_velocity_to_level(v1, map1to4, Nv=x_4.features.size(0))
        x_4, traj_score4 = self.traj_prior4(x_4, v4)

        x_up4 = self.ACRBlock(x_4, x_4, self.mlp4, self.channel_att4, self.invconv4)

        x_up4 = self.align4(x_up4, v3)   # 上采样到 level-3，用 v3 对齐
        x_up3 = self.ACRBlock(x_3, x_up4, self.mlp3, self.channel_att3, self.invconv3)

        x_up3 = self.align3(x_up3, v2)   # 上采样到 level-2，用 v2 对齐
        x_up2 = self.ACRBlock(x_2, x_up3, self.mlp2, self.channel_att2, self.invconv2)

        x_up2 = self.align2(x_up2, v1)   # 上采样到 level-1，用 v1 对齐
        x_up1 = self.ACRBlock(x_1, x_up2, self.mlp1, self.channel_att1, self.conv5)

        output = self.semantic_linear(x_up1.features)
        voxel = x_up1.replace_feature(output)

        return output, voxel, v1, idx_v1