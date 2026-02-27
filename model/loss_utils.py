# -*- coding: utf-8 -*-
from typing import Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

@torch.no_grad()
def build_p2v_map_for_level(ev_bxyt: torch.Tensor,
                            idx_lvl: torch.Tensor,
                            stride: tuple):
    device = idx_lvl.device
    sx, sy, st = map(int, stride)

    b_ev, x_ev, y_ev, t_ev = ev_bxyt.unbind(dim=1)
    b_lv, x_lv, y_lv, t_lv = idx_lvl.long().unbind(dim=1)

    # 降采样后构造 key
    x_c = torch.div(x_ev, sx, rounding_mode="floor")
    y_c = torch.div(y_ev, sy, rounding_mode="floor")
    t_c = torch.div(t_ev, st, rounding_mode="floor")

    def hash4(b, x, y, t):
        return (((b & 0xFFFF) << 48) |
                ((x & 0xFFFF) << 32) |
                ((y & 0xFFFF) << 16) |
                (t & 0xFFFF)).long()

    key_ev = hash4(b_ev, x_c, y_c, t_c)
    key_lv = hash4(b_lv, x_lv, y_lv, t_lv)

    key_lv_sorted, perm = torch.sort(key_lv)
    pos = torch.searchsorted(key_lv_sorted, key_ev, right=False)
    posc = pos.clamp(max=key_lv_sorted.numel() - 1)

    hit = key_lv_sorted[posc] == key_ev
    p2v_map = torch.full((ev_bxyt.size(0),), -1, dtype=torch.long, device=device)
    p2v_map[hit] = perm[posc[hit]]

    return p2v_map


@torch.no_grad()
def majority_labels_for_level(seg_ev: torch.Tensor,
                              p2v_map_level: torch.Tensor,
                              Nv: int,
                              num_classes: int = None):
    valid = (p2v_map_level >= 0)
    if not valid.any():
        return (torch.zeros(Nv, dtype=torch.long, device=seg_ev.device),
                torch.zeros(Nv, dtype=torch.long, device=seg_ev.device))
    v_id = p2v_map_level[valid]
    y_ev = seg_ev[valid].long()
    if num_classes is None:
        num_classes = int(y_ev.max().item()) + 1 if y_ev.numel() > 0 else 1
    counts = torch.zeros(Nv, num_classes, device=seg_ev.device, dtype=torch.int32)
    onehot = F.one_hot(y_ev.clamp_min(0), num_classes).to(counts.dtype)
    counts.index_add_(0, v_id, onehot)
    voxel_label = counts.argmax(dim=1)
    support = counts.gather(1, voxel_label.view(-1, 1)).squeeze(1)
    return voxel_label, support


@torch.no_grad()
def temporal_nearest_velocity_labels(idx_lvl: torch.Tensor,
                                     voxel_label: torch.Tensor,
                                     support: torch.Tensor,
                                     min_support: int = 2,
                                     dt_max: int = 256):
    device = idx_lvl.device
    Nv = idx_lvl.size(0)
    v_tar = torch.zeros(Nv, 2, device=device)
    valid = torch.zeros(Nv, dtype=torch.bool, device=device)
    dt_used = torch.zeros(Nv, device=device)

    keep = (support >= min_support) & (voxel_label >= 0)
    if keep.sum() < 2:
        return v_tar, valid, dt_used

    idx = idx_lvl[keep]
    label = voxel_label[keep]
    ids = keep.nonzero(as_tuple=False).squeeze(1)

    key = idx[:, 0] * 10000 + label
    time = idx[:, 3].float()
    order = torch.argsort(key * 1e6 + time)

    idx_sorted = idx[order]
    ids_sorted = ids[order]
    k = key[order]
    t = idx_sorted[:, 3]

    dt = torch.zeros_like(t)
    nxt = torch.full_like(t, -1, dtype=torch.long)
    for i in range(len(t) - 1):
        if k[i] == k[i + 1]:
            d = t[i + 1] - t[i]
            if 0 < d <= dt_max:
                dt[i] = d
                nxt[i] = i + 1

    m = nxt >= 0
    if not m.any():
        return v_tar, valid, dt_used

    src = ids_sorted[m]
    tgt = ids_sorted[nxt[m]]
    dx = (idx_lvl[tgt][:, 1].float() - idx_lvl[src][:, 1].float())
    dy = (idx_lvl[tgt][:, 2].float() - idx_lvl[src][:, 2].float())
    dt_f = dt[m].clamp_min(1.0)

    v_tar[src, 0] = dx / dt_f
    v_tar[src, 1] = dy / dt_f
    dt_used[src] = dt_f
    valid[src] = True
    return v_tar, valid, dt_used


def vel_loss_temporal_nearest(preds_vel: dict,
                              lvl_idx: dict,
                              batch: dict,
                              scale_weights=(1.0, 0.7, 0.5),
                              min_support=(5, 4, 3),
                              huber_delta=0.5,
                              log_dt_stats=True):
    """
    现在固定使用 batch['p2v_map_levels'][k] 做速度监督的投票/配对。
    min_support 可为 tuple/list (按 v2/v3/v4 顺序) 或 dict {'v2':..,'v3':..,'v4':..}
    """
    device = next(iter(preds_vel.values())).device
    seg_ev = batch['seg_label'].to(device).long()
    p2v_levels = batch['p2v_map_levels']     # <--- 关键：分层映射

    ks = ['v2', 'v3', 'v4']
    if isinstance(min_support, dict):
        ms = [min_support.get(k, 1) for k in ks]
    else:
        ms = list(min_support)

    total = 0.0
    logs = {}
    dt_all = []

    for i, (k, w) in enumerate(zip(ks, scale_weights)):
        if k not in preds_vel:
            continue
        v_pred = preds_vel[k]
        idx = lvl_idx[k].to(device).long()
        Nv = v_pred.size(0)

        vox_label, support = majority_labels_for_level(seg_ev, p2v_levels[k], Nv)
        v_star, valid_mask, dt_used = temporal_nearest_velocity_labels(idx, vox_label, support, min_support=ms[i])

        if valid_mask.any():
            diff = (v_pred[valid_mask] - v_star[valid_mask]).abs()
            delta = huber_delta
            quad = torch.minimum(diff, torch.tensor(delta, device=device))
            lin = diff - quad
            loss_k = (0.5 * quad**2 + delta * lin).sum(dim=1).mean()
            dt_all.append(dt_used[valid_mask].detach())
        else:
            loss_k = torch.tensor(0.0, device=device)

        total = total + w * loss_k
        logs[f'loss_{k}'] = float(loss_k.detach().cpu())

    if log_dt_stats and len(dt_all) > 0:
        dt_all = torch.cat(dt_all)
        logs.update(dict(
            dt_mean=float(dt_all.mean().cpu()),
            dt_median=float(dt_all.median().cpu()),
            dt_max=float(dt_all.max().cpu()),
            num_pairs=int(dt_all.numel())
        ))
        print(f"[Δt统计] mean={logs['dt_mean']:.3f}, median={logs['dt_median']:.3f}, max={logs['dt_max']:.3f}, count={logs['num_pairs']}", flush=True)

    return total, logs


def background_zero_velocity_loss(preds_vel: dict,
                                  lvl_idx: dict,
                                  batch: dict,
                                  p2v_map: dict,
                                  min_support_bg=(5, 4, 3),
                                  bg_label: int = 0):
    """
    背景零速度约束：依旧使用分层映射 p2v_map[k] （这里就是 p2v_map_levels）
    """
    device = next(iter(preds_vel.values())).device
    seg_ev = batch['seg_label'].to(device).long()
    ks = ['v2', 'v3', 'v4']
    if isinstance(min_support_bg, dict):
        ms = [min_support_bg.get(k, 1) for k in ks]
    else:
        ms = list(min_support_bg)

    loss = 0.0
    for i, k in enumerate(ks):
        if k not in preds_vel:
            continue
        Nv = preds_vel[k].size(0)
        vox_label, support = majority_labels_for_level(seg_ev, p2v_map[k], Nv)
        bg = (vox_label == bg_label) & (support >= ms[i])
        if bg.any():
            v = preds_vel[k][bg]
            loss = loss + (v.pow(2).sum(dim=1).mean())
    return loss


@torch.no_grad()
def build_p2v_map_v1(ev_bxyt: torch.Tensor, idx_v1: torch.Tensor):
    # stride=(1,1,1) 的特例包装，便于缓存/复用
    return build_p2v_map_for_level(ev_bxyt, idx_v1, stride=(1,1,1))

def vel_loss_v1_only(v1: torch.Tensor,
                     idx_v1: torch.Tensor,
                     batch: dict,
                     p2v_map_v1: torch.Tensor = None,
                     bg_label: int = 0,
                     ignore_index: int = -100,
                     min_support: int = 5,
                     huber_delta: float = 0.5,
                     return_logs: bool = True):
    """
    只监督 v1（高分辨率）的速度场。
    掩码规则：
      - seg == ignore_index 的事件剔除
      - seg == bg_label 的事件不参与（前景监督）
      - voxel 支持数 < min_support 的体素不参与
    """
    device = v1.device

    # 事件级标签与坐标（必须一一对应：长度 Ne）
    seg_ev_full  = batch['seg_label'].to(device).long()      # [Ne]
    ev_bxyt_full = batch['ev_bxyt'].to(device).long()        # [Ne,4]

    # —— 前景/有效事件掩码
    valid_ev = torch.ones_like(seg_ev_full, dtype=torch.bool, device=device)
    if ignore_index is not None:
        valid_ev &= (seg_ev_full != ignore_index)
    if bg_label is not None:
        valid_ev &= (seg_ev_full != bg_label)

    seg_ev  = seg_ev_full[valid_ev]           # [Ne_fg]
    ev_bxyt = ev_bxyt_full[valid_ev, :]       # [Ne_fg,4]

    # —— p2v 映射：若外部传入，就同步用同一掩码过滤；否则内部重建（基于过滤后的事件）
    if p2v_map_v1 is None:
        p2v_map_v1 = build_p2v_map_for_level(ev_bxyt, idx_v1.long(), stride=(1, 1, 1))  # [Ne_fg]
    else:
        p2v_map_v1 = p2v_map_v1.to(device).long()
        p2v_map_v1 = p2v_map_v1[valid_ev]  # 关键：用同一掩码对齐到前景事件

    Nv1 = idx_v1.size(0)
    if seg_ev.numel() == 0:
        return (v1.new_zeros(()), dict(loss_v1=0.0, used_pairs=0)) if return_logs else v1.new_zeros(())

    # 多数投票：事件域 -> 体素域
    vox_label, support = majority_labels_for_level(seg_ev, p2v_map_v1, Nv1, num_classes=None)

    # 时间最近邻监督速度
    v_star, valid_mask, dt_used = temporal_nearest_velocity_labels(
        idx_v1.long(), vox_label, support, min_support=min_support
    )

    if not valid_mask.any():
        return (v1.new_zeros(()), dict(loss_v1=0.0, used_pairs=0)) if return_logs else v1.new_zeros(())

    # Huber (L1 平滑)
    diff  = (v1[valid_mask] - v_star[valid_mask]).abs()
    delta = v1.new_tensor(huber_delta)
    quad  = torch.minimum(diff, delta)
    lin   = diff - quad
    loss  = (0.5 * quad**2 + delta * lin).sum(dim=1).mean()

    if return_logs:
        logs = dict(
            loss_v1=float(loss.detach().cpu()),
            used_pairs=int(valid_mask.sum().item()),
            dt_mean=float(dt_used[valid_mask].mean().detach().cpu()) if dt_used[valid_mask].numel() else 0.0,
        )
        return loss, logs
    else:
        return loss


# @torch.no_grad()
# def build_p2v_map_for_level(ev_bxyt: torch.Tensor,
#                             idx_lvl: torch.Tensor,
#                             stride: tuple):
#     device = idx_lvl.device
#     sx, sy, st = map(int, stride)
#
#     b_ev, x_ev, y_ev, t_ev = ev_bxyt.unbind(dim=1)
#     b_lv, x_lv, y_lv, t_lv = idx_lvl.long().unbind(dim=1)
#
#     # 降采样后构造 key
#     x_c = torch.div(x_ev, sx, rounding_mode="floor")
#     y_c = torch.div(y_ev, sy, rounding_mode="floor")
#     t_c = torch.div(t_ev, st, rounding_mode="floor")
#
#     def hash4(b, x, y, t):
#         return (((b & 0xFFFF) << 48) |
#                 ((x & 0xFFFF) << 32) |
#                 ((y & 0xFFFF) << 16) |
#                 (t & 0xFFFF)).long()
#
#     key_ev = hash4(b_ev, x_c, y_c, t_c)
#     key_lv = hash4(b_lv, x_lv, y_lv, t_lv)
#
#     key_lv_sorted, perm = torch.sort(key_lv)
#     pos = torch.searchsorted(key_lv_sorted, key_ev, right=False)
#     posc = pos.clamp(max=key_lv_sorted.numel() - 1)
#
#     hit = key_lv_sorted[posc] == key_ev
#     p2v_map = torch.full((ev_bxyt.size(0),), -1, dtype=torch.long, device=device)
#     p2v_map[hit] = perm[posc[hit]]
#
#     return p2v_map
#
#
# @torch.no_grad()
# def majority_labels_for_level(seg_ev: torch.Tensor,
#                               p2v_map_level: torch.Tensor,
#                               Nv: int,
#                               num_classes: int = None):
#     valid = (p2v_map_level >= 0)
#     if not valid.any():
#         return (torch.zeros(Nv, dtype=torch.long, device=seg_ev.device),
#                 torch.zeros(Nv, dtype=torch.long, device=seg_ev.device))
#     v_id = p2v_map_level[valid]
#     y_ev = seg_ev[valid].long()
#     if num_classes is None:
#         num_classes = int(y_ev.max().item()) + 1 if y_ev.numel() > 0 else 1
#     counts = torch.zeros(Nv, num_classes, device=seg_ev.device, dtype=torch.int32)
#     onehot = F.one_hot(y_ev.clamp_min(0), num_classes).to(counts.dtype)
#     counts.index_add_(0, v_id, onehot)
#     voxel_label = counts.argmax(dim=1)
#     support = counts.gather(1, voxel_label.view(-1, 1)).squeeze(1)
#     return voxel_label, support
#
#
# @torch.no_grad()
# def temporal_nearest_velocity_labels(idx_lvl: torch.Tensor,
#                                      voxel_label: torch.Tensor,
#                                      support: torch.Tensor,
#                                      min_support: int = 2,
#                                      dt_max: int = 256):
#     device = idx_lvl.device
#     Nv = idx_lvl.size(0)
#     v_tar = torch.zeros(Nv, 2, device=device)
#     valid = torch.zeros(Nv, dtype=torch.bool, device=device)
#     dt_used = torch.zeros(Nv, device=device)
#
#     keep = (support >= min_support) & (voxel_label >= 0)
#     if keep.sum() < 2:
#         return v_tar, valid, dt_used
#
#     idx = idx_lvl[keep]
#     label = voxel_label[keep]
#     ids = keep.nonzero(as_tuple=False).squeeze(1)
#
#     key = idx[:, 0] * 10000 + label
#     time = idx[:, 3].float()
#     order = torch.argsort(key * 1e6 + time)
#
#     idx_sorted = idx[order]
#     ids_sorted = ids[order]
#     k = key[order]
#     t = idx_sorted[:, 3]
#
#     dt = torch.zeros_like(t)
#     nxt = torch.full_like(t, -1, dtype=torch.long)
#     for i in range(len(t) - 1):
#         if k[i] == k[i + 1]:
#             d = t[i + 1] - t[i]
#             if 0 < d <= dt_max:
#                 dt[i] = d
#                 nxt[i] = i + 1
#
#     m = nxt >= 0
#     if not m.any():
#         return v_tar, valid, dt_used
#
#     src = ids_sorted[m]
#     tgt = ids_sorted[nxt[m]]
#     dx = (idx_lvl[tgt][:, 1].float() - idx_lvl[src][:, 1].float())
#     dy = (idx_lvl[tgt][:, 2].float() - idx_lvl[src][:, 2].float())
#     dt_f = dt[m].clamp_min(1.0)
#
#     v_tar[src, 0] = dx / dt_f
#     v_tar[src, 1] = dy / dt_f
#     dt_used[src] = dt_f
#     valid[src] = True
#     return v_tar, valid, dt_used
#
#
# def vel_loss_temporal_nearest(preds_vel: dict,
#                               lvl_idx: dict,
#                               batch: dict,
#                               scale_weights=(1.0, 0.7, 0.5),
#                               min_support=(5, 4, 3),
#                               huber_delta=0.5,
#                               log_dt_stats=True):
#     """
#     现在固定使用 batch['p2v_map_levels'][k] 做速度监督的投票/配对。
#     min_support 可为 tuple/list (按 v2/v3/v4 顺序) 或 dict {'v2':..,'v3':..,'v4':..}
#     """
#     device = next(iter(preds_vel.values())).device
#     seg_ev = batch['seg_label'].to(device).long()
#     p2v_levels = batch['p2v_map_levels']     # <--- 关键：分层映射
#
#     ks = ['v2', 'v3', 'v4']
#     if isinstance(min_support, dict):
#         ms = [min_support.get(k, 1) for k in ks]
#     else:
#         ms = list(min_support)
#
#     total = 0.0
#     logs = {}
#     dt_all = []
#
#     for i, (k, w) in enumerate(zip(ks, scale_weights)):
#         if k not in preds_vel:
#             continue
#         v_pred = preds_vel[k]
#         idx = lvl_idx[k].to(device).long()
#         Nv = v_pred.size(0)
#
#         vox_label, support = majority_labels_for_level(seg_ev, p2v_levels[k], Nv)
#         v_star, valid_mask, dt_used = temporal_nearest_velocity_labels(idx, vox_label, support, min_support=ms[i])
#
#         if valid_mask.any():
#             diff = (v_pred[valid_mask] - v_star[valid_mask]).abs()
#             delta = huber_delta
#             quad = torch.minimum(diff, torch.tensor(delta, device=device))
#             lin = diff - quad
#             loss_k = (0.5 * quad**2 + delta * lin).sum(dim=1).mean()
#             dt_all.append(dt_used[valid_mask].detach())
#         else:
#             loss_k = torch.tensor(0.0, device=device)
#
#         total = total + w * loss_k
#         logs[f'loss_{k}'] = float(loss_k.detach().cpu())
#
#     if log_dt_stats and len(dt_all) > 0:
#         dt_all = torch.cat(dt_all)
#         logs.update(dict(
#             dt_mean=float(dt_all.mean().cpu()),
#             dt_median=float(dt_all.median().cpu()),
#             dt_max=float(dt_all.max().cpu()),
#             num_pairs=int(dt_all.numel())
#         ))
#         print(f"[Δt统计] mean={logs['dt_mean']:.3f}, median={logs['dt_median']:.3f}, max={logs['dt_max']:.3f}, count={logs['num_pairs']}", flush=True)
#
#     return total, logs
#
#
# def background_zero_velocity_loss(preds_vel: dict,
#                                   lvl_idx: dict,
#                                   batch: dict,
#                                   p2v_map: dict,
#                                   min_support_bg=(5, 4, 3),
#                                   bg_label: int = 0):
#     """
#     背景零速度约束：依旧使用分层映射 p2v_map[k] （这里就是 p2v_map_levels）
#     """
#     device = next(iter(preds_vel.values())).device
#     seg_ev = batch['seg_label'].to(device).long()
#     ks = ['v2', 'v3', 'v4']
#     if isinstance(min_support_bg, dict):
#         ms = [min_support_bg.get(k, 1) for k in ks]
#     else:
#         ms = list(min_support_bg)
#
#     loss = 0.0
#     for i, k in enumerate(ks):
#         if k not in preds_vel:
#             continue
#         Nv = preds_vel[k].size(0)
#         vox_label, support = majority_labels_for_level(seg_ev, p2v_map[k], Nv)
#         bg = (vox_label == bg_label) & (support >= ms[i])
#         if bg.any():
#             v = preds_vel[k][bg]
#             loss = loss + (v.pow(2).sum(dim=1).mean())
#     return loss
#
#
# @torch.no_grad()
# def build_p2v_map_v1(ev_bxyt: torch.Tensor, idx_v1: torch.Tensor):
#     # stride=(1,1,1) 的特例包装，便于缓存/复用
#     return build_p2v_map_for_level(ev_bxyt, idx_v1, stride=(1,1,1))
#
# def vel_loss_v1_only(v1: torch.Tensor,
#                      idx_v1: torch.Tensor,
#                      batch: dict,
#                      p2v_map_v1: torch.Tensor = None,
#                      bg_label: int = 0,
#                      ignore_index: int = -100,
#                      min_support: int = 5,
#                      huber_delta: float = 0.5,
#                      return_logs: bool = True):
#     """
#     只监督 v1（高分辨率）的速度场。
#     掩码规则：
#       - seg == ignore_index 的事件剔除
#       - seg == bg_label 的事件不参与（前景监督）
#       - voxel 支持数 < min_support 的体素不参与
#     """
#     device = v1.device
#
#     # 事件级标签与坐标（必须一一对应：长度 Ne）
#     seg_ev_full  = batch['seg_label'].to(device).long()      # [Ne]
#     ev_bxyt_full = batch['ev_bxyt'].to(device).long()        # [Ne,4]
#
#     # —— 前景/有效事件掩码
#     valid_ev = torch.ones_like(seg_ev_full, dtype=torch.bool, device=device)
#     if ignore_index is not None:
#         valid_ev &= (seg_ev_full != ignore_index)
#     if bg_label is not None:
#         valid_ev &= (seg_ev_full != bg_label)
#
#     seg_ev  = seg_ev_full[valid_ev]           # [Ne_fg]
#     ev_bxyt = ev_bxyt_full[valid_ev, :]       # [Ne_fg,4]
#
#     # —— p2v 映射：若外部传入，就同步用同一掩码过滤；否则内部重建（基于过滤后的事件）
#     if p2v_map_v1 is None:
#         p2v_map_v1 = build_p2v_map_for_level(ev_bxyt, idx_v1.long(), stride=(1, 1, 1))  # [Ne_fg]
#     else:
#         p2v_map_v1 = p2v_map_v1.to(device).long()
#         p2v_map_v1 = p2v_map_v1[valid_ev]  # 关键：用同一掩码对齐到前景事件
#
#     Nv1 = idx_v1.size(0)
#     if seg_ev.numel() == 0:
#         return (v1.new_zeros(()), dict(loss_v1=0.0, used_pairs=0)) if return_logs else v1.new_zeros(())
#
#     # 多数投票：事件域 -> 体素域
#     vox_label, support = majority_labels_for_level(seg_ev, p2v_map_v1, Nv1, num_classes=None)
#
#     # 时间最近邻监督速度
#     v_star, valid_mask, dt_used = temporal_nearest_velocity_labels(
#         idx_v1.long(), vox_label, support, min_support=min_support
#     )
#
#     if not valid_mask.any():
#         return (v1.new_zeros(()), dict(loss_v1=0.0, used_pairs=0)) if return_logs else v1.new_zeros(())
#
#     # Huber (L1 平滑)
#     diff  = (v1[valid_mask] - v_star[valid_mask]).abs()
#     delta = v1.new_tensor(huber_delta)
#     quad  = torch.minimum(diff, delta)
#     lin   = diff - quad
#     loss  = (0.5 * quad**2 + delta * lin).sum(dim=1).mean()
#
#     if return_logs:
#         logs = dict(
#             loss_v1=float(loss.detach().cpu()),
#             used_pairs=int(valid_mask.sum().item()),
#             dt_mean=float(dt_used[valid_mask].mean().detach().cpu()) if dt_used[valid_mask].numel() else 0.0,
#         )
#         return loss, logs
#     else:
#         return loss
#

# ---------------------------
# 统一网格: 从 spatial_shape 取 W,H,T
# indices 约定: [b, x, y, t]
# spatial_shape 对应: [W, H, T]
# ---------------------------
def _get_WH_T_from_spatial_shape(sp):
    W = int(sp.spatial_shape[0])
    H = int(sp.spatial_shape[1])
    T = int(sp.spatial_shape[2])
    assert W > 0 and H > 0 and T > 0, f"Invalid spatial_shape={sp.spatial_shape}"
    return W, H, T

# ---------------------------
# 稳定的 64-bit key 打包 (仅 x,y,t；b 用外部分片)
# ---------------------------
@torch.no_grad()
def pack_key_xyzt(coords: torch.Tensor, strides=None):
    x = coords[:, 1].to(torch.int64)
    y = coords[:, 2].to(torch.int64)
    t = coords[:, 3].to(torch.int64)
    if strides is None:
        Sx = int(x.max().item()) + 1
        Sy = int(y.max().item()) + 1
        St = int(t.max().item()) + 1
    else:
        Sx, Sy, St = strides
    key = (((x * Sy) + y) * St) + t
    return key, (Sx, Sy, St)

# ---------------------------
# 网格散射 (固定 H,W) & 差分
# ---------------------------
@torch.no_grad()
def scatter_count_grid_fixed(sp_t, H: int, W: int):
    coords = sp_t.indices.to(torch.long)
    dev = sp_t.features.device
    x = coords[:, 1].clamp(0, W - 1)
    y = coords[:, 2].clamp(0, H - 1)
    mu = torch.zeros((1, 1, H, W), device=dev, dtype=sp_t.features.dtype)
    mu[0, 0, y, x] += 1.0
    return mu

@torch.no_grad()
def scatter_mean_vector_to_grid_fixed(sp_t, vec_xy, H: int, W: int):
    coords = sp_t.indices.to(torch.long)
    dev = sp_t.features.device
    x = coords[:, 1].clamp(0, W - 1)
    y = coords[:, 2].clamp(0, H - 1)
    vg = torch.zeros((1, 2, H, W), device=dev, dtype=sp_t.features.dtype)
    cnt= torch.zeros((1, 1, H, W), device=dev, dtype=sp_t.features.dtype)
    vg[0, 0, y, x] += vec_xy[:, 0]
    vg[0, 1, y, x] += vec_xy[:, 1]
    cnt[0, 0, y, x] += 1.0
    vg = vg / (cnt + 1e-6)
    return vg, cnt

class _GridDiffOps2D(torch.nn.Module):
    """div(mx,my) with simple central-diff; 边界退化为一阶差分"""
    def __init__(self):
        super().__init__()
        kx = torch.tensor([[0,0,0],[-1,0,1],[0,0,0]], dtype=torch.float32).view(1,1,3,3)*0.5
        ky = torch.tensor([[0,-1,0],[0, 0,0],[0, 1,0]], dtype=torch.float32).view(1,1,3,3)*0.5
        self.register_buffer("kx", kx)
        self.register_buffer("ky", ky)
    def div(self, mx, my):
        dx = F.conv2d(mx, self.kx, padding=1)
        dy = F.conv2d(my, self.ky, padding=1)
        return dx + dy

# ---------------------------
# 局部整数位移估计 (±radius 邻域; Δt=1)
# ---------------------------
@torch.no_grad()
def estimate_local_integer_flow_xy(sp_t, sp_t1, radius_xy: int = 1):
    cf = sp_t.indices.to(torch.int64)    # [N0,4]
    ct = sp_t1.indices.to(torch.int64)   # [N1,4]
    dev = sp_t.features.device
    N0  = cf.shape[0]

    # 仅在同一 b 内调用，本函数不跨 batch
    # 为 t+1 切片建 key (x,y,t)
    key_t, strides = pack_key_xyzt(ct)
    sort_key, sort_idx = torch.sort(key_t)

    # 生成从小到大的曼哈顿半径偏移，不含(0,0)
    offs = []
    for d in range(1, radius_xy + 1):
        for dx in range(-d, d + 1):
            dy1 =  d - abs(dx)
            dy2 = -d + abs(dx)
            offs.append((dx, dy1))
            if dy1 != dy2:
                offs.append((dx, dy2))
    # 把(0,0)放最前（优先生命中）
    offs = [(0,0)] + offs

    v = torch.zeros((N0, 2), device=dev, dtype=sp_t.features.dtype)
    taken = torch.zeros(N0, dtype=torch.bool, device=dev)

    for dx, dy in offs:
        q = cf.clone()
        q[:, 1] = q[:, 1] + int(dx)
        q[:, 2] = q[:, 2] + int(dy)
        q[:, 3] = q[:, 3] + 1
        key_q, _ = pack_key_xyzt(q, strides)
        pos = torch.searchsorted(sort_key, key_q)
        posc = (pos - 1).clamp(0, sort_key.numel() - 1)
        hit = (pos > 0) & (pos <= sort_key.numel()) & (sort_key[posc] == key_q)
        new = hit & (~taken)
        if new.any():
            v[new, 0] = float(dx)
            v[new, 1] = float(dy)
            taken[new] = True
        if taken.all():
            break
    return v

# ---------------------------
# μ 的双线性搬运
# ---------------------------
def _advect_grid_bilinear(mu_t: torch.Tensor, v_grid: torch.Tensor):
    B, _, H, W = mu_t.shape
    yy, xx = torch.meshgrid(
        torch.arange(H, device=mu_t.device, dtype=mu_t.dtype),
        torch.arange(W, device=mu_t.device, dtype=mu_t.dtype),
        indexing='ij'
    )
    gx = (xx + v_grid[:, 0]) / max(1.0, W - 1) * 2 - 1
    gy = (yy + v_grid[:, 1]) / max(1.0, H - 1) * 2 - 1
    grid = torch.stack([gx, gy], dim=-1)  # [1,H,W,2]
    mu_adv = F.grid_sample(mu_t, grid, mode='bilinear', padding_mode='zeros', align_corners=True)
    return mu_adv


@torch.no_grad()
def _list_valid_pairs_in_batch(sp, dt=1):
    c = sp.indices
    t_vals = torch.unique(c[:, 3]).tolist()
    t_set  = set(t_vals)
    pairs = [(t, t + dt) for t in t_vals if (t + dt) in t_set]
    return pairs

def _build_slice(sp, mask):
    return sp.__class__(sp.features[mask], sp.indices[mask], sp.spatial_shape, sp.batch_size)

def compute_temporal_losses_on_sparse(
    sp,                      # SparseConvTensor（包含多个 b,t）
    diff_ops,                # _GridDiffOps2D 实例
    dt=1,
    radius_xy=1,
    max_pairs_per_batch=2,   # 控制计算量
    use_integer_flow=True,   # True: 用局部整数位移；False: 改为自定义 v_provider
    v_provider=None
):
    device = sp.features.device
    dtype  = sp.features.dtype
    W, H, T = _get_WH_T_from_spatial_shape(sp)

    coords = sp.indices
    batches = torch.unique(coords[:, 0]).tolist()

    Lc_all, Lw_all, Lk_all = [], [], []
    for b in batches:
        mask_b = (coords[:, 0] == b)
        sp_b = sp.__class__(sp.features[mask_b], coords[mask_b], sp.spatial_shape, sp.batch_size)

        pairs = _list_valid_pairs_in_batch(sp_b, dt=dt)
        if not pairs:
            Lc_all.append(torch.tensor(0., device=device, dtype=dtype))
            Lw_all.append(torch.tensor(0., device=device, dtype=dtype))
            Lk_all.append(torch.tensor(0., device=device, dtype=dtype))
            continue
        pairs = pairs[:max_pairs_per_batch]

        Lc_b, Lw_b, Lk_b = [], [], []
        cb = sp_b.indices
        for (t0, t1) in pairs:
            m0 = (cb[:, 3] == t0)
            m1 = (cb[:, 3] == t1)
            if (m0.sum() == 0) or (m1.sum() == 0):
                continue
            sp_t  = _build_slice(sp_b, m0)
            sp_t1 = _build_slice(sp_b, m1)

            if use_integer_flow:
                v_xy = estimate_local_integer_flow_xy(sp_t, sp_t1, radius_xy=radius_xy)  # [N0,2]
            else:
                assert v_provider is not None, "set v_provider when use_integer_flow=False"
                v_xy = v_provider(sp_t, sp_t1)  # 单位=像素/Δt

            # 固定 H×W 网格
            mu_t  = scatter_count_grid_fixed(sp_t,  H, W)
            mu_t1 = scatter_count_grid_fixed(sp_t1, H, W)
            v_grid, _ = scatter_mean_vector_to_grid_fixed(sp_t, v_xy, H, W)

            # 连续性残差
            mx = mu_t * v_grid[:, :1]
            my = mu_t * v_grid[:, 1:]
            R  = mu_t1 - mu_t + diff_ops.div(mx, my)
            Lc = (R ** 2).mean()

            # Warp 一致
            mu_adv = _advect_grid_bilinear(mu_t, v_grid)
            Lw = (mu_adv - mu_t1).pow(2).mean()

            # 动能正则（点速度）
            Lk = v_xy.pow(2).sum(dim=1, keepdim=True).mean()

            Lc_b.append(Lc); Lw_b.append(Lw); Lk_b.append(Lk)

        if len(Lc_b) == 0:
            Lc_all.append(torch.tensor(0., device=device, dtype=dtype))
            Lw_all.append(torch.tensor(0., device=device, dtype=dtype))
            Lk_all.append(torch.tensor(0., device=device, dtype=dtype))
        else:
            Lc_all.append(torch.stack(Lc_b).mean())
            Lw_all.append(torch.stack(Lw_b).mean())
            Lk_all.append(torch.stack(Lk_b).mean())

    L_cont = torch.stack(Lc_all).mean()
    L_warp = torch.stack(Lw_all).mean()
    L_kin  = torch.stack(Lk_all).mean()
    return L_cont, L_warp, L_kin
