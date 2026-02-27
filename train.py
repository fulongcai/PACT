# -*- coding: utf-8 -*-
import os
import random
import numpy as np
import torch
import torch.optim as optim
from tqdm.auto import tqdm
import mlflow

from configs.configs import cfg
from dataset.ev_uav import EvUAV
from model.PACT import PACT
from utils.stcloss import STCLoss
from utils.eval import evalute
from model.loss_utils import vel_loss_v1_only, build_p2v_map_for_level

def setup(seed: int):
    seed_n = seed
    print('random seed:' + str(seed_n))
    g = torch.Generator()
    g.manual_seed(seed_n)
    random.seed(seed_n)
    np.random.seed(seed_n)
    torch.manual_seed(seed_n)
    torch.cuda.manual_seed(seed_n)
    torch.cuda.manual_seed_all(seed_n)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    torch.use_deterministic_algorithms(True)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
    os.environ['PYTHONHASHSEED'] = str(seed_n)


if __name__ == '__main__':
    # --------- reproducibility ----------
    seed = 37
    setup(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --------- model ----------
    net = PACT(cfg).train().to(device)

    # --------- data ----------
    dataset = EvUAV(cfg, mode='train')
    train_sampler = torch.utils.data.sampler.RandomSampler(list(range(len(dataset))))
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=cfg.batch_size, collate_fn=dataset.custom_collate, sampler=train_sampler)

    val_dataset = EvUAV(cfg, mode='val')
    val_loader  = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        collate_fn=val_dataset.custom_collate,
        num_workers=0,
        pin_memory=True
    )

    # --------- losses ----------
    stc_criterion = STCLoss(k=cfg.k, t=cfg.t, cfg=cfg).to(device)

    # --------- optim ----------
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=cfg.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    best_loss = float("inf")
    best_iou  = 0.0

    # --------- evaluator ----------
    evaluter = evalute(cfg)

    # --------- mlflow ----------
    mlflow.set_experiment('train')
    mlflow.start_run(run_name='train')

    for epoch in range(cfg.epochs):
        net.train()
        pbar = tqdm(total=len(train_loader), unit="Batch", unit_scale=True,
                    desc=f"Epoch: {epoch}", position=0, leave=True)

        for batch_idx, batch in enumerate(train_loader):
            # ---- inputs ----
            x = batch['voxel_ev']
            label = batch['seg_label_ev'].float().to(device)

            p2v_map_ev = batch['p2v_map'].long().to(device)

            # spconv 的 SparseConvTensor 支持 .to
            try:
                x = x.to(device)
            except AttributeError:
                pass

            # ---- forward ----
            ev_bxyt = torch.as_tensor(batch['locs'], device=device, dtype=torch.long)  # [Ne,4]
            seg_label_ev = torch.as_tensor(batch['seg_label_ev'], device=device, dtype=torch.long)  # [Ne]
            p2v_map_ev = batch['p2v_map'].to(device).long()  # [Ne] 事件→预测体素

            preds, voxel, v1, idx_v1 = net(x)
            # 事件坐标与标签（dataset给出的原始分辨率体素坐标）
            assert set(ev_bxyt[:, 0].unique().tolist()) == set(idx_v1[:, 0].unique().tolist())
            p2v_v1 = build_p2v_map_for_level(ev_bxyt, idx_v1.long(), stride=(1, 1, 1))


            # 主任务：STC 权重交叉熵 —— 这里仍然用**张量** p2v_map_ev
            loss_task = stc_criterion(voxel, p2v_map_ev, preds, seg_label_ev.float())

            # v1 速度监督（事件级 → v1 体素）
            loss_v1, logs_v1 = vel_loss_v1_only(
                v1=v1,
                idx_v1=idx_v1,
                batch={'ev_bxyt': ev_bxyt, 'seg_label': seg_label_ev},
                p2v_map_v1=p2v_v1,
                huber_delta=0.5
            )

            loss = loss_task + 0.3 * loss_v1

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            # ---- logs ----
            pbar.set_postfix(loss=float(loss), task=float(loss_task))
            pbar.update(1)

            with torch.no_grad():
                # ================== 日志部分保持不变 ================== #
                mlflow.log_metric('loss', float(loss))
                mlflow.log_metric('loss_task', float(loss_task))

                if float(loss) < best_loss:
                    best_path = os.path.join(cfg.model_save_root, f'best_loss_seed{seed}.pt')
                    os.makedirs(cfg.model_save_root, exist_ok=True)
                    torch.save(net.state_dict(), best_path)
                    best_loss = float(loss)


            # 释放显存碎片
            del preds, voxel

        pbar.close()
        scheduler.step()

        # ---------------- 验证（保持你原有逻辑） ----------------
        net.eval()
        try:
            evaluter.reset()
        except Exception:
            evaluter.matches = {}

        with torch.no_grad():
            for sample_idx, batch in enumerate(val_loader):
                x_val = batch['voxel_ev']
                label_val = batch['seg_label_ev'].float().to(device)
                p2v_map_val = batch['p2v_map'].long().to(device)  # 验证集仍是事件->体素**张量**

                try:
                    x_val = x_val.to(device)
                except AttributeError:
                    pass

                preds_val, voxel_val, v1, idx_v1 = net(x_val)

                seg_pred = preds_val[p2v_map_val].squeeze().detach().cpu()
                seg_gt   = label_val.cpu()

                evaluter.matches[str(sample_idx)] = {}
                evaluter.matches[str(sample_idx)]['seg_pred'] = seg_pred
                evaluter.matches[str(sample_idx)]['seg_gt']   = seg_gt

                del preds_val, voxel_val, x_val
                torch.cuda.empty_cache()

            # 计算 mIoU
            iou = evaluter.evaluate_semantic_segmantation_miou()
            mlflow.log_metric('val_mIoU', float(iou))

            if float(iou) >= best_iou:
                best_iou = float(iou)
                iou_path = os.path.join(cfg.model_save_root, f'best_iou_seed{seed}.pt')
                os.makedirs(cfg.model_save_root, exist_ok=True)
                torch.save(net.state_dict(), iou_path)
            print(f"[Val] New best mIoU: {best_iou:.4f}. Iou: {float(iou):.4f}. Saved to {iou_path}")
            epoch_save_path = os.path.join(cfg.model_save_root, f'epoch_{epoch}_seed{seed}.pt')

            # 确保目录存在
            os.makedirs(cfg.model_save_root, exist_ok=True)

            # 保存权重
            torch.save(net.state_dict(), epoch_save_path)
            print(f"Checkpoints saved to {epoch_save_path}")
            torch.cuda.empty_cache()
