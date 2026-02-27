import torch
from configs.configs import cfg
from dataset.ev_uav import EvUAV
from model.PACT import PACT
from utils.eval import evalute
import tqdm


if __name__ == '__main__':
    device = "cuda:0"

    net = PACT(cfg).eval()

    torch.set_grad_enabled(False)
    net.cuda()

    dataset = EvUAV(cfg, mode='test')

    test_dataloader = torch.utils.data.DataLoader(dataset, batch_size=cfg.batch_size,collate_fn=dataset.custom_collate)

    net.load_state_dict(torch.load(cfg.model_path))
    print('dict load: ',cfg.model_path)


    pbar = tqdm.tqdm(total=len(test_dataloader), desc='video', unit='video',unit_scale=True,position=0, leave=True)

    evaluter = evalute(cfg)

    for sample,ev in enumerate(test_dataloader):
        with torch.no_grad():
            x = ev['voxel_ev']
            label = ev['seg_label_ev'].float().cuda()
            p2v_map = ev['p2v_map'].long().cuda()
            ev_locs = ev['locs'].float().requires_grad_()
            idx = ev['idx_label']
            ts = ev_locs[:,3]

            preds, voxel, v1, idx_v1 = net(x)

            ev_bxyt = torch.as_tensor(ev['locs'], device=device).long()  # [N_ev,4] = [b,x,y,t]，原始分辨率
            seg_ev = ev['seg_label_ev'].to(device).long()  # [N_ev]

            preds = preds[p2v_map].squeeze().cpu()

            if cfg.eval:
                preds_cpu = preds.detach().cpu()
                label_cpu = label.detach().cpu()
                evaluter.matches[str(sample)] = {}
                evaluter.matches[str(sample)]['seg_pred']= preds_cpu
                evaluter.matches[str(sample)]['seg_gt'] = label_cpu
                if cfg.roc:
                    evaluter.roc_update(ts,preds,idx,label.cpu(),ev_locs)
        pbar.update(1)

    if cfg.eval:
        iou = evaluter.evaluate_semantic_segmantation_miou()
        seg_acc = evaluter.evaluate_semantic_segmantation_accuracy()
        if cfg.roc:
            pd, fa= evaluter.cal_roc()
        print('iou:{},seg_acc:{},pd:{},fa:{}'.format(iou,seg_acc,pd,fa))