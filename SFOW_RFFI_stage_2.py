import argparse
import os
import random
import time
import math
import csv
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm
from datetime import datetime
from scipy.optimize import linear_sum_assignment

from models.build_model import build_model_pu
from datasets.datasets import get_dataset_trans
from utils.evaluate_utils import hungarian_evaluate_fixed
from utils.utils import *
from utils.losses import *
from utils.sinkhorn_knopp import SinkhornKnopp


def load_stage1_checkpoint(checkpoint_path, model, args):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    
    if isinstance(checkpoint, dict):
        state_dict = checkpoint.get('state_dict', checkpoint)
    else:
        state_dict = checkpoint.state_dict()

    stage1_ori_M = None
    if isinstance(checkpoint, dict) and 'cur_m' in checkpoint:
        stage1_ori_M = checkpoint['cur_m'].float()
    else:
        for k in state_dict.keys():
            if 'classifier_new.ori_M' in k:
                stage1_ori_M = state_dict[k].detach().cpu().float()
                break
    
    model_sd = model.state_dict()
    filtered_sd = {k: v for k, v in state_dict.items() 
                   if k in model_sd and v.shape == model_sd[k].shape and not k.startswith("fc")}
    
    model.load_state_dict(filtered_sd, strict=False)

    if stage1_ori_M is not None:
        stage1_ori_M = stage1_ori_M.cuda()

    return model, stage1_ori_M


def align_new_classes_to_etf(model, data_loader, args):
    model.eval()
    features_list = []
    pseudo_labels_list = []
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Aligning", leave=False):
            if isinstance(batch, (tuple, list)):
                inputs = batch[0]
                if isinstance(inputs, (tuple, list)): inputs = inputs[0]
            else:
                inputs = batch
            
            inputs = inputs.cuda()
            _, logits, feat_norm, _, _ = model(inputs)
            features_list.append(feat_norm.cpu())
            pseudo_labels_list.append(logits.argmax(dim=1).cpu())

    features_all = torch.cat(features_list, dim=0).cuda()
    pseudo_labels_all = torch.cat(pseudo_labels_list, dim=0).cuda()
    
    new_class_indices = list(range(args.no_known, args.no_class))
    empirical_centers = []
    
    for c in new_class_indices:
        mask = pseudo_labels_all == c
        if mask.sum() > 0:
            center = F.normalize(features_all[mask].mean(dim=0), dim=0)
        else:
            center = F.normalize(torch.randn(features_all.shape[1]).cuda(), dim=0)
        empirical_centers.append(center)

    empirical_centers = torch.stack(empirical_centers)
    
    if hasattr(model, "module"):
        current_etf = model.module.classifier_new.ori_M[:, args.no_known:].T.clone()
    else:
        current_etf = model.classifier_new.ori_M[:, args.no_known:].T.clone()
    
    cost_matrix = torch.mm(empirical_centers, current_etf.T).cpu().numpy()
    _, col_ind = linear_sum_assignment(-cost_matrix)
    
    sorted_etf = current_etf[col_ind]
    new_etf_block = sorted_etf.T 
    
    if hasattr(model, "module"):
        model.module.classifier_new.ori_M[:, args.no_known:].data.copy_(new_etf_block)
    else:
        model.classifier_new.ori_M[:, args.no_known:].data.copy_(new_etf_block)


def train_epoch(args, unlbl_loader, model, old_model, optimizer, ema_optimizer, scheduler, epoch, sinkhorn, train_stat):
    losses = AverageMeter()
    
    w_t = linear_rampup(epoch, args.warmup, args.epochs)
    
    if not args.no_progress:
        p_bar = tqdm(range(len(unlbl_loader)))

    for batch_idx, data_unlbl in enumerate(unlbl_loader):
        (inputs_u_w, inputs_u_s), targets_u, _, index_u = data_unlbl 
        
        inputs = interleave(torch.cat((inputs_u_w, inputs_u_s)), 2).cuda()
        batch_u = inputs_u_w.shape[0]
        
        model.train()
        old_model.eval()
        
        _, logits, feat_norm, feat_con, _ = model(inputs)
        
        with torch.no_grad():
            _, logits_old, _, _, _ = old_model(inputs)

        probs_old = torch.softmax(logits_old, dim=1)
        max_prob_old, target_old_pl = torch.max(probs_old, dim=1)
        
        mask_reliable = (max_prob_old > 0.75) & (target_old_pl < args.no_known)
        
        if mask_reliable.sum() > 0:
            loss_ret = F.cross_entropy(logits[mask_reliable], target_old_pl[mask_reliable])
        else:
            loss_ret = torch.tensor(0.0).cuda()

        logits = de_interleave(logits, 2)
        logits_u_w, logits_u_s = logits.chunk(2)
        
        with torch.no_grad():
            pseudo_label_all = sinkhorn(logits_u_w.detach())
            _, targets_u_pl = torch.max(pseudo_label_all, dim=-1)

        feat_con = de_interleave(feat_con, 2)
        feat_con_u_w, _ = feat_con.chunk(2)
        
        train_stat['feature_con_bank'][index_u] = feat_con_u_w.detach().clone()
        with torch.no_grad():
            cosine_corr = torch.matmul(feat_con_u_w, train_stat['feature_con_bank'].T)
            _, knn_index = torch.topk(cosine_corr, k=args.chosen_neighbors, dim=-1, largest=True)
            mask_knn = torch.scatter(torch.zeros([batch_u, len(unlbl_loader.dataset)]).cuda(), 
                                     1, knn_index[:, 1:], 1).detach()
        
        loss_local = supcon_knn(features=feat_con_u_w, 
                                features_all=train_stat['feature_con_bank'].detach(), 
                                mask=mask_knn)
        
        loss_global = F.cross_entropy(logits_u_s / args.temparature, targets_u_pl)

        loss_align = w_t * loss_global + (1 - w_t) * loss_local
        final_loss = loss_align + 20.0 * loss_ret
        
        losses.update(final_loss.item(), batch_u)
        
        optimizer.zero_grad()
        final_loss.backward()
        optimizer.step()
        ema_optimizer.step()
        scheduler.step()

        if not args.no_progress and batch_idx % 20 == 0:
            p_bar.set_description(f"Loss: {losses.avg:.3f}")
            p_bar.update(20)

    if not args.no_progress:
        p_bar.close()
        
    return train_stat


def test_routine(args, loader, model, mode='known', offset=0):
    model.eval()
    preds_list = []
    targets_list = []
    
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.cuda(), targets.cuda()
            _, outputs, _, _, _ = model(inputs)
            
            if mode == 'known':
                prec1, _ = accuracy(outputs, targets, topk=(1, 5))
                return prec1.item()
            else:
                _, max_idx = torch.max(outputs, dim=1)
                preds_list.extend(max_idx.cpu().numpy().tolist())
                targets_list.extend(targets.cpu().numpy().tolist())

    if mode != 'known':
        preds = torch.from_numpy(np.array(preds_list))
        tars = torch.from_numpy(np.array(targets_list))
        return hungarian_evaluate_fixed(preds, tars, offset)


def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--dataset', default='wisig', type=str, choices=['wisig', 'oracle', 'lora'])
    parser.add_argument('--data-root', default='./data')
    parser.add_argument('--split-root', default='./random_splits')
    parser.add_argument('--out', default='./outputs')
    parser.add_argument('--stage1-ckpt', default='./checkpoints/stage1_best.pth', type=str)

    parser.add_argument('--no-class', default=10, type=int)
    parser.add_argument('--lbl-percent', type=int, default=50)
    parser.add_argument('--novel-percent', default=50, type=int)
    
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--batch-size', default=200, type=int)
    parser.add_argument('--lr', default=5e-4, type=float)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--warmup', default=15, type=int)
    parser.add_argument('--temparature', default=0.3, type=float)
    parser.add_argument('--chosen_neighbors', default=100, type=int)
    parser.add_argument('--rho', default='0.3,0.9', type=str)
    parser.add_argument('--rff-method', default='spectrogram', type=str)
    
    parser.add_argument('--resume', default='', type=str)
    parser.add_argument('--no-progress', action='store_true')

    args = parser.parse_args()
    
    run_id = datetime.today().strftime('%m%d_%H%M')
    args.ssl_indexes = f'{args.split_root}/{args.dataset}_{args.lbl_percent}_{args.novel_percent}.pkl'
    args.exp_name = f'{args.dataset}_{args.lbl_percent}_{args.novel_percent}_{run_id}'
    args.out = os.path.join(args.out, args.exp_name)
    os.makedirs(args.out, exist_ok=True)

    csv_path = os.path.join(args.out, "log.csv")
    with open(csv_path, mode='w', newline='') as f:
        csv.writer(f).writerow(["Epoch", "Known_Acc", "Novel_Acc", "All_Acc"])

    args.n_gpu = torch.cuda.device_count()
    if args.seed != -1:
        set_seed(args)

    args.data_root = os.path.join(args.data_root, args.dataset)
    os.makedirs(args.data_root, exist_ok=True)
    os.makedirs(args.split_root, exist_ok=True)
    
    args.no_known = args.no_class - int((args.novel_percent * args.no_class) / 100)

    print(f"Dataset: {args.dataset} | Known: {args.no_known} | Total: {args.no_class}")

    lbl_dataset, unlbl_dataset, pl_dataset, test_known, test_novel, test_all = get_dataset_trans(args)
    
    u_bs = int((float(args.batch_size) * len(unlbl_dataset))/(len(lbl_dataset) + len(unlbl_dataset)))
    l_bs = args.batch_size - u_bs
    
    lbl_loader = DataLoader(lbl_dataset, sampler=RandomSampler(lbl_dataset), batch_size=l_bs, num_workers=4, drop_last=True)
    unlbl_loader = DataLoader(unlbl_dataset, sampler=RandomSampler(unlbl_dataset), batch_size=u_bs, num_workers=4, drop_last=True)
    pl_loader = DataLoader(pl_dataset, sampler=SequentialSampler(pl_dataset), batch_size=args.batch_size, num_workers=4)
    
    test_loader_known = DataLoader(test_known, batch_size=args.batch_size, num_workers=4)
    test_loader_novel = DataLoader(test_novel, batch_size=args.batch_size, num_workers=4)
    test_loader_all = DataLoader(test_all, batch_size=args.batch_size, num_workers=4)

    temp_model = build_model_pu(args)
    temp_model, stage1_etf = load_stage1_checkpoint(args.stage1_ckpt, temp_model, args)
    
    old_etf_known = stage1_etf[:, :args.no_known]
    
    model = build_model_pu(args, old_etf=old_etf_known).cuda()
    ema_model = build_model_pu(args, ema=True, old_etf=old_etf_known).cuda()
    
    model.load_state_dict(temp_model.state_dict(), strict=False)
    ema_model.load_state_dict(temp_model.state_dict(), strict=False)
    
    target_mod = model.module if hasattr(model, "module") else model
    target_mod.classifier_new.ori_M[:, :args.no_known].data.copy_(stage1_etf[:, :args.no_known])

    import copy
    old_model = copy.deepcopy(model)
    old_model.eval()
    for param in old_model.parameters():
        param.requires_grad = False

    align_new_classes_to_etf(model, unlbl_loader, args)

    ema_optimizer = WeightEMA(0.95, model, ema_model)
    sinkhorn = SinkhornKnopp(num_iters_sk=3, epsilon_sk=0.05, imb_factor=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=0.01*args.lr)

    train_stat = {
        'all_prototype': target_mod.classifier_new.ori_M.T[:args.no_known, :].cuda(),
        'feature_con_bank': torch.zeros(len(unlbl_loader.dataset), 128).cuda(),
    }

    best_all_acc = 0.0
    final_ckpt_path = os.path.join(args.out, "best_model.pth")

    for epoch in range(args.epochs):
        train_stat = train_epoch(args, unlbl_loader, model, old_model, optimizer, ema_optimizer, scheduler, epoch, sinkhorn, train_stat)
        
        acc_known = test_routine(args, test_loader_known, model, mode='known')
        res_novel = test_routine(args, test_loader_novel, model, mode='cluster', offset=args.no_known)
        res_all = test_routine(args, test_loader_all, model, mode='cluster', offset=0)
        
        acc_novel = res_novel["acc"]
        acc_all = res_all["acc"]

        print(f'Epoch: {epoch} | Known: {acc_known:.2f}% | Novel: {acc_novel:.2f}% | All: {acc_all:.2f}%')
        
        with open(csv_path, mode='a', newline='') as f:
            csv.writer(f).writerow([epoch, acc_known, acc_novel, acc_all])

        if acc_all >= best_all_acc:
            best_all_acc = acc_all
            save_dict = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'cur_m': target_mod.classifier_new.ori_M.cpu(),
                'acc': acc_all,
            }
            torch.save(save_dict, final_ckpt_path)
            print(f"  --> New Best Saved: {acc_all:.2f}%")

    ckpt = torch.load(final_ckpt_path)
    model.load_state_dict(ckpt['state_dict'], strict=False)
    target_mod.classifier_new.ori_M.data.copy_(ckpt['cur_m'].cuda())
    
    align_new_classes_to_etf(model, pl_loader, args)
    
    print(f"Training Finished. Results at: {args.out}")

if __name__ == '__main__':
    cudnn.benchmark = True
    main()
