import argparse
import os
import random
import time
import math
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn 
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from models.build_model import build_model_pu
from datasets.datasets import get_dataset_trans
from utils.utils import AverageMeter, set_seed

def compute_accuracy(model, loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.cuda(), targets.cuda()
            _, logits, _, _, _ = model(inputs)
            preds = logits.argmax(1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)
    return 100. * correct / (total + 1e-8)

def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--dataset', default='wisig', type=str, choices=['lora', 'wisig', 'oracle'])
    parser.add_argument('--data-root', default='./data', type=str)
    parser.add_argument('--split-root', default='./random_splits')
    
    parser.add_argument('--epochs', default=50, type=int) 
    parser.add_argument('--batch-size', default=64, type=int)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--checkpoint-dir', default='./checkpoints/stage1')
    parser.add_argument('--tag', default='stage1_sup')
    
    parser.add_argument('--no-class', type=int, default=10) 
    parser.add_argument('--no-known', type=int, default=10) 
    
    parser.add_argument('--lbl-percent', type=int, default=100)
    parser.add_argument('--novel-percent', type=int, default=0)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--rff-method', default='spectrogram', type=str)

    args = parser.parse_args()
    
    if args.dataset == 'wisig': args.lr = 2e-4
    elif args.dataset == 'oracle': args.lr = 3e-3
    elif args.dataset == 'lora': args.lr = 1e-4

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    if args.seed != -1: set_seed(args)
    
    args.split_id = f'split_{args.seed}'
    args.ssl_indexes = f'{args.split_root}/{args.dataset}_stage1_{args.seed}.pkl'

    print("=" * 60)
    print(f"STAGE 1: Source Pre-training")
    print(f"Dataset: {args.dataset} | Known Classes: {args.no_known}")
    print("=" * 60)

    lbl_dataset, _, _, _, _, _ = get_dataset_trans(args)
    
    train_size = int(0.8 * len(lbl_dataset))
    val_size = len(lbl_dataset) - train_size
    train_ds, val_ds = random_split(lbl_dataset, [train_size, val_size], 
                                    generator=torch.Generator().manual_seed(args.seed))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, 
                              num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, 
                            num_workers=args.num_workers, pin_memory=True)

    print(f"Data Loaded: Train={len(train_ds)}, Val={len(val_ds)}")

    print(f"Building Model...")
    model = build_model_pu(args) 
    model = model.cuda()
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_acc = 0.0
    final_ckpt_path = os.path.join(args.checkpoint_dir, f"{args.dataset}_{args.tag}_best.pth")

    for epoch in range(args.epochs):
        model.train()
        train_loss = AverageMeter()
        train_acc = AverageMeter()
        
        loop = tqdm(train_loader, leave=False)
        for batch in loop:
            if isinstance(batch, (list, tuple)):
                inputs, targets = batch[0], batch[1]
                if isinstance(inputs, (list, tuple)): 
                    inputs = inputs[0] 
            else:
                inputs, targets = batch

            inputs, targets = inputs.cuda(), targets.cuda()
            
            _, logits, _, _, _ = model(inputs)
            
            loss = F.cross_entropy(logits / 0.1, targets)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            acc = (logits.argmax(1) == targets).float().mean()
            train_loss.update(loss.item(), inputs.size(0))
            train_acc.update(acc.item(), inputs.size(0))
            
            loop.set_description(f"Epoch {epoch+1}/{args.epochs}")
            loop.set_postfix(loss=train_loss.avg, acc=train_acc.avg)
        
        scheduler.step()
        
        val_acc = compute_accuracy(model, val_loader)
        print(f"[Epoch {epoch+1}] Train Acc: {train_acc.avg*100:.2f}% | Val Acc: {val_acc:.2f}%")
        
        if val_acc >= best_acc:
            best_acc = val_acc
            
            if hasattr(model, "module"):
                cur_m = model.module.classifier_new.ori_M.detach().cpu()
            else:
                cur_m = model.classifier_new.ori_M.detach().cpu()

            save_dict = {
                'state_dict': model.state_dict(),
                'cur_m': cur_m,
                'acc': val_acc,
                'epoch': epoch,
                'no_class': args.no_class, 
            }
            torch.save(save_dict, final_ckpt_path)
            print(f"  >>> Best Model Saved: {val_acc:.2f}%")

    print(f"\nTraining Completed. Best Val Acc: {best_acc:.2f}%")
    print(f"Checkpoint saved to: {final_ckpt_path}")

if __name__ == '__main__':
    cudnn.benchmark = True
    main()