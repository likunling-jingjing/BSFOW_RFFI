import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

class ETF_Classifier(nn.Module):
    def __init__(self, feat_in, num_classes, fix_bn=False, old_etf=None, device='cuda', try_assert=True):
        super(ETF_Classifier, self).__init__()
        self.feat_in = feat_in
        self.num_classes = num_classes
        self.device = device
        
        if old_etf is not None:
            self.known_classes = old_etf.shape[1]
            if old_etf.shape[0] != feat_in:
                 raise ValueError(f"Dimension Mismatch: {old_etf.shape[0]} != {feat_in}")
        else:
            self.known_classes = 0 
        
        self.new_classes = num_classes - self.known_classes

        if self.known_classes > 0:
            self.ori_M_known = old_etf.clone().detach().to(device)
            self.ori_M_known = F.normalize(self.ori_M_known, dim=0)
        else:
            self.ori_M_known = self._create_standard_etf(num_classes).to(device)
            self.new_classes = 0 

        if self.new_classes > 0:
            new_M = self._optimize_incremental_etf(self.ori_M_known, self.new_classes)
            final_M = torch.cat([self.ori_M_known, new_M], dim=1)
        else:
            final_M = self.ori_M_known

        self.register_buffer('ori_M', final_M)

        self.BN_H = nn.BatchNorm1d(feat_in)
        if fix_bn:
            for param in self.BN_H.parameters():
                param.requires_grad = False
        
        self.to(device) 

    def _create_standard_etf(self, n_classes):
        if n_classes <= 1:
            return torch.ones(self.feat_in, n_classes)

        I = torch.eye(n_classes)
        one = torch.ones(n_classes, n_classes)
        M = math.sqrt(n_classes / (n_classes - 1)) * (I - one / n_classes)

        if self.feat_in >= n_classes - 1:
            U = torch.randn(self.feat_in, n_classes)
            U, _ = torch.linalg.qr(U)
            M = U[:, :n_classes] @ M
        else:
            U = torch.randn(self.feat_in, n_classes)
            M = F.normalize(U, dim=0)
        
        return F.normalize(M, dim=0)

    def _optimize_incremental_etf(self, old_etf, new_classes, max_iter=2000, lr=0.1):
        feat_dim = old_etf.shape[0]
        
        new_vec = torch.randn(feat_dim, new_classes).to(self.device)
        new_vec = F.normalize(new_vec, dim=0)
        new_vec.requires_grad_(True)

        optimizer = torch.optim.Adam([new_vec], lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_iter, eta_min=0.001)

        for i in range(max_iter):
            optimizer.zero_grad()
            curr_new = F.normalize(new_vec, dim=0)
            
            sim_old_new = old_etf.detach().T @ curr_new
            loss_old = torch.logsumexp(sim_old_new * 10.0, dim=(0, 1))

            if new_classes > 1:
                sim_new_new = curr_new.T @ curr_new
                mask = ~torch.eye(new_classes, dtype=bool, device=self.device)
                sim_new_off = sim_new_new[mask]
                loss_new = torch.logsumexp(sim_new_off * 10.0, dim=0)
            else:
                loss_new = torch.tensor(0.0).to(self.device)

            loss = loss_old + loss_new
            loss.backward()
            optimizer.step()
            scheduler.step()

            with torch.no_grad():
                new_vec.data = F.normalize(new_vec.data, dim=0)

        return F.normalize(new_vec.detach(), dim=0)  

    def forward(self, x, return_logits=False):
        x = self.BN_H(x)
        x = F.normalize(x, dim=1)
        
        if return_logits:
            return x @ self.ori_M
        else:
            return x
    
    def get_logits(self, x):
        if x.shape[1] == self.feat_in:
            x = self.BN_H(x)
            x = F.normalize(x, dim=1)
        return x @ self.ori_M