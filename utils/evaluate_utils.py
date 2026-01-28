import numpy as np
import torch
from sklearn import metrics
from scipy.optimize import linear_sum_assignment

@torch.no_grad()
def hungarian_evaluate(predictions, targets, offset=0):
    targets = targets - offset
    predictions = predictions - offset
    predictions_np = predictions.numpy()
    num_elems = targets.size(0)

    valid_idx = np.where(predictions_np >= 0)[0]
    predictions_sel = predictions[valid_idx]
    targets_sel = targets[valid_idx]
    num_classes = torch.unique(targets).numel()
    num_classes_pred = torch.unique(predictions_sel).numel()

    match = _hungarian_match(predictions_sel, targets_sel, preds_k=num_classes_pred, targets_k=num_classes)
    reordered_preds = torch.zeros(predictions_sel.size(0), dtype=predictions_sel.dtype)
    for pred_i, target_i in match:
        reordered_preds[predictions_sel == int(pred_i)] = int(target_i)

    reordered_preds = reordered_preds.numpy()
    acc = int((reordered_preds == targets_sel.numpy()).sum()) / float(num_elems) if float(num_elems) else -1
    nmi = metrics.normalized_mutual_info_score(targets.numpy(), predictions.numpy())
    ari = metrics.adjusted_rand_score(targets.numpy(), predictions.numpy())
    
    return {'acc': acc * 100, 'ari': ari, 'nmi': nmi, 'hungarian_match': match}

@torch.no_grad()
def _hungarian_match(flat_preds, flat_targets, preds_k, targets_k):
    num_samples = flat_targets.shape[0]
    num_k = preds_k
    num_correct = np.zeros((num_k, num_k))

    for c1 in range(num_k):
        for c2 in range(num_k):
            votes = int(((flat_preds == c1) * (flat_targets == c2)).sum())
            num_correct[c1, c2] = votes

    match = linear_sum_assignment(num_samples - num_correct)
    match = np.array(list(zip(*match)))

    res = []
    for out_c, gt_c in match:
        res.append((out_c, gt_c))

    return res

@torch.no_grad()
def hungarian_evaluate_fixed(predictions, targets, offset=0):
    num_elems = targets.size(0)

    known_mask = targets < offset
    novel_mask = targets >= offset

    known_acc = 0
    if known_mask.sum() > 0:
        known_targets = targets[known_mask]
        known_preds = predictions[known_mask]

        known_matrix = _build_confusion_matrix(known_preds, known_targets)
        known_row_ind, known_col_ind = linear_sum_assignment(known_matrix.max() - known_matrix)
        known_acc = known_matrix[known_row_ind, known_col_ind].sum() / known_mask.sum()
    
    novel_acc = 0
    if novel_mask.sum() > 0:
        novel_targets = targets[novel_mask]
        novel_preds = predictions[novel_mask]
        
        novel_targets_remapped = novel_targets - offset
        novel_preds_remapped = novel_preds.clone()
        
        unique_novel_preds = torch.unique(novel_preds)
        for i, pred_label in enumerate(unique_novel_preds):
            novel_preds_remapped[novel_preds == pred_label] = i
        
        novel_matrix = _build_confusion_matrix(novel_preds_remapped, novel_targets_remapped)
        novel_row_ind, novel_col_ind = linear_sum_assignment(novel_matrix.max() - novel_matrix)
        novel_acc = novel_matrix[novel_row_ind, novel_col_ind].sum() / novel_mask.sum()
    
    total_acc = (known_acc * known_mask.sum() + novel_acc * novel_mask.sum()) / num_elems
    
    nmi = metrics.normalized_mutual_info_score(targets.numpy(), predictions.numpy())
    ari = metrics.adjusted_rand_score(targets.numpy(), predictions.numpy())
    
    return {
        'acc': total_acc * 100, 
        'ari': ari, 
        'nmi': nmi, 
        'known_acc': known_acc * 100, 
        'novel_acc': novel_acc * 100
    }

def _build_confusion_matrix(preds, targets):
    preds_k = preds.max().item() + 1
    targets_k = targets.max().item() + 1
    matrix_size = max(preds_k, targets_k)
    matrix = np.zeros((matrix_size, matrix_size), dtype=np.int64)
    
    for i in range(len(preds)):
        matrix[preds[i].item(), targets[i].item()] += 1
    
    return matrix