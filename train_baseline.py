#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# 导入 baseline 模型
from baseline_models import get_model



class JointDataset(Dataset):

    def __init__(self, data_dir: str, split: str = 'train'):
        csv_path = f"{data_dir}/{split}.csv"
        print(f"Loading {split} data from {csv_path}...")
        df = pd.read_csv(csv_path)
        
        need = {'page_key', 'sentence_id', 'sentence_index', 'x0', 'y0', 'x1', 'y1'}
        miss = need - set(df.columns)
        if miss:
            raise ValueError(f"Missing required columns in CSV: {miss}")

        # 将 sentence_id 规范成页内连续整数（仅作分组键）
        sid_all_sorted = sorted(df['sentence_id'].unique())
        sid2rank_global = {sid: i for i, sid in enumerate(sid_all_sorted)}
        df['sid_rank'] = df['sentence_id'].map(sid2rank_global).astype(int)

        self.pages: List[Dict] = []

        for page_key, g in df.groupby('page_key'):
            g = g.copy()
            sent_items = []  # 按"输入索引"顺序保存
            
            for sid_r, sg in g.groupby('sid_rank'):
                sg = sg.copy()

                # 按 sentence_index 排序（人工标注的句内顺序）
                sg = sg.sort_values('sentence_index').copy()
                sg['sentence_index'] = range(len(sg))  

                # 句框 = union
                x0 = sg['x0'].min()
                y0 = sg['y0'].min()
                x1 = sg['x1'].max()
                y1 = sg['y1'].max()
                s_box = np.array([x0, y0, x1, y1], dtype=np.float32)

                sent_items.append(dict(
                    sid_input=len(sent_items),
                    sentence_id=int(sg['sentence_id'].iloc[0]),  # 人工标注的顺序
                    s_box=s_box,
                    sg=sg
                ))

            # 句输入数组
            S = len(sent_items)
            sentence_bboxes = []
            sentence_ids = []  #  收集人工标注的sentence_id
            
            for it in sent_items:
                sentence_bboxes.append(it['s_box'].tolist())
                sentence_ids.append(it['sentence_id'])
            
            sentence_bboxes = torch.tensor(sentence_bboxes, dtype=torch.float32)
            
            sentence_labels = torch.tensor(sentence_ids, dtype=torch.long)
            sent_items_sorted = sorted(enumerate(sent_items), 
                                       key=lambda x: x[1]['sentence_id'])
            
            char_bboxes_list = []
            char_to_sentence_idx_list = []
            
            # 按人工标注的句顺序遍历
            for orig_idx, it in sent_items_sorted:
                sg = it['sg'] 
                
                # 累计到全局
                for _, row in sg.iterrows():
                    char_bboxes_list.append([
                        float(row['x0']), float(row['y0']),
                        float(row['x1']), float(row['y1'])
                    ])
                    char_to_sentence_idx_list.append(orig_idx)  # 指向原始输入位置

            char_bboxes = torch.tensor(char_bboxes_list, dtype=torch.float32)
            char_to_sentence_idx = torch.tensor(char_to_sentence_idx_list, dtype=torch.long)

            # 全局字符"阅读顺序名次"= 0..N-1（按人工标注拼出来的）
            char_labels = torch.arange(char_bboxes.shape[0], dtype=torch.long)

            # 安全检查
            if char_to_sentence_idx.numel() > 0:
                cmin = int(char_to_sentence_idx.min().item())
                cmax = int(char_to_sentence_idx.max().item())
                assert 0 <= cmin and cmax < S, \
                    f"[{page_key}] char_to_sentence_idx out of range: min={cmin}, max={cmax}, S={S}"

            self.pages.append(dict(
                page_key=page_key,
                sentence_bboxes=sentence_bboxes,
                sentence_labels=sentence_labels,
                char_bboxes=char_bboxes,
                char_labels=char_labels,
                char_to_sentence_idx=char_to_sentence_idx
            ))

        print(f"  ✓ Loaded {len(self.pages)} pages (using human annotations)")

    def __len__(self):
        return len(self.pages)
    
    def __getitem__(self, idx):
        return self.pages[idx]


def joint_collate_fn(batch: List[Dict]):
    return batch[0]


# ============================== 排序损失 ==============================
def ranknet_pairwise_loss(scores: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    N = scores.shape[0]
    if N <= 1:
        return scores.sum() * 0.0
    loss_terms = []
    for i in range(N):
        for j in range(i + 1, N):
            if labels[i] < labels[j]:
                diff = scores[j] - scores[i]
                loss_terms.append(torch.nn.functional.softplus(-diff))
            elif labels[j] < labels[i]:
                diff = scores[i] - scores[j]
                loss_terms.append(torch.nn.functional.softplus(-diff))
    return torch.stack(loss_terms).mean() if loss_terms else scores.sum() * 0.0


def hinge_pairwise_loss(scores: torch.Tensor, labels: torch.Tensor, margin: float = 1.0) -> torch.Tensor:
    N = scores.shape[0]
    if N <= 1:
        return scores.sum() * 0.0
    loss_terms = []
    for i in range(N):
        for j in range(i + 1, N):
            if labels[i] < labels[j]:
                loss_terms.append(torch.relu(scores[i] - scores[j] + margin))
            elif labels[j] < labels[i]:
                loss_terms.append(torch.relu(scores[j] - scores[i] + margin))
    return torch.stack(loss_terms).mean() if loss_terms else scores.sum() * 0.0


def make_loss_fn(loss_type: str):
    if loss_type == 'ranknet':
        return ranknet_pairwise_loss
    if loss_type == 'hinge':
        return lambda s, y: hinge_pairwise_loss(s, y, margin=1.0)
    raise ValueError(f"Unknown loss_type: {loss_type}")


# ============================== 评估指标==============================
def _kendall_tau(order_true: np.ndarray, order_pred: np.ndarray) -> float:
    n = len(order_true)
    conc = disc = 0
    for i in range(n):
        for j in range(i + 1, n):
            a = order_true[i] - order_true[j]
            b = order_pred[i] - order_pred[j]
            if a * b > 0:
                conc += 1
            elif a * b < 0:
                disc += 1
    denom = conc + disc
    return float((conc - disc) / denom) if denom > 0 else 0.0


def _spearman_rho(order_true: np.ndarray, order_pred: np.ndarray) -> float:
    t = order_true.astype(np.float64)
    p = order_pred.astype(np.float64)
    t = (t - t.mean()) / (t.std() + 1e-9)
    p = (p - p.mean()) / (p.std() + 1e-9)
    return float(np.clip((t * p).mean(), -1.0, 1.0))


def _topk_accuracy(order_true: np.ndarray, order_pred: np.ndarray, k: int) -> float:
    diff = np.abs(order_true - order_pred)
    return float(np.mean(diff <= (k - 1)))


def _pairwise_accuracy(order_true: np.ndarray, order_pred: np.ndarray) -> float:
    n = len(order_true)
    correct = total = 0
    for i in range(n):
        for j in range(i + 1, n):
            total += 1
            correct += int((order_true[i] - order_true[j]) * (order_pred[i] - order_pred[j]) > 0)
    return float(correct / total) if total > 0 else 0.0


def compute_five_metrics(order_true: np.ndarray, order_pred: np.ndarray) -> Dict[str, float]:
    return {
        'kendall_tau': _kendall_tau(order_true, order_pred),
        'spearman_rho': _spearman_rho(order_true, order_pred),
        'top1_accuracy': _topk_accuracy(order_true, order_pred, k=1),
        'top3_accuracy': _topk_accuracy(order_true, order_pred, k=3),
        'pairwise_accuracy': _pairwise_accuracy(order_true, order_pred),
    }


# ============================== 三级评估：sentence / intra / global ==============================
@torch.no_grad()
def evaluate(model, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model_eval = getattr(model, 'eval', None)
    if callable(model_eval):
        model.eval()

    sent_metrics: List[Dict[str, float]] = []
    intra_metrics: List[Dict[str, float]] = []
    global_metrics: List[Dict[str, float]] = []

    for batch in tqdm(loader, desc='Evaluating', leave=False):
        # ---- move data ----
        sb = batch['sentence_bboxes'].to(device)
        sl = batch['sentence_labels'].to(device)
        cb = batch['char_bboxes'].to(device)
        cl = batch['char_labels'].to(device)
        cs = batch['char_to_sentence_idx'].to(device)

        # ---- forward ----
        batch_gpu = {
            'sentence_bboxes': sb,
            'sentence_labels': sl,
            'char_bboxes': cb,
            'char_labels': cl,
            'char_to_sentence_idx': cs
        }
        outputs = model(batch_gpu)

        sent_scores = outputs['sent_scores']
        char_scores = outputs['char_scores']

        # ---- move everything to CPU for slicing ----
        sent_scores = sent_scores.cpu()
        char_scores = char_scores.cpu()
        sl_cpu = sl.cpu()
        cl_cpu = cl.cpu()
        cs_cpu = cs.cpu()

        # ---------------- sentence level ----------------
        s_pred_idx = torch.argsort(sent_scores, dim=0)
        s_order_pred = torch.empty_like(s_pred_idx)
        s_order_pred[s_pred_idx] = torch.arange(len(s_pred_idx))

        m_s = compute_five_metrics(
            sl_cpu.numpy(),
            s_order_pred.numpy()
        )
        sent_metrics.append(m_s)

        # ---------------- intra-sentence ----------------
        intra_accum = {k: 0.0 for k in ['kendall_tau','spearman_rho','top1_accuracy','top3_accuracy','pairwise_accuracy']}
        total_weight = 0

        for sid in torch.unique(cs_cpu).tolist():
            idx = (cs_cpu == sid).nonzero(as_tuple=False).squeeze(-1)
            if idx.numel() <= 1:
                continue

            # predicted ranks within sentence
            c_idx_sorted = torch.argsort(char_scores[idx], dim=0)
            c_order_pred = torch.empty_like(c_idx_sorted)
            c_order_pred[c_idx_sorted] = torch.arange(len(c_idx_sorted))

            # true ranks within sentence
            cl_group = cl_cpu[idx]
            cl_order = torch.argsort(torch.argsort(cl_group))

            m_g = compute_five_metrics(cl_order.numpy(), c_order_pred.numpy())

            w = int(idx.numel())
            total_weight += w
            for k in intra_accum:
                intra_accum[k] += m_g[k] * w

        if total_weight > 0:
            m_intra = {k: intra_accum[k] / total_weight for k in intra_accum}
        else:
            m_intra = {k: 0.0 for k in intra_accum}

        intra_metrics.append(m_intra)

        # ---------------- global ----------------
        s_sorted = torch.argsort(sent_scores, dim=0)
        pred_global_idx_list = []
        for sid in s_sorted.tolist():
            idx = (cs_cpu == sid).nonzero(as_tuple=False).squeeze(-1)
            if idx.numel() == 0:
                continue
            c_idx_sorted = torch.argsort(char_scores[idx], dim=0)
            pred_global_idx_list.extend(idx[c_idx_sorted].tolist())

        order_pred_global = torch.empty_like(cl_cpu)
        order_pred_global[pred_global_idx_list] = torch.arange(len(pred_global_idx_list))

        m_global = compute_five_metrics(cl_cpu.numpy(), order_pred_global.numpy())
        global_metrics.append(m_global)

    # ------- average all pages -------
    def _avg(lst):
        keys = lst[0].keys()
        return {k: float(np.mean([x[k] for x in lst])) for k in keys}

    out = {}
    out.update({f"sentence/{k}": v for k, v in _avg(sent_metrics).items()})
    out.update({f"intra/{k}": v for k, v in _avg(intra_metrics).items()})
    out.update({f"global/{k}": v for k, v in _avg(global_metrics).items()})
    return out


# ============================== 训练 ==============================
class TrainConfig:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def dynamic_lambda_linear(epoch, base_epoch, lambda_max, warmup):
    if epoch < base_epoch:
        return 0.0
    t = min(max(epoch - base_epoch + 1, 0), warmup)
    return float(lambda_max * t / max(warmup, 1))


def dynamic_lambda_cosine(epoch, base_epoch, lambda_max, total_epochs):
    if epoch < base_epoch:
        return 0.0
    progress = (epoch - base_epoch) / max(total_epochs - base_epoch, 1)
    return float(lambda_max * 0.5 * (1 - math.cos(math.pi * progress)))


def train(cfg: TrainConfig):
    device = torch.device(cfg.device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # 数据
    train_set = JointDataset(cfg.data_dir, 'train')
    val_set = JointDataset(cfg.data_dir, 'val')
    test_set = JointDataset(cfg.data_dir, 'test')

    train_loader = DataLoader(train_set, batch_size=1, shuffle=True, collate_fn=joint_collate_fn)
    val_loader   = DataLoader(val_set,   batch_size=1, shuffle=False, collate_fn=joint_collate_fn)
    test_loader  = DataLoader(test_set,  batch_size=1, shuffle=False, collate_fn=joint_collate_fn)

    # 模型
    model = get_model(cfg.model, {
        'rule_type': cfg.rule_type,
        'hidden_dim': cfg.hidden_dim,
        'dropout': cfg.dropout,
        'num_heads': cfg.num_heads
    })
    if hasattr(model, 'to'):
        model.to(device)

    # 优化器/调度（仅NN）
    if hasattr(model, 'parameters'):
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    else:
        optimizer = None
        scheduler = None

    loss_fn = make_loss_fn(cfg.loss_type)

    save_dir = Path(cfg.save_dir) / f"baseline_{cfg.model}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    save_dir.mkdir(parents=True, exist_ok=True)
    best_val_tau = -1.0

    # ----------------- Stage I：句级 -----------------
    if hasattr(model, 'parameters'):
        print("\n===== Stage I: Sentence-only pretrain =====")
        for epoch in range(1, cfg.stage1_epochs + 1):
            model.train()
            ep_loss = 0.0
            for batch in tqdm(train_loader, desc=f"Stage I Epoch {epoch}/{cfg.stage1_epochs}"):
                # Tensors to device
                batch = {
                    k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                    for k, v in batch.items()
                }
                sb = batch['sentence_bboxes']
                sl = batch['sentence_labels']
                optimizer.zero_grad()
                outputs = model(batch)
                sent_scores = outputs['sent_scores']
                loss_s = loss_fn(sent_scores, sl)
                loss_s.backward()
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                optimizer.step()
                ep_loss += float(loss_s.item())
            if scheduler is not None:
                scheduler.step()
            print(f"  Stage I avg loss: {ep_loss / max(len(train_loader), 1):.4f}")
    else:
        print("\n[Rule-based baseline] Skip Stage I.")

    # ----------------- Stage II：联合 -----------------
    print("\n===== Stage II: Joint training =====")
    for epoch in range(cfg.stage1_epochs + 1, cfg.epochs + 1):
        if hasattr(model, 'train'):
            model.train()
        ep_s_loss = ep_c_loss = ep_total = 0.0

        # λ 调度
        if cfg.lambda_schedule == 'cosine':
            lam = dynamic_lambda_cosine(epoch, cfg.stage1_epochs + 1, cfg.lambda_max, cfg.epochs)
        else:
            lam = dynamic_lambda_linear(epoch, cfg.stage1_epochs + 1, cfg.lambda_max, cfg.lambda_warmup)

        if hasattr(model, 'parameters'):
            for batch in tqdm(train_loader, desc=f"Stage II Epoch {epoch}/{cfg.epochs} (λ={lam:.3f})"):
                batch = {
                    k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                    for k, v in batch.items()
                }
                sb = batch['sentence_bboxes']; sl = batch['sentence_labels']
                cb = batch['char_bboxes'];     cl = batch['char_labels']

                optimizer.zero_grad()
                outputs = model(batch)
                sent_scores = outputs['sent_scores']
                char_scores = outputs['char_scores']
                loss_s = loss_fn(sent_scores, sl)
                loss_c = loss_fn(char_scores, cl)
                loss = loss_s + lam * loss_c
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                optimizer.step()

                ep_s_loss += float(loss_s.item())
                ep_c_loss += float(loss_c.item())
                ep_total  += float(loss.item())
            if scheduler is not None:
                scheduler.step()
            nb = max(len(train_loader), 1)
            print(f"  Train: total {ep_total/nb:.4f} | sent {ep_s_loss/nb:.4f} | char {ep_c_loss/nb:.4f}")

        # 验证（三级指标）
        val_metrics = evaluate(model, val_loader, device)
        print("  Val Metrics:", json.dumps(val_metrics, ensure_ascii=False))
        val_global_tau = float(val_metrics.get('global/kendall_tau', -1.0))
        if val_global_tau > best_val_tau:
            best_val_tau = val_global_tau
            # 保存
            if hasattr(model, 'state_dict'):
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict() if hasattr(model, 'state_dict') else None,
                    'val_metrics': val_metrics,
                    'config': cfg.__dict__
                }, save_dir / 'best_model.pth')
            else:
                # 规则模型无需 state_dict
                with open(save_dir / 'best_rule.json', 'w') as f:
                    json.dump({'epoch': epoch, 'val_metrics': val_metrics, 'config': cfg.__dict__}, f, indent=2, ensure_ascii=False)
            print(f"  ✓ Best model saved (Val global τ={best_val_tau:.4f})")

    # ----------------- 测试 -----------------
    print("\n===== Final Test =====")
    if (save_dir / 'best_model.pth').exists() and hasattr(model, 'load_state_dict'):
        ckpt = torch.load(save_dir / 'best_model.pth', map_location=device)
        if ckpt.get('model_state_dict') is not None:
            model.load_state_dict(ckpt['model_state_dict'])
    test_metrics = evaluate(model, test_loader, device)
    print("[Test Set]", json.dumps(test_metrics, ensure_ascii=False, indent=2))

    with open(save_dir / 'results.json', 'w') as f:
        json.dump({'val_best_global_tau': best_val_tau, 'test': test_metrics, 'config': cfg.__dict__},
                  f, indent=2, ensure_ascii=False)
    print(f"✓ Results saved to {save_dir}")


def main():
    ap = argparse.ArgumentParser()
    # 模型
    ap.add_argument('--model', type=str, default='position_regressor',
                    choices=['rule_based', 'position_only', 'position_regressor', 'char_attention_mlp'])
    ap.add_argument('--rule_type', type=str, default='right_to_left',
                    choices=['right_to_left', 'left_to_right', 'top_to_bottom'])

    # 数据/设备
    ap.add_argument('--data_dir', type=str, default='/home/zhengxiaoying/DBManuscripts/DongbaHRO')
    ap.add_argument('--device', type=str, default='cuda')

    # 结构/训练
    ap.add_argument('--hidden_dim', type=int, default=128)
    ap.add_argument('--num_heads', type=int, default=4)
    ap.add_argument('--dropout', type=float, default=0.1)
    ap.add_argument('--epochs', type=int, default=60)
    ap.add_argument('--stage1_epochs', type=int, default=8)
    ap.add_argument('--loss_type', type=str, choices=['ranknet', 'hinge'], default='ranknet')
    ap.add_argument('--lambda_max', type=float, default=0.7)
    ap.add_argument('--lambda_warmup', type=int, default=6)
    ap.add_argument('--lambda_schedule', type=str, choices=['linear', 'cosine'], default='linear')
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--weight_decay', type=float, default=1e-5)
    ap.add_argument('--grad_clip', type=float, default=1.0)
    ap.add_argument('--save_dir', type=str, default='/home/zhengxiaoying/DBManuscripts/DongbaHRO/checkpoints')

    args = ap.parse_args()
    cfg = TrainConfig(**vars(args))
    train(cfg)


if __name__ == "__main__":
    main()