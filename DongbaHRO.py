#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
import argparse, json, math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# ============================== Metrics ==============================
def _kendall_tau(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    n = len(y_true)
    conc = disc = 0
    for i in range(n):
        for j in range(i+1, n):
            s = (y_true[i] - y_true[j]) * (y_pred[i] - y_pred[j])
            conc += int(s > 0)
            disc += int(s < 0)
    d = conc + disc
    return (conc - disc) / d if d > 0 else 0.0

def _spearman_rho(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) <= 1:
        return 0.0
    t = (y_true - y_true.mean()) / (y_true.std() + 1e-8)
    p = (y_pred - y_pred.mean()) / (y_pred.std() + 1e-8)
    return float(np.clip(np.mean(t * p), -1.0, 1.0))

def _topk_accuracy(y_true_rank: np.ndarray, y_pred_rank: np.ndarray, k: int) -> float:
    if len(y_true_rank) == 0:
        return 0.0
    true_best = int(np.argmin(y_true_rank))
    pred_sorted_idx = np.argsort(y_pred_rank)
    return float(true_best in pred_sorted_idx[:max(1, k)])

def _pairwise_accuracy(y_true_rank: np.ndarray, y_pred_rank: np.ndarray) -> float:
    n = len(y_true_rank)
    ok = tot = 0
    for i in range(n):
        for j in range(i+1, n):
            tot += 1
            ok += int((y_true_rank[i] < y_true_rank[j]) == (y_pred_rank[i] < y_pred_rank[j]))
    return ok / tot if tot > 0 else 0.0

def compute_five_metrics(y_true_rank: np.ndarray, y_pred_rank: np.ndarray) -> Dict[str, float]:
    return {
        'kendall_tau': _kendall_tau(y_true_rank, y_pred_rank),
        'spearman_rho': _spearman_rho(y_true_rank, y_pred_rank),
        'top1_accuracy': _topk_accuracy(y_true_rank, y_pred_rank, k=1),
        'top3_accuracy': _topk_accuracy(y_true_rank, y_pred_rank, k=3),
        'pairwise_accuracy': _pairwise_accuracy(y_true_rank, y_pred_rank),
    }

# ============================== Dataset ==============================
class JointDataset(Dataset):
    """
    
    每页=1样本：
      sentence_bboxes: [S,4]              # 输入顺序
      sentence_labels: [S]                # 人工标注的sentence_id
      char_bboxes: [N,4]                  # 按人工标注顺序排列
      char_labels: [N] (0..N-1)           # 全局阅读顺序（人工标注）
      char_to_sentence_idx: [N]           # 字符所属句子的输入位置索引
      sentence_dir_feat: [S,4]            # 方向特征（辅助特征）
      char_dir_feat: [N,2]                # 方向特征（辅助特征）
    """
    def __init__(self, data_dir: str, split: str = 'train', use_direction: bool = True):
        import pandas as pd
        csv_path = f"{data_dir}/{split}.csv"
        print(f"Loading {split} data from {csv_path}...")
        df = pd.read_csv(csv_path)
        
        need = {'page_key', 'sentence_id', 'sentence_index', 'x0', 'y0', 'x1', 'y1'}
        miss = need - set(df.columns)
        if miss:
            raise ValueError(f"Missing required columns in CSV: {miss}")

        sid_all_sorted = sorted(df['sentence_id'].unique())
        sid2rank_global = {sid: i for i, sid in enumerate(sid_all_sorted)}
        df['sid_rank'] = df['sentence_id'].map(sid2rank_global).astype(int)

        self.pages: List[Dict] = []

        for page_key, g in df.groupby('page_key'):
            g = g.copy()
            sent_items = []  # 按"输入索引"顺序保存
            
            for sid_r, sg in g.groupby('sid_rank'):
                sg = sg.copy()

                # ✅ 按 sentence_index 排序（人工标注的句内顺序）
                sg = sg.sort_values('sentence_index').copy()
                sg['sentence_index'] = range(len(sg))  

                # 句框 = union
                x0 = sg['x0'].min()
                y0 = sg['y0'].min()
                x1 = sg['x1'].max()
                y1 = sg['y1'].max()
                s_box = np.array([x0, y0, x1, y1], dtype=np.float32)

                # 方向特征：PCA 主方向
                cx = (sg['x0'].values + sg['x1'].values) * 0.5
                cy = (sg['y0'].values + sg['y1'].values) * 0.5
                pts = np.stack([cx, cy], axis=1)
                
                if use_direction and pts.shape[0] >= 2:
                    ctr = pts.mean(0)
                    pts0 = pts - ctr
                    cov = np.cov(pts0.T)
                    eigvals, eigvecs = np.linalg.eigh(cov)
                    u = eigvecs[:, -1]
                    u = u / (np.linalg.norm(u) + 1e-8)
                else:
                    u = np.array([1.0, 0.0], dtype=np.float32)
                    ctr = pts.mean(0)

                ang = math.atan2(u[1], u[0])
                s_dir = [float(u[0]), float(u[1]), math.sin(ang), math.cos(ang)]

                # 字层方向（t、p 归一化）
                u_perp = np.array([-u[1], u[0]], dtype=np.float32)
                proj_t = (pts - ctr) @ u
                proj_p = (pts - ctr) @ u_perp
                vmax_t = np.max(np.abs(proj_t)) + 1e-6
                vmax_p = np.max(np.abs(proj_p)) + 1e-6
                t_norm = (proj_t / vmax_t).astype(np.float32)
                p_norm = (proj_p / vmax_p).astype(np.float32)
                char_dir = np.stack([t_norm, p_norm], axis=1)

                sent_items.append(dict(
                    sid_input=len(sent_items),
                    sentence_id=int(sg['sentence_id'].iloc[0]),  # ✅ 人工标注的顺序
                    s_box=s_box,
                    s_dir=s_dir,
                    sg=sg,
                    char_dir=char_dir
                ))

            # 句输入数组
            S = len(sent_items)
            sentence_bboxes = []
            sentence_dir_feat = []
            sentence_ids = []  # ✅ 收集人工标注的sentence_id
            
            for it in sent_items:
                sentence_bboxes.append(it['s_box'].tolist())
                sentence_dir_feat.append(it['s_dir'])
                sentence_ids.append(it['sentence_id'])
            
            sentence_bboxes = torch.tensor(sentence_bboxes, dtype=torch.float32)
            sentence_dir_feat = torch.tensor(sentence_dir_feat, dtype=torch.float32)
            
            # ✅ 直接用人工标注的sentence_id作为标签
            sentence_labels = torch.tensor(sentence_ids, dtype=torch.long)

            # ✅ 按人工标注的sentence_id顺序生成全局字符标签
            # 首先按sentence_id排序sent_items
            sent_items_sorted = sorted(enumerate(sent_items), 
                                       key=lambda x: x[1]['sentence_id'])
            
            char_bboxes_list = []
            char_to_sentence_idx_list = []
            char_dir_blocks = []
            
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
                
                char_dir_blocks.append(it['char_dir'])

            char_bboxes = torch.tensor(char_bboxes_list, dtype=torch.float32)
            char_to_sentence_idx = torch.tensor(char_to_sentence_idx_list, dtype=torch.long)
            
            if use_direction and len(char_dir_blocks) > 0:
                char_dir_feat = torch.tensor(
                    np.concatenate(char_dir_blocks, axis=0), 
                    dtype=torch.float32
                )
            else:
                char_dir_feat = torch.zeros((char_bboxes.shape[0], 2), dtype=torch.float32)

            # ✅ 全局字符"阅读顺序名次"= 0..N-1（按人工标注拼出来的）
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
                sentence_dir_feat=sentence_dir_feat,
                char_bboxes=char_bboxes,
                char_labels=char_labels,
                char_to_sentence_idx=char_to_sentence_idx,
                char_dir_feat=char_dir_feat
            ))

        print(f"  ✓ Loaded {len(self.pages)} pages (using human annotations)")

    def __len__(self):
        return len(self.pages)
    
    def __getitem__(self, idx):
        return self.pages[idx]

def joint_collate_fn(batch: List[Dict]):
    return batch[0]

# ============================== Losses ==============================
def ranknet_pairwise_loss(scores: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    N = scores.numel()
    if N <= 1:
        return scores.sum() * 0.0
    terms = []
    for i in range(N):
        for j in range(i+1, N):
            if labels[i] < labels[j]:
                terms.append(F.softplus(-(scores[j] - scores[i])))
            elif labels[j] < labels[i]:
                terms.append(F.softplus(-(scores[i] - scores[j])))
    return torch.stack(terms).mean() if terms else scores.sum() * 0.0

def hinge_pairwise_loss(scores: torch.Tensor, labels: torch.Tensor, margin: float = 1.0) -> torch.Tensor:
    N = scores.numel()
    if N <= 1:
        return scores.sum() * 0.0
    terms = []
    for i in range(N):
        for j in range(i+1, N):
            if labels[i] < labels[j]:
                terms.append(torch.relu(scores[i] - scores[j] + margin))
            elif labels[j] < labels[i]:
                terms.append(torch.relu(scores[j] - scores[i] + margin))
    return torch.stack(terms).mean() if terms else scores.sum() * 0.0

def listmle_top1_loss(scores: torch.Tensor, labels: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    top_idx = torch.argmin(labels).view(1)
    return F.cross_entropy((scores / temperature).unsqueeze(0), top_idx)

def make_loss_fn(loss_type: str):
    if loss_type == 'ranknet':
        return ranknet_pairwise_loss
    if loss_type == 'hinge':
        return lambda s, y: hinge_pairwise_loss(s, y, margin=1.0)
    if loss_type == 'listmle':
        return listmle_top1_loss
    raise ValueError(f"Unknown loss_type: {loss_type}")

# ============================== Model ==============================
def _bbox_to_geom_feats_page_norm(bboxes: torch.Tensor) -> torch.Tensor:
    """整页归一化的几何特征"""
    x0, y0, x1, y1 = [bboxes[:, i] for i in range(4)]
    w = (x1 - x0).clamp(min=1.0)
    h = (y1 - y0).clamp(min=1.0)
    cx = (x0 + x1) * 0.5
    cy = (y0 + y1) * 0.5
    
    eps = 1e-6
    cxn = (cx - cx.min()) / (cx.max() - cx.min() + eps) * 2 - 1
    cyn = (cy - cy.min()) / (cy.max() - cy.min() + eps) * 2 - 1
    
    feats = torch.stack([
        cxn, cyn,
        torch.log(w), torch.log(h),
        torch.log(w / h),
        torch.log(w * h)
    ], dim=1)
    return feats  # [M, 6]

class SharedBackbone(nn.Module):
    def __init__(self, in_dim: int, hidden: int, drop: float):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(drop),
            nn.Linear(hidden, hidden), nn.ReLU()
        )
    
    def forward(self, x):
        return self.mlp(x)

class JointModelV4(nn.Module):
    def __init__(self, hidden_dim=256, num_heads=4, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.sent_in_dim = 6 + 4  # geom + dir
        self.char_in_dim = 6 + 2  # geom + dir

        self.sent_backbone = SharedBackbone(self.sent_in_dim, hidden_dim, dropout)
        self.sent_self_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.sent_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 1)
        )

        self.char_backbone = SharedBackbone(self.char_in_dim, hidden_dim, dropout)
        self.char_self_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.char_cross_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )

        self.fuse = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.char_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        sb = batch["sentence_bboxes"]
        sd = batch["sentence_dir_feat"]
        cb = batch["char_bboxes"]
        cd = batch["char_dir_feat"]
        cs = batch["char_to_sentence_idx"]

        device = sb.device
        S = sb.shape[0]
        N = cb.shape[0]

        # Sentence branch
        s_geom = _bbox_to_geom_feats_page_norm(sb)
        s_in = torch.cat([s_geom, sd], dim=1)
        s_feat = self.sent_backbone(s_in).unsqueeze(0)
        s_out, _ = self.sent_self_attn(s_feat, s_feat, s_feat)
        s_out = s_out.squeeze(0)
        sent_scores = self.sent_head(s_out).squeeze(-1)

        # Character branch
        c_geom = _bbox_to_geom_feats_page_norm(cb)
        c_in = torch.cat([c_geom, cd], dim=1)
        c_feat = self.char_backbone(c_in).unsqueeze(0)
        c_self, _ = self.char_self_attn(c_feat, c_feat, c_feat)

        # Masked cross-attention
        assert cs.min() >= 0 and cs.max() < S, \
            f"char_to_sentence_idx out of range: min={int(cs.min())}, max={int(cs.max())}, S={S}"
        
        attn_mask = torch.full((N, S), float('-inf'), device=device)
        for s_id in range(S):
            idx = (cs == s_id).nonzero(as_tuple=False).squeeze(-1)
            if idx.numel() > 0:
                attn_mask[idx, s_id] = 0.0

        c_cross, _ = self.char_cross_attn(
            query=c_self,
            key=s_out.unsqueeze(0),
            value=s_out.unsqueeze(0),
            attn_mask=attn_mask
        )

        c_fused = self.fuse(torch.cat([c_self.squeeze(0), c_cross.squeeze(0)], dim=1))
        char_scores = self.char_head(c_fused).squeeze(-1)

        return {"sent_scores": sent_scores, "char_scores": char_scores}

# ============================== Evaluation ==============================
@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    sent_list = []
    intra_list = []
    glob_list = []
    
    for batch in tqdm(loader, desc='Evaluating', leave=False):
        batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) 
                 for k, v in batch.items()}
        out = model(batch)
        
        sent_scores = out['sent_scores'].detach().cpu()
        char_scores = out['char_scores'].detach().cpu()
        sl = batch['sentence_labels'].detach().cpu()
        cl = batch['char_labels'].detach().cpu()
        cs = batch['char_to_sentence_idx'].detach().cpu()

        # Sentence-level metrics
        s_idx = torch.argsort(sent_scores, dim=0)
        s_rank = torch.empty_like(s_idx)
        s_rank[s_idx] = torch.arange(len(s_idx))
        m_s = compute_five_metrics(sl.numpy(), s_rank.numpy())
        sent_list.append(m_s)

        # Intra-sentence metrics (weighted)
        acc = {
            'kendall_tau': 0.0, 'spearman_rho': 0.0,
            'top1_accuracy': 0.0, 'top3_accuracy': 0.0,
            'pairwise_accuracy': 0.0
        }
        weight = 0
        
        for sid in torch.unique(cs).tolist():
            idx = (cs == sid).nonzero(as_tuple=False).squeeze(-1)
            if idx.numel() <= 1:
                continue
            
            c_sorted = torch.argsort(char_scores[idx], dim=0)
            c_rank = torch.empty_like(c_sorted)
            c_rank[c_sorted] = torch.arange(len(c_sorted))
            
            cl_grp = cl[idx]
            cl_rank = torch.argsort(torch.argsort(cl_grp))
            
            mg = compute_five_metrics(cl_rank.numpy(), c_rank.numpy())
            w = int(idx.numel())
            weight += w
            for k in acc:
                acc[k] += mg[k] * w
        
        m_intra = {k: (acc[k] / weight if weight > 0 else 0.0) for k in acc}
        intra_list.append(m_intra)

        # Global composition
        s_sorted = torch.argsort(sent_scores, dim=0)
        pred_idx = []
        for sid in s_sorted.tolist():
            idx = (cs == sid).nonzero(as_tuple=False).squeeze(-1)
            if idx.numel() == 0:
                continue
            c_sorted = torch.argsort(char_scores[idx], dim=0)
            pred_idx.extend(idx[c_sorted].tolist())
        
        glob_rank = torch.empty_like(cl)
        glob_rank[pred_idx] = torch.arange(len(pred_idx))
        m_g = compute_five_metrics(cl.numpy(), glob_rank.numpy())
        glob_list.append(m_g)

    def _avg(L: List[Dict[str, float]]):
        keys = L[0].keys()
        return {k: float(np.mean([x[k] for x in L])) for k in keys}
    
    out = {}
    out.update({f"sentence/{k}": v for k, v in _avg(sent_list).items()})
    out.update({f"intra/{k}": v for k, v in _avg(intra_list).items()})
    out.update({f"global/{k}": v for k, v in _avg(glob_list).items()})
    return out

# ============================== Training ==============================
@dataclass
class TrainConfig:
    data_dir: str
    device: str = 'cuda'
    hidden_dim: int = 256
    num_heads: int = 4
    dropout: float = 0.1
    epochs: int = 60
    stage1_epochs: int = 8
    loss_type: str = 'ranknet'
    lambda_max: float = 0.7
    lambda_warmup: int = 6
    lambda_schedule: str = 'linear'
    lr: float = 1e-3
    weight_decay: float = 1e-5
    grad_clip: float = 1.0
    patience: int = 10
    use_direction: bool = True
    save_dir: str = "/home/zhengxiaoying/DBManuscripts/reading_order_project/checkpoints"

def dynamic_lambda_linear(epoch, base_epoch, lambda_max, warmup):
    if epoch < base_epoch:
        return 0.0
    t = min(max(epoch - base_epoch + 1, 0), warmup)
    return float(lambda_max * t / max(warmup, 1))

def dynamic_lambda_cosine(epoch, base_epoch, lambda_max, total_epochs):
    if epoch < base_epoch:
        return 0.0
    prog = (epoch - base_epoch) / max(total_epochs - base_epoch, 1)
    return float(lambda_max * 0.5 * (1 - math.cos(math.pi * prog)))

def train(cfg: TrainConfig):
    device = torch.device(cfg.device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print("\n✅ Using human-annotated sentence_id and sentence_index")

    train_set = JointDataset(cfg.data_dir, 'train', use_direction=cfg.use_direction)
    val_set = JointDataset(cfg.data_dir, 'val', use_direction=cfg.use_direction)
    test_set = JointDataset(cfg.data_dir, 'test', use_direction=cfg.use_direction)
    
    train_loader = DataLoader(train_set, batch_size=1, shuffle=True, collate_fn=joint_collate_fn)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, collate_fn=joint_collate_fn)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, collate_fn=joint_collate_fn)

    model = JointModelV4(cfg.hidden_dim, cfg.num_heads, cfg.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    loss_fn = make_loss_fn(cfg.loss_type)

    save_dir = Path(cfg.save_dir) / f"joint_v4_corrected_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    save_dir.mkdir(parents=True, exist_ok=True)
    best_tau = -1.0

    # ============ Stage I: Sentence-only pretrain ============
    print("\n===== Stage I: Sentence-only pretrain =====")
    for ep in range(1, cfg.stage1_epochs + 1):
        model.train()
        ep_loss = 0.0
        
        for batch in tqdm(train_loader, desc=f"Stage I Epoch {ep}/{cfg.stage1_epochs}"):
            batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) 
                     for k, v in batch.items()}
            
            optimizer.zero_grad()
            out = model(batch)
            loss = loss_fn(out['sent_scores'], batch['sentence_labels'])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            ep_loss += float(loss.item())
        
        scheduler.step()
        print(f"  Stage I avg loss: {ep_loss / max(len(train_loader), 1):.4f}")

    # ============ Stage II: Joint training ============
    print("\n===== Stage II: Joint training =====")
    for ep in range(cfg.stage1_epochs + 1, cfg.epochs + 1):
        model.train()
        tot, s_l, c_l = 0.0, 0.0, 0.0

        # λ schedule
        if cfg.lambda_schedule == 'cosine':
            lam = dynamic_lambda_cosine(ep, cfg.stage1_epochs + 1, cfg.lambda_max, cfg.epochs)
        else:
            lam = dynamic_lambda_linear(ep, cfg.stage1_epochs + 1, cfg.lambda_max, cfg.lambda_warmup)

        for batch in tqdm(train_loader, desc=f"Stage II Epoch {ep}/{cfg.epochs} (λ={lam:.3f})"):
            batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) 
                     for k, v in batch.items()}
            
            optimizer.zero_grad()
            out = model(batch)

            loss_s = loss_fn(out['sent_scores'], batch['sentence_labels'])
            loss_c = loss_fn(out['char_scores'], batch['char_labels'])
            loss = loss_s + lam * loss_c

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()

            tot += float(loss.item())
            s_l += float(loss_s.item())
            c_l += float(loss_c.item())

        scheduler.step()
        
        nb = max(len(train_loader), 1)
        print(f"  Train: total {tot/nb:.4f} | sent {s_l/nb:.4f} | char {c_l/nb:.4f}")

        # Validation
        val = evaluate(model, val_loader, device)
        print("  Val Metrics:", json.dumps(val, ensure_ascii=False))
        
        val_tau = float(val.get('global/kendall_tau', -1.0))
        if val_tau > best_tau:
            best_tau = val_tau
            torch.save({
                'epoch': ep,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_metrics': val,
                'config': cfg.__dict__
            }, save_dir / 'best_model.pth')
            print(f"  ✓ Best model saved (Val global τ={best_tau:.4f})")

    # ============ Final Test ============
    print("\n===== Final Test =====")
    ckpt = torch.load(save_dir / 'best_model.pth', map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    test = evaluate(model, test_loader, device)
    print("[Test Set]", json.dumps(test, ensure_ascii=False, indent=2))
    
    with open(save_dir / 'results.json', 'w') as f:
        json.dump({
            'model': 'JointModelV4 (Human Annotations)',
            'val_best': ckpt['val_metrics'],
            'test': test,
            'config': cfg.__dict__
        }, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Results saved to {save_dir}")

# ============================== Main ==============================
if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', type=str,
                   default='/home/zhengxiaoying/DBManuscripts/dongba_data_splits_by_page')
    ap.add_argument('--device', type=str, default='cuda')
    ap.add_argument('--hidden_dim', type=int, default=256)
    ap.add_argument('--num_heads', type=int, default=4)
    ap.add_argument('--dropout', type=float, default=0.1)
    ap.add_argument('--epochs', type=int, default=60)
    ap.add_argument('--stage1_epochs', type=int, default=8)
    ap.add_argument('--loss_type', type=str, 
                   choices=['ranknet', 'hinge', 'listmle'], 
                   default='ranknet')
    ap.add_argument('--lambda_max', type=float, default=0.7)
    ap.add_argument('--lambda_warmup', type=int, default=6)
    ap.add_argument('--lambda_schedule', type=str, 
                   choices=['linear', 'cosine'], 
                   default='linear')
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--weight_decay', type=float, default=1e-5)
    ap.add_argument('--grad_clip', type=float, default=1.0)
    ap.add_argument('--patience', type=int, default=10)
    ap.add_argument('--use_direction', action='store_true')
    ap.add_argument('--save_dir', type=str,
                   default='/home/zhengxiaoying/DBManuscripts/reading_order_project/checkpoints')
    
    args = ap.parse_args()
    cfg = TrainConfig(**vars(args))
    train(cfg)