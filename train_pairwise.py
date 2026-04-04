#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

使用示例:
    python train_pairwise_fixed.py --use_direction --epochs 60
"""

from __future__ import annotations
import argparse, json, math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List

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
    
    def __init__(self, data_dir: str, split: str = 'train', use_direction: bool = True):
        import pandas as pd
        csv_path = f"{data_dir}/{split}.csv"
        print(f"Loading {split} data from {csv_path}...")
        df = pd.read_csv(csv_path)
        
        self.split = split
        self.use_direction = use_direction
        
        need = {'page_key', 'sentence_id', 'sentence_index', 'x0', 'y0', 'x1', 'y1'}
        miss = need - set(df.columns)
        if miss:
            raise ValueError(f"Missing required columns: {miss}")

        sid_all_sorted = sorted(df['sentence_id'].unique())
        sid2rank_global = {sid: i for i, sid in enumerate(sid_all_sorted)}
        df['sid_rank'] = df['sentence_id'].map(sid2rank_global).astype(int)

        self.pages: List[Dict] = []

        for page_key, g in df.groupby('page_key'):
            g = g.copy()
            sent_items = []
            
            for sid_r, sg in g.groupby('sid_rank'):
                sg = sg.copy()
                sg = sg.sort_values('sentence_index').copy()
                sg['sentence_index'] = range(len(sg))

                x0 = sg['x0'].min()
                y0 = sg['y0'].min()
                x1 = sg['x1'].max()
                y1 = sg['y1'].max()
                s_box = np.array([x0, y0, x1, y1], dtype=np.float32)

                # 计算PCA方向特征（与JointModelV4完全相同）
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

                # 字符级方向特征
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
                    sentence_id=int(sg['sentence_id'].iloc[0]),
                    s_box=s_box,
                    s_dir=s_dir,
                    sg=sg,
                    char_dir=char_dir
                ))

            S = len(sent_items)
            sentence_bboxes = []
            sentence_dir_feat = []
            sentence_ids = []
            
            for it in sent_items:
                sentence_bboxes.append(it['s_box'].tolist())
                sentence_dir_feat.append(it['s_dir'])
                sentence_ids.append(it['sentence_id'])
            
            sentence_bboxes = torch.tensor(sentence_bboxes, dtype=torch.float32)
            sentence_dir_feat = torch.tensor(sentence_dir_feat, dtype=torch.float32)
            sentence_labels = torch.tensor(sentence_ids, dtype=torch.long)

            # 按人工标注顺序生成字符数据
            sent_items_sorted = sorted(enumerate(sent_items), 
                                       key=lambda x: sentence_ids[x[0]])
            
            char_bboxes_list = []
            char_to_sentence_idx_list = []
            char_dir_blocks = []
            
            for orig_idx, it in sent_items_sorted:
                sg = it['sg']
                for _, row in sg.iterrows():
                    char_bboxes_list.append([
                        float(row['x0']), float(row['y0']),
                        float(row['x1']), float(row['y1'])
                    ])
                    char_to_sentence_idx_list.append(orig_idx)
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

            char_labels = torch.arange(char_bboxes.shape[0], dtype=torch.long)

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

        print(f"  ✓ Loaded {len(self.pages)} pages")
        if split == 'train':
            print("  ⚠️  Training set: will shuffle inputs to prevent order leakage")

    def __len__(self):
        return len(self.pages)
    
    def __getitem__(self, idx):
        page = {k: v.clone() if isinstance(v, torch.Tensor) else v 
                for k, v in self.pages[idx].items()}
        
        if self.split == 'train':
            # 打乱句子
            S = page['sentence_bboxes'].shape[0]
            if S > 1:
                sent_perm = torch.randperm(S)
                page['sentence_bboxes'] = page['sentence_bboxes'][sent_perm]
                page['sentence_dir_feat'] = page['sentence_dir_feat'][sent_perm]
                page['sentence_labels'] = page['sentence_labels'][sent_perm]  # 直接重排
                
                # 更新char_to_sentence_idx映射
                old_to_new = torch.empty(S, dtype=torch.long)
                old_to_new[sent_perm] = torch.arange(S)
                page['char_to_sentence_idx'] = old_to_new[page['char_to_sentence_idx']]
            
            # 打乱字符
            N = page['char_bboxes'].shape[0]
            if N > 1:
                char_perm = torch.randperm(N)
                page['char_bboxes'] = page['char_bboxes'][char_perm]
                page['char_dir_feat'] = page['char_dir_feat'][char_perm]
                page['char_labels'] = page['char_labels'][char_perm]  # 直接重排
                page['char_to_sentence_idx'] = page['char_to_sentence_idx'][char_perm]
        
        return page


def collate_fn(batch):
    return batch[0]


# ============================== Model ==============================
def _bbox_to_geom_feats_page_norm(bboxes: torch.Tensor) -> torch.Tensor:
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


class PairwiseRankingModel(nn.Module):
    
    def __init__(self, hidden_dim=256, dropout=0.1, use_direction=True):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.use_direction = use_direction
        
        # 与JointModelV4相同的特征维度
        self.sent_in_dim = 6 + (4 if use_direction else 0)
        self.char_in_dim = 6 + (2 if use_direction else 0)
        
        # 句子级比较器
        self.sent_comparator = nn.Sequential(
            nn.Linear(self.sent_in_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # 字符级比较器
        self.char_comparator = nn.Sequential(
            nn.Linear(self.char_in_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, batch: Dict) -> Dict:
        """
        输出：
          sent_scores: [S] - 句子排序分数
          char_scores: [N] - 字符排序分数
        """
        sb = batch['sentence_bboxes']
        sd = batch['sentence_dir_feat']
        cb = batch['char_bboxes']
        cd = batch['char_dir_feat']
        
        device = sb.device
        S = sb.shape[0]
        N = cb.shape[0]
        
        # 提取几何特征（与JointModelV4相同）
        s_geom = _bbox_to_geom_feats_page_norm(sb)
        c_geom = _bbox_to_geom_feats_page_norm(cb)
        
        # 拼接方向特征
        if self.use_direction:
            s_feat = torch.cat([s_geom, sd], dim=1)  # [S, 10]
            c_feat = torch.cat([c_geom, cd], dim=1)  # [N, 8]
        else:
            s_feat = s_geom
            c_feat = c_geom
        
        # 句子级：两两比较累积分数
        sent_scores = torch.zeros(S, device=device)
        if S > 1:
            for i in range(S):
                for j in range(S):
                    if i != j:
                        pair_feat = torch.cat([s_feat[i], s_feat[j]])
                        score_ij = self.sent_comparator(pair_feat).squeeze()
                        sent_scores[i] += score_ij
        
        # 字符级：两两比较累积分数
        char_scores = torch.zeros(N, device=device)
        if N > 1:
            for i in range(N):
                for j in range(N):
                    if i != j:
                        pair_feat = torch.cat([c_feat[i], c_feat[j]])
                        score_ij = self.char_comparator(pair_feat).squeeze()
                        char_scores[i] += score_ij
        
        return {
            'sent_scores': sent_scores,
            'char_scores': char_scores
        }


# ============================== Loss ==============================
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


# ============================== Evaluation ==============================
@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    """✅ 与JointModelV4完全相同的三级评估"""
    model.eval()
    sent_list, intra_list, glob_list = [], [], []
    
    for batch in tqdm(loader, desc='Evaluating', leave=False):
        batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) 
                 for k, v in batch.items()}
        out = model(batch)
        
        sent_scores = out['sent_scores'].detach().cpu()
        char_scores = out['char_scores'].detach().cpu()
        sl = batch['sentence_labels'].detach().cpu()
        cl = batch['char_labels'].detach().cpu()
        cs = batch['char_to_sentence_idx'].detach().cpu()

        # Sentence-level
        s_idx = torch.argsort(sent_scores, dim=0)
        s_rank = torch.empty_like(s_idx)
        s_rank[s_idx] = torch.arange(len(s_idx))
        m_s = compute_five_metrics(sl.numpy(), s_rank.numpy())
        sent_list.append(m_s)

        # Intra-sentence
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

        # Global
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

    def _avg(L):
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
    dropout: float = 0.1
    epochs: int = 60  
    lr: float = 1e-3
    weight_decay: float = 1e-5
    grad_clip: float = 1.0
    use_direction: bool = True
    save_dir: str = "/home/zhengxiaoying/DBManuscripts/DongbaHRO/checkpoints"


def train(cfg: TrainConfig):
    device = torch.device(cfg.device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print("\n✅ Pairwise Ranking Model (Fixed)")
    print(f"  - Using human annotations")
    print(f"  - Hidden dim: {cfg.hidden_dim}")
    print(f"  - Direction features: {cfg.use_direction}")
    print(f"  - Epochs: {cfg.epochs}")

    train_set = JointDataset(cfg.data_dir, 'train', use_direction=cfg.use_direction)
    val_set = JointDataset(cfg.data_dir, 'val', use_direction=cfg.use_direction)
    test_set = JointDataset(cfg.data_dir, 'test', use_direction=cfg.use_direction)
    
    train_loader = DataLoader(train_set, batch_size=1, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, collate_fn=collate_fn)

    model = PairwiseRankingModel(
        hidden_dim=cfg.hidden_dim,
        dropout=cfg.dropout,
        use_direction=cfg.use_direction
    ).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)

    save_dir = Path(cfg.save_dir) / f"pairwise_fixed_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    save_dir.mkdir(parents=True, exist_ok=True)
    best_tau = -1.0

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        ep_loss = 0.0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{cfg.epochs}"):
            batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) 
                     for k, v in batch.items()}
            
            optimizer.zero_grad()
            out = model(batch)
            
            loss_s = ranknet_pairwise_loss(out['sent_scores'], batch['sentence_labels'])
            loss_c = ranknet_pairwise_loss(out['char_scores'], batch['char_labels'])
            loss = loss_s + loss_c
            
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            ep_loss += float(loss.item())
        
        scheduler.step()
        print(f"  Train loss: {ep_loss / max(len(train_loader), 1):.4f}")

        val = evaluate(model, val_loader, device)
        print("  Val Metrics:", json.dumps(val, ensure_ascii=False))
        
        val_tau = float(val.get('global/kendall_tau', -1.0))
        if val_tau > best_tau:
            best_tau = val_tau
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_metrics': val,
                'config': cfg.__dict__
            }, save_dir / 'best_model.pth')
            print(f"  ✓ Best model saved (Val global τ={best_tau:.4f})")

    print("\n===== Final Test =====")
    ckpt = torch.load(save_dir / 'best_model.pth', map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    test = evaluate(model, test_loader, device)
    print("[Test Set]", json.dumps(test, ensure_ascii=False, indent=2))
    
    with open(save_dir / 'results.json', 'w') as f:
        json.dump({
            'model': 'PairwiseRanking (Fixed)',
            'val_best': ckpt['val_metrics'],
            'test': test,
            'config': cfg.__dict__
        }, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Results saved to {save_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str,
                        default='/home/zhengxiaoying/DBManuscripts/DongbaHRO')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--use_direction', action='store_true',
                        help='使用PCA方向特征')
    parser.add_argument('--save_dir', type=str,
                        default='/home/zhengxiaoying/DBManuscripts/DongbaHRO/checkpoints')
    
    args = parser.parse_args()
    cfg = TrainConfig(**vars(args))
    train(cfg)


if __name__ == "__main__":
    main()