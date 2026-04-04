#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
import argparse
import json
import math
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
class TransformerDataset(Dataset):

    def __init__(self, data_dir: str, split: str = 'train', 
                 mode: str = 'character', use_direction: bool = True):
        import pandas as pd
        assert mode in ['sentence', 'character'], f"mode must be 'sentence' or 'character', got {mode}"
        
        self.mode = mode
        csv_path = f"{data_dir}/{split}.csv"
        print(f"Loading {split} data from {csv_path} (mode={mode})...")
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
            
            if mode == 'sentence':
                sent_items = []
                for sid_r, sg in g.groupby('sid_rank'):
                    sg = sg.sort_values('sentence_index').copy()
                    
                    # 句框 = union
                    x0 = sg['x0'].min()
                    y0 = sg['y0'].min()
                    x1 = sg['x1'].max()
                    y1 = sg['y1'].max()
                    s_box = np.array([x0, y0, x1, y1], dtype=np.float32)
                    
                    # 方向特征
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
                    
                    ang = math.atan2(u[1], u[0])
                    s_dir = [float(u[0]), float(u[1]), math.sin(ang), math.cos(ang)]
                    
                    sent_items.append({
                        'bbox': s_box,
                        'dir_feat': s_dir,
                        'sentence_id': int(sg['sentence_id'].iloc[0])
                    })
                
                # 构建句子级别的batch
                bboxes = torch.tensor([it['bbox'] for it in sent_items], dtype=torch.float32)
                dir_feats = torch.tensor([it['dir_feat'] for it in sent_items], dtype=torch.float32)
                labels = torch.tensor([it['sentence_id'] for it in sent_items], dtype=torch.long)
                
                self.pages.append({
                    'page_key': page_key,
                    'bboxes': bboxes,
                    'dir_feats': dir_feats,
                    'labels': labels,
                    'mode': 'sentence'
                })
            
            else:  # character模式
                for sid_r, sg in g.groupby('sid_rank'):
                    sg = sg.copy()
                    sg = sg.sort_values('sentence_index').copy()
                    sg['sentence_index'] = range(len(sg))
                    
                    # 计算方向特征
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
                        u_perp = np.array([-u[1], u[0]], dtype=np.float32)
                        
                        proj_t = (pts - ctr) @ u
                        proj_p = (pts - ctr) @ u_perp
                        vmax_t = np.max(np.abs(proj_t)) + 1e-6
                        vmax_p = np.max(np.abs(proj_p)) + 1e-6
                        t_norm = (proj_t / vmax_t).astype(np.float32)
                        p_norm = (proj_p / vmax_p).astype(np.float32)
                    else:
                        t_norm = np.zeros(len(sg), dtype=np.float32)
                        p_norm = np.zeros(len(sg), dtype=np.float32)
                    
                    char_dir = np.stack([t_norm, p_norm], axis=1)
                    
                    sent_items.append({
                        'sid_input': len(sent_items),
                        'sentence_id': int(sg['sentence_id'].iloc[0]),
                        'sg': sg,
                        'char_dir': char_dir
                    })
   
                sent_items_sorted = sorted(enumerate(sent_items), 
                                          key=lambda x: x[1]['sentence_id'])
                
                char_bboxes_list = []
                char_to_sent_list = []
                char_dir_blocks = []
                
                for orig_idx, it in sent_items_sorted:
                    sg = it['sg']
                    
                    for _, row in sg.iterrows():
                        char_bboxes_list.append([
                            float(row['x0']), float(row['y0']),
                            float(row['x1']), float(row['y1'])
                        ])
                        char_to_sent_list.append(orig_idx)  # 指向原始输入位置
                    
                    char_dir_blocks.append(it['char_dir'])
                
                bboxes = torch.tensor(char_bboxes_list, dtype=torch.float32)
                char_to_sent = torch.tensor(char_to_sent_list, dtype=torch.long)
                
                if use_direction and len(char_dir_blocks) > 0:
                    dir_feats = torch.tensor(
                        np.concatenate(char_dir_blocks, axis=0), 
                        dtype=torch.float32
                    )
                else:
                    dir_feats = torch.zeros((bboxes.shape[0], 2), dtype=torch.float32)

                labels = torch.arange(bboxes.shape[0], dtype=torch.long)
                
                self.pages.append({
                    'page_key': page_key,
                    'bboxes': bboxes,
                    'dir_feats': dir_feats,
                    'labels': labels,
                    'char_to_sent': char_to_sent,
                    'mode': 'character'
                })

        print(f"  ✓ Loaded {len(self.pages)} pages (mode={mode}, aligned with JointModelV4)")

    def __len__(self):
        return len(self.pages)
    
    def __getitem__(self, idx):
        return self.pages[idx]

def transformer_collate_fn(batch: List[Dict]):
    return batch[0]

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
    return feats

class TransformerBaseline(nn.Module):
    def __init__(self, mode: str = 'character', hidden_dim: int = 256, 
                 num_layers: int = 3, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.mode = mode
        self.hidden_dim = hidden_dim
        
        # 输入维度
        if mode == 'sentence':
            self.in_dim = 6 + 4  # geom + dir
        else:  # character
            self.in_dim = 6 + 2  # geom + dir
        
        # 输入投影
        self.input_proj = nn.Sequential(
            nn.Linear(self.in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation='relu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 输出头
        self.output_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        bboxes = batch['bboxes']
        dir_feats = batch['dir_feats']
        
        # 特征编码
        geom_feats = _bbox_to_geom_feats_page_norm(bboxes)
        input_feats = torch.cat([geom_feats, dir_feats], dim=1)
        
        # [N, in_dim] -> [1, N, hidden_dim]
        x = self.input_proj(input_feats).unsqueeze(0)
        
        # Transformer编码
        x = self.transformer(x)  # [1, N, hidden_dim]
        
        # 预测分数
        scores = self.output_head(x.squeeze(0)).squeeze(-1)  # [N]
        
        return {'scores': scores}

# ============================== Evaluation==============================
@torch.no_grad()
def evaluate_transformer(model: nn.Module, loader: DataLoader, 
                         device: torch.device, mode: str) -> Dict[str, float]:
    model.eval()
    
    if mode == 'sentence':
        sent_list = []
        
        for batch in tqdm(loader, desc='Evaluating', leave=False):
            batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) 
                     for k, v in batch.items()}
            out = model(batch)
            
            scores = out['scores'].detach().cpu()
            labels = batch['labels'].detach().cpu()
            
            # Sentence-level metrics
            s_idx = torch.argsort(scores, dim=0)
            s_rank = torch.empty_like(s_idx)
            s_rank[s_idx] = torch.arange(len(s_idx))
            m_s = compute_five_metrics(labels.numpy(), s_rank.numpy())
            sent_list.append(m_s)
        
        def _avg(L: List[Dict[str, float]]):
            keys = L[0].keys()
            return {k: float(np.mean([x[k] for x in L])) for k in keys}
        
        out = {}
        out.update({f"sentence/{k}": v for k, v in _avg(sent_list).items()})
        return out
    
    else:  # character模式
        sent_list = []
        intra_list = []
        glob_list = []
        
        for batch in tqdm(loader, desc='Evaluating', leave=False):
            batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) 
                     for k, v in batch.items()}
            out = model(batch)
            
            char_scores = out['scores'].detach().cpu()
            char_labels = batch['labels'].detach().cpu()
            char_to_sent = batch['char_to_sent'].detach().cpu()
            
            # Sentence-level metrics（通过聚合字符分数）
            sent_scores_list = []
            sent_labels_list = []
            
            for sid in torch.unique(char_to_sent).tolist():
                idx = (char_to_sent == sid).nonzero(as_tuple=False).squeeze(-1)
                if idx.numel() == 0:
                    continue
                # 句子分数 = 该句子所有字符的平均分数
                sent_score = float(char_scores[idx].mean().item())
                sent_scores_list.append(sent_score)
                # 句子标签 = 该句子第一个字符的标签
                sent_label = int(char_labels[idx].min().item())
                sent_labels_list.append(sent_label)
            
            if len(sent_scores_list) > 0:
                sent_scores_arr = np.array(sent_scores_list)
                sent_labels_arr = np.array(sent_labels_list)
                
                s_idx = np.argsort(sent_scores_arr)
                s_rank = np.empty_like(s_idx)
                s_rank[s_idx] = np.arange(len(s_idx))
                m_s = compute_five_metrics(sent_labels_arr, s_rank)
                sent_list.append(m_s)
            
            # Intra-sentence metrics (weighted)
            acc = {
                'kendall_tau': 0.0, 'spearman_rho': 0.0,
                'top1_accuracy': 0.0, 'top3_accuracy': 0.0,
                'pairwise_accuracy': 0.0
            }
            weight = 0
            
            for sid in torch.unique(char_to_sent).tolist():
                idx = (char_to_sent == sid).nonzero(as_tuple=False).squeeze(-1)
                if idx.numel() <= 1:
                    continue
                
                c_sorted = torch.argsort(char_scores[idx], dim=0)
                c_rank = torch.empty_like(c_sorted)
                c_rank[c_sorted] = torch.arange(len(c_sorted))
                
                cl_grp = char_labels[idx]
                cl_rank = torch.argsort(torch.argsort(cl_grp))
                
                mg = compute_five_metrics(cl_rank.numpy(), c_rank.numpy())
                w = int(idx.numel())
                weight += w
                for k in acc:
                    acc[k] += mg[k] * w
            
            m_intra = {k: (acc[k] / weight if weight > 0 else 0.0) for k in acc}
            intra_list.append(m_intra)
            
            # Global metrics
            c_idx = torch.argsort(char_scores, dim=0)
            c_rank = torch.empty_like(c_idx)
            c_rank[c_idx] = torch.arange(len(c_idx))
            m_g = compute_five_metrics(char_labels.numpy(), c_rank.numpy())
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
class TransformerConfig:
    data_dir: str
    mode: str = 'character'
    device: str = 'cuda'
    hidden_dim: int = 256
    num_layers: int = 3
    num_heads: int = 4
    dropout: float = 0.1
    epochs: int = 60
    lr: float = 1e-3
    weight_decay: float = 1e-5
    grad_clip: float = 1.0
    use_direction: bool = True
    save_dir: str = "/home/zhengxiaoying/DBManuscripts/DongbaHRO/checkpoints"

def train_transformer(cfg: TransformerConfig):
    device = torch.device(cfg.device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"\n✅ Training Transformer Baseline ({cfg.mode} mode)")
    print("✅ Data processing: ALIGNED with JointModelV4")
    
    # 数据集
    train_set = TransformerDataset(cfg.data_dir, 'train', cfg.mode, cfg.use_direction)
    val_set = TransformerDataset(cfg.data_dir, 'val', cfg.mode, cfg.use_direction)
    test_set = TransformerDataset(cfg.data_dir, 'test', cfg.mode, cfg.use_direction)
    
    train_loader = DataLoader(train_set, batch_size=1, shuffle=True, 
                              collate_fn=transformer_collate_fn)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, 
                           collate_fn=transformer_collate_fn)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, 
                            collate_fn=transformer_collate_fn)
    
    # 模型
    model = TransformerBaseline(
        mode=cfg.mode,
        hidden_dim=cfg.hidden_dim,
        num_layers=cfg.num_layers,
        num_heads=cfg.num_heads,
        dropout=cfg.dropout
    ).to(device)
    
    # 优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, 
                                 weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    
    # 保存目录
    save_dir = Path(cfg.save_dir) / f"transformer_{cfg.mode}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    save_dir.mkdir(parents=True, exist_ok=True)
    
    best_tau = -1.0
    patience_counter = 0
    patience = 10
    
    print("\n===== Training =====")
    for ep in range(1, cfg.epochs + 1):
        model.train()
        ep_loss = 0.0
        
        for batch in tqdm(train_loader, desc=f"Epoch {ep}/{cfg.epochs}"):
            batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) 
                     for k, v in batch.items()}
            
            optimizer.zero_grad()
            out = model(batch)
            loss = ranknet_pairwise_loss(out['scores'], batch['labels'])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            
            ep_loss += float(loss.item())
        
        scheduler.step()
        avg_loss = ep_loss / max(len(train_loader), 1)
        print(f"  Epoch {ep}: avg loss = {avg_loss:.4f}")
        
        # 验证
        val_metrics = evaluate_transformer(model, val_loader, device, cfg.mode)
        print("  Val Metrics:", json.dumps(val_metrics, ensure_ascii=False))
        
        # 保存最佳模型
        if cfg.mode == 'character':
            val_tau = float(val_metrics.get('global/kendall_tau', -1.0))
        else:
            val_tau = float(val_metrics.get('sentence/kendall_tau', -1.0))
        
        if val_tau > best_tau:
            best_tau = val_tau
            patience_counter = 0
            torch.save({
                'epoch': ep,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_metrics': val_metrics,
                'config': cfg.__dict__
            }, save_dir / 'best_model.pth')
            print(f"  ✓ Best model saved (Val τ={best_tau:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping triggered (patience={patience})")
                break
    
    # 测试
    print("\n===== Final Test =====")
    ckpt = torch.load(save_dir / 'best_model.pth', map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    test_metrics = evaluate_transformer(model, test_loader, device, cfg.mode)
    print("[Test Set]", json.dumps(test_metrics, ensure_ascii=False, indent=2))
    
    # 保存结果
    with open(save_dir / 'results.json', 'w') as f:
        json.dump({
            'model': f'TransformerBaseline ({cfg.mode})',
            'note': 'Data processing ALIGNED with JointModelV4',
            'val_best': ckpt['val_metrics'],
            'test': test_metrics,
            'config': cfg.__dict__
        }, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Results saved to {save_dir}")

# ============================== Main ==============================
if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='Train Transformer Baseline ')
    ap.add_argument('--data_dir', type=str,
                   default='/home/zhengxiaoying/DBManuscripts/DongbaHRO')
    ap.add_argument('--mode', type=str, choices=['sentence', 'character'], 
                   default='character',
                   help='Prediction granularity: sentence or character level')
    ap.add_argument('--device', type=str, default='cuda')
    ap.add_argument('--hidden_dim', type=int, default=256)
    ap.add_argument('--num_layers', type=int, default=3,
                   help='Number of Transformer encoder layers')
    ap.add_argument('--num_heads', type=int, default=4)
    ap.add_argument('--dropout', type=float, default=0.1)
    ap.add_argument('--epochs', type=int, default=60)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--weight_decay', type=float, default=1e-5)
    ap.add_argument('--grad_clip', type=float, default=1.0)
    ap.add_argument('--use_direction', action='store_true',
                   help='Use PCA direction features')
    ap.add_argument('--save_dir', type=str,
                   default='/home/zhengxiaoying/DBManuscripts/DongbaHRO/checkpoints')
    
    args = ap.parse_args()
    cfg = TransformerConfig(**vars(args))
    train_transformer(cfg)