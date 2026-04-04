#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用方法：
    python train_gnn_baseline.py --data_dir /path/to/data --use_direction --gnn_type gat
"""

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

def _centers_xyxy(b: np.ndarray) -> np.ndarray:
    return (b[:, :2] + b[:, 2:]) / 2.0

def sentence_rule_order_idx(centers: np.ndarray, rule: str = 'top_to_bottom') -> np.ndarray:
    if rule == 'top_to_bottom':
        return np.lexsort((-centers[:, 0], centers[:, 1]))
    elif rule == 'left_to_right':
        return np.lexsort((centers[:, 1], centers[:, 0]))
    elif rule == 'right_to_left':
        return np.lexsort((centers[:, 1], -centers[:, 0]))
    else:
        raise ValueError(f'Unknown rule: {rule}')

# ==================== 数据集====================
class JointDataset(Dataset):
    """与 train_joint_v4_fixed.py 完全相同的数据集"""
    def __init__(self, data_dir: str, split: str = 'train', 
                 use_direction: bool = True, rule_for_sent: str = 'top_to_bottom'):
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
            sent_items = []
            
            for sid_r, sg in g.groupby('sid_rank'):
                sg = sg.copy()
                idxs = sorted(sg['sentence_index'].astype(int).tolist())
                ok = (len(idxs) == len(sg) and idxs[0] == 0 and 
                      idxs[-1] == len(sg) - 1 and len(set(idxs)) == len(idxs))
                if not ok:
                    sg = sg.sort_values(['y0', 'x0', 'x1']).copy()
                    sg['sentence_index'] = range(len(sg))

                x0 = sg['x0'].min()
                y0 = sg['y0'].min()
                x1 = sg['x1'].max()
                y1 = sg['y1'].max()
                s_box = np.array([x0, y0, x1, y1], dtype=np.float32)

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
                    s_box=s_box,
                    s_dir=s_dir,
                    sg=sg,
                    char_dir=char_dir
                ))

            S = len(sent_items)
            sentence_bboxes = []
            sentence_dir_feat = []
            sentence_ids = []  # 收集人工标注的sentence_id
            for it in sent_items:
                sentence_bboxes.append(it['s_box'].tolist())
                sentence_dir_feat.append(it['s_dir'])
                # 获取该句子的sentence_id（人工标注的阅读顺序）
                sentence_ids.append(int(it['sg']['sentence_id'].iloc[0]))
            
            sentence_bboxes = torch.tensor(sentence_bboxes, dtype=torch.float32)
            sentence_dir_feat = torch.tensor(sentence_dir_feat, dtype=torch.float32)
            
            # 直接用人工标注的sentence_id作为句子顺序标签
            sentence_labels = torch.tensor(sentence_ids, dtype=torch.long)

            # 按人工标注的顺序（sentence_id升序）生成全局字符标签
            # 首先按sentence_id排序sent_items
            sent_items_sorted = sorted(enumerate(sent_items), 
                                       key=lambda x: sentence_ids[x[0]])
            
            char_bboxes_list = []
            char_to_sentence_idx_list = []
            char_dir_blocks = []
            
            for orig_idx, it in sent_items_sorted:
                sg = it['sg'].copy()
                # 按人工标注的sentence_index排序
                sg = sg.sort_values('sentence_index').copy()
                
                for _, row in sg.iterrows():
                    char_bboxes_list.append([
                        float(row['x0']), float(row['y0']),
                        float(row['x1']), float(row['y1'])
                    ])
                    char_to_sentence_idx_list.append(orig_idx)  # 指向原始输入位置
                
                # 方向特征仍然保留（作为额外特征）
                char_dir_blocks.append(it['char_dir'][np.argsort(sg['sentence_index'].values)])

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

            if char_to_sentence_idx.numel() > 0:
                cmin = int(char_to_sentence_idx.min().item())
                cmax = int(char_to_sentence_idx.max().item())
                assert 0 <= cmin and cmax < S

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

    def __len__(self):
        return len(self.pages)
    
    def __getitem__(self, idx):
        return self.pages[idx]

def joint_collate_fn(batch: List[Dict]):
    return batch[0]

# ==================== 图构建工具 ====================
def build_knn_graph(bboxes: torch.Tensor, k: int = 5) -> torch.Tensor:
    """
    基于K近邻构建图的边
    
    Args:
        bboxes: [N, 4] bbox坐标
        k: 每个节点连接k个最近邻居
    
    Returns:
        edge_index: [2, E] 边的索引
    """
    N = bboxes.shape[0]
    
    if N <= 1:
        return torch.zeros((2, 0), dtype=torch.long, device=bboxes.device)
    
    # 计算中心点
    centers = (bboxes[:, :2] + bboxes[:, 2:]) / 2.0  # [N, 2]
    
    # 计算距离矩阵
    dist = torch.cdist(centers, centers, p=2)  # [N, N]
    
    # 对角线设为无穷大（不自连）
    dist.fill_diagonal_(float('inf'))
    
    # 找到每个节点的k个最近邻
    k_actual = min(k, N - 1)
    _, indices = torch.topk(dist, k_actual, dim=1, largest=False)  # [N, k]
    
    # 构建边
    src = torch.arange(N, device=bboxes.device).unsqueeze(1).repeat(1, k_actual)  # [N, k]
    dst = indices  # [N, k]
    
    edge_index = torch.stack([src.flatten(), dst.flatten()], dim=0)  # [2, N*k]
    
    return edge_index


def build_spatial_graph(bboxes: torch.Tensor, threshold: float = 0.3) -> torch.Tensor:
    """
    基于空间邻近度构建图（连接距离小于阈值的节点）
    
    Args:
        bboxes: [N, 4] bbox坐标（归一化后）
        threshold: 距离阈值
    
    Returns:
        edge_index: [2, E]
    """
    N = bboxes.shape[0]
    
    if N <= 1:
        return torch.zeros((2, 0), dtype=torch.long, device=bboxes.device)
    
    centers = (bboxes[:, :2] + bboxes[:, 2:]) / 2.0
    dist = torch.cdist(centers, centers, p=2)
    
    # 找到距离小于阈值的节点对
    mask = (dist < threshold) & (dist > 0)  # 排除自连接
    edge_index = mask.nonzero(as_tuple=False).t()  # [2, E]
    
    return edge_index


# ==================== GNN 层 ====================
class GCNLayer(nn.Module):
    """图卷积层（GCN）"""
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [N, in_dim] 节点特征
            edge_index: [2, E] 边索引
        
        Returns:
            out: [N, out_dim] 更新后的节点特征
        """
        N = x.shape[0]
        
        if edge_index.shape[1] == 0:
            # 没有边，直接线性变换
            return self.linear(x)
        
        # 聚合邻居信息
        src, dst = edge_index
        
        # 计算度数（用于归一化）
        deg = torch.zeros(N, device=x.device)
        deg.scatter_add_(0, dst, torch.ones_like(dst, dtype=torch.float32))
        deg = deg.clamp(min=1.0)
        
        # 消息传递：dst节点聚合src节点的信息
        messages = torch.zeros_like(x)
        messages.scatter_add_(0, dst.unsqueeze(1).expand(-1, x.shape[1]), x[src])
        
        # 归一化
        messages = messages / deg.unsqueeze(1)
        
        # 加上自身
        out = self.linear(x + messages)
        
        return out


class GATLayer(nn.Module):
    """图注意力层（GAT）"""
    def __init__(self, in_dim: int, out_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.num_heads = num_heads
        self.out_dim = out_dim
        self.head_dim = out_dim // num_heads
        
        assert out_dim % num_heads == 0, "out_dim must be divisible by num_heads"
        
        self.linear_q = nn.Linear(in_dim, out_dim)
        self.linear_k = nn.Linear(in_dim, out_dim)
        self.linear_v = nn.Linear(in_dim, out_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [N, in_dim]
            edge_index: [2, E]
        
        Returns:
            out: [N, out_dim]
        """
        N = x.shape[0]
        
        if edge_index.shape[1] == 0:
            return self.linear_v(x)
        
        # 多头注意力
        Q = self.linear_q(x).view(N, self.num_heads, self.head_dim)  # [N, H, D]
        K = self.linear_k(x).view(N, self.num_heads, self.head_dim)
        V = self.linear_v(x).view(N, self.num_heads, self.head_dim)
        
        src, dst = edge_index
        
        # 计算注意力分数
        # Q[dst] * K[src]
        scores = (Q[dst] * K[src]).sum(dim=-1) / math.sqrt(self.head_dim)  # [E, H]
        
        # Softmax归一化（对每个dst节点的所有邻居）
        # 为每个dst计算softmax
        unique_dst = torch.unique(dst)
        attn_weights = torch.zeros_like(scores)
        
        for d in unique_dst:
            mask = (dst == d)
            attn_weights[mask] = F.softmax(scores[mask], dim=0)
        
        attn_weights = self.dropout(attn_weights)
        
        # 聚合邻居信息
        messages = V[src] * attn_weights.unsqueeze(-1)  # [E, H, D]
        
        out = torch.zeros(N, self.num_heads, self.head_dim, device=x.device)
        out.scatter_add_(0, dst.unsqueeze(1).unsqueeze(2).expand(-1, self.num_heads, self.head_dim), messages)
        
        out = out.view(N, self.out_dim)
        
        # 残差连接
        if x.shape[1] == out.shape[1]:
            out = out + x
        
        return out


# ==================== GNN 模型 ====================
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


class HierarchicalGNN(nn.Module):
    """
    层次化GNN模型
    
    架构：
    1. 句子图：句子作为节点，基于空间邻近构建边
    2. 字符图：字符作为节点，基于K近邻构建边
    3. 每个分支独立进行图卷积
    """
    def __init__(self, hidden_dim=256, num_layers=2, dropout=0.1, 
                 gnn_type='gat', k_neighbors=5):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.gnn_type = gnn_type
        self.k_neighbors = k_neighbors
        
        self.sent_in_dim = 6 + 4  # geom + dir
        self.char_in_dim = 6 + 2
        
        # 句子分支
        self.sent_input_proj = nn.Linear(self.sent_in_dim, hidden_dim)
        self.sent_gnn_layers = nn.ModuleList()
        for _ in range(num_layers):
            if gnn_type == 'gat':
                self.sent_gnn_layers.append(GATLayer(hidden_dim, hidden_dim, num_heads=4, dropout=dropout))
            else:  # gcn
                self.sent_gnn_layers.append(GCNLayer(hidden_dim, hidden_dim))
        
        self.sent_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 1)
        )
        
        # 字符分支
        self.char_input_proj = nn.Linear(self.char_in_dim, hidden_dim)
        self.char_gnn_layers = nn.ModuleList()
        for _ in range(num_layers):
            if gnn_type == 'gat':
                self.char_gnn_layers.append(GATLayer(hidden_dim, hidden_dim, num_heads=4, dropout=dropout))
            else:  # gcn
                self.char_gnn_layers.append(GCNLayer(hidden_dim, hidden_dim))
        
        self.char_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        sb = batch["sentence_bboxes"]
        sd = batch["sentence_dir_feat"]
        cb = batch["char_bboxes"]
        cd = batch["char_dir_feat"]
        
        device = sb.device
        
        # ========== 句子分支 ==========
        s_geom = _bbox_to_geom_feats_page_norm(sb)
        s_in = torch.cat([s_geom, sd], dim=1)
        s_feat = self.sent_input_proj(s_in)
        
        # 构建句子图
        s_edge_index = build_knn_graph(sb, k=self.k_neighbors)
        
        # 图卷积
        for gnn_layer in self.sent_gnn_layers:
            s_feat = F.relu(gnn_layer(s_feat, s_edge_index))
        
        sent_scores = self.sent_head(s_feat).squeeze(-1)
        
        # ========== 字符分支 ==========
        c_geom = _bbox_to_geom_feats_page_norm(cb)
        c_in = torch.cat([c_geom, cd], dim=1)
        c_feat = self.char_input_proj(c_in)
        
        # 构建字符图
        c_edge_index = build_knn_graph(cb, k=self.k_neighbors)
        
        # 图卷积
        for gnn_layer in self.char_gnn_layers:
            c_feat = F.relu(gnn_layer(c_feat, c_edge_index))
        
        char_scores = self.char_head(c_feat).squeeze(-1)
        
        return {
            "sent_scores": sent_scores,
            "char_scores": char_scores
        }


# ==================== 损失函数====================
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


# ==================== 评估函数====================
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

        # Sentence-level
        s_idx = torch.argsort(sent_scores, descending=False)  # RankNet: 分数小的靠前
        s_rank = torch.empty_like(s_idx)
        s_rank[s_idx] = torch.arange(len(s_idx))
        m_s = compute_five_metrics(sl.numpy(), s_rank.numpy())
        sent_list.append(m_s)

        # Intra-sentence
        acc = {k: 0.0 for k in ['kendall_tau', 'spearman_rho', 'top1_accuracy', 
                                'top3_accuracy', 'pairwise_accuracy']}
        weight = 0
        
        for sid in torch.unique(cs).tolist():
            idx = (cs == sid).nonzero(as_tuple=False).squeeze(-1)
            if idx.numel() <= 1:
                continue
            
            c_sorted = torch.argsort(char_scores[idx], descending=False)  # RankNet: 分数小的靠前
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
        s_sorted = torch.argsort(sent_scores, descending=False)  # RankNet: 分数小的靠前
        pred_idx = []
        for sid in s_sorted.tolist():
            idx = (cs == sid).nonzero(as_tuple=False).squeeze(-1)
            if idx.numel() == 0:
                continue
            c_sorted = torch.argsort(char_scores[idx], descending=False)  # RankNet: 分数小的靠前
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


# ==================== 训练 ====================
@dataclass
class TrainConfig:
    data_dir: str
    device: str = 'cuda'
    hidden_dim: int = 256
    num_layers: int = 2
    dropout: float = 0.1
    gnn_type: str = 'gat'  # 'gat' or 'gcn'
    k_neighbors: int = 5
    epochs: int = 60
    lr: float = 1e-3
    weight_decay: float = 1e-5
    grad_clip: float = 1.0
    use_direction: bool = True
    save_dir: str = "/home/zhengxiaoying/DBManuscripts/DongbaHRO/checkpoints"
    lambda_char: float = 0.5


def train(cfg: TrainConfig):
    device = torch.device(cfg.device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"\n{'='*60}")
    print(f"Training: Hierarchical GNN ({cfg.gnn_type.upper()})")
    print(f"{'='*60}\n")

    train_set = JointDataset(cfg.data_dir, 'train', use_direction=cfg.use_direction)
    val_set = JointDataset(cfg.data_dir, 'val', use_direction=cfg.use_direction)
    test_set = JointDataset(cfg.data_dir, 'test', use_direction=cfg.use_direction)
    
    train_loader = DataLoader(train_set, batch_size=1, shuffle=True, collate_fn=joint_collate_fn)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, collate_fn=joint_collate_fn)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, collate_fn=joint_collate_fn)

    model = HierarchicalGNN(
        cfg.hidden_dim, cfg.num_layers, cfg.dropout, 
        cfg.gnn_type, cfg.k_neighbors
    ).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)

    save_dir = Path(cfg.save_dir) / f"gnn_{cfg.gnn_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    save_dir.mkdir(parents=True, exist_ok=True)
    best_tau = -1.0

    print("\n===== Training =====")
    for ep in range(1, cfg.epochs + 1):
        model.train()
        tot, s_l, c_l = 0.0, 0.0, 0.0

        for batch in tqdm(train_loader, desc=f"Epoch {ep}/{cfg.epochs}"):
            batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) 
                     for k, v in batch.items()}
            
            optimizer.zero_grad()
            out = model(batch)

            loss_s = ranknet_pairwise_loss(out['sent_scores'], batch['sentence_labels'])
            loss_c = ranknet_pairwise_loss(out['char_scores'], batch['char_labels'])
            loss = loss_s + cfg.lambda_char * loss_c

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()

            tot += float(loss.item())
            s_l += float(loss_s.item())
            c_l += float(loss_c.item())

        scheduler.step()
        
        nb = max(len(train_loader), 1)
        print(f"  Train: total {tot/nb:.4f} | sent {s_l/nb:.4f} | char {c_l/nb:.4f}")

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

    print("\n===== Final Test =====")
    ckpt = torch.load(save_dir / 'best_model.pth', map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    test = evaluate(model, test_loader, device)
    print("[Test Set]", json.dumps(test, ensure_ascii=False, indent=2))
    
    with open(save_dir / 'results.json', 'w') as f:
        json.dump({
            'model': f'HierarchicalGNN-{cfg.gnn_type.upper()}',
            'val_best': ckpt['val_metrics'],
            'test': test,
            'config': cfg.__dict__
        }, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Results saved to {save_dir}")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', type=str,
                   default='/home/zhengxiaoying/DBManuscripts/DongbaHRO')
    ap.add_argument('--device', type=str, default='cuda')
    ap.add_argument('--hidden_dim', type=int, default=256)
    ap.add_argument('--num_layers', type=int, default=2)
    ap.add_argument('--dropout', type=float, default=0.1)
    ap.add_argument('--gnn_type', type=str, choices=['gat', 'gcn'], default='gat')
    ap.add_argument('--k_neighbors', type=int, default=5)
    ap.add_argument('--epochs', type=int, default=60)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--weight_decay', type=float, default=1e-5)
    ap.add_argument('--grad_clip', type=float, default=1.0)
    ap.add_argument('--use_direction', action='store_true')
    ap.add_argument('--save_dir', type=str,
                   default='/home/zhengxiaoying/DBManuscripts/DongbaHRO/checkpoints')
    ap.add_argument('--lambda_char', type=float, default=0.5)
    
    args = ap.parse_args()
    cfg = TrainConfig(**vars(args))
    train(cfg)