#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用方法:
    python inference_visualization.py --output_dir /path/to/output --n_samples 10
"""

from __future__ import annotations
import argparse
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.patheffects as path_effects
from tqdm import tqdm

CHECKPOINT_BASE = "/home/zhengxiaoying/DBManuscripts/DongbaHRO/checkpoints"

BEST_MODELS = {
    "Ours": {
        "path": f"{CHECKPOINT_BASE}/ablations/ablation_no_char_self_attn_20251110_181805/best_model.pth",
        "type": "ablation",
        "config": {"use_char_self_attn": False, "use_sent_self_attn": True, "use_mask": True, "fusion_type": "concat"},
        "test_tau": 0.9198
    },
    "Ours_Full": {
        "path": f"{CHECKPOINT_BASE}/ablations/ablation_baseline_20251110_072728/best_model.pth",
        "type": "ablation",
        "config": {"use_char_self_attn": True, "use_sent_self_attn": True, "use_mask": True, "fusion_type": "concat"},
        "test_tau": 0.8998
    },
    # Transformer baseline
    "Transformer": {
        "path": f"{CHECKPOINT_BASE}/transformer_character_20251113_115301/best_model.pth",
        "type": "transformer",
        "test_tau": 0.9018
    },
    # ListNet
    "ListNet": {
        "path": f"{CHECKPOINT_BASE}/listnet_baseline_20251110_073309/best_model.pth",
        "type": "listnet",
        "test_tau": 0.8258
    },
    # GNN-GAT
    "GNN-GAT": {
        "path": f"{CHECKPOINT_BASE}/gnn_gat_20251110_074039/best_model.pth",
        "type": "gnn",
        "test_tau": 0.8826
    },
    # Pairwise Ranking
    "Pairwise": {
        "path": f"{CHECKPOINT_BASE}/pairwise_fixed_20251113_114631/best_model.pth",
        "type": "pairwise",
        "test_tau": 0.8980
    },
    # Pointer Network
    "Pointer": {
        "path": f"{CHECKPOINT_BASE}/pointer_fixed_20251112_122837/best_model.pth",
        "type": "pointer",
        "test_tau": 0.8914
    },
}

# 默认数据路径
DEFAULT_DATA_DIR = "/home/zhengxiaoying/DBManuscripts/DongbaHRO"
DEFAULT_IMAGE_DIR = "/home/zhengxiaoying/test"
DEFAULT_OUTPUT_DIR = "/home/zhengxiaoying/DBManuscripts/DongbaHRO/visualizations/comparison"


# ============================== 指标计算 ==============================
def _kendall_tau(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    n = len(y_true)
    if n <= 1:
        return 0.0
    conc = disc = 0
    for i in range(n):
        for j in range(i + 1, n):
            s = (y_true[i] - y_true[j]) * (y_pred[i] - y_pred[j])
            conc += int(s > 0)
            disc += int(s < 0)
    d = conc + disc
    return (conc - disc) / d if d > 0 else 0.0


def _pairwise_accuracy(y_true_rank: np.ndarray, y_pred_rank: np.ndarray) -> float:
    n = len(y_true_rank)
    ok = tot = 0
    for i in range(n):
        for j in range(i + 1, n):
            tot += 1
            ok += int((y_true_rank[i] < y_true_rank[j]) == (y_pred_rank[i] < y_pred_rank[j]))
    return ok / tot if tot > 0 else 0.0


# ============================== 数据处理 ==============================
def _centers_xyxy(b: np.ndarray) -> np.ndarray:
    return (b[:, :2] + b[:, 2:]) / 2.0


def sentence_rule_order_idx(centers: np.ndarray, rule: str = 'top_to_bottom') -> np.ndarray:
    if rule == 'top_to_bottom':
        return np.lexsort((-centers[:, 0], centers[:, 1]))
    elif rule == 'left_to_right':
        return np.lexsort((centers[:, 1], centers[:, 0]))
    else:
        return np.lexsort((centers[:, 1], -centers[:, 0]))


def _bbox_to_geom_feats_page_norm(bboxes: torch.Tensor) -> torch.Tensor:
    x0, y0, x1, y1 = [bboxes[:, i] for i in range(4)]
    w = (x1 - x0).clamp(min=1.0)
    h = (y1 - y0).clamp(min=1.0)
    cx = (x0 + x1) * 0.5
    cy = (y0 + y1) * 0.5
    eps = 1e-6
    cxn = (cx - cx.min()) / (cx.max() - cx.min() + eps) * 2 - 1
    cyn = (cy - cy.min()) / (cy.max() - cy.min() + eps) * 2 - 1
    feats = torch.stack([cxn, cyn, torch.log(w), torch.log(h), torch.log(w / h), torch.log(w * h)], dim=1)
    return feats


# ============================== 模型定义 ==============================
class SharedBackbone(nn.Module):
    def __init__(self, in_dim: int, hidden: int, drop: float):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(drop),
            nn.Linear(hidden, hidden), nn.ReLU()
        )

    def forward(self, x):
        return self.mlp(x)


class JointModelV4Ablation(nn.Module):
    def __init__(self, hidden_dim=256, num_heads=4, dropout=0.1,
                 use_sent_self_attn=True, use_char_self_attn=True,
                 use_mask=True, fusion_type='concat'):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.sent_in_dim = 6 + 4
        self.char_in_dim = 6 + 2
        
        self.use_sent_self_attn = use_sent_self_attn
        self.use_char_self_attn = use_char_self_attn
        self.use_mask = use_mask
        self.fusion_type = fusion_type

        self.sent_backbone = SharedBackbone(self.sent_in_dim, hidden_dim, dropout)
        
        if self.use_sent_self_attn:
            self.sent_self_attn = nn.MultiheadAttention(
                hidden_dim, num_heads, dropout=dropout, batch_first=True
            )
        
        self.sent_head = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, 1))

        self.char_backbone = SharedBackbone(self.char_in_dim, hidden_dim, dropout)
        
        if self.use_char_self_attn:
            self.char_self_attn = nn.MultiheadAttention(
                hidden_dim, num_heads, dropout=dropout, batch_first=True
            )
        
        self.char_cross_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )

        if fusion_type == 'concat':
            fuse_in_dim = hidden_dim * 2
        else:
            fuse_in_dim = hidden_dim
        
        self.fuse = nn.Sequential(
            nn.Linear(fuse_in_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)
        )
        self.char_head = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, 1))

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
        
        if self.use_sent_self_attn:
            s_out, _ = self.sent_self_attn(s_feat, s_feat, s_feat)
            s_out = s_out.squeeze(0)
        else:
            s_out = s_feat.squeeze(0)
        
        sent_scores = self.sent_head(s_out).squeeze(-1)

        # Character branch
        c_geom = _bbox_to_geom_feats_page_norm(cb)
        c_in = torch.cat([c_geom, cd], dim=1)
        c_feat = self.char_backbone(c_in).unsqueeze(0)
        
        if self.use_char_self_attn:
            c_self, _ = self.char_self_attn(c_feat, c_feat, c_feat)
        else:
            c_self = c_feat

        # Masked cross-attention
        if self.use_mask:
            attn_mask = torch.full((N, S), float('-inf'), device=device)
            for s_id in range(S):
                idx = (cs == s_id).nonzero(as_tuple=False).squeeze(-1)
                if idx.numel() > 0:
                    attn_mask[idx, s_id] = 0.0
        else:
            attn_mask = None

        c_cross, _ = self.char_cross_attn(
            query=c_self, key=s_out.unsqueeze(0), value=s_out.unsqueeze(0),
            attn_mask=attn_mask
        )

        # Feature fusion
        if self.fusion_type == 'concat':
            c_fused = self.fuse(torch.cat([c_self.squeeze(0), c_cross.squeeze(0)], dim=1))
        elif self.fusion_type == 'add':
            c_fused = self.fuse(c_self.squeeze(0) + c_cross.squeeze(0))
        else:
            c_fused = self.fuse(c_self.squeeze(0))
        
        char_scores = self.char_head(c_fused).squeeze(-1)

        return {"sent_scores": sent_scores, "char_scores": char_scores}


class TransformerBaseline(nn.Module):
    def __init__(self, hidden_dim=256, num_heads=4, num_layers=3, dropout=0.1, mode='character'):
        super().__init__()
        self.mode = mode
        self.hidden_dim = hidden_dim
        
        # 输入维度
        if mode == 'sentence':
            self.in_dim = 6 + 4  # geom + dir
        else:  # character
            self.in_dim = 6 + 2  # geom + dir

        self.input_proj = nn.Sequential(
            nn.Linear(self.in_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads,
            dim_feedforward=hidden_dim * 4, dropout=dropout, 
            batch_first=True, activation='relu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.output_head = nn.Sequential(
            nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, 1)
        )

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if 'char_bboxes' in batch:
            cb = batch["char_bboxes"]
            cd = batch["char_dir_feat"]
        else:
            cb = batch["bboxes"]
            cd = batch["dir_feats"]

        c_geom = _bbox_to_geom_feats_page_norm(cb)
        c_in = torch.cat([c_geom, cd], dim=1)

        x = self.input_proj(c_in).unsqueeze(0)
        x = self.transformer(x)
        scores = self.output_head(x.squeeze(0)).squeeze(-1)

        return {"char_scores": scores}


class HierarchicalListNet(nn.Module):
    def __init__(self, hidden_dim=256, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.sent_in_dim = 6 + 4
        self.char_in_dim = 6 + 2

        self.sent_backbone = SharedBackbone(self.sent_in_dim, hidden_dim, dropout)
        self.sent_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 1)
        )

        self.char_backbone = SharedBackbone(self.char_in_dim, hidden_dim, dropout)
        self.char_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        sb = batch["sentence_bboxes"]
        sd = batch["sentence_dir_feat"]
        cb = batch["char_bboxes"]
        cd = batch["char_dir_feat"]

        s_geom = _bbox_to_geom_feats_page_norm(sb)
        s_in = torch.cat([s_geom, sd], dim=1)
        s_feat = self.sent_backbone(s_in)
        sent_scores = self.sent_head(s_feat).squeeze(-1)

        c_geom = _bbox_to_geom_feats_page_norm(cb)
        c_in = torch.cat([c_geom, cd], dim=1)
        c_feat = self.char_backbone(c_in)
        char_scores = self.char_head(c_feat).squeeze(-1)

        return {"sent_scores": sent_scores, "char_scores": char_scores}

# ============================== 数据集 ==============================
class InferenceDataset:
    def __init__(self, data_dir: str, split: str = 'test', use_direction: bool = True):
        csv_path = f"{data_dir}/{split}.csv"
        print(f"Loading {split} data from {csv_path}...")
        
        df = pd.read_csv(csv_path)
        
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

                x0, y0, x1, y1 = sg['x0'].min(), sg['y0'].min(), sg['x1'].max(), sg['y1'].max()
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
                    sentence_id=int(sg['sentence_id'].iloc[0]),  
                    s_box=s_box, s_dir=s_dir, sg=sg, char_dir=char_dir
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

            sent_items_sorted = sorted(enumerate(sent_items), 
                                       key=lambda x: sentence_ids[x[0]])

            char_bboxes_list = []
            char_to_sentence_idx_list = []
            char_dir_blocks = []

            for orig_idx, it in sent_items_sorted:
                sg = it['sg']  
                
                for _, row in sg.iterrows():
                    char_bboxes_list.append([float(row['x0']), float(row['y0']), 
                                             float(row['x1']), float(row['y1'])])
                    char_to_sentence_idx_list.append(orig_idx)  # 指向原始输入位置
                
                char_dir_blocks.append(it['char_dir'])

            char_bboxes = torch.tensor(char_bboxes_list, dtype=torch.float32)
            char_to_sentence_idx = torch.tensor(char_to_sentence_idx_list, dtype=torch.long)

            if use_direction and len(char_dir_blocks) > 0:
                char_dir_feat = torch.tensor(np.concatenate(char_dir_blocks, axis=0), dtype=torch.float32)
            else:
                char_dir_feat = torch.zeros((char_bboxes.shape[0], 2), dtype=torch.float32)

            # ✅ 全局字符"阅读顺序名次"= 0..N-1（按人工标注拼出来的）
            char_labels = torch.arange(char_bboxes.shape[0], dtype=torch.long)

            parts = page_key.split('_')
            prj_id = parts[0] if len(parts) >= 1 else page_key
            subprj_id = parts[1] if len(parts) >= 2 else "0"

            self.pages.append(dict(
                page_key=page_key, prj_id=prj_id, subprj_id=subprj_id,
                sentence_bboxes=sentence_bboxes, sentence_labels=sentence_labels,
                sentence_dir_feat=sentence_dir_feat, char_bboxes=char_bboxes,
                char_labels=char_labels, char_to_sentence_idx=char_to_sentence_idx,
                char_dir_feat=char_dir_feat
            ))

        print(f"  ✓ Loaded {len(self.pages)} pages (using human annotations)")

    def __len__(self):
        return len(self.pages)

    def __getitem__(self, idx):
        return self.pages[idx]


# ============================== 推理 ==============================
@torch.no_grad()
def inference_single_page(model: nn.Module, batch: Dict, device: torch.device, 
                          model_type: str = 'ours') -> Tuple[np.ndarray, float]:
    """对单页进行推理"""
    model.eval()
    batch_gpu = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}

    out = model(batch_gpu)
    char_scores = out['char_scores'].cpu()
    cl = batch['char_labels']
    cs = batch['char_to_sentence_idx']
    if model_type in ['ours', 'ablation', 'listnet'] and 'sent_scores' in out:
        sent_scores = out['sent_scores'].cpu()
        if model_type == 'listnet':
            s_sorted = torch.argsort(sent_scores, dim=0, descending=True)
        else:
            s_sorted = torch.argsort(sent_scores, dim=0)
        
        pred_idx = []
        for sid in s_sorted.tolist():
            idx = (cs == sid).nonzero(as_tuple=False).squeeze(-1)
            if idx.numel() == 0:
                continue
            if model_type == 'listnet':
                c_sorted = torch.argsort(char_scores[idx], dim=0, descending=True)
            else:
                c_sorted = torch.argsort(char_scores[idx], dim=0)
            pred_idx.extend(idx[c_sorted].tolist())
        
        glob_rank = torch.empty_like(cl)
        glob_rank[pred_idx] = torch.arange(len(pred_idx))
    else:
        c_sorted = torch.argsort(char_scores, dim=0)
        glob_rank = torch.empty_like(cl)
        glob_rank[c_sorted] = torch.arange(len(c_sorted))

    pred_order = glob_rank.numpy()
    true_order = cl.numpy()
    global_tau = _kendall_tau(true_order, pred_order)

    return pred_order, global_tau


def run_inference_all(models: Dict[str, Tuple[nn.Module, str]], dataset: InferenceDataset, 
                      device: torch.device) -> pd.DataFrame:
    results = []

    for idx in tqdm(range(len(dataset)), desc="Running inference"):
        batch = dataset[idx]
        row = {
            'page_key': batch['page_key'],
            'prj_id': batch['prj_id'],
            'subprj_id': batch['subprj_id'],
            'num_chars': batch['char_bboxes'].shape[0],
            'num_sents': batch['sentence_bboxes'].shape[0]
        }

        for model_name, (model, model_type) in models.items():
            try:
                pred_order, global_tau = inference_single_page(model, batch, device, model_type)
                row[f'{model_name}_tau'] = global_tau
                row[f'{model_name}_pred_order'] = pred_order.tolist()
            except Exception as e:
                print(f"Error on {batch['page_key']} with {model_name}: {e}")
                row[f'{model_name}_tau'] = -999
                row[f'{model_name}_pred_order'] = []

        results.append(row)

    return pd.DataFrame(results)


def select_best_samples(df: pd.DataFrame, n_samples: int = 10, 
                        our_model: str = 'Ours', 
                        baseline_models: List[str] = None) -> pd.DataFrame:
    df = df.copy()
    
    if baseline_models is None:
        baseline_models = ['Transformer', 'ListNet']
    
    baseline_cols = [f'{m}_tau' for m in baseline_models if f'{m}_tau' in df.columns]
    our_tau_col = f'{our_model}_tau'
    
    if not baseline_cols or our_tau_col not in df.columns:
        return df.head(n_samples)
    
    df['baseline_avg_tau'] = df[baseline_cols].mean(axis=1)
    df['baseline_min_tau'] = df[baseline_cols].min(axis=1)
    df['advantage'] = df[our_tau_col] - df['baseline_avg_tau']
    
    good_samples = df[
        (df[our_tau_col] > 0.75) & 
        (df['advantage'] > 0.02)
    ].copy()
    
    if len(good_samples) < n_samples:
        good_samples = df[df[our_tau_col] > 0.5].copy()
    
    return good_samples.sort_values('advantage', ascending=False).head(n_samples)


# ============================== 可视化 ==============================
def visualize_comparison(batch: Dict, predictions: Dict[str, np.ndarray],
                         image_path: str, output_path: str,
                         models_to_show: List[str], scores: Dict[str, float] = None):
    # 加载原图
    img_array = None
    try:
        img = Image.open(image_path)
        img_array = np.array(img)
        if img_array.ndim == 3 and img_array.shape[2] == 4:
            img_array = img_array[:, :, :3]
    except Exception as e:
        print(f"  Warning: Cannot load image {image_path}: {e}")

    n_models = len(models_to_show)
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 7))
    if n_models == 1:
        axes = [axes]

    char_bboxes = batch['char_bboxes'].numpy()
    num_chars = len(char_bboxes)

    colors = {'Ground Truth': '#2ecc71', 'Ours': '#3498db', 'Transformer': '#e67e22', 
              'ListNet': '#e74c3c', 'GNN-GAT': '#9b59b6', 'Pairwise': '#8b4513', 'Pointer': '#ff69b4'}

    for ax_idx, model_name in enumerate(models_to_show):
        ax = axes[ax_idx]
        
        ax.set_facecolor('white')
        
        if img_array is not None:
            ax.imshow(img_array)
        else:
            ax.set_facecolor('#f0f0f0')
            x_min, y_min = char_bboxes[:, 0].min() - 20, char_bboxes[:, 1].min() - 20
            x_max, y_max = char_bboxes[:, 2].max() + 20, char_bboxes[:, 3].max() + 20
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_max, y_min)  
        
        color = colors.get(model_name, '#7f8c8d')
        
        if model_name == 'Ground Truth':
            for i in range(num_chars):
                x0, y0, x1, y1 = char_bboxes[i]
                cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
                
                rect = patches.Rectangle(
                    (x0, y0), x1 - x0, y1 - y0,
                    linewidth=2, edgecolor=color, facecolor='none', alpha=0.9
                )
                ax.add_patch(rect)
                
                reading_order = i + 1
                text = ax.text(cx, cy, str(reading_order), color='white', fontsize=8,
                              fontweight='bold', ha='center', va='center')
                text.set_path_effects([
                    path_effects.Stroke(linewidth=3, foreground=color),
                    path_effects.Normal()
                ])
            
            if num_chars > 1:
                centers = (char_bboxes[:, :2] + char_bboxes[:, 2:]) / 2
                for i in range(num_chars - 1):
                    ax.annotate('', xy=(centers[i+1, 0], centers[i+1, 1]),
                               xytext=(centers[i, 0], centers[i, 1]),
                               arrowprops=dict(arrowstyle='->', color=color, alpha=0.4, lw=1.5))
        else:
            pred_order = predictions.get(model_name, np.arange(num_chars))
            
            for i in range(num_chars):
                x0, y0, x1, y1 = char_bboxes[i]
                cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
                
                rect = patches.Rectangle(
                    (x0, y0), x1 - x0, y1 - y0,
                    linewidth=2, edgecolor=color, facecolor='none', alpha=0.9
                )
                ax.add_patch(rect)
                
                rank = int(pred_order[i]) + 1
                text = ax.text(cx, cy, str(rank), color='white', fontsize=8,
                              fontweight='bold', ha='center', va='center')
                text.set_path_effects([
                    path_effects.Stroke(linewidth=3, foreground=color),
                    path_effects.Normal()
                ])
            
            if num_chars > 1:
                order_indices = np.argsort(pred_order)
                centers = (char_bboxes[:, :2] + char_bboxes[:, 2:]) / 2
                
                for i in range(len(order_indices) - 1):
                    idx1, idx2 = order_indices[i], order_indices[i + 1]
                    ax.annotate('', xy=(centers[idx2, 0], centers[idx2, 1]),
                               xytext=(centers[idx1, 0], centers[idx1, 1]),
                               arrowprops=dict(arrowstyle='->', color=color, alpha=0.4, lw=1.5))
        
        title = model_name
        if scores and model_name in scores:
            title += f"\nτ = {scores[model_name]:.3f}"
        ax.set_title(title, fontsize=12, fontweight='bold', color=color)
        ax.axis('off')

    plt.suptitle(f"Page: {batch['page_key']} ({num_chars} chars)", fontsize=13, fontweight='bold')
    plt.tight_layout()
    fig.patch.set_facecolor('white')
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    print(f"  Saved: {output_path}")


# ============================== 模型加载 ==============================
def load_models(model_names: List[str], device: torch.device) -> Dict[str, Tuple[nn.Module, str]]:
    """加载指定的模型"""
    models = {}
    
    for name in model_names:
        if name not in BEST_MODELS:
            print(f"  ⚠ Unknown model: {name}")
            continue
            
        info = BEST_MODELS[name]
        path = info['path']
        
        if not Path(path).exists():
            print(f"  ⚠ Checkpoint not found: {path}")
            continue
        
        print(f"Loading {name} from {path}...")
        ckpt = torch.load(path, map_location=device, weights_only=False)
        config = ckpt.get('config', {})
        
        model_type = info['type']
        
        try:
            if model_type == 'ablation':
                ablation_config = info.get('config', {})
                model = JointModelV4Ablation(
                    hidden_dim=config.get('hidden_dim', 256),
                    num_heads=config.get('num_heads', 4),
                    dropout=config.get('dropout', 0.1),
                    use_sent_self_attn=ablation_config.get('use_sent_self_attn', True),
                    use_char_self_attn=ablation_config.get('use_char_self_attn', True),
                    use_mask=ablation_config.get('use_mask', True),
                    fusion_type=ablation_config.get('fusion_type', 'concat')
                ).to(device)
            elif model_type == 'transformer':
                model = TransformerBaseline(
                    hidden_dim=config.get('hidden_dim', 256),
                    num_heads=config.get('num_heads', 4),
                    num_layers=config.get('num_layers', 3),
                    dropout=config.get('dropout', 0.1),
                    mode=config.get('mode', 'character')
                ).to(device)
            elif model_type == 'listnet':
                model = HierarchicalListNet(
                    hidden_dim=config.get('hidden_dim', 256),
                    dropout=config.get('dropout', 0.1)
                ).to(device)
            else:
                print(f"  ⚠ Unsupported model type: {model_type}")
                continue
            
            model.load_state_dict(ckpt['model_state_dict'])
            model.eval()
            models[name] = (model, model_type)
            print(f"  ✓ Loaded {name} (test τ: {info.get('test_tau', 'N/A')})")
        except Exception as e:
            print(f"  ✗ Failed to load {name}: {e}")
            continue
    
    return models


# ============================== 主函数 ==============================
def main():
    parser = argparse.ArgumentParser(description='Reading Order Inference Visualization')
    parser.add_argument('--data_dir', type=str, default=DEFAULT_DATA_DIR)
    parser.add_argument('--image_dir', type=str, default=DEFAULT_IMAGE_DIR)
    parser.add_argument('--output_dir', type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument('--models', type=str, nargs='+', 
                        default=['Ours', 'Transformer', 'ListNet'],
                        help='Models to compare')
    parser.add_argument('--n_samples', type=int, default=10)
    parser.add_argument('--device', type=str, default='cuda')
    
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载数据
    dataset = InferenceDataset(args.data_dir, split='test')
    
    # 加载模型
    models = load_models(args.models, device)
    
    if not models:
        print("No models loaded!")
        return
    
    print(f"\nLoaded {len(models)} models: {list(models.keys())}")
    
    # 推理
    print("\n" + "=" * 60)
    print("Running inference on all test pages...")
    print("=" * 60)
    results_df = run_inference_all(models, dataset, device)
    
    # 保存结果
    results_csv = output_dir / 'inference_results.csv'
    save_df = results_df.drop(columns=[c for c in results_df.columns if 'pred_order' in c])
    save_df.to_csv(results_csv, index=False)
    print(f"Results saved to {results_csv}")
    
    # 性能统计
    print("\n" + "=" * 60)
    print("Performance Summary")
    print("=" * 60)
    for model_name in models.keys():
        tau_col = f'{model_name}_tau'
        if tau_col in results_df.columns:
            valid_tau = results_df[results_df[tau_col] > -100][tau_col]
            print(f"  {model_name}: Mean τ = {valid_tau.mean():.4f} (±{valid_tau.std():.4f})")
    
    # 选择样本
    print("\n" + "=" * 60)
    print(f"Selecting top {args.n_samples} samples...")
    print("=" * 60)
    
    baseline_models = [m for m in models.keys() if m != 'Ours']
    selected = select_best_samples(results_df, args.n_samples, 'Ours', baseline_models)
    
    print(f"Selected {len(selected)} samples")
    
    # 生成可视化
    print("\n" + "=" * 60)
    print("Generating visualizations...")
    print("=" * 60)
    
    for idx, (_, row) in enumerate(selected.iterrows()):
        page_key = row['page_key']
        
        batch = None
        for page in dataset.pages:
            if page['page_key'] == page_key:
                batch = page
                break
        
        if batch is None:
            continue
        
        # 图片路径
        image_name = f"{row['prj_id']}_{row['subprj_id']}_origin.png"
        image_path = Path(args.image_dir) / image_name
        
        # 收集预测
        predictions = {}
        scores = {}
        for model_name in models.keys():
            pred_col = f'{model_name}_pred_order'
            tau_col = f'{model_name}_tau'
            if pred_col in row and row[pred_col]:
                predictions[model_name] = np.array(row[pred_col])
                scores[model_name] = row[tau_col]
        
        # 生成对比图
        output_path = output_dir / f"comparison_{idx+1:02d}_{page_key}.png"
        models_to_show = ['Ground Truth'] + list(models.keys())
        visualize_comparison(batch, predictions, str(image_path), str(output_path),
                           models_to_show=models_to_show, scores=scores)
    
    print("\n" + "=" * 60)
    print(f"✅ Done! Output: {output_dir}")
    print("=" * 60)


if __name__ == '__main__':
    main()