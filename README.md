# HiRO-Net and DongbaHRO

Official implementation of **HiRO-Net: Hierarchy-Aware Reading Order Prediction for Irregular Dongba Manuscripts** and the **DongbaHRO** hierarchical reading-order benchmark.

## Overview

HiRO-Net models reading order under a two-level hierarchy. It contains:

- a **sentence branch** for parent-level ordering;
- a **character branch** for lower-level ordering;
- **Masked Cross-Hierarchical Attention (MCHA)**, where character representations are queries and sentence representations are keys/values;
- a hierarchy-derived mask constructed from explicit sentence-character membership;
- hierarchical inference that first orders sentences and then orders characters within each predicted parent sentence.

The final HiRO-Net configuration used in the revised manuscript:
- keeps sentence self-attention;
- does **not** use character self-attention;
- uses page-normalized geometric features and PCA-derived directional features;
- uses masked character-to-sentence cross-attention with concatenation fusion;
- uses LayerNorm + Linear ranking heads;
- optimizes sentence-level and character-level RankNet objectives jointly from the beginning;
- uses exhaustive within-page ranking pairs, without pair subsampling or hard-negative mining;
- uses a fixed character-loss weight `lambda = 0.7`.

## DongbaHRO

DongbaHRO contains:

| Item | Count |
|---|---:|
| Pages | 440 |
| Sentences | 3,637 |
| Characters | 30,857 |
| Training pages | 308 |
| Validation pages | 66 |
| Test pages | 66 |

The dataset is available on Hugging Face:

https://huggingface.co/datasets/zhengxiaoying/DongbaHRO

### Annotation fields

The training splits use page-level CSV files. The core fields used by HiRO-Net are:

| Field | Description |
|---|---|
| `page_key` | Page identifier |
| `sentence_id` | Sentence identifier / page-level sentence-order label used by the reading-order task |
| `sentence_index` | Intra-sentence character reading-order index |
| `x0, y0, x1, y1` | Character bounding box in image coordinates |

Each CSV row corresponds to one Dongba character. Characters sharing the same `sentence_id` belong to the same parent sentence, and `sentence_index` specifies their annotated order within that sentence.

The sentence bounding box used by HiRO-Net is computed as the union of the character boxes belonging to the sentence.

## Repository structure

```text
.
├── train_hiro_net.py
├── train_pairwise.py
├── train_pointer.py
├── train_gnn_baseline.py
├── train_listnet_baseline.py
├── train_transformer_baseline.py
├── inference_visualization.py
├── analysis/
│   ├── analyze_error_decomposition.py
│   ├── attention_leakage_analysis.py
│   ├── evaluate_detection_robustness_v4.py
│   └── extract_attention_multiseed.py
├── scripts/
│   ├── run_hiro_net_5seed.sh
│   └── run_ablation_5seed.sh
└── requirements.txt
```

## Main training command

The command below corresponds to the final HiRO-Net configuration reported in the revised manuscript:

```bash
python train_hiro_net.py \
  --data_dir /path/to/dongba_data_splits_by_page \
  --device cuda \
  --hidden_dim 256 \
  --num_heads 4 \
  --dropout 0.1 \
  --epochs 60 \
  --stage1_epochs 0 \
  --loss_type ranknet \
  --lambda_max 0.7 \
  --lambda_schedule fixed \
  --lr 1e-3 \
  --weight_decay 1e-5 \
  --grad_clip 1.0 \
  --patience 10 \
  --use_direction \
  --no_char_self_attn \
  --fusion_type concat \
  --seed 42 \
  --save_dir outputs \
  --exp_name hiro_net_seed42
```

The five random seeds used in the revised manuscript are:

```text
42, 123, 2025, 2026, 3407
```

To run all five seeds:

```bash
bash scripts/run_hiro_net_5seed.sh /path/to/dongba_data_splits_by_page outputs/hiro_net
```

## Final HiRO-Net results

Kendall's tau, mean ± standard deviation over five runs:

| Dataset | Sentence / Parent | In-Sentence / Within-Parent | Global |
|---|---:|---:|---:|
| DongbaHRO | 0.8721 ± 0.0037 | 0.8391 ± 0.0070 | 0.9201 ± 0.0034 |
| cBAD | 0.9867 ± 0.0000 | 0.9989 ± 0.0015 | 0.9895 ± 0.0002 |
| FCR-500 | 0.8391 ± 0.0149 | 0.9964 ± 0.0004 | 0.9915 ± 0.0018 |
| OHG | 0.9710 ± 0.0021 | 0.9966 ± 0.0004 | 0.9982 ± 0.0004 |

The public benchmarks use the same two-level formulation: `TextRegion -> TextLine` is mapped to the parent-child hierarchy used by HiRO-Net.

## Ablation study on DongbaHRO

Kendall's tau, mean ± standard deviation over five runs:

| Configuration | Sentence | In-Sentence | Global |
|---|---:|---:|---:|
| HiRO-Net | 0.8721 ± 0.0037 | 0.8391 ± 0.0070 | 0.9201 ± 0.0034 |
| w/o Direction Features | 0.8617 ± 0.0040 | 0.8035 ± 0.0101 | 0.9090 ± 0.0028 |
| w/o Sentence Self-Attention | 0.8664 ± 0.0030 | 0.8451 ± 0.0072 | 0.9158 ± 0.0046 |
| + Character Self-Attention | 0.8671 ± 0.0029 | 0.8225 ± 0.0113 | 0.9142 ± 0.0019 |
| w/o MCHA | 0.8636 ± 0.0031 | 0.7460 ± 0.0019 | 0.8737 ± 0.0027 |
| w/o Hierarchical Mask | 0.8567 ± 0.0036 | 0.7708 ± 0.0217 | 0.8981 ± 0.0020 |

`w/o MCHA` removes sentence-to-character cross-level context entirely. `w/o Hierarchical Mask` retains character-to-sentence cross-attention but removes the hierarchy-derived membership mask.

## Training-strategy comparison

| Strategy | Sentence | In-Sentence | Global |
|---|---:|---:|---:|
| Sentence init. + linear warm-up | 0.863 ± 0.008 | 0.812 ± 0.020 | 0.908 ± 0.011 |
| Sentence init. + fixed lambda | 0.867 ± 0.007 | 0.813 ± 0.018 | 0.911 ± 0.008 |
| Joint training + fixed lambda | **0.872 ± 0.004** | **0.839 ± 0.007** | **0.920 ± 0.003** |

The final model therefore uses joint optimization from the beginning with fixed `lambda = 0.7`.

## Reproducing the main ablations

The release script supports the following controlled variants:

```text
w/o Direction Features      -> omit --use_direction
w/o Sentence Self-Attention -> add --no_sent_self_attn
+ Character Self-Attention  -> omit --no_char_self_attn
w/o MCHA                    -> --fusion_type self_only
w/o Hierarchical Mask       -> add --no_mask
```

For the strict `w/o MCHA` variant, `self_only` must bypass the cross-attention computation rather than merely discard its output.

## Error and robustness analysis

The revised manuscript additionally reports:

- residual global-error decomposition into inter-sentence and intra-sentence pair errors;
- bounding-box jitter experiments;
- false-negative and false-positive detection perturbations;
- normalized sequence similarity based on Levenshtein distance;
- five-seed attention-leakage statistics;
- page-level Spearman correlation between off-parent attention leakage and ordering performance.

The corresponding scripts are included under `analysis/`.

## License

The DongbaHRO dataset is released under the CC BY-NC 4.0 license for academic research purposes.

## Citation

Citation information will be updated after publication.
