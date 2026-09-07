# HiRO-Net and DongbaHRO

Official implementation of **HiRO-Net: Hierarchy-Aware Reading Order Prediction for Irregular Dongba Manuscripts** and the **DongbaHRO** hierarchical reading-order benchmark.

## Overview

HiRO-Net models reading order under a two-level hierarchy. It contains:

- a **sentence branch** for parent-level ordering;
- a **character branch** for lower-level ordering;
- **Masked Cross-Hierarchical Attention (MCHA)**, where character representations are used as queries and sentence representations are used as keys/values;
- a hierarchy-derived attention mask constructed from explicit sentence-character membership;
- hierarchical inference that first orders sentences and then orders characters within each predicted parent sentence.

The final HiRO-Net configuration used in the revised manuscript:

- keeps sentence self-attention;
- does **not** use character self-attention;
- uses page-normalized geometric features and PCA-derived directional features;
- uses masked character-to-sentence cross-attention with concatenation-based fusion;
- uses `LayerNorm + Linear` ranking heads;
- jointly optimizes sentence-level and character-level RankNet objectives from the beginning of training;
- uses exhaustive within-page ranking pairs, without pair subsampling or hard-negative mining;
- uses a fixed character-loss weight of `lambda = 0.7`.

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
| `sentence_id` | Sentence identifier / page-level sentence-order label |
| `sentence_index` | Intra-sentence character reading-order index |
| `x0, y0, x1, y1` | Character bounding-box coordinates |

Each CSV row corresponds to one Dongba character. Characters sharing the same `sentence_id` belong to the same parent sentence, and `sentence_index` specifies their annotated order within that sentence.

The sentence bounding box used by HiRO-Net is computed as the union of the character bounding boxes belonging to that sentence.

## Main implementation

The main training and ablation implementation is:

```text
DongbaHRO.py
```

The script supports both the final HiRO-Net configuration and the controlled ablations reported in the revised manuscript.

## Main training command

The following command reproduces the final HiRO-Net configuration reported in the revised manuscript:

```bash
python DongbaHRO.py \
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
  --use_direction \
  --no_char_self_attn \
  --fusion_type concat \
  --seed 42 \
  --save_dir ./checkpoints \
  --exp_name hiro_net_seed42
```

The final model therefore uses:

```text
Directional features              enabled
Sentence self-attention           enabled
Character self-attention          disabled
MCHA                              enabled
Hierarchical mask                 enabled
Fusion type                       concat
Loss                              RankNet
Sentence-only pretraining         disabled
Training strategy                 joint from the beginning
Character-loss weight             fixed lambda = 0.7
```

## Random seeds

The five random seeds used in the revised manuscript are:

```text
42, 123, 2025, 2026, 3407
```

All repeated experiments use the same fixed train, validation, and test splits.

## Final HiRO-Net results

Kendall's tau, reported as mean ± standard deviation over five runs:

| Dataset | Sentence / Parent | In-Sentence / Within-Parent | Global |
|---|---:|---:|---:|
| DongbaHRO | 0.8721 ± 0.0037 | 0.8391 ± 0.0070 | 0.9201 ± 0.0034 |
| cBAD | 0.9867 ± 0.0000 | 0.9989 ± 0.0015 | 0.9895 ± 0.0002 |
| FCR-500 | 0.8391 ± 0.0149 | 0.9964 ± 0.0004 | 0.9915 ± 0.0018 |
| OHG | 0.9710 ± 0.0021 | 0.9966 ± 0.0004 | 0.9982 ± 0.0004 |

For cross-dataset evaluation, `TextRegion` and `TextLine` annotations in the public benchmarks are mapped to the parent-level and child-level units required by HiRO-Net, respectively.

## Ablation study on DongbaHRO

Kendall's tau, reported as mean ± standard deviation over five runs:

| Configuration | Sentence | In-Sentence | Global |
|---|---:|---:|---:|
| HiRO-Net | 0.8721 ± 0.0037 | 0.8391 ± 0.0070 | 0.9201 ± 0.0034 |
| w/o Direction Features | 0.8617 ± 0.0040 | 0.8035 ± 0.0101 | 0.9090 ± 0.0028 |
| w/o Sentence Self-Attention | 0.8664 ± 0.0030 | **0.8451 ± 0.0072** | 0.9158 ± 0.0046 |
| + Character Self-Attention | 0.8671 ± 0.0029 | 0.8225 ± 0.0113 | 0.9142 ± 0.0019 |
| w/o MCHA | 0.8636 ± 0.0031 | 0.7460 ± 0.0019 | 0.8737 ± 0.0027 |
| w/o Hierarchical Mask | 0.8567 ± 0.0036 | 0.7708 ± 0.0217 | 0.8981 ± 0.0020 |

`w/o MCHA` removes sentence-to-character cross-level context entirely.

`w/o Hierarchical Mask` retains character-to-sentence cross-attention but removes the hierarchy-derived membership mask.

## Reproducing the main ablations

Starting from the final HiRO-Net command above, the main ablations are obtained by modifying the following options:

```text
w/o Direction Features
    remove --use_direction

w/o Sentence Self-Attention
    add --no_sent_self_attn

+ Character Self-Attention
    remove --no_char_self_attn

w/o MCHA
    set --fusion_type self_only

w/o Hierarchical Mask
    add --no_mask
```

For the strict `w/o MCHA` configuration, `self_only` bypasses the character-to-sentence cross-attention computation entirely, so character ranking relies only on the local character representation.

## Training-strategy comparison

Kendall's tau, reported as mean ± standard deviation over five runs:

| Strategy | Sentence | In-Sentence | Global |
|---|---:|---:|---:|
| Sentence initialization + linear lambda warm-up | 0.8631 ± 0.0081 | 0.8116 ± 0.0200 | 0.9077 ± 0.0112 |
| Sentence initialization + fixed lambda | 0.8671 ± 0.0069 | 0.8125 ± 0.0182 | 0.9108 ± 0.0076 |
| Joint training + fixed lambda | **0.8721 ± 0.0037** | **0.8391 ± 0.0070** | **0.9201 ± 0.0034** |

These results motivate the final training strategy: joint optimization from the beginning with a fixed `lambda = 0.7`.

## RankNet pair construction

HiRO-Net uses exhaustive within-page pair construction.

For the sentence branch:

- all unordered sentence pairs on a page are included;
- supervision is derived from the annotated sentence order.

For the character branch:

- the full-page ground-truth character order is first constructed from the annotated sentence order and intra-sentence character order;
- RankNet is then applied to all unordered character pairs on the page.

No pair subsampling or hard-negative selection is used.

## MCHA

MCHA follows standard scaled dot-product cross-attention but constrains its attention scope using the explicit sentence-character hierarchy.

For character `i`:

- the character representation is used as the query;
- sentence representations are used as keys and values;
- the hierarchy-derived mask permits attention only to the annotated parent sentence.

The mask therefore has size:

```text
number of characters × number of sentences
```

Under the masked configuration, off-parent attention is zero by construction.

## Additional analyses in the revised manuscript

The revised manuscript additionally reports:

- residual global-error decomposition into inter-sentence and intra-sentence pair errors;
- bounding-box jitter robustness analysis;
- false-negative and false-positive detection perturbations;
- normalized sequence similarity based on Levenshtein distance;
- five-seed attention-leakage statistics;
- page-level Spearman correlation between off-parent attention leakage and ordering performance.

These analyses were added to evaluate model stability, detection robustness, and the effect of hierarchy-constrained cross-level attention.

## License

DongbaHRO is released under the **CC BY-NC 4.0** license for academic research purposes.

## Citation

Citation information will be updated after publication.
