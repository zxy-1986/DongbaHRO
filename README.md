# DongbaHRO: A Hierarchical Reading Order Dataset for Dongba Manuscripts

## Overview

DongbaHRO is the first character-level hierarchical reading order dataset for Dongba pictographic manuscripts. It provides both sentence-level and character-level reading order annotations, supporting research on reading order prediction, document layout analysis, and digital preservation of endangered cultural heritage.

## Dataset

### Download

The full dataset (images + annotations) is hosted on Hugging Face:

👉 **[https://huggingface.co/datasets/zhengxiaoying/DongbaHRO](https://huggingface.co/datasets/zhengxiaoying/DongbaHRO)**

### Statistics

| Item | Count |
|------|-------|
| Total pages | 440 |
| Sentences | 3,637 |
| Characters | 30,857 |
| Annotation levels | 2 (sentence + character) |

### Data Splits

| Split | Pages |
|-------|-------|
| Training | 308 (70%) |
| Validation | 66 (15%) |
| Test | 66 (15%) |

### Annotation Format

Each page is stored as a CSV file with the following fields:

| Field | Description |
|-------|-------------|
| `char_id` | Unique character identifier |
| `x0, y0, x1, y1` | Bounding box coordinates (top-left and bottom-right) |
| `sentence_id` | The sentence to which the character belongs |
| `sentence_index` | Reading order index of the sentence within the page |
| `char_index` | Reading order index of the character within its sentence |
| `global_index` | Reading order index of the character within the full page |

### Annotation Process

Three domain experts in Dongba script annotated the manuscripts using a custom-built hierarchical annotation tool. The annotation procedure follows two levels: (1) sentence-level ordering across the full page, and (2) character-level ordering within each sentence.

## Code

### Training & Evaluation

`DongbaHRO.py` provides the unified training and evaluation pipeline.

### Models

| File | Model |
|------|-------|
| `train_baseline_corrected.py` | Baseline |
| `train_gnn_baseline.py` | GNN / GCN (use `--gnn_type gnn` or `--gnn_type gcn`) |
| `train_listnet_baseline.py` | ListNet |
| `train_pairwise.py` | Pairwise |
| `train_pointer.py` | Pointer Network |
| `train_transformer_baseline.py` | Transformer |

All models use `--hidden_dim 256` by default.

## License

This dataset is released under the [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) license for academic research purposes only.
