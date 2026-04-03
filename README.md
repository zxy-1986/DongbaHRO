# DongbaHRO: A Hierarchical Reading Order Dataset for Dongba Manuscripts

## Overview

DongbaHRO is the first character-level hierarchical reading order dataset for Dongba pictographic manuscripts. It provides both sentence-level and character-level reading order annotations, supporting research on reading order prediction, document layout analysis, and digital preservation of endangered cultural heritage.

## Dataset Statistics

| Item | Count |
|------|-------|
| Total pages | 440 |
| Sentences | 3,637 |
| Characters | 30,857 |
| Annotation levels | 2 (sentence + character) |

## Annotation Format

Each page is stored as a CSV file with the following fields:

| Field | Description |
|-------|-------------|
| `char_id` | Unique character identifier |
| `x0, y0, x1, y1` | Bounding box coordinates (top-left and bottom-right) |
| `sentence_id` | The sentence to which the character belongs |
| `sentence_index` | Reading order index of the sentence within the page |
| `char_index` | Reading order index of the character within its sentence |
| `global_index` | Reading order index of the character within the full page |

## Data Splits

| Split | Pages |
|-------|-------|
| Training | 308 (70%) |
| Validation | 66 (15%) |
| Test | 66 (15%) |

## Annotation Process

Three domain experts in Dongba script annotated the manuscripts using a custom-built hierarchical annotation tool. The annotation procedure follows two levels: (1) sentence-level ordering across the full page, and (2) character-level ordering within each sentence.

## Usage

```python
import pandas as pd

# Load a single page annotation
page = pd.read_csv("data/train/page_001.csv")

# Get sentence-level order
sentences = page.groupby("sentence_id").first().sort_values("sentence_index")

# Get character-level order within a sentence
sent_1 = page[page["sentence_id"] == 1].sort_values("char_index")

# Get global reading order
global_order = page.sort_values("global_index")
```

## Citation

If you use this dataset in your research, please cite:

```bibtex
@article{zheng2025dongbahro,
  title={Hierarchical Reading Order Prediction for Dongba Manuscripts via Masked Cross-Attention},
  author={Zheng, Xiaoying and Cheng, Bodong and Shi, Jinxin and Zhou, Aimin},
  journal={International Journal on Document Analysis and Recognition (IJDAR)},
  year={2025}
}
```

## License

This dataset is released under the [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) license for academic research purposes only.

## Contact

For questions or access requests, please contact the corresponding author: Aimin Zhou (amzhou@cs.ecnu.edu.cn).
