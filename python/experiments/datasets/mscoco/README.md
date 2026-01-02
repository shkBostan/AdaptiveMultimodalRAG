# MS COCO Dataset (Multimodal RAG Setup)

## Overview
This directory contains the MS COCO 2017 dataset prepared for multimodal
Retrieval-Augmented Generation (RAG) experiments.

MS COCO (Microsoft Common Objects in Context) is a large-scale, real-world
dataset consisting of images paired with human-annotated textual descriptions.
In this project, MS COCO is used to evaluate text–image retrieval, multimodal
embedding fusion, and grounded generation within an adaptive RAG pipeline.

## Dataset Source
- Official website: https://cocodataset.org
- Version: MS COCO 2017
- Modalities: Image, Text (captions)
- License: Creative Commons Attribution 4.0 (CC BY 4.0)

## Directory Structure
`
raw/
├── images/
│ ├── train2017/
│ └── val2017/
└── annotations/
  ├── captions_train2017.json
  └── captions_val2017.json

processed/
├── documents.jsonl
├── queries.jsonl
└── splits.json
`

### raw/
Contains the original, unmodified MS COCO files downloaded from the official
source. These files must remain unchanged to ensure experimental reproducibility.

### processed/
Contains files generated automatically by preprocessing pipelines.
This directory is intentionally empty in the repository and populated at runtime.

- `documents.jsonl`: RAG-ready multimodal documents (text, image references, metadata)
- `queries.jsonl`: Evaluation and retrieval queries derived from captions
- `splits.json`: Deterministic train/validation/test splits

## Usage in This Project
MS COCO is used as the primary benchmark for:
- Multimodal embedding evaluation
- Cross-modal retrieval (text ↔ image)
- Adaptive fusion strategy analysis
- Retrieval-augmented generation with visual grounding

The dataset enables controlled ablation studies on fusion strategies such as
concatenation, weighted sum, and adaptive modality selection.

## Preprocessing
All preprocessing steps are performed via dedicated scripts to avoid manual bias.
The preprocessing pipeline:
1. Loads raw images and caption annotations
2. Converts data into a unified JSONL schema
3. Generates deterministic dataset splits
4. Stores outputs in the `processed/` directory

No processed data is committed to the repository.

## Citation
If you use MS COCO in academic work, please cite:

Lin, T.-Y., et al.  
*Microsoft COCO: Common Objects in Context.*  
ECCV 2014.

## Notes
- Only the `train2017` and `val2017` splits are used.
- Test and segmentation annotations are intentionally excluded.
- This dataset setup is designed for research-grade multimodal RAG experiments
  and is fully compatible with reproducibility and peer-review requirements.
