# TIF-Contrast: Triplet-aligned Text-Image-Formula Contrastive Learning

## Overview 

**Abstract**: Academic similarity detection is fundamental to research integrity.Yet, effective detection in multimodal documents is hindered by the semantic heterogeneity across text, figures, and mathematical expressions. Conventional approaches, which are largely confined to unimodal text matching or limited dual-modal settings, often
fail to capture complex cross-modal semantic correlations and lack the adaptability to cope with the variable
modal proportion distribution in real-world academic documents.To bridge this gap, we propose the Triplet aligned Text-Image-Formula Contrastive Similarity Framework (TIF-Contrast). Central to our approach is a
triplet-aligned embedding mechanism that projects diverse modalities into a unified semantic space, ensuring
precise alignment between text, images, and formulas.Specifically, triplet-aligned embeddings refer to our
strategy of constructing contrastive “anchor-positive-negative” modality triplets to constrain feature distribution, forcing semantically consistent cross-modal content to cluster closely in the shared space. By leveraging
contrastive learning, the model maximizes the affinity of matched cross-modal triplets while distinguishing mismatched pairs, thereby refining fine-grained semantic representations. Additionally, we incorporate
Adaptive-attention fusion and interpretability modules to augment system stability. Extensive experiments
on the Teddy Cup and SciOL datasets demonstrate that TIF-Contrast significantly outperforms text-only and
text-image baselines, achieving accuracy gains of 12.3% and 8.7%, respectively. Ablation studies further
validate the efficacy of the proposed alignment and fusion strategies.


## Installation

### Prerequisites

- Python 3.8+
- CUDA 11.8+ (for GPU support)
- 16GB+ RAM (32GB recommended for training)

### Step 1: Clone Repository

```bash
git clone https://github.com/yourusername/TIF-Contrast.git
cd TIF-Contrast
```

### Step 2: Create Environment

```bash
# Using conda (recommended)
conda create -n tif python=3.8
conda activate tif

# Or using venv
python -m venv tif_env
source tif_env/bin/activate  # On Windows: tif_env\Scripts\activate
```

### Step 3: Install Dependencies

```bash
# Install PyTorch (adjust for your CUDA version)
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118

# Install other requirements
pip install -r requirements.txt

# Install PaddleOCR for formula extraction
pip install paddlepaddle-gpu paddleocr
```

### Step 4: Download Pre-trained Weights (Optional)

```bash
bash scripts/download_weights.sh
```

---

## 🎬 Quick Start

### Single Document Pair Similarity

```python
from tif_contrast.inference import compare_documents

# Compare two academic documents
result = compare_documents(
    doc1_path='data/documents/doc_0001',
    doc2_path='data/documents/doc_0002',
    model_path='checkpoints/tif-contrast-best.pth',
    verbose=True
)

print(f"Similarity: {result['similarity']:.4f}")
print(f"Prediction: {result['prediction']}")
```

### Batch Processing

```python
from tif_contrast.inference import batch_inference

# Process multiple document pairs
results = batch_inference(
    document_pairs='data/test_pairs.csv',
    model_path='checkpoints/tif-contrast-best.pth',
    output_file='results/similarity_scores.csv',
    batch_size=16
)
```

### Command Line Inference

```bash
# Single comparison
python -m tif_contrast.inference \
    --mode single \
    --model_path checkpoints/tif-contrast-best.pth \
    --doc1 data/documents/doc_0001 \
    --doc2 data/documents/doc_0002

# Batch processing
python -m tif_contrast.inference \
    --mode batch \
    --model_path checkpoints/tif-contrast-best.pth \
    --pairs_file data/test_pairs.csv \
    --output_file results/predictions.csv
```


### Generate Training Pairs

```bash
python scripts/generate_triplets.py \
    --data_dir data/processed \
    --output_file data/train/pairs.csv \
    --ratio 1.0  # positive:negative ratio
```

### Download SciOL Dataset

```bash
python scripts/download_sciol.py \
    --output_dir data/sciol
```


## 🔧 Training

### Basic Training

```bash
python train.py \
    --config configs/tif_contrast.yaml \
    --data_dir data/train \
    --val_dir data/val \
    --output_dir checkpoints/tif_contrast \
    --batch_size 32 \
    --epochs 400 \
    --lr 2e-5 \
    --num_gpus 2
```

### Advanced Options

```bash
python train.py \
    --config configs/tif_contrast.yaml \
    --data_dir data/train \
    --val_dir data/val \
    --output_dir checkpoints/experiment_1 \
    --batch_size 32 \
    --epochs 400 \
    --lr 2e-5 \
    --lambda1 0.6 \          # Pairwise loss weight
    --lambda2 0.4 \          # Triplet loss weight
    --num_gpus 2 \
    --use_wandb \            # Enable W&B logging
    --resume checkpoints/checkpoint_epoch_100.pth  # Resume training
```

##  Evaluation

### Similarity Detection

```bash
python evaluate.py \
    --model_path checkpoints/tif-contrast-best.pth \
    --test_dir data/test \
    --output_file results/eval_results.json \
    --threshold 0.5
```

**Output metrics:**
- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix

### Cross-Modal Retrieval

```bash
python evaluate_retrieval.py \
    --model_path checkpoints/tif-contrast-best.pth \
    --dataset sciol \
    --metrics recall@1 recall@5 recall@10 \
    --output_file results/retrieval_results.json
```

### Ablation Studies

Test model variants:

```bash
# Without formula modality
python evaluate.py \
    --model_path checkpoints/tif-contrast-best.pth \
    --test_dir data/test \
    --ablation without_formula

# Without triplet loss
python evaluate.py \
    --model_path checkpoints/tif-contrast-best.pth \
    --test_dir data/test \
    --ablation without_triplet

# Without adaptive fusion
python evaluate.py \
    --model_path checkpoints/tif-contrast-best.pth \
    --test_dir data/test \
    --ablation without_adaptive
```

---

##  Results

### Similarity Detection Performance

**Self-Collected Dataset (12,000 samples)**

| Method | Modalities | Precision | Recall | F1 | Accuracy |
|--------|-----------|-----------|--------|-----|----------|
| TF-IDF | T | 41.3 | 38.7 | 39.9 | 40.8 |
| SciBERT | T | 53.6 | 50.9 | 52.2 | **53.1** |
| ViT-B | I | 47.8 | 45.2 | 46.5 | 47.1 |
| CLIP | T+I | 64.5 | 61.2 | 62.8 | 63.4 |
| BLIP-2 | T+I | 68.9 | 65.7 | 67.3 | 68.1 |
| BioMed-CLIP | T+I | 71.6 | 68.9 | 70.2 | **71.0** |
| **TIF-Contrast** | **T+I+F** | **77.3** | **75.4** | **76.6** | **77.8** |

**Key Improvements:**
- **+24.7%** vs. SciBERT (text-only)
- **+6.8%** vs. BioMed-CLIP (best bimodal)
- **+30.7%** vs. ViT-B (image-only)

### Cross-Modal Retrieval (SciOL)

| Model | Text→Image<br>R@1 | Text→Image<br>R@10 | Image→Text<br>R@1 | Image→Text<br>R@10 |
|-------|---------|----------|---------|----------|
| CLIP | 1.4 | 3.1 | 1.1 | 2.3 |
| OpenCLIP | 2.5 | 6.0 | 2.2 | 5.5 |
| BLIP-2 | 4.8 | 12.8 | 4.5 | 11.9 |
| **TIF-Contrast** | **5.7** | **13.8** | **5.6** | **12.6** |

### Ablation Study Results

| Variant | Accuracy | ΔAcc | Impact |
|---------|----------|------|--------|
| Full Model | **92.5** | - | - |
| w/o Formula | 88.4 | -4.1 | Moderate |
| w/o Image | 86.2 | -6.3 | High |
| w/o Text | 72.1 | **-20.4** | Critical |
| w/o Triplet Loss | 89.8 | -2.7 | Moderate |
| w/o Adaptive Fusion | 90.5 | -2.0 | Low |

**Key Findings:**
- Text is the most critical modality (-20.4% when removed)
- All three modalities contribute meaningfully
- Triplet alignment and adaptive fusion both improve performance

---

### Core Components

1. **Multi-Modal Encoders**
   - **SciBERT**: Domain-adapted BERT for scientific text
   - **OpenCLIP-ViT**: Vision transformer for scientific figures
   - **YOLOv11**: Core region detection in images
   - **MathBERT**: Structure-aware encoder for formulas

2. **Triplet-Aligned Embedding**
   - Projects all modalities into unified 512-dim space
   - Enforces semantic consistency via contrastive learning
   - InfoNCE loss for pairwise alignment

3. **Adaptive Attention Fusion**
   - Learns dynamic weights for each modality
   - Modality dropout (10%) for robustness
   - Handles variable modal distributions

4. **Loss Function**
   ```
   L_total = λ₁ · L_pair + λ₂ · L_triplet
   
   L_pair = (L_TI + L_TF + L_IF) / 3
   L_triplet = InfoNCE(fused_embeddings)
   ```


##  Project Structure

```
TIF-Contrast/
├── configs/                    # Configuration files
│   └── tif_contrast.yaml      # Main config
├── data/                      # Dataset directory
│   ├── train/
│   ├── val/
│   └── test/
├── tif_contrast/              # Main package
│   ├── __init__.py
│   ├── models/                # Model implementations
│   │   ├── encoders.py        # Text/Image/Formula encoders
│   │   ├── fusion.py          # Adaptive attention fusion
│   │   └── tif_model.py       # Main TIF-Contrast model
│   ├── losses/                # Loss functions
│   │   ├── contrastive.py     # InfoNCE loss
│   │   └── triplet.py         # Triplet alignment loss
│   ├── data/                  # Data processing
│   │   ├── dataset.py         # Dataset classes
│   │   └── preprocessing.py   # Preprocessing utilities
│   ├── utils/                 # Utilities
│   │   ├── metrics.py         # Evaluation metrics
│   │   
│   └── inference.py           # Inference pipeline
├── scripts/                   # Utility scripts
│   ├── preprocess_dataset.py  # PDF preprocessing
│   ├── generate_triplets.py   # Generate training pairs
│   └── download_weights.sh    # Download pre-trained weights
├── train.py                   # Training script
├── evaluate.py                # Evaluation script
├── evaluate_retrieval.py      # Retrieval evaluation
├── requirements.txt           # Dependencies
└── README.md                  # This file
```

---

##  Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.

---

##  Contact

- **Li Xue** - lixue@gcu.edu.cn
- **Project**: https://github.com/yourusername/TIF-Contrast
- **Issues**: https://github.com/yourusername/TIF-Contrast/issues

---

##  Acknowledgments

This work builds upon several excellent projects:

- [SciBERT](https://github.com/allenai/scibert) - Scientific text encoding
- [OpenCLIP](https://github.com/mlfoundations/open_clip) - Vision-language alignment
- [MathBERT](https://github.com/tbs17/MathBERT) - Mathematical formula understanding
- [Ultralytics YOLOv11](https://github.com/ultralytics/ultralytics) - Object detection
- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) - Formula extraction
- [SciOL Dataset](https://huggingface.co/datasets/boschresearch/SciOL) - Evaluation benchmark

---

##  Citation

If you find this work useful, please cite:

```bibtex
@article{li2025tifcontrast,
  title={Similarity Detection for Academic Documents via Triplet-aligned 
         Text-Image-Formula Embeddings and Contrastive Learning},
  author={Li, Xue and Liu, Ziyuan and Wu, Junyao and Lin, Yangzhao and Huang, Yingyuan},
  journal={arXiv preprint arXiv:2402.xxxxx},
  year={2025}
}
```
