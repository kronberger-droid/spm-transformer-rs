# Reference: Target Paper & Benchmarks

## Target Paper: Gordon et al. 2020

**Title**: "Embedding Human Heuristics in Machine Learning for Probe Microscopy"

**Target Performance**:
- **Mean AUROC: 0.960**
- **Balanced Accuracy: 0.847**

### Architecture: LRCN (CNN + LSTM)

**Best Model Configuration**:
- **Window size**: W=20 scan lines (rolling window over 128-line images)
- **CNN**: VGG-like architecture
  - Filters: 32, 64, 128
  - Kernels: 3×3
  - Max pooling after each conv block
- **LSTM**: 256 hidden units for temporal context
- **Classes**: 4 tip states

### Classes (4 total)

1. **Individual atoms** - Highest/atomic resolution
2. **Asymmetries/Dimers** - Combined category
3. **Rows** - Lower resolution
4. **Generic defects** - Tip quality issues

### Key Innovation

Processes consecutive scan lines in windows, using LSTM to learn temporal evolution. This mimics human operator behavior of assessing tips by observing line-by-line scan evolution.

---

## Reference Implementations

### Python Vision Transformer (Baseline)
**Location**: `~/Programming/python/transformer-line-classification/`

**Architecture**:
- Vision Transformer (ViT)
- 6 transformer encoder layers
- 8 attention heads, 256 embedding dims, 1024 FFN dims
- 0.1 dropout, pre-normalization

**Training Setup**:
- Optimizer: AdamW (lr=1e-4, weight_decay=0.05)
- LR Schedule: 5-epoch warmup + ReduceLROnPlateau
- Class weights: [2.845, 0.569, 1.478, 0.823]
- Per-scanline normalization (mean=0, std=1)
- Augmentation: horizontal flip, Gaussian noise, line dropout
- Batch size: 32

**Performance**:
- **Validation Accuracy: 39.23%** (epoch 39)
- **Validation AUROC: 0.6633**
- Training stopped at epoch 54 (early stopping)

**Status**: Significantly underperforms target (39% vs 85% balanced accuracy)

---

## Dataset Details

**Format**: 128×128 STM images
- 128 scanlines × 128 pixels per line
- Stored as NPZ files (pre-augmented)

**Class Distribution** (severe imbalance):
- Class 0 (atoms): 8.79%
- Class 1 (asymmetries/dimers): 43.90% ← majority class
- Class 2 (rows): 16.91%
- Class 3 (defects): 30.39%

**Computed Class Weights** (inverse frequency):
- [2.845, 0.569, 1.478, 0.823]

---

## Performance Gap Analysis

### Current vs Target

| Metric | Python ViT | Rust ViT (before fixes) | Target (LRCN) | Gap |
|--------|-----------|------------------------|---------------|-----|
| Accuracy | 39.23% | ~36% | 84.7% | ~45-49 pp |
| AUROC | 0.6633 | ~0.66 | 0.960 | ~0.30 |
| Architecture | Global attention | Global attention | Sequential (CNN+LSTM) | Paradigm mismatch |

### Why the Gap Exists

**Architectural paradigm mismatch**:
- **Current (ViT)**: Global attention over all 128 lines simultaneously
- **Target (LRCN)**: Rolling window of W=20 lines with sequential processing
- **Missing**: Temporal/sequential modeling of scan line evolution
- **Human heuristic**: Not captured - ViT doesn't mimic line-by-line assessment

---

## Next Performance Milestones

1. **Phase 1**: Fix baseline ViT → expect ~50-60% accuracy
2. **Phase 2**: Add incremental improvements → expect ~60-70% accuracy
3. **Phase 3**: Sequential transformer → target ≥85% accuracy, ≥0.96 AUROC
