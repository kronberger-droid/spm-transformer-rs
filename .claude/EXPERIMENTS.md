# Training Experiments Log

## Experiment Naming Convention

`YYYYMMDD_HHMMSS` or `YYYYMMDD_HHMMSS_jSLURM_JOB_ID`

---

## Baseline Experiments (Before Fixes)

### Experiment: Rust ViT Baseline (Pre-fixes)
**Date**: 2025-12-01 (approximate)
**Location**: `checkpoints/` (old runs)

**Configuration**:
- Model: ScanLineEncoder (ViT)
- Layers: 4
- d_model: 256
- Heads: 8
- LR: 1e-3 (constant, no scheduling)
- Batch size: 32
- Epochs: 50
- Class weights: ❌ NOT used
- Normalization: ❌ None
- Num classes: 6 ❌ (should be 4)

**Results**:
```
Train accuracy: ~36% (epoch 50)
Valid accuracy: ~36% (epoch 50)
Train loss: ~1.60
Valid loss: ~1.60
```

**Analysis**:
- No overfitting (train ≈ valid)
- Model appears stuck/plateaued
- Close to random guessing for 4 classes (25%)
- Multiple critical issues identified

**Issues**:
1. Class weights not used → imbalanced learning
2. No normalization → unstable gradients
3. LR too high → overshooting
4. Wrong num_classes → incorrect output layer
5. Only 4 layers → less capacity than Python (6 layers)

---

### Experiment: Python ViT Reference
**Date**: 2025-11 (approximate)
**Location**: `~/Programming/python/transformer-line-classification/`

**Configuration**:
- Model: Vision Transformer
- Layers: 6
- d_model: 256
- Heads: 8
- FFN dims: 1024
- LR: 1e-4 (warmup + ReduceLROnPlateau)
- Batch size: 32
- Epochs: 54 (early stopping)
- Class weights: ✅ [2.845, 0.569, 1.478, 0.823]
- Normalization: ✅ Per-scanline (mean=0, std=1)
- Augmentation: ✅ Horizontal flip, Gaussian noise, line dropout

**Results**:
```
Best validation accuracy: 39.23% (epoch 39)
Best validation AUROC: 0.6633
Training stopped: epoch 54 (early stopping)
```

**Analysis**:
- Still significantly underperforms target (39% vs 85%)
- Proper training setup but wrong architecture paradigm
- ViT approach may not capture sequential nature of STM scanning

---

## Experiments with Fixes (Phase 1)

### Experiment: TBD - First test with all fixes
**Date**: Pending
**Status**: Not yet run

**Configuration** (Planned):
- Model: ScanLineEncoder (ViT)
- Layers: 6 ✅ (increased from 4)
- d_model: 256
- Heads: 8
- LR: 1e-4 ✅ (lowered from 1e-3)
- LR Schedule: ✅ Linear warmup (1e-6 → 1e-4 over 5 epochs)
- Batch size: 32
- Epochs: 10-20 (initial test)
- Class weights: ✅ Computed from data
- Normalization: ✅ Per-scanline (mean=0, std=1)
- Num classes: ✅ 4 (inferred from data)

**Expected Results**:
- Validation accuracy: 50-60% (if fixes work)
- Training should be more stable
- Should see improvement over baseline

**Success Criteria**:
- Accuracy > 45% → fixes are helping
- Accuracy > 55% → proceed with incremental improvements
- Accuracy < 45% → need debugging

---

## Experiment Template

### Experiment: [Name]
**Date**: YYYY-MM-DD
**Checkpoint**: `checkpoints/[id]/`
**Status**: [Pending / Running / Complete]

**Configuration**:
- Model:
- Layers:
- d_model:
- Heads:
- LR:
- LR Schedule:
- Batch size:
- Epochs:
- Class weights:
- Normalization:
- Other:

**Results**:
```
[Paste analyzer output or key metrics]
```

**Analysis**:
- [What worked well]
- [What didn't work]
- [Unexpected findings]

**Next Steps**:
- [Planned follow-up experiments]

---

## Ablation Studies (Future)

### Planned Ablations

1. **Class Weights Impact**
   - Run with and without class weights
   - Expected: Significant impact on minority classes

2. **Normalization Impact**
   - Run with and without per-scanline normalization
   - Expected: Affects training stability

3. **Learning Rate**
   - Test: 1e-5, 1e-4, 5e-4, 1e-3
   - Expected: 1e-4 optimal

4. **Number of Layers**
   - Test: 4, 6, 8, 12
   - Expected: Diminishing returns after 6-8

5. **Model Size**
   - Test d_model: 128, 256, 512
   - Expected: 256 optimal for dataset size

---

## Notes

- Always use `cargo run --release --bin analyze -- --checkpoints-path checkpoints/[id]/` to analyze results
- Compare train vs valid metrics to detect overfitting
- Track best epoch, not just final epoch
- Save config.json with each checkpoint for reproducibility
